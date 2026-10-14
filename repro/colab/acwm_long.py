"""DINO-WM on ACWM-Phys: long training of the predictor and the decoder on cached DINOv2 tokens (one big GPU, e.g. Colab G4).

The same model, losses, optimizers, learning rates and batch size as train.py (see the slim's config); only the frozen
encoder runs once ahead of time. One process per environment does, in every iteration,
  - a predictor step on a batch of training slices (as predictor_long.py: z loss, AdamW for the predictor and the
    action/proprio encoders), until the predictor has made --pred_steps optimizer steps in total;
  - a decoder step on batch_size * (num_hist + num_pred) training frames drawn uniformly (train.py decodes the frames of
    its slices; loss = MSE to the frame in [-1, 1] + 0.25 * VQ term, which is 0 with quantize: false), Adam, until
    --dec_steps in total or until the validation reconstruction loss has not improved by --dec_min_delta (relative)
    for --dec_patience evaluations.
Step counts include the steps of the run the weights come from (epoch x steps per epoch of train.py's loader).

Weights: the slim's (trained run), or --init_pred <predictor_long weights file> for the predictor, or --from_scratch.
Every --val_every steps: predictor validation loss on all validation slices, decoder validation loss on up to
--dec_val_frames validation frames; the best weights so far are kept as <name>_pred_best.pt / <name>_dec_best.pt.
Every --state_every steps <name>_state.pt (weights, optimizers, counters, RNG) is written; --resume continues from it.
<out>/<name>.override.json (optional, read at the start and at every validation) can change pred_steps, dec_steps,
dec_patience and dec_min_delta of a running job.

usage (cwd = repo root):
  python repro/colab/acwm_long.py <slim.pt> <ACWM-Phys root> <out dir> --pred_steps 933334 --dec_steps 933334
         [--init_pred weights.pt] [--from_scratch] [--resume] [--tf32] [--amp] [--tokens_cpu] [--images_gpu]
         [--val_every 20000] [--state_every 20000] [--max_episodes 0] [--tag NAME]
"""
import os, sys, json, time, argparse, itertools
import torch
import torch.nn.functional as F
import hydra
from omegaconf import OmegaConf
from torchvision import transforms

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor_long import CachedTokens, slices, remap, n_obs_of, save_atomic
from models.visual_world_model import VWorldModel


class Frames(torch.utils.data.Dataset):
    def __init__(self, eps):
        self.eps = eps

    def __len__(self):
        return len(self.eps)

    def __getitem__(self, k):
        d, i = self.eps[k]
        return k, d.get_frames(i, range(n_obs_of(d, i)))[0]["visual"]


@torch.no_grad()
def encode_all(eps, encoder, resize, dev, workers, tdev, keep_images):
    """-> tokens (frames, patches, dim) float16 on tdev, first-frame offset of every episode, images uint8 (frames, 3, 224, 224) on CPU"""
    n_obs = [n_obs_of(d, i) for d, i in eps]
    offs = torch.tensor([0] + list(itertools.accumulate(n_obs))[:-1])
    probe = encoder(resize(torch.zeros(1, 3, 224, 224, device=dev)))
    tok = torch.empty(sum(n_obs), probe.shape[1], probe.shape[2], dtype=torch.float16, device=tdev)
    img = torch.empty(sum(n_obs), 3, 224, 224, dtype=torch.uint8) if keep_images else None
    loader = torch.utils.data.DataLoader(Frames(eps), batch_size=None, shuffle=False, num_workers=workers)
    t0 = time.time()
    for n, (k, vis) in enumerate(loader):
        if keep_images:
            img[offs[k] : offs[k] + len(vis)] = ((vis.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        vis = vis.to(dev)
        tok[offs[k] : offs[k] + len(vis)] = torch.cat([encoder(resize(vis[s : s + 256])) for s in range(0, len(vis), 256)]).half().to(tdev)
        if n % 500 == 0:
            print(f"  encoded {n}/{len(eps)} episodes {time.time() - t0:.0f}s", flush=True)
    return tok, offs, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slim"); ap.add_argument("data_root"); ap.add_argument("out")
    ap.add_argument("--pred_steps", type=int, required=True, help="total predictor optimizer steps (including the loaded run's)")
    ap.add_argument("--dec_steps", type=int, required=True, help="maximum total decoder steps (including the loaded run's)")
    ap.add_argument("--dec_patience", type=int, default=10)
    ap.add_argument("--dec_min_delta", type=float, default=0.005)
    ap.add_argument("--dec_val_frames", type=int, default=4096)
    ap.add_argument("--init_pred", default="")
    ap.add_argument("--from_scratch", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--val_every", type=int, default=20000)
    ap.add_argument("--state_every", type=int, default=20000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast for the forward passes (losses in float32)")
    ap.add_argument("--tokens_cpu", action="store_true")
    ap.add_argument("--images_gpu", action="store_true", help="keep the uint8 frames for the decoder on the GPU")
    ap.add_argument("--max_episodes", type=int, default=0, help="smoke test")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(args.seed)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and dev.startswith("cuda"))
    tdev = "cpu" if args.tokens_cpu else dev
    S = torch.load(args.slim, map_location="cpu")
    cfg = OmegaConf.create(S["cfg"])
    name = S["model_name"] + (f"_{args.tag}" if args.tag else "")
    H, P, f = cfg.num_hist, cfg.num_pred, cfg.frameskip
    B = cfg.training.batch_size
    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, f"{name}.jsonl"), "a")

    def emit(rec):
        rec = {"model": name, "time": time.strftime("%F %T"), **rec}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)

    # ---- data: the env's own dataset target, same split and action statistics as the slim
    ds = OmegaConf.create(OmegaConf.to_container(cfg.env.dataset))
    ds.data_path = remap(ds.data_path, args.data_root)
    _, trajs = hydra.utils.call(ds, num_hist=H, num_pred=P, frameskip=f)
    dset = trajs["train"].dataset
    assert torch.allclose(dset.action_mean, S["action_mean"]) and torch.allclose(dset.action_std, S["action_std"]), "action statistics differ"
    by_name = {n: i for i, n in enumerate(dset.video_names)}
    tr_idx, va_idx = [by_name[n] for n in S["train_names"]], [by_name[n] for n in S["val_names"]]
    assert list(trajs["train"].indices) == tr_idx and list(trajs["valid"].indices) == va_idx, "episode split differs from the slim"
    if args.max_episodes:
        tr_idx, va_idx = tr_idx[: args.max_episodes], va_idx[: max(args.max_episodes // 4, 2)]
    sets = {"train": [(dset, i) for i in tr_idx], "valid": [(dset, i) for i in va_idx]}

    encoder = hydra.utils.instantiate(cfg.encoder).to(dev).eval()
    resize = transforms.Resize(cfg.img_size // 16 * encoder.patch_size)  # VWorldModel: decoder_scale 16
    data, t0 = {}, time.time()
    for split, eps in sets.items():
        tok, offs, img = encode_all(eps, encoder, resize, dev, args.workers, tdev, keep_images=True)
        acts = [d.actions[i] for d, i in eps]
        a_offs = torch.tensor([0] + list(itertools.accumulate(len(a) for a in acts))[:-1])
        fr, ac = slices(eps, offs, a_offs, f, H + P)
        if args.images_gpu:
            img = img.to(dev)
        elif dev.startswith("cuda"):
            img = img.pin_memory()
        data[split] = {"tok": tok, "act": torch.cat(acts).to(dev), "fr": fr.to(tdev), "ac": ac.to(dev), "img": img}
        emit({"event": "encoded", "split": split, "episodes": len(eps), "frames": len(tok), "slices": len(fr),
              "tokens_gb": round(tok.numel() * 2 / 2**30, 1), "images_gb": round(img.numel() / 2**30, 1)})
    emb_dim, patch_size = encoder.emb_dim, encoder.patch_size
    del encoder
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()
    emit({"event": "encoding_done", "seconds": round(time.time() - t0)})

    # ---- models: train.py's, the encoder replaced by the cached tokens
    if not dev.startswith("cuda"):  # models/vit.py puts its attention mask on 'cuda' in the constructor; keep it on the CPU for a smoke test
        import models.vit as vit

        class _Stay(torch.Tensor):
            def to(self, *a, **k):
                return self.as_subclass(torch.Tensor)

        _gen = vit.generate_mask_matrix
        vit.generate_mask_matrix = lambda npatch, nwindow: _gen(npatch, nwindow).as_subclass(_Stay)
    A = dset.action_dim * f
    proprio_encoder = hydra.utils.instantiate(cfg.proprio_encoder, in_chans=1, emb_dim=cfg.proprio_emb_dim)
    action_encoder = hydra.utils.instantiate(cfg.action_encoder, in_chans=A, emb_dim=cfg.action_emb_dim)
    predictor = hydra.utils.instantiate(cfg.predictor, num_patches=(cfg.img_size // 16) ** 2, num_frames=H,
                                        dim=emb_dim + (cfg.proprio_emb_dim * cfg.num_proprio_repeat + cfg.action_emb_dim * cfg.num_action_repeat) * cfg.concat_dim)
    decoder = hydra.utils.instantiate(cfg.decoder, emb_dim=emb_dim)
    init = torch.load(args.init_pred, map_location="cpu") if args.init_pred else None
    steps_per_epoch = (len(data["train"]["fr"]) + B - 1) // B
    if args.from_scratch:
        pred_step = dec_step = 0
    else:
        Wp = init if init else S
        predictor.load_state_dict(Wp["predictor"]); action_encoder.load_state_dict(Wp["action_encoder"]); proprio_encoder.load_state_dict(Wp["proprio_encoder"])
        decoder.load_state_dict(S["decoder"])
        pred_step, dec_step = int(Wp["epoch"]) * steps_per_epoch, int(S["epoch"]) * steps_per_epoch
    model = VWorldModel(image_size=cfg.img_size, num_hist=H, num_pred=P, encoder=CachedTokens(emb_dim, patch_size), proprio_encoder=proprio_encoder,
                        action_encoder=action_encoder, decoder=None, predictor=predictor, proprio_dim=cfg.proprio_emb_dim, action_dim=cfg.action_emb_dim,
                        concat_dim=cfg.concat_dim, num_action_repeat=cfg.num_action_repeat, num_proprio_repeat=cfg.num_proprio_repeat,
                        train_encoder=False, train_predictor=True, train_decoder=False).to(dev)
    decoder.to(dev)
    opt_p = torch.optim.AdamW(predictor.parameters(), lr=cfg.training.predictor_lr)
    opt_a = torch.optim.AdamW(itertools.chain(action_encoder.parameters(), proprio_encoder.parameters()), lr=cfg.training.action_encoder_lr)
    opt_d = torch.optim.Adam(decoder.parameters(), lr=cfg.training.decoder_lr)  # train.py: Adam for the decoder, AdamW for the predictor
    if not args.from_scratch:
        if not init and S.get("predictor_optimizer") is not None:
            opt_p.load_state_dict(S["predictor_optimizer"])
        if S.get("decoder_optimizer") is not None:
            opt_d.load_state_dict(S["decoder_optimizer"])

    # ---- batches
    def pred_batch(split, idx):
        D = data[split]
        obs = {"visual": D["tok"][D["fr"][idx]].to(dev).float(), "proprio": torch.zeros(len(idx), H + P, 1, device=dev)}
        return obs, D["act"][D["ac"][idx.to(dev)]].reshape(len(idx), H + P, A)

    def dec_batch(split, idx):
        D = data[split]
        z = D["tok"][idx.to(tdev)].to(dev, non_blocking=True).float()[:, None]  # (b, 1, patches, dim)
        x = D["img"][idx.to(D["img"].device)].to(dev, non_blocking=True).float() / 127.5 - 1.0
        return z, x

    def dec_loss(z, x):
        with amp():
            out, diff = decoder(z)
        return F.mse_loss(out.float(), x) + 0.25 * diff.float().mean()

    n_val_frames = len(data["valid"]["tok"])
    dec_val_idx = torch.linspace(0, n_val_frames - 1, min(args.dec_val_frames, n_val_frames)).long()

    @torch.no_grad()
    def evaluate():
        model.eval(); decoder.eval()
        n = len(data["valid"]["fr"])
        tot = {"z_visual_loss": 0.0, "z_loss": 0.0}
        for s in range(0, n, 256):
            idx = torch.arange(s, min(s + 256, n), device=tdev)
            with amp():
                _, _, _, _, comp = model(*pred_batch("valid", idx))
            for k in tot:
                tot[k] += comp[k].item() * len(idx)
        rec = {"val_z_visual_loss": tot["z_visual_loss"] / n, "val_z_loss": tot["z_loss"] / n}
        dl = 0.0
        for s in range(0, len(dec_val_idx), 128):
            idx = dec_val_idx[s : s + 128]
            dl += dec_loss(*dec_batch("valid", idx)).item() * len(idx)
        rec["val_dec_recon_loss"] = dl / len(dec_val_idx)
        model.train(); decoder.train()
        return rec

    def pred_weights():
        return {"model_name": name, "epoch": pred_step / steps_per_epoch, "steps": pred_step, "predictor": predictor.state_dict(),
                "action_encoder": action_encoder.state_dict(), "proprio_encoder": proprio_encoder.state_dict()}

    def dec_weights():
        return {"model_name": name, "epoch": dec_step / steps_per_epoch, "steps": dec_step, "decoder": decoder.state_dict()}

    state_path = os.path.join(args.out, f"{name}_state.pt")
    best = {"pred": [float("inf"), None], "dec": [float("inf"), None], "dec_bad_evals": 0, "dec_stopped": False}
    if args.resume and os.path.exists(state_path):
        R = torch.load(state_path, map_location="cpu")
        predictor.load_state_dict(R["predictor"]); action_encoder.load_state_dict(R["action_encoder"]); proprio_encoder.load_state_dict(R["proprio_encoder"])
        decoder.load_state_dict(R["decoder"])
        opt_p.load_state_dict(R["opt_p"]); opt_a.load_state_dict(R["opt_a"]); opt_d.load_state_dict(R["opt_d"])
        pred_step, dec_step, best = R["pred_step"], R["dec_step"], R["best"]
        torch.set_rng_state(R["rng_cpu"])
        if dev.startswith("cuda") and R.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(R["rng_cuda"])
        emit({"event": "resumed", "pred_step": pred_step, "dec_step": dec_step, "best": best})
    else:
        rec = evaluate()
        emit({"event": "start", "pred_step": pred_step, "dec_step": dec_step, "steps_per_epoch": steps_per_epoch,
              "note": "random initialization" if args.from_scratch else ("predictor from " + args.init_pred if init else "slim weights"),
              "slim_val_z_visual_loss": S.get("val_z_visual_loss"), "slim_val_decoder_recon_loss": S.get("val_decoder_recon_loss"), **rec})
        if not args.from_scratch:  # the loaded weights are the best so far until something beats them
            best["pred"], best["dec"] = [rec["val_z_visual_loss"], pred_step], [rec["val_dec_recon_loss"], dec_step]
            save_atomic(pred_weights(), os.path.join(args.out, f"{name}_pred_best.pt"))
            save_atomic(dec_weights(), os.path.join(args.out, f"{name}_dec_best.pt"))

    def save_state():
        save_atomic({**pred_weights(), "decoder": decoder.state_dict(), "opt_p": opt_p.state_dict(), "opt_a": opt_a.state_dict(),
                     "opt_d": opt_d.state_dict(), "pred_step": pred_step, "dec_step": dec_step, "best": best,
                     "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state() if dev.startswith("cuda") else None}, state_path)

    override_path, override_seen = os.path.join(args.out, f"{name}.override.json"), [None]

    def apply_override():
        if not os.path.exists(override_path) or os.path.getmtime(override_path) == override_seen[0]:
            return
        override_seen[0] = os.path.getmtime(override_path)
        try:
            o = json.load(open(override_path))
        except Exception as e:
            emit({"event": "override_unreadable", "error": repr(e)[:200]}); return
        for k in ("pred_steps", "dec_steps", "dec_patience", "dec_min_delta"):
            if k in o:
                setattr(args, k, type(getattr(args, k))(o[k]))
        emit({"event": "override", **{k: getattr(args, k) for k in ("pred_steps", "dec_steps", "dec_patience", "dec_min_delta")}})

    apply_override()

    # ---- training
    n_tr, n_tr_frames = len(data["train"]["fr"]), len(data["train"]["tok"])
    model.train(); decoder.train()
    perm, pos = torch.randperm(n_tr, device=tdev), 0
    t0, it0, run = time.time(), 0, {"z": [], "dec": []}
    while pred_step < args.pred_steps or (dec_step < args.dec_steps and not best["dec_stopped"]):
        if pred_step < args.pred_steps:
            if pos >= n_tr:
                perm, pos = torch.randperm(n_tr, device=tdev), 0
            idx = perm[pos : pos + B]; pos += B
            with amp():
                _, _, _, loss, comp = model(*pred_batch("train", idx))
            opt_p.zero_grad(set_to_none=True); opt_a.zero_grad(set_to_none=True)
            loss.backward()
            opt_p.step(); opt_a.step()
            pred_step += 1
            if pred_step % 50 == 0:
                run["z"].append(comp["z_visual_loss"].item())
        if dec_step < args.dec_steps and not best["dec_stopped"]:
            idx = torch.randint(n_tr_frames, (B * (H + P),))
            loss_d = dec_loss(*dec_batch("train", idx))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
            dec_step += 1
            if dec_step % 50 == 0:
                run["dec"].append(loss_d.item())
        it0 += 1
        last = not (pred_step < args.pred_steps or (dec_step < args.dec_steps and not best["dec_stopped"]))
        if it0 % args.val_every == 0 or last:
            apply_override()
            rec = evaluate()
            if rec["val_z_visual_loss"] < best["pred"][0]:
                best["pred"] = [rec["val_z_visual_loss"], pred_step]
                save_atomic(pred_weights(), os.path.join(args.out, f"{name}_pred_best.pt"))
            if not best["dec_stopped"]:
                if rec["val_dec_recon_loss"] < best["dec"][0] * (1 - args.dec_min_delta):
                    best["dec_bad_evals"] = 0
                else:
                    best["dec_bad_evals"] += 1
                if rec["val_dec_recon_loss"] < best["dec"][0]:
                    best["dec"] = [rec["val_dec_recon_loss"], dec_step]
                    save_atomic(dec_weights(), os.path.join(args.out, f"{name}_dec_best.pt"))
                if best["dec_bad_evals"] >= args.dec_patience:
                    best["dec_stopped"] = True
            el = time.time() - t0
            emit({"event": "val", "pred_step": pred_step, "dec_step": dec_step, "pred_epoch": round(pred_step / steps_per_epoch, 2),
                  "train_z_visual_loss_sampled": sum(run["z"]) / max(len(run["z"]), 1), "train_dec_loss_sampled": sum(run["dec"]) / max(len(run["dec"]), 1),
                  **rec, "best_pred": best["pred"], "best_dec": best["dec"], "dec_bad_evals": best["dec_bad_evals"], "dec_stopped": best["dec_stopped"],
                  "it_per_s": round(it0 / el, 2), "hours": round(el / 3600, 3)})
            run = {"z": [], "dec": []}
        if it0 % args.state_every == 0 or last:
            save_state()
    save_atomic(pred_weights(), os.path.join(args.out, f"{name}_pred_last.pt"))
    save_atomic(dec_weights(), os.path.join(args.out, f"{name}_dec_last.pt"))
    emit({"event": "done", "pred_step": pred_step, "dec_step": dec_step, "best": best})


if __name__ == "__main__":
    main()
