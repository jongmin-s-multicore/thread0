"""ACWM-Phys baseline: DINO-WM 의 predictor 와 디코더를 정해진 step 수까지 학습한다 (GPU 한 장, 이슈 #4 의 계획).

인코더(DINOv2 ViT-S/14)는 동결이라 모든 프레임의 패치 토큰을 한 번 계산해 둔다 (float16). 디코더의 목표 영상도
dataset transform 을 거친 224 px 프레임을 uint8 로 한 번 만들어 둔다. 그 뒤 매 step 에
  predictor  train.py 와 같은 모델(VWorldModel, 인코더 자리에 캐시 토큰)·손실·옵티마이저(AdamW lr 5e-5, action/proprio encoder AdamW lr 5e-4)
  디코더     train.py 와 같은 손실(MSE + 0.25 VQ diff)·옵티마이저(Adam lr 3e-4). 입력은 같은 배치 창의 실제 프레임 토큰 (z.detach() 와 같다)
를 한 번씩 갱신한다. 배치는 train.py 와 같은 창(num_hist + num_pred 프레임, frameskip 간격) 24 개다.

평가(--eval_every step 마다): 검증 에피소드의 모든 창에서 z_visual_loss·z_loss, 그 창의 프레임 재구성 MSE.
  predictor 는 검증 z_visual_loss 가 가장 낮은 가중치를 <name>_pred_best.pt 로, 디코더는 재구성 MSE 가 가장 낮은 가중치를 <name>_dec_best.pt 로 둔다.
  디코더는 --dec_cap step 에서 멈추고, 그 전에도 재구성 MSE 가 직전 최저값보다 --dec_min_rel 이상 줄지 않는 평가가 --dec_patience 번 이어지면 멈춘다.
  매 평가마다 이어 학습용 <name>_state.pt (가중치, 옵티마이저 3개, step, best, RNG) 를 쓴다. 다시 실행하면 그 파일에서 이어 간다.
state 형식은 이슈 #4 의 Colab 실행(HF hanbin5/dino-wm-acwm-phys baseline_x15/*_state.pt)과 같아서 그 파일을 <out>/<name>_state.pt 로 두면 이어 학습한다.

usage (cwd = repo root, env sourced):
  python repro/colab/acwm_long.py <hydra.yaml> <out dir> --name NAME --pred_target 933334 --dec_cap 250000
         [--init <state.pt>] [--eval_every 20000] [--tf32] [--amp] [--images_cpu] [--tokens_cpu] [--max_episodes 0]
"""
import os, sys, json, time, argparse, itertools, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
from omegaconf import OmegaConf
from torchvision import transforms

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.visual_world_model import VWorldModel
from predictor_long import CachedTokens, slices


class Frames(torch.utils.data.Dataset):
    """eps: list of (dataset, episode index) -> transformed frames of the whole episode"""
    def __init__(self, eps):
        self.eps = eps

    def __len__(self):
        return len(self.eps)

    def __getitem__(self, k):
        d, i = self.eps[k]
        return k, d.get_frames(i, range(d.n_obs[i]))[0]["visual"]  # ACWMPhysDataset and ACWMPourWaterDataset: element 0 is the obs dict


def to_u8(x):
    """[-1, 1] -> uint8"""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def from_u8(x):
    return x.float() / 127.5 - 1


@torch.no_grad()
def encode_all(eps, encoder, resize, dev, workers, tdev, idev):
    """-> tokens (frames, patches, dim) float16 on tdev, images (frames, 3, H, W) uint8 on idev, first-frame offset of every episode"""
    n_obs = [d.n_obs[i] for d, i in eps]
    offs = torch.tensor([0] + list(itertools.accumulate(n_obs))[:-1])
    _, first = Frames(eps[:1])[0]
    probe = encoder(resize(first[:1].to(dev)))
    tok = torch.empty(sum(n_obs), probe.shape[1], probe.shape[2], dtype=torch.float16, device=tdev)
    img = torch.empty(sum(n_obs), *first.shape[1:], dtype=torch.uint8, device=idev)
    loader = torch.utils.data.DataLoader(Frames(eps), batch_size=None, shuffle=False, num_workers=workers)
    t0 = time.time()
    for n, (k, vis) in enumerate(loader):
        o = int(offs[k])
        img[o : o + len(vis)] = to_u8(vis).to(idev)
        vis = vis.to(dev)
        tok[o : o + len(vis)] = torch.cat([encoder(resize(vis[s : s + 256])) for s in range(0, len(vis), 256)]).half().to(tdev)
        if n % 500 == 0:
            print(f"  encoded {n}/{len(eps)} episodes {time.time() - t0:.0f}s", flush=True)
    return tok, img, offs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg"); ap.add_argument("out")
    ap.add_argument("--name", required=True)
    ap.add_argument("--pred_target", type=int, default=933334)
    ap.add_argument("--dec_cap", type=int, required=True)
    ap.add_argument("--init", default="", help="state file to start from when <out>/<name>_state.pt does not exist (weights, and optimizers/steps if present)")
    ap.add_argument("--eval_every", type=int, default=20000)
    ap.add_argument("--dec_patience", type=int, default=5)
    ap.add_argument("--dec_min_rel", type=float, default=0.005)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_episodes", type=int, default=0, help="smoke test: use only this many training episodes")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast for the predictor and decoder forward (losses in float32)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--tokens_cpu", action="store_true", help="keep the cached tokens in RAM instead of on the GPU")
    ap.add_argument("--images_cpu", action="store_true", help="keep the cached decoder targets in RAM instead of on the GPU")
    args = ap.parse_args()
    dev, name = args.device, args.name
    cfg = OmegaConf.load(args.cfg)
    H, P, f, B = cfg.num_hist, cfg.num_pred, cfg.frameskip, cfg.training.batch_size
    os.makedirs(args.out, exist_ok=True)
    state_path = os.path.join(args.out, f"{name}_state.pt")
    torch.manual_seed(args.seed)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp)
    tdev = "cpu" if args.tokens_cpu else dev
    idev = "cpu" if args.images_cpu else dev
    log = open(os.path.join(args.out, f"{name}.jsonl"), "a")

    def emit(rec):
        rec = {"model": name, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)

    # ---- data: train.py's episodes, split (seed 42) and action normalization
    _, trajs = hydra.utils.call(cfg.env.dataset, num_hist=H, num_pred=P, frameskip=f)
    base = trajs["train"].dataset
    if not hasattr(base, "n_obs"):  # Pour Water: one frame per action
        base.n_obs = [int(x) for x in base.seq_lengths]
    tr_idx, va_idx = list(trajs["train"].indices), list(trajs["valid"].indices)
    print(json.dumps([tr_idx, va_idx]), flush=True)  # the split, to compare with other runs
    if args.max_episodes:
        tr_idx, va_idx = tr_idx[: args.max_episodes], va_idx[: max(args.max_episodes // 4, 2)]
    encoder = hydra.utils.instantiate(cfg.encoder).to(dev).eval()
    resize = transforms.Resize(cfg.img_size // 16 * encoder.patch_size)  # VWorldModel.encoder_transform (decoder_scale 16)
    t0 = time.time()
    data = {}
    for split, idx in (("train", tr_idx), ("valid", va_idx)):
        eps = [(base, i) for i in idx]
        tok, img, offs = encode_all(eps, encoder, resize, dev, args.workers, tdev, idev)
        acts = [base.actions[i] for i in idx]
        a_offs = torch.tensor([0] + list(itertools.accumulate(len(a) for a in acts))[:-1])
        fr, ac = slices(eps, offs, a_offs, f, H + P)
        data[split] = (tok, img, torch.cat(acts).to(dev), fr.to(tdev), ac.to(dev))
        emit({"event": "encoded", "split": split, "episodes": len(eps), "frames": len(tok), "slices": len(fr),
              "tokens_gb": round(tok.numel() * 2 / 2**30, 1), "images_gb": round(img.numel() / 2**30, 1)})
    emb_dim, patch_size = encoder.emb_dim, encoder.patch_size
    del encoder
    torch.cuda.empty_cache()
    emit({"event": "encoding_done", "seconds": round(time.time() - t0)})

    # ---- model: train.py's, the encoder replaced by the cached tokens
    A = base.action_dim * f
    proprio_encoder = hydra.utils.instantiate(cfg.proprio_encoder, in_chans=1, emb_dim=cfg.proprio_emb_dim)
    action_encoder = hydra.utils.instantiate(cfg.action_encoder, in_chans=A, emb_dim=cfg.action_emb_dim)
    predictor = hydra.utils.instantiate(cfg.predictor, num_patches=(cfg.img_size // 16) ** 2, num_frames=H,
                                        dim=emb_dim + (cfg.proprio_emb_dim * cfg.num_proprio_repeat + cfg.action_emb_dim * cfg.num_action_repeat) * cfg.concat_dim)
    decoder = hydra.utils.instantiate(cfg.decoder, emb_dim=emb_dim)
    model = VWorldModel(image_size=cfg.img_size, num_hist=H, num_pred=P, encoder=CachedTokens(emb_dim, patch_size), proprio_encoder=proprio_encoder,
                        action_encoder=action_encoder, decoder=None, predictor=predictor, proprio_dim=cfg.proprio_emb_dim, action_dim=cfg.action_emb_dim,
                        concat_dim=cfg.concat_dim, num_action_repeat=cfg.num_action_repeat, num_proprio_repeat=cfg.num_proprio_repeat,
                        train_encoder=False, train_predictor=True, train_decoder=False).to(dev)
    decoder.to(dev)
    opt_p = torch.optim.AdamW(predictor.parameters(), lr=cfg.training.predictor_lr)
    opt_a = torch.optim.AdamW(itertools.chain(action_encoder.parameters(), proprio_encoder.parameters()), lr=cfg.training.action_encoder_lr)
    opt_d = torch.optim.Adam(decoder.parameters(), lr=cfg.training.decoder_lr)
    pred_step = dec_step = 0
    best = {"pred": [math.inf, 0], "dec": [math.inf, 0], "dec_bad_evals": 0, "dec_stopped": False}
    resumed = os.path.exists(state_path)
    src = state_path if resumed else args.init
    note = "random initialization"
    if src:
        S = torch.load(src, map_location="cpu", weights_only=False)
        predictor.load_state_dict(S["predictor"]); action_encoder.load_state_dict(S["action_encoder"]); proprio_encoder.load_state_dict(S["proprio_encoder"])
        if "decoder" in S:
            decoder.load_state_dict(S["decoder"])
        for k, o in (("opt_p", opt_p), ("opt_a", opt_a), ("opt_d", opt_d)):
            if k in S:
                o.load_state_dict(S[k])
        pred_step, dec_step = int(S.get("pred_step", 0)), int(S.get("dec_step", 0))
        if "best" in S:
            best = S["best"]
        if resumed and "rng_cpu" in S:
            torch.set_rng_state(S["rng_cpu"]); torch.cuda.set_rng_state(S["rng_cuda"])
        note = f"{'resumed from' if resumed else 'initialized from'} {src}: " + ", ".join(k for k in ("decoder", "opt_p", "opt_a", "opt_d", "best") if k in S)
        del S
    if args.compile:
        model.predictor = torch.compile(predictor, dynamic=False)
        dec_fwd = torch.compile(decoder, dynamic=False)
    else:
        dec_fwd = decoder

    def batch(split, idx):
        tok, img, acts, fr, ac = data[split]
        f_idx = fr[idx]
        obs = {"visual": tok[f_idx].to(dev, non_blocking=True).float(), "proprio": torch.zeros(len(idx), H + P, 1, device=dev)}
        return obs, acts[ac[idx.to(dev)]].reshape(len(idx), H + P, A), f_idx

    def dec_loss(z, f_idx, split):
        """z (b, t, patches, dim) tokens of the frames f_idx (b, t) -> train.py's decoder loss and its reconstruction MSE"""
        img = data[split][1]
        tgt = from_u8(img[f_idx.reshape(-1).to(img.device)].to(dev, non_blocking=True))
        with amp():
            rec, diff = dec_fwd(z.reshape(-1, 1, *z.shape[2:]))
        mse = F.mse_loss(rec.float(), tgt)
        return mse + model.decoder_latent_loss_weight * diff, mse

    @torch.no_grad()
    def evaluate(with_dec):
        model.eval(); decoder.eval()
        n = len(data["valid"][3])
        tot = {"z_visual_loss": 0.0, "z_loss": 0.0, "dec": 0.0}
        for s in range(0, n, 128):
            idx = torch.arange(s, min(s + 128, n), device=tdev)
            obs, act, f_idx = batch("valid", idx)
            with amp():
                _, _, _, _, comp = model(obs, act)
            tot["z_visual_loss"] += comp["z_visual_loss"].item() * len(idx)
            tot["z_loss"] += comp["z_loss"].item() * len(idx)
            if with_dec:
                tot["dec"] += dec_loss(obs["visual"], f_idx, "valid")[1].item() * len(idx)
        model.train(); decoder.train()
        r = {"val_z_visual_loss": tot["z_visual_loss"] / n, "val_z_loss": tot["z_loss"] / n}
        if with_dec:
            r["val_dec_recon_loss"] = tot["dec"] / n
        return r

    def save(path, **kw):
        torch.save({"model_name": name, "epoch": pred_step / steps_per_epoch, "steps": pred_step, **kw}, path + ".tmp")
        os.replace(path + ".tmp", path)

    def save_state():
        save(state_path, predictor=predictor.state_dict(), action_encoder=action_encoder.state_dict(), proprio_encoder=proprio_encoder.state_dict(),
             decoder=decoder.state_dict(), opt_p=opt_p.state_dict(), opt_a=opt_a.state_dict(), opt_d=opt_d.state_dict(),
             pred_step=pred_step, dec_step=dec_step, best=best, rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state())

    n_tr = len(data["train"][3])
    steps_per_epoch = (n_tr + B - 1) // B
    dec_on = lambda: not best["dec_stopped"] and dec_step < args.dec_cap
    if resumed:
        emit({"event": "resumed", "pred_step": pred_step, "dec_step": dec_step, "best": best, "note": note})
    else:
        emit({"event": "start", "pred_step": pred_step, "dec_step": dec_step, "steps_per_epoch": steps_per_epoch, "note": note, **evaluate(True)})
    model.train(); decoder.train()
    t0, t_last, steps_run, n_last = time.time(), time.time(), 0, 0
    run = {"z": [], "dec": []}
    while pred_step < args.pred_target:
        perm = torch.randperm(n_tr, device=tdev)
        for s in range(0, n_tr, B):
            obs, act, f_idx = batch("train", perm[s : s + B])
            with amp():
                _, _, _, loss, comp = model(obs, act)
            opt_p.zero_grad(set_to_none=True); opt_a.zero_grad(set_to_none=True)
            loss.backward()
            opt_p.step(); opt_a.step()
            pred_step += 1
            steps_run += 1
            if dec_on():
                dl, _ = dec_loss(obs["visual"], f_idx, "train")
                opt_d.zero_grad(set_to_none=True)
                dl.backward()
                opt_d.step()
                dec_step += 1
                if pred_step % 50 == 0:
                    run["dec"].append(dl.item())
            if pred_step % 50 == 0:
                run["z"].append(comp["z_visual_loss"].item())
            if pred_step % 2000 == 0:
                print(f"  step {pred_step} dec {dec_step} {(steps_run - n_last) / (time.time() - t_last):.2f} it/s", flush=True)
                t_last, n_last = time.time(), steps_run
            if pred_step % args.eval_every == 0 or pred_step >= args.pred_target:
                with_dec = not best["dec_stopped"]  # the decoder trained since the last evaluation
                r = evaluate(with_dec)
                if r["val_z_visual_loss"] < best["pred"][0]:
                    best["pred"] = [r["val_z_visual_loss"], pred_step]
                    save(os.path.join(args.out, f"{name}_pred_best.pt"), predictor=predictor.state_dict(), action_encoder=action_encoder.state_dict(),
                         proprio_encoder=proprio_encoder.state_dict())
                if with_dec:
                    v, prev = r["val_dec_recon_loss"], best["dec"][0]
                    if v < prev:
                        best["dec"] = [v, dec_step]
                        save(os.path.join(args.out, f"{name}_dec_best.pt"), decoder=decoder.state_dict(), steps=dec_step, epoch=dec_step / steps_per_epoch)
                    best["dec_bad_evals"] = 0 if v < prev * (1 - args.dec_min_rel) else best["dec_bad_evals"] + 1
                    if best["dec_bad_evals"] >= args.dec_patience or dec_step >= args.dec_cap:
                        best["dec_stopped"] = True
                hours = (time.time() - t0) / 3600
                emit({"event": "val", "pred_step": pred_step, "dec_step": dec_step, "pred_epoch": round(pred_step / steps_per_epoch, 2),
                      "train_z_visual_loss_sampled": sum(run["z"]) / max(len(run["z"]), 1),
                      "train_dec_loss_sampled": sum(run["dec"]) / len(run["dec"]) if run["dec"] else None, **r,
                      "best_pred": best["pred"], "best_dec": best["dec"], "dec_bad_evals": best["dec_bad_evals"], "dec_stopped": best["dec_stopped"],
                      "it_per_s": round(steps_run / (time.time() - t0), 2), "hours": round(hours, 3)})
                run = {"z": [], "dec": []}
                save_state()
            if pred_step >= args.pred_target:
                break
    emit({"event": "done", "pred_step": pred_step, "dec_step": dec_step, "best": best, "hours": round((time.time() - t0) / 3600, 3)})


if __name__ == "__main__":
    main()
