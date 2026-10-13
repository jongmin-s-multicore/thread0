"""ACWM-Phys world model 의 predictor 만 이어서 오래 학습한다 (큰 GPU 한 장, 예: Colab G4 = RTX PRO 6000 96 GB).

인코더(DINOv2)는 동결이고 디코더는 predictor 와 따로 학습되므로(정답 프레임의 잠재만 재구성), predictor 의 학습에는
"프레임의 DINOv2 패치 토큰"과 action 만 있으면 된다. 그래서
  1) 학습·검증 에피소드의 모든 프레임을 한 번 인코딩해 토큰을 GPU 메모리에 올려 두고 (float16, 환경당 12-20 GB),
  2) train.py 와 같은 모델(VWorldModel, 디코더 없이)·같은 손실·같은 옵티마이저 설정·같은 배치 크기로 predictor 를 이어 학습한다.
영상 디코딩, 인코더 forward, 디코더 forward/backward 가 스텝에서 빠져 train.py 보다 훨씬 빠르다.
시작할 때 불러온 가중치의 검증 손실을 다시 재서 원래 학습 로그의 값과 비교한다 (경로가 같은지 확인).

입력은 ckpt_tools.py slim 이 만든 파일, 출력은 --epochs 의 각 epoch 에서의 가중치(<out>/<model>_ep<N>.pt)와 로그(<out>/<model>.jsonl).
레포의 고정 환경이 아니어도 돈다: torch, torchvision, einops, omegaconf, hydra-core, decord 만 쓴다 (plan.py·train.py 를 import 하지 않는다).

다른 split 의 에피소드를 학습에 섞을 수도 있다 (예: OoD 테스트 에피소드의 일부). --extra_split ood_test --extra_train smaller:even,larger:even
이면 파일 이름이 smaller_*, larger_* 인 에피소드 중 번호가 짝수인 것을 학습에 넣고 나머지는 검증 세트 ood_smaller, ood_larger 로 따로 잰다
(규칙: all, none, even, odd). --from_scratch 는 불러온 가중치를 쓰지 않고 epoch 0 부터 새로 학습한다 (설정·분할·action 통계만 slim 에서 가져온다).

usage (cwd = repo root):
  python repro/colab/predictor_long.py <slim.pt> <ACWM-Phys root> <out dir> --epochs 30,60,120,240
         [--device cuda] [--max_episodes 0] [--batch_size 0] [--seed 0] [--tf32] [--amp] [--compile]
         [--extra_split ood_test --extra_train smaller:even,larger:even] [--from_scratch] [--eval_only] [--tag NAME] [--tokens_cpu]
"""
import os, sys, json, time, argparse, itertools
import torch
import torch.nn as nn
import hydra
from omegaconf import OmegaConf
from torchvision import transforms

sys.path.insert(0, os.getcwd())
from datasets.acwm_phys_dset import ACWMPhysDataset
from datasets.traj_dset import split_traj_datasets
from models.visual_world_model import VWorldModel

SUBDIR = {"acwm_cloth_move": "deformable/clothmove", "acwm_push_sand": "particle/push_sand", "acwm_stack_cube": "rigid_dynamics/stack_cube"}


class CachedTokens(nn.Module):
    """Stands in for the frozen encoder: the "images" it receives are already its patch tokens."""
    name = "cached_tokens"
    latent_ndim = 2

    def __init__(self, emb_dim, patch_size):
        super().__init__()
        self.emb_dim, self.patch_size = emb_dim, patch_size

    def forward(self, x):
        return x


class Frames(torch.utils.data.Dataset):
    """eps: list of (dataset, episode index)"""
    def __init__(self, eps):
        self.eps = eps

    def __len__(self):
        return len(self.eps)

    def __getitem__(self, k):
        d, i = self.eps[k]
        return k, d.get_frames(i, range(d.n_obs[i]))[0]["visual"]


@torch.no_grad()
def encode_all(eps, encoder, resize, dev, workers, tdev):
    """-> tokens (n_frames_total, patches, dim) float16 on tdev, first-frame offset of every episode"""
    n_obs = [d.n_obs[i] for d, i in eps]
    offs = torch.tensor([0] + list(itertools.accumulate(n_obs))[:-1])
    probe = encoder(resize(torch.zeros(1, 3, 224, 224, device=dev)))
    tok = torch.empty(sum(n_obs), probe.shape[1], probe.shape[2], dtype=torch.float16, device=tdev)
    loader = torch.utils.data.DataLoader(Frames(eps), batch_size=None, shuffle=False, num_workers=workers)
    t0 = time.time()
    for n, (k, vis) in enumerate(loader):
        vis = vis.to(dev)
        tok[offs[k] : offs[k] + len(vis)] = torch.cat([encoder(resize(vis[s : s + 256])) for s in range(0, len(vis), 256)]).half().to(tdev)
        if n % 200 == 0:
            print(f"  encoded {n}/{len(eps)} episodes {time.time() - t0:.0f}s", flush=True)
    return tok, offs


def slices(eps, offs, a_offs, f, num_frames):
    """every training window of train.py's slicer: frame indices (n, num_frames) and action indices (n, num_frames * f), global"""
    fr, ac = [], []
    for k, (d, i) in enumerate(eps):
        T = int(d.seq_lengths[i])
        starts = torch.arange(max(T - num_frames * f + 1, 0))
        fr.append(offs[k] + starts[:, None] + f * torch.arange(num_frames)[None])
        ac.append(a_offs[k] + starts[:, None] + torch.arange(num_frames * f)[None])
    return torch.cat(fr), torch.cat(ac)


def pick(names, rules):
    """names of the extra split -> (indices for training, {group: held-out indices}); group = file-name prefix before '_episode'"""
    import re
    train, held = [], {}
    for i, n in enumerate(names):
        g = n.split("_episode")[0] if "_episode" in n and not n.startswith("episode") else "all"
        num = int(re.search(r"(\d+)", n.split("episode")[-1]).group(1))
        rule = rules.get(g, "none")
        use = rule == "all" or (rule == "even" and num % 2 == 0) or (rule == "odd" and num % 2 == 1)
        if use:
            train.append(i)
        else:
            held.setdefault(g, []).append(i)
    return train, held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slim"); ap.add_argument("data_root"); ap.add_argument("out")
    ap.add_argument("--epochs", default="30,60,120,240")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_episodes", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tf32", action="store_true", help="TF32 matmul (faster on recent GPUs; float32 weights, about 1e-3 relative matmul error)")
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast for the predictor forward (the loss is still computed in float32)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the predictor (same math, less per-step overhead)")
    ap.add_argument("--extra_split", default="", help="another split of the same environment whose episodes can join the training set")
    ap.add_argument("--extra_train", default="", help="group:rule[,group:rule] with rule all|none|even|odd; the other episodes become validation sets ood_<group>")
    ap.add_argument("--from_scratch", action="store_true")
    ap.add_argument("--eval_only", action="store_true", help="only measure the loaded weights on the validation sets (the training set is not encoded)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--tokens_cpu", action="store_true", help="keep the cached tokens in RAM instead of on the GPU (when the GPU is shared; a little slower)")
    args = ap.parse_args()
    dev = args.device
    S = torch.load(args.slim, map_location="cpu")
    cfg = OmegaConf.create(S["cfg"])
    name, env, H, P, f = S["model_name"] + (f"_{args.tag}" if args.tag else ""), cfg.env.name, cfg.num_hist, cfg.num_pred, cfg.frameskip
    start_epoch = 0 if args.from_scratch else S["epoch"]
    B = args.batch_size or cfg.training.batch_size
    milestones = sorted(int(e) for e in args.epochs.split(","))
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp and dev.startswith("cuda"))
    tdev = "cpu" if args.tokens_cpu else dev  # where the cached tokens and their slice indices live

    # ---- data: same episodes, split and action normalization as the original run
    ds = cfg.env.dataset
    dset = ACWMPhysDataset(os.path.join(args.data_root, SUBDIR[env]), "ind_train", n_rollout=ds.n_rollout, exclude=list(ds.exclude),
                           transform=hydra.utils.instantiate(ds.transform), normalize_action=cfg.normalize_action,
                           action_dims=None if ds.action_dims is None else list(ds.action_dims), action_format=ds.action_format)
    assert torch.allclose(dset.action_mean, S["action_mean"]) and torch.allclose(dset.action_std, S["action_std"]), "action statistics differ"
    by_name = {n: i for i, n in enumerate(dset.video_names)}
    tr_idx, va_idx = [by_name[n] for n in S["train_names"]], [by_name[n] for n in S["val_names"]]
    tr_split, _ = split_traj_datasets(dset, train_fraction=ds.split_ratio, random_seed=42)
    print("split identical to torch's own on this machine:", list(tr_split.indices) == tr_idx, flush=True)
    if args.max_episodes:
        tr_idx, va_idx = tr_idx[: args.max_episodes], va_idx[: max(args.max_episodes // 4, 2)]
    sets = {"train": [(dset, i) for i in tr_idx], "valid": [(dset, i) for i in va_idx]}
    if args.extra_split:  # same action normalization as the training data
        extra = ACWMPhysDataset(os.path.join(args.data_root, SUBDIR[env]), args.extra_split, transform=hydra.utils.instantiate(ds.transform),
                                normalize_action=cfg.normalize_action, action_dims=None if ds.action_dims is None else list(ds.action_dims),
                                action_format=ds.action_format, action_stats=(dset.action_mean, dset.action_std))
        ex_train, ex_held = pick(extra.video_names, dict(r.split(":") for r in args.extra_train.split(",") if r))
        if args.max_episodes:
            ex_train, ex_held = ex_train[:4], {g: v[:4] for g, v in ex_held.items()}
        sets["train"] += [(extra, i) for i in ex_train]
        for g, v in ex_held.items():
            sets[f"ood_{g}"] = [(extra, i) for i in v]
        json.dump({"extra_split": args.extra_split, "train": [extra.video_names[i] for i in ex_train],
                   "held_out": {g: [extra.video_names[i] for i in v] for g, v in ex_held.items()}}, open(os.path.join(args.out, f"{name}_split.json"), "w"), indent=1)
        print(f"extra split {args.extra_split}: {len(ex_train)} episodes into training, held out " + ", ".join(f"{g} {len(v)}" for g, v in ex_held.items()), flush=True)

    # ---- tokens of every frame, on the GPU
    if args.eval_only:
        sets.pop("train")
    encoder = hydra.utils.instantiate(cfg.encoder).to(dev).eval()
    enc_size = cfg.img_size // 16 * encoder.patch_size  # VWorldModel: decoder_scale 16
    resize = transforms.Resize(enc_size)
    t0 = time.time()
    data = {}
    for split, eps in sets.items():
        tok, offs = encode_all(eps, encoder, resize, dev, args.workers, tdev)
        acts = [d.actions[i] for d, i in eps]
        a_offs = torch.tensor([0] + list(itertools.accumulate(len(a) for a in acts))[:-1])
        fr, ac = slices(eps, offs, a_offs, f, H + P)
        data[split] = (tok, torch.cat(acts).to(dev), fr.to(tdev), ac.to(dev))
        print(f"{split}: {len(eps)} episodes, {len(tok)} frames, {len(fr)} slices, tokens {tok.numel() * 2 / 2**30:.1f} GB", flush=True)
    emb_dim, patch_size = encoder.emb_dim, encoder.patch_size
    del encoder
    torch.cuda.empty_cache() if dev.startswith("cuda") else None
    print(f"encoding took {time.time() - t0:.0f}s", flush=True)

    # ---- model: train.py's, without encoder and decoder
    if not dev.startswith("cuda"):  # models/vit.py puts its attention mask on 'cuda' in the constructor; keep it where it is for a CPU test
        import models.vit as vit

        class _Stay(torch.Tensor):
            def to(self, *a, **k):
                return self.as_subclass(torch.Tensor)

        _gen = vit.generate_mask_matrix
        vit.generate_mask_matrix = lambda npatch, nwindow: _gen(npatch, nwindow).as_subclass(_Stay)
    A = dset.action_dim * f
    proprio_encoder = hydra.utils.instantiate(cfg.proprio_encoder, in_chans=1, emb_dim=cfg.proprio_emb_dim)
    action_encoder = hydra.utils.instantiate(cfg.action_encoder, in_chans=A, emb_dim=cfg.action_emb_dim)
    n_patches = (cfg.img_size // 16) ** 2
    predictor = hydra.utils.instantiate(cfg.predictor, num_patches=n_patches, num_frames=H,
                                        dim=emb_dim + (cfg.proprio_emb_dim * cfg.num_proprio_repeat + cfg.action_emb_dim * cfg.num_action_repeat) * cfg.concat_dim)
    if not args.from_scratch:
        predictor.load_state_dict(S["predictor"]); action_encoder.load_state_dict(S["action_encoder"]); proprio_encoder.load_state_dict(S["proprio_encoder"])
    model = VWorldModel(image_size=cfg.img_size, num_hist=H, num_pred=P, encoder=CachedTokens(emb_dim, patch_size), proprio_encoder=proprio_encoder,
                        action_encoder=action_encoder, decoder=None, predictor=predictor, proprio_dim=cfg.proprio_emb_dim, action_dim=cfg.action_emb_dim,
                        concat_dim=cfg.concat_dim, num_action_repeat=cfg.num_action_repeat, num_proprio_repeat=cfg.num_proprio_repeat,
                        train_encoder=False, train_predictor=True, train_decoder=False).to(dev)
    if args.compile:
        model.predictor = torch.compile(predictor)  # shares the parameters; state dicts are taken from `predictor`
    opt_p = torch.optim.AdamW(predictor.parameters(), lr=cfg.training.predictor_lr)
    if not args.from_scratch:
        opt_p.load_state_dict(S["predictor_optimizer"])
    opt_a = torch.optim.AdamW(itertools.chain(action_encoder.parameters(), proprio_encoder.parameters()), lr=cfg.training.action_encoder_lr)  # its state is not in the checkpoint

    def batch(split, idx):
        tok, acts, fr, ac = data[split]
        obs = {"visual": tok[fr[idx]].to(dev).float(), "proprio": torch.zeros(len(idx), H + P, 1, device=dev)}
        return obs, acts[ac[idx.to(dev)]].reshape(len(idx), H + P, A)

    @torch.no_grad()
    def evaluate(split, cap=None):
        model.eval()
        n = len(data[split][2]) if cap is None else min(cap, len(data[split][2]))
        tot = {"z_visual_loss": 0.0, "z_loss": 0.0}
        for s in range(0, n, 256):
            idx = torch.arange(s, min(s + 256, n), device=tdev)
            with amp():
                _, _, _, _, comp = model(*batch(split, idx))
            for k in tot:
                tot[k] += comp[k].item() * len(idx)
        return {k: v / n for k, v in tot.items()}

    log = open(os.path.join(args.out, f"{name}.jsonl"), "a")
    def emit(rec):
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)

    def evaluate_all():
        """InD validation loss, and the same loss on every held-out group of the extra split"""
        v = evaluate("valid")
        rec = {"val_z_visual_loss": v["z_visual_loss"], "val_z_loss": v["z_loss"]}
        for k in data:
            if k not in ("train", "valid"):
                rec[f"val_{k}_z_visual_loss"] = evaluate(k)["z_visual_loss"]
        return rec

    emit({"model": name, "epoch": start_epoch, **evaluate_all(), "original_log_val_z_visual_loss": S["val_z_visual_loss"],
          "note": "random initialization" if args.from_scratch else "loaded weights, before any further training"})

    if args.eval_only:
        return
    n_tr = len(data["train"][2])
    steps_per_epoch = (n_tr + B - 1) // B
    t0, step = time.time(), 0
    for epoch in range(start_epoch + 1, milestones[-1] + 1):
        model.train()
        perm = torch.randperm(n_tr, device=tdev)
        run = 0.0
        for s in range(0, n_tr, B):
            with amp():
                _, _, _, loss, comp = model(*batch("train", perm[s : s + B]))
            opt_p.zero_grad(set_to_none=True); opt_a.zero_grad(set_to_none=True)
            loss.backward()
            opt_p.step(); opt_a.step()
            run += comp["z_visual_loss"].item() if (s // B) % 50 == 0 else 0.0
            step += 1
        rec = {"model": name, "epoch": epoch, "steps_total": epoch * steps_per_epoch, "train_z_visual_loss_sampled": run / ((steps_per_epoch + 49) // 50),
               "it_per_s": step / (time.time() - t0), "minutes": (time.time() - t0) / 60}
        if epoch in milestones or epoch % 5 == 0:
            rec.update(evaluate_all())
        emit(rec)
        if epoch in milestones:
            torch.save({"model_name": name, "epoch": epoch, "predictor": predictor.state_dict(), "action_encoder": action_encoder.state_dict(),
                        "proprio_encoder": proprio_encoder.state_dict()}, os.path.join(args.out, f"{name}_ep{epoch}.pt"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
