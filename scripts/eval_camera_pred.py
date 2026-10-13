"""Evaluate latent prediction error of several world models under camera motion.

Evaluation frames are rendered by scripts/render_camera_data.py --episodes val --counterfactual,
one folder per condition. The visual latent is the output of the frozen DINOv2 encoder, which
is the same for all models. Targets are encoded once per condition. For each validation
episode and each step t (the context ends at frame t):

  err_true     MSE(pred, z[state t+1, camera t+1])         actual next frame
  err_copy     MSE(z[frame t], z[state t+1, camera t+1])   current frame used as prediction
  err_camonly  MSE(pred, z[state t,   camera t+1])         camera moved, object unchanged
  err_objonly  MSE(pred, z[state t+1, camera t])           object moved, camera unchanged
  open_h<k>    open-loop error k steps ahead (k = 1..horizon)

Models with num_hist = 2 need two context frames. All metrics use t >= 1 so that all models
are evaluated on the same (episode, t) pairs.

    python scripts/eval_camera_pred.py --object rope \\
        --models fixed=train_rope@50 told=cam_rope_move_told_h1@50 \\
        --conds eval_fixed eval_static_rand eval_move eval_move_fast --out results.json

Model format: <label>=<name>@<epoch>, read from
$DINO_TRAIN/outputs/<name>/{hydra.yaml,checkpoints/model_<epoch>.pth}.
--decoder <name>@<epoch> adds LPIPS/SSIM of the decoded 1-step prediction. The same decoder
is used for all models.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from datasets.deformable_cam_dset import DeformCamDataset  # noqa: E402
from datasets.img_transforms import default_transform  # noqa: E402
from plan import load_model  # noqa: E402

DEVICE = "cuda"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--object", required=True, choices=["rope", "granular"])
    p.add_argument("--models", nargs="+", required=True, help="label=name@epoch")
    p.add_argument("--conds", nargs="+", required=True, help="folders under deformable_cam/<object>/")
    p.add_argument("--decoder", default=None, help="name@epoch of the model whose decoder is used")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--n_episodes", type=int, default=None)
    p.add_argument("--data", default=None, help="default: $DATASET_DIR/deformable_cam")
    p.add_argument("--ckpt_base", default=None, help="default: $DINO_TRAIN")
    p.add_argument("--out", required=True)
    return p.parse_args()


def mse(a, b):
    return ((a - b) ** 2).flatten(1).mean(dim=1)


class Model:
    """World model with the action normalization statistics of its training data."""

    def __init__(self, spec, obj, data_root, ckpt_base):
        self.label, rest = spec.split("=", 1)
        self.name, epoch = rest.split("@")
        self.epoch = epoch
        path = Path(ckpt_base) / "outputs" / self.name
        cfg = OmegaConf.load(path / "hydra.yaml")
        assert cfg.frameskip == 1, "written for frameskip 1"
        self.num_hist = int(cfg.num_hist)
        ds = cfg.env.dataset
        self.cam_action = bool(ds.get("cam_action", False))
        self.train_variant = ds.get("variant", "fixed")
        # Action statistics of the training data. Push statistics are the same for all
        # variants. For a model trained on the original dataset, the `move` set is used.
        stats = DeformCamDataset(
            data_path=data_root, object_name=obj, normalize_action=True, cam_action=self.cam_action,
            variant=self.train_variant if self.train_variant != "fixed" else "move",
        )
        self.action_mean, self.action_std = stats.action_mean, stats.action_std
        ckpt = path / "checkpoints" / f"model_{epoch}.pth"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        self.wm = load_model(ckpt, cfg, cfg.num_action_repeat, DEVICE)
        self.wm.eval()
        self.cfg = cfg

    def actions(self, push, pose):
        """Normalize (N, T, 4) push actions and (N, T, 3) poses into model actions."""
        a = torch.from_numpy(push).float()
        if self.cam_action:
            p = torch.from_numpy(pose).float()
            d = torch.zeros_like(p)
            d[:, :-1] = p[:, 1:] - p[:, :-1]
            d[:, -1] = d[:, -2]
            a = torch.cat([a, d], dim=-1)
        return ((a - self.action_mean) / self.action_std).to(DEVICE)

    @torch.no_grad()
    def predict(self, frames, act, t, k):
        """Open-loop latents for steps t+1..t+k, context ending at frame t, for one episode.

        frames (T, 3, H, W) normalized, act (T, A), t a list of ints. Returns (len(t), k, P, D)
        visual latents and the matching proprio latents.
        """
        n = self.num_hist
        obs0 = {
            "visual": torch.stack([frames[i - n + 1:i + 1] for i in t]),
            "proprio": torch.zeros(len(t), n, 1, device=DEVICE),
        }
        a = torch.stack([act[i - n + 1:i + k] for i in t])
        z_obs, _ = self.wm.rollout(obs0, a)
        return z_obs["visual"][:, n:], z_obs["proprio"][:, n:]


def load_condition(root, n_episodes):
    def frames(name):
        return np.load(root / name, mmap_mode="r")

    d = {
        "frames": frames("frames.npy"),
        "camonly": frames("frames_camonly.npy"),
        "objonly": frames("frames_objonly.npy"),
        "pose": np.load(root / "cam_pose.npy"),
        "push": np.load(root / "actions.npy"),
        "episodes": np.load(root / "episodes.npy"),
    }
    n = min(n_episodes or len(d["pose"]), len(d["pose"]))
    return {k: v[:n] for k, v in d.items()}


def main():
    args = parse_args()
    data_root = Path(args.data or Path(os.environ["DATASET_DIR"]) / "deformable_cam")
    ckpt_base = args.ckpt_base or os.environ["DINO_TRAIN"]
    transform = default_transform(224)
    models = [Model(s, args.object, data_root, ckpt_base) for s in args.models]
    dec = None
    if args.decoder:
        from metrics.image_metrics import eval_images

        dec = Model("decoder=" + args.decoder, args.object, data_root, ckpt_base).wm
        assert dec.decoder is not None, "the --decoder model has no decoder"
    enc = models[0].wm  # frozen DINOv2, same for all models

    def to_model(x):  # (n, H, W, 3) uint8 -> normalized (n, 3, H, W) on the GPU
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float() / 255.0
        return transform(x).to(DEVICE)

    @torch.no_grad()
    def encode(x):
        out = []
        for i in range(0, len(x), 64):
            xb = x[i:i + 64]
            obs = {"visual": xb[:, None], "proprio": torch.zeros(len(xb), 1, 1, device=DEVICE)}
            out.append(enc.encode_obs(obs)["visual"][:, 0])
        return torch.cat(out)

    K = args.horizon
    res = {"object": args.object, "horizon": K, "conditions": {},
           "models": {m.label: {"name": m.name, "epoch": m.epoch, "num_hist": m.num_hist,
                                "cam_action": m.cam_action, "train_variant": m.train_variant} for m in models}}
    t_start = time.time()
    for cond in args.conds:
        c = load_condition(data_root / args.object / cond, args.n_episodes)
        N, T = c["pose"].shape[:2]
        acc = {m.label: {} for m in models}
        ref = {}

        def add(store, key, val):
            store.setdefault(key, []).append(val.detach().float().cpu())

        ts = list(range(1, T - 1))            # 1-step: context ends at t, target t+1
        ts_open = list(range(1, T - K))       # open loop needs K future frames
        torch.set_grad_enabled(False)          # evaluation only
        for i in range(N):
            F = to_model(c["frames"][i])
            zF = encode(F)
            zC = encode(to_model(c["camonly"][i]))   # index t: state t,   camera t+1
            zO = encode(to_model(c["objonly"][i]))   # index t: state t+1, camera t
            tgt = zF[[t + 1 for t in ts]]
            add(ref, "err_copy", mse(zF[ts], tgt))
            add(ref, "true_vs_camonly", mse(tgt, zC[ts]))   # change due to object motion
            add(ref, "true_vs_objonly", mse(tgt, zO[ts]))   # change due to camera motion
            for k in range(1, K + 1):
                add(ref, f"open_copy_h{k}", mse(zF[ts_open], zF[[t + k for t in ts_open]]))
            for m in models:
                a = m.actions(c["push"][i:i + 1], c["pose"][i:i + 1])[0]
                pred, pred_prop = m.predict(F, a, ts, 1)
                pred = pred[:, 0]
                e_true, e_cam, e_obj = mse(pred, tgt), mse(pred, zC[ts]), mse(pred, zO[ts])
                st = acc[m.label]
                add(st, "err_true", e_true); add(st, "err_camonly", e_cam); add(st, "err_objonly", e_obj)
                add(st, "closest_true", ((e_true < e_cam) & (e_true < e_obj)).float())
                add(st, "closest_camonly", ((e_cam < e_true) & (e_cam < e_obj)).float())
                add(st, "closest_objonly", ((e_obj < e_true) & (e_obj < e_cam)).float())
                if dec is not None:
                    img, _ = dec.decode_obs({"visual": pred[:, None], "proprio": pred_prop[:, :1]})
                    im = eval_images(img["visual"][:, 0], F[[t + 1 for t in ts]])
                    add(st, "img_lpips", torch.as_tensor(im["lpips"]).reshape(-1))
                    add(st, "img_ssim", torch.as_tensor(im["ssim"]).reshape(-1))
                zr, _ = m.predict(F, a, ts_open, K)
                for k in range(1, K + 1):
                    add(st, f"open_h{k}", mse(zr[:, k - 1], zF[[t + k for t in ts_open]]))
        mean = lambda d: {k: float(torch.cat(v).mean()) for k, v in d.items()}
        dpose = np.abs(np.diff(c["pose"], axis=1))
        res["conditions"][cond] = {
            "n_episodes": int(N), "n_steps": int(T), "n_samples_1step": int(N * len(ts)),
            "azimuth_deg_per_step": float(np.rad2deg(dpose[..., 0]).mean()),
            "reference": mean(ref), "models": {lab: mean(d) for lab, d in acc.items()},
        }
        r = res["conditions"][cond]
        print(f"== {cond}: {N} episodes, copy {r['reference']['err_copy']:.3f}, "
              f"object-motion effect {r['reference']['true_vs_camonly']:.3f}, camera-motion effect {r['reference']['true_vs_objonly']:.3f}", flush=True)
        for lab, d in r["models"].items():
            print(f"   {lab:16s} true {d['err_true']:.3f}  camonly {d['err_camonly']:.3f}  objonly {d['err_objonly']:.3f}  "
                  f"closest true/cam/obj {d['closest_true']:.2f}/{d['closest_camonly']:.2f}/{d['closest_objonly']:.2f}  "
                  f"open h{K} {d[f'open_h{K}']:.3f}" + (f"  lpips {d['img_lpips']:.3f} ssim {d['img_ssim']:.3f}" if "img_lpips" in d else ""), flush=True)
    res["seconds"] = round(time.time() - t_start, 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
