"""Open-loop rollout error of the released PushT world model over whole validation demos.

For each validation demo: encode the first frame, roll the model forward with the demo's own actions (1-frame
context as in planning, later frames are never re-encoded), and compare with the real frames after every model step
(= 5 env steps). Per step, averaged over the demos that are long enough:
  z_visual_err    MSE between predicted and real visual latents (the visual term of the planning objective)
  z_visual_copy   the same for a "nothing moves" predictor that keeps the first frame's latent
  pred_lpips/ssim decoded prediction vs real frame;  recon_*: decoded real latent vs real frame (decoder floor);
  copy_lpips      first real frame vs real frame (how much the scene changed)

usage (after `source repro/env.sh`): CUDA_VISIBLE_DEVICES=<gpu> python repro/eval/pusht_openloop_error.py <out_dir>
Writes <out_dir>/openloop_error.json and <out_dir>/openloop_rollout.png (Real vs Model for the three longest demos).
"""
import os, sys, json
import numpy as np

THREAD0 = os.environ["THREAD0"]
sys.path.insert(0, THREAD0)
os.chdir(THREAD0)

import torch
import hydra
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf
from plan import load_model
from metrics.image_metrics import ssim
from metrics.lpipsPyTorch.modules.lpips import LPIPS

out_dir = os.path.abspath(sys.argv[1])
model_path = Path(os.environ["DINO_CKPT"]) / "outputs" / "pusht"
cfg = OmegaConf.load(model_path / "hydra.yaml")
_, dset = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred, frameskip=cfg.frameskip)
dset = dset["valid"]
device = "cuda"
model = load_model(model_path / "checkpoints" / "model_latest.pth", cfg, cfg.num_action_repeat, device)
model.eval()
lpips_fn = LPIPS("vgg", "0.1").to(device)
fs = cfg.frameskip

per_demo = []
with torch.no_grad():
    for i in range(len(dset)):
        obs, act, _, _ = dset[i]
        H = (obs["visual"].shape[0] - 1) // fs
        obs = {k: v[: H * fs + 1 : fs].unsqueeze(0).to(device) for k, v in obs.items()}  # (1, H+1, ...)
        act = rearrange(act[: H * fs], "(h f) d -> h (f d)", f=fs).unsqueeze(0).to(device)
        z_pred, _ = model.rollout({k: v[:, :1] for k, v in obs.items()}, act)
        z_gt = model.encode_obs(obs)
        img_pred = model.decode_obs(z_pred)[0]["visual"][0]
        img_rec = model.decode_obs(z_gt)[0]["visual"][0]
        img = obs["visual"][0]
        mse = lambda a, b: ((a - b) ** 2).flatten(1).mean(1).cpu().numpy()
        each = lambda fn, a, b: np.array([float(fn(a[k : k + 1], b[k : k + 1])) for k in range(a.shape[0])])
        per_demo.append({
            "H": H,
            "z_visual_err": mse(z_pred["visual"][0], z_gt["visual"][0]),
            "z_visual_copy": mse(z_gt["visual"][0, :1].expand_as(z_gt["visual"][0]), z_gt["visual"][0]),
            "z_proprio_err": mse(z_pred["proprio"][0], z_gt["proprio"][0]),
            "z_proprio_copy": mse(z_gt["proprio"][0, :1].expand_as(z_gt["proprio"][0]), z_gt["proprio"][0]),
            "pred_lpips": each(lpips_fn, img_pred, img), "pred_ssim": each(ssim, img_pred, img),
            "recon_lpips": each(lpips_fn, img_rec, img), "recon_ssim": each(ssim, img_rec, img),
            "copy_lpips": each(lpips_fn, img[:1].expand_as(img), img),
            "_img": img.cpu(), "_img_pred": img_pred.cpu(),
        })
        print(f"demo {i}: {H} model steps", flush=True)

keys = [k for k in per_demo[0] if k not in ("H",) and not k.startswith("_")]
max_h = max(d["H"] for d in per_demo)
by_step = []
for k in range(max_h + 1):
    ds = [d for d in per_demo if d["H"] >= k]
    by_step.append({"model_step": k, "env_step": k * fs, "n_demos": len(ds),
                    **{key: float(np.mean([d[key][k] for d in ds])) for key in keys}})
res = {"model": "pusht", "context_frames": 1, "frameskip": fs, "n_demos": len(per_demo),
       "by_step": by_step,
       "per_demo": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items() if not k.startswith("_")}
                    for d in per_demo]}
os.makedirs(out_dir, exist_ok=True)
json.dump(res, open(os.path.join(out_dir, "openloop_error.json"), "w"), indent=1)

print("env step  n   z_vis err / copy   ratio | pred LPIPS  SSIM | recon LPIPS | copy LPIPS")
for r in by_step:
    if r["model_step"] in (1, 2, 3, 5, 10, 15, 20, 25, 30) or r["model_step"] == max_h:
        print("%7d %3d   %.4f / %.4f   %5.2f | %.4f  %.4f | %.4f | %.4f" % (
            r["env_step"], r["n_demos"], r["z_visual_err"], r["z_visual_copy"],
            r["z_visual_err"] / max(r["z_visual_copy"], 1e-12), r["pred_lpips"], r["pred_ssim"],
            r["recon_lpips"], r["copy_lpips"]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

show = sorted(range(len(per_demo)), key=lambda j: -per_demo[j]["H"])[:3]
steps = [0, 5, 10, 15, 20, 25, 30]
fig, axes = plt.subplots(2 * len(show), len(steps), figsize=(1.6 * len(steps), 1.6 * 2 * len(show)))
for r, j in enumerate(show):
    for c, k in enumerate(steps):
        for row, key, name in ((2 * r, "_img", "Real"), (2 * r + 1, "_img_pred", "Model")):
            ax = axes[row, c]
            ax.imshow(((per_demo[j][key][k].permute(1, 2, 0) + 1) / 2).clamp(0, 1).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"demo {j}\n{name}", fontsize=8)
            if row == 0:
                ax.set_title(f"step {k * fs}", fontsize=8)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(out_dir, "openloop_rollout.png"), dpi=150)
