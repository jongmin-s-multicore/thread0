"""Prediction-quality eval (paper Tables 4 / 9): LPIPS & SSIM of decoded 1-step predictions.

Mirrors train.py's validation computation (model.eval(), model(obs, act), eval_images on
visual_out[:, t - num_pred] vs obs['visual'][:, t] in the normalized [-1, 1] image space),
but over the whole validation split (seed-42 trajectory split used in training; PushT: val folder)
instead of only the first validation batch. The dataset is built after seed(training.seed), as in
train.py, so the slice order and the evenly spaced subset (when capped) are deterministic.

usage (cwd = $DINO_WM, env sourced): python eval_pred_quality.py <ckpt_base_path> <model_name> <out_json> [max_samples]
"""
import os, sys, json, time
import numpy as np
import torch
import hydra
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
from plan import load_model
from metrics.image_metrics import eval_images
from utils import seed

ckpt_base, model_name, out_json = sys.argv[1:4]
max_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 0
model_path = Path(ckpt_base) / "outputs" / model_name
cfg = OmegaConf.load(model_path / "hydra.yaml")
# train.py 와 같은 seed 로 데이터셋을 만들어 슬라이스 순서(np.random.permutation)를 학습 때와 같게, 부분 표본을 결정적으로 한다
seed(cfg.training.seed)
datasets, _ = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred, frameskip=cfg.frameskip)
dset = datasets["valid"]
n = len(dset)
idx = np.arange(n)
if max_samples and n > max_samples:  # uniform deterministic subsample
    idx = np.linspace(0, n - 1, max_samples).round().astype(int)
sub = torch.utils.data.Subset(dset, idx.tolist())
loader = torch.utils.data.DataLoader(sub, batch_size=32, shuffle=False, num_workers=8)

device = "cuda"
model = load_model(model_path / "checkpoints" / "model_latest.pth", cfg, cfg.num_action_repeat, device)
model.eval()
epoch = torch.load(model_path / "checkpoints" / "model_latest.pth", map_location="cpu")["epoch"]

sums, count = {}, 0
t0 = time.time()
with torch.no_grad():
    for obs, act, state in loader:
        obs = {k: v.to(device) for k, v in obs.items()}
        act = act.to(device)
        z_out, visual_out, visual_rec, loss, loss_components = model(obs, act)
        b = act.shape[0]
        for t in range(cfg.num_hist, cfg.num_hist + cfg.num_pred):
            s = eval_images(visual_out[:, t - cfg.num_pred], obs["visual"][:, t])
            for k, v in s.items():
                sums[f"pred_{k}"] = sums.get(f"pred_{k}", 0.0) + float(v.mean()) * b
        s = eval_images(visual_rec[:, -1], obs["visual"][:, -1])
        for k, v in s.items():
            sums[f"recon_{k}"] = sums.get(f"recon_{k}", 0.0) + float(v.mean()) * b
        for k, v in loss_components.items():
            sums[f"loss_{k}"] = sums.get(f"loss_{k}", 0.0) + float(v) * b
        count += b
res = {k: v / count for k, v in sums.items()}
res.update(dict(model=model_name, ckpt_epoch=int(epoch), n_val_slices=n, n_evaluated=count,
                num_hist=cfg.num_hist, frameskip=cfg.frameskip, seconds=round(time.time() - t0, 1)))
print(json.dumps(res, indent=1))
os.makedirs(os.path.dirname(out_json), exist_ok=True)
json.dump(res, open(out_json, "w"), indent=1)
