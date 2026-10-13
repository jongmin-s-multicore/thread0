"""Videos of real frames next to the decoded predictions of several world models under camera motion.

One mp4 per (condition, kind, episode). Panels: Real, then one per model. Two kinds:

  1step     each panel shows the model's prediction of frame t+1 from the real frames up to t
  openloop  each panel shows the open-loop rollout from the first context frames, with the
            dataset actions (and the camera motion for models that take it as input)

All predictions are decoded with one decoder (--decoder), also for models trained without one.
Episodes are evenly spaced over the validation episodes of the condition.

    python scripts/make_camera_videos.py --object rope \\
        --models fixed=train_rope@50 told_h1=cam_rope_move_told_h1@50 \\
        --decoder cam_rope_move_told_h1@50 --conds eval_fixed eval_move --out videos/rope

Model format and data layout are the same as in scripts/eval_camera_pred.py.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_camera_pred import DEVICE, Model, load_condition  # noqa: E402

from datasets.img_transforms import default_transform  # noqa: E402

LABELS = {
    "fixed": "fixed camera",
    "static": "static random view",
    "blind_h1": "moving, not told, 1 frame",
    "blind_h2": "moving, not told, 2 frames",
    "told_h1": "moving, told",
}
FPS, HOLD_SEC, SCALE, COLS = 4, 3, 2, 3


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--object", required=True, choices=["rope", "granular"])
    p.add_argument("--models", nargs="+", required=True, help="label=name@epoch")
    p.add_argument("--decoder", required=True, help="name@epoch of the model whose decoder is used")
    p.add_argument("--conds", nargs="+", required=True)
    p.add_argument("--kinds", nargs="+", default=["1step", "openloop"], choices=["1step", "openloop"])
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--data", default=None, help="default: $DATASET_DIR/deformable_cam")
    p.add_argument("--ckpt_base", default=None, help="default: $DINO_TRAIN")
    p.add_argument("--out", required=True, help="output folder")
    return p.parse_args()


def panel(img, text):
    """(H, W, 3) uint8 -> upscaled panel with a label box in the top-left corner."""
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(img, (0, 0), (tw + 10, th + base + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (5, th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def grid(panels):
    """List of equal panels -> one frame with COLS columns (black cells fill the last row)."""
    blank = np.zeros_like(panels[0])
    panels = panels + [blank] * (-len(panels) % COLS)
    rows = [np.concatenate(panels[i:i + COLS], axis=1) for i in range(0, len(panels), COLS)]
    return np.concatenate(rows, axis=0)


def main():
    args = parse_args()
    data_root = Path(args.data or Path(os.environ["DATASET_DIR"]) / "deformable_cam")
    ckpt_base = args.ckpt_base or os.environ["DINO_TRAIN"]
    transform = default_transform(224)
    models = [Model(s, args.object, data_root, ckpt_base) for s in args.models]
    dec = Model("decoder=" + args.decoder, args.object, data_root, ckpt_base).wm
    assert dec.decoder is not None, "the --decoder model has no decoder"
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(False)

    def to_model(x):
        x = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float() / 255.0
        return transform(x).to(DEVICE)

    def decode(z_vis, z_prop):  # (n, P, D) latents -> (n, H, W, 3) uint8
        img, _ = dec.decode_obs({"visual": z_vis[:, None], "proprio": z_prop[:, None]})
        img = (img["visual"][:, 0] * 0.5 + 0.5).clamp(0, 1)
        return (img.permute(0, 2, 3, 1) * 255).round().byte().cpu().numpy()

    n_written = 0
    for cond in args.conds:
        c = load_condition(data_root / args.object / cond, None)
        N, T = c["pose"].shape[:2]
        name = cond.replace("eval_", "")
        for i in np.linspace(0, N - 1, min(args.n_episodes, N)).round().astype(int):
            real = np.asarray(c["frames"][i])
            F = to_model(real)
            ep = int(c["episodes"][i])
            preds = {k: {} for k in args.kinds}
            for m in models:
                a = m.actions(c["push"][i:i + 1], c["pose"][i:i + 1])[0]
                if "1step" in args.kinds:      # prediction of frame t+1 for t = 1..T-2
                    z, zp = m.predict(F, a, list(range(1, T - 1)), 1)
                    preds["1step"][m.label] = decode(z[:, 0], zp[:, 0])
                if "openloop" in args.kinds:   # rollout from the context ending at frame 1
                    z, zp = m.predict(F, a, [1], T - 2)
                    preds["openloop"][m.label] = decode(z[0], zp[0])
            for kind in args.kinds:
                frames = []
                for j in range(T - 2):          # shown frame index j + 2
                    cells = [panel(real[j + 2].copy(), f"Real, step {j + 2}")]
                    cells += [panel(preds[kind][m.label][j].copy(), LABELS.get(m.label, m.label)) for m in models]
                    frames.append(grid(cells))
                frames += [frames[-1]] * (FPS * HOLD_SEC)
                path = os.path.join(args.out, f"{name}_{kind}_ep{ep:04d}.mp4")
                imageio.mimsave(path, frames, fps=FPS, codec="libx264", quality=7, macro_block_size=None)
                n_written += 1
        print(f"{cond}: done", flush=True)
    print(f"wrote {n_written} videos to {args.out}")


if __name__ == "__main__":
    main()
