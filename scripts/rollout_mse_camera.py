"""Slide/paper figure: rollout latent MSE of the single-camera model, training camera vs moving camera.

Model: train_<object>@50 (original DINO-WM, fixed camera). Conditions: eval_fixed (dataset camera)
and eval_move (camera turns 2-8° per step). Both have the same 100 validation episodes and actions.
For each episode, rollouts start at steps t = 1..14 (as in eval_camera_pred.py) with the push
actions; MSE between the predicted and the encoded real DINOv2 patch features at step t+k.
Per-episode means over t, then mean and 95% CI (1.96 * sd / sqrt(n)) over the 100 episodes.

    source repro/env.sh
    python scripts/rollout_mse_camera.py --out $DINO_WORK/results/camera_motion/talk
    python scripts/rollout_mse_camera.py --out $DINO_WORK/results/camera_motion/talk --plot_only

rollout_mse.png (slide "Train: single / Eval: moving"): the single-camera model on the training camera
and on the moving camera. rollout_mse_fixedcam.png (slide "Train: moving / Eval: single camera"): both
models on the dataset camera, y range 0-1 (_shared_y: the 0-4 range of the other two figures). rollout_mse_moving.png (slide "Train: moving / Eval: moving"): on the moving
camera, the single-camera model and the moving-camera model cam_<object>_move_told_h1@50, which also
gets the camera pose change (Δazimuth, Δheight, Δradius) as action. Both figures use the same y range.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
K = 5
OBJECTS = ["rope", "granular"]
CONDS = {"eval_fixed": "Training camera", "eval_move": "Moving camera"}
COLORS = {"eval_fixed": "#2a78d6", "eval_move": "#eb6834", "told": "#1baf7a"}
INK, MUTED, GRID = "#1a1a19", "#5f5e5a", "#e4e3df"


ENTRIES = [(f"{o}/{c}", f"train_{o}@50", o, c) for o in OBJECTS for c in CONDS] + \
          [(f"told/{o}/{c}", f"cam_{o}_move_told_h1@50", o, c) for o in OBJECTS for c in CONDS]
YLIM = (0, 4.0)


def compute(path, have=None):
    import torch

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo / "scripts"))
    sys.path.insert(0, str(repo))
    from eval_camera_pred import DEVICE, Model, load_condition, mse
    from datasets.img_transforms import default_transform

    root = Path(os.environ["DATASET_DIR"]) / "deformable_cam"
    tf = default_transform(224)
    torch.set_grad_enabled(False)
    res = have or {"model": "train_<object>@50", "horizon": K, "starts": "t = 1..T-K-1", "per_episode": {}}
    res["models"] = {key: spec for key, spec, _, _ in ENTRIES}
    for key, spec, obj, cond in ENTRIES:
        if key in res["per_episode"]:
            continue
        m = Model("m=" + spec, obj, root, os.environ["DINO_TRAIN"])
        enc = m.wm
        if True:
            c = load_condition(root / obj / cond, None)
            N, T = c["pose"].shape[:2]
            ts = list(range(1, T - K))
            per_ep = np.zeros((N, K))
            for i in range(N):
                x = torch.from_numpy(np.ascontiguousarray(c["frames"][i])).permute(0, 3, 1, 2).float() / 255.0
                F = tf(x).to(DEVICE)
                zF = enc.encode_obs({"visual": F[:, None], "proprio": torch.zeros(T, 1, 1, device=DEVICE)})["visual"][:, 0]
                a = m.actions(c["push"][i:i + 1], c["pose"][i:i + 1])[0]
                zr, _ = m.predict(F, a, ts, K)
                for k in range(1, K + 1):
                    per_ep[i, k - 1] = float(mse(zr[:, k - 1], zF[[t + k for t in ts]]).mean())
            res["per_episode"][key] = per_ep.tolist()
            print(key, "mean per step:", np.round(per_ep.mean(0), 3), flush=True)
    json.dump(res, open(path, "w"))
    return res


def plot(res, out, name, lines, ylim=YLIM):
    """lines: list of (key with {obj}, legend label, color)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 14,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.6), sharey=True, facecolor="white")
    x = np.arange(1, K + 1)
    for ax, obj in zip(axes, OBJECTS):
        for key, _, col in lines:
            e = np.array(res["per_episode"][key.format(obj=obj)])
            mu = e.mean(0)
            ci = 1.96 * e.std(0, ddof=1) / np.sqrt(len(e))
            ax.fill_between(x, mu - ci, mu + ci, color=col, alpha=0.15, lw=0)
            ax.plot(x, mu, color=col, lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
            ax.scatter(x, mu, s=36, color=col, edgecolor="white", linewidth=1.5, zorder=4)
            ax.annotate(f"{mu[-1]:.2f}", (x[-1], mu[-1]), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=12, color=INK)
        ax.set_title(obj.capitalize(), fontsize=15, weight="bold", color=INK, pad=8)
        ax.set_xticks(x)
        ax.set_xlim(0.7, K + 0.75)
        ax.set_xlabel("Rollout step", color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
            ax.spines[s].set_linewidth(0.8)
        ax.tick_params(colors=INK, length=3, color=MUTED)
    axes[0].set_ylabel("Latent MSE (DINOv2 features)", color=INK)
    axes[0].set_ylim(*ylim)
    handles = [plt.Line2D([], [], color=col, lw=2, marker="o", markersize=6, markeredgecolor="white")
               for _, _, col in lines]
    fig.legend(handles, [lab for _, lab, _ in lines], loc="lower center", ncol=2, frameon=False, fontsize=13,
               bbox_to_anchor=(0.54, 0.0), handlelength=1.8, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.2)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300 if ext == "png" else None, facecolor="white")
    print("wrote", out / f"{name}.{{png,svg,pdf}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot_only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rollout_mse_fixed_model.json"
    have = json.load(open(path)) if path.exists() else None
    res = have if args.plot_only else compute(path, have)
    plot(res, out, "rollout_mse", [("{obj}/eval_fixed", "Training camera", COLORS["eval_fixed"]),
                                   ("{obj}/eval_move", "Moving camera", COLORS["eval_move"])])
    plot(res, out, "rollout_mse_moving", [("{obj}/eval_move", "Train: single camera", COLORS["eval_move"]),
                                          ("told/{obj}/eval_move", "Train: moving camera", COLORS["told"])])
    # slide "Train: moving / Eval: single camera": both models on the dataset camera
    fixedcam = [("{obj}/eval_fixed", "Train: single camera", COLORS["eval_fixed"]),
                ("told/{obj}/eval_fixed", "Train: moving camera", COLORS["told"])]
    plot(res, out, "rollout_mse_fixedcam", fixedcam, ylim=(0, 1.0))
    plot(res, out, "rollout_mse_fixedcam_shared_y", fixedcam)


if __name__ == "__main__":
    main()
