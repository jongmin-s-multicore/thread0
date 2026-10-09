"""Pour Water 결과 그림 (이슈 첨부용). eval_pour_water.py 의 JSON, eval_pour_water_volume.py 의 npz, 학습 epoch_logs.jsonl 에서 그린다.

usage: python repro/eval/plot_pour_water.py <results_dir> <train_outputs_dir> <out_dir>
  results_dir        h3f5_ep6.json h1f5_ep6.json h3f5_volume.npz h1f5_volume.npz 가 있는 곳
  train_outputs_dir  train_pw_h3f5/ train_pw_h1f5/ 가 있는 곳 (epoch_logs.jsonl)
모델 A = num_hist 3 (파랑), B = num_hist 1 (주황). 기준선은 회색.
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e8e7e3"
C_A, C_B = "#2a78d6", "#eb6834"  # categorical slots 1, 2 (validated light palette)
C_BASE, C_CEIL, C_GT = "#8a8984", "#c3c2b7", "#3a3a38"
LW, MS = 1.6, 5.5
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 8,
})
GROUPS = [("ind_test", "all", "InD test (50 episodes)"), ("ood_test", "low", "OoD low volume (26)"), ("ood_test", "high", "OoD high volume (24)")]


def line(ax, x, y, color, label=None, marker=True, ls="-", lw=LW, zorder=3):
    ax.plot(x, y, color=color, lw=lw, ls=ls, label=label, solid_capstyle="round", solid_joinstyle="round", zorder=zorder)
    if marker:
        ax.plot(x, y, color=color, ls="", marker="o", ms=MS, mec=SURFACE, mew=1.2, zorder=zorder + 1)


def end_label(ax, x, y, text, dx=0.12, color=INK2, va="center"):
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset fontsize", color=color, fontsize=8, va=va, annotation_clip=False)


def fig_training(out, tr_a, tr_b):
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for rows, color, nm in ((tr_a, C_A, "A  num_hist 3"), (tr_b, C_B, "B  num_hist 1")):
        ep = [r["epoch"] for r in rows]
        line(ax, ep, [r["val_loss"] for r in rows], color, f"{nm}  validation")
        line(ax, ep, [r["train_loss"] for r in rows], color, f"{nm}  train", marker=False, ls=(0, (1.2, 2.2)), lw=1.3, zorder=2)
    ax.set_xlabel("epoch (1 epoch = every slice once, batch 24)"); ax.set_ylabel("latent MSE loss (visual + proprio)")
    ax.set_xticks(range(1, 7)); ax.set_ylim(0, None)
    ax.set_title("Training: A and B reach the same validation loss")
    ax.legend(loc="upper right", ncol=1)
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def fig_steps(out, A, B):
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.4), sharex=True)
    for j, (sp, g, title) in enumerate(GROUPS):
        for i, (metric, ylab) in enumerate((("lpips", "LPIPS (VGG)"), ("mse", "MSE on [-1, 1] images"))):
            ax = axes[i, j]
            ca, cb = A["sets"][sp]["m2_curve"][g], B["sets"][sp]["m2_curve"][g]
            line(ax, range(1, 6), ca["recon_ceiling"][metric], C_CEIL, "decoder floor (reconstruct the true frame)", marker=False, lw=1.4)
            line(ax, range(1, 6), ca["recon_copy"][metric], C_BASE, "copy the last context frame, through the decoder", marker=False, lw=1.4)
            line(ax, range(1, 6), ca["model"][metric], C_A, "A  num_hist 3")
            line(ax, range(1, 8), cb["model"][metric], C_B, "B  num_hist 1")
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 1:
                ax.set_xlabel("steps ahead (1 step = 0.5 s)")
            ax.set_xticks(range(1, 8)); ax.set_ylim(0, None)
    h, l = axes[0, 0].get_legend_handles_labels()
    order = [2, 3, 1, 0]  # A, B, copy baseline, decoder floor
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", ncol=2, bbox_to_anchor=(0.005, 0.955), columnspacing=2.5)
    fig.suptitle("Open-loop rollouts on 37-frame ACWM-Phys windows (10 starts per episode)", x=0.01, ha="left", fontsize=10.5, fontweight="semibold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(out, dpi=200); plt.close(fig)


def fig_full(out, A, B):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), sharey=True)
    top = 0
    for ax, (sp, g, title) in zip(axes, GROUPS):
        for R, color, nm in ((A, C_A, "A"), (B, C_B, "B")):
            c = R["sets"][sp]["m3_curve"][g]; steps = np.array(R["sets"][sp]["m3_steps"]); t = steps * R["frameskip"] / 10.0
            line(ax, t, c["model"]["lpips"], color, f"{nm}  num_hist {R['num_hist']}", marker=False)
        c = A["sets"][sp]["m3_curve"][g]; t = np.array(A["sets"][sp]["m3_steps"]) * 0.5
        line(ax, t, c["copy_last"]["lpips"], C_BASE, "copy the last context frame (raw)", marker=False, lw=1.4)
        line(ax, t, c["recon_ceiling"]["lpips"], C_CEIL, "decoder floor", marker=False, lw=1.4)
        top = max(top, max(c["copy_last"]["lpips"]))
        ax.set_title(title); ax.set_xlabel("time in the episode (s)")
    axes[0].set_ylim(0, top * 1.05); axes[0].set_ylabel("LPIPS (VGG)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=4, bbox_to_anchor=(0.005, 0.925), columnspacing=1.6, handlelength=1.8)
    fig.suptitle("Whole-episode rollout (59 steps = 30 s) from the first frame(s) with all actions", x=0.01, ha="left", fontsize=10.5, fontweight="semibold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig(out, dpi=200); plt.close(fig)


def fig_fill(out, VA, VB):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), sharex=True, sharey=True)
    marks = {"ind_test": ("o", "InD test"), "ood_low": ("v", "OoD low volume"), "ood_high": ("^", "OoD high volume")}
    for ax, V, color, nm in ((axes[0], VA, C_A, "A  num_hist 3"), (axes[1], VB, C_B, "B  num_hist 1")):
        w0 = V["ood_test/w0"]
        sel = {"ind_test": (V["ind_test/band_gt"], V["ind_test/band_pred"]),
               "ood_low": (V["ood_test/band_gt"][w0 < 4000], V["ood_test/band_pred"][w0 < 4000]),
               "ood_high": (V["ood_test/band_gt"][w0 >= 4000], V["ood_test/band_pred"][w0 >= 4000])}
        lim = [500, 3000]
        ax.plot(lim, lim, color=C_BASE, lw=1.0, zorder=1)
        for key, (gt, pr) in sel.items():
            gt, pr = gt[:, -10:].mean(1), pr[:, -10:].mean(1)
            m, lab = marks[key]
            ax.plot(gt, pr, ls="", marker=m, ms=5, color=color, alpha=0.55, mec=SURFACE, mew=0.8, zorder=3)
            ax.plot(gt.mean(), pr.mean(), ls="", marker=m, ms=10, color=color, mec=INK, mew=1.2, zorder=4)
            ax.annotate(lab, (gt.mean(), pr.mean()), xytext=(0, {"ind_test": 27, "ood_low": 13, "ood_high": -17}[key]), textcoords="offset points", ha="center", fontsize=8, color=INK2)
        ax.set_title(nm); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal"); ax.set_xticks(range(500, 3001, 500)); ax.set_yticks(range(500, 3001, 500))
        ax.set_xlabel("ground truth: water pixels in the box, last 5 s")
    axes[0].set_ylabel("model rollout: water pixels in the box, last 5 s")
    axes[0].annotate("model = ground truth", (2750, 2750), xytext=(-6, 6), textcoords="offset points", ha="right", rotation=0, fontsize=8, color=INK2)
    fig.suptitle("Final water amount in the destination box (one point per episode, large marker = group mean)", x=0.01, ha="left", fontsize=10.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out, dpi=200); plt.close(fig)


def fig_time(out, VA, VB):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), sharey=True)
    t = np.arange(60) * 0.5
    top = 0
    for ax, (sp, g, title) in zip(axes, GROUPS):
        for V, color, nm in ((VA, C_A, "A  num_hist 3"), (VB, C_B, "B  num_hist 1")):
            w0 = V[f"{sp}/w0"]; m = np.ones(len(w0), bool) if g == "all" else (w0 < 4000) if g == "low" else (w0 >= 4000)
            line(ax, t, V[f"{sp}/tot_pred"][m].mean(0), color, nm, marker=False)
        gt = VA[f"{sp}/tot_gt"][m].mean(0)
        line(ax, t, gt, C_GT, "ground truth", marker=False, lw=2.0)
        top = max(top, gt.max(), VA[f"{sp}/tot_pred"][m].mean(0).max(), VB[f"{sp}/tot_pred"][m].mean(0).max())
        ax.set_title(title); ax.set_xlabel("time in the episode (s)")
    axes[0].set_ylim(0, top * 1.06); axes[0].set_ylabel("water pixels in the frame (224 px, mean)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, bbox_to_anchor=(0.995, 1.0))
    fig.suptitle("Total water in the whole-episode rollout vs ground truth", x=0.01, ha="left", fontsize=10.5, fontweight="semibold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(out, dpi=200); plt.close(fig)


def main():
    res, tro, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    A, B = json.load(open(res / "h3f5_ep6.json")), json.load(open(res / "h1f5_ep6.json"))
    rows = lambda j: [json.loads(l) for l in open(tro / j / "epoch_logs.jsonl")]
    fig_training(out / "fig1_training.png", rows("train_pw_h3f5"), rows("train_pw_h1f5"))
    fig_steps(out / "fig2_rollout_steps.png", A, B)
    fig_full(out / "fig3_full_episode.png", A, B)
    if (res / "h3f5_volume.npz").exists() and (res / "h1f5_volume.npz").exists():
        VA, VB = np.load(res / "h3f5_volume.npz"), np.load(res / "h1f5_volume.npz")
        fig_fill(out / "fig4_fill_level.png", VA, VB)
        fig_time(out / "fig5_water_over_time.png", VA, VB)
    print("figures in", out)


if __name__ == "__main__":
    main()
