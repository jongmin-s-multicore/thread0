"""rollout_diff.json (rollout_diff_analyze.py) 로 그림 세 장을 만든다.

  acwm_oodmix_mse_ratio.png  모델 스텝마다 InD+OoD 모델의 MSE / InD 만 학습한 모델의 MSE (1 보다 작으면 InD+OoD 가 낫다). 띠 = 95 % 구간.
  acwm_oodmix_retention.png  Cloth Move, Push Sand 에서 OoD 차이를 얼마나 남기는가 r(k). 파란 선 = Real (실제 프레임의 OoD 차이 / Encoder->Decoder 의
                             OoD 차이, 1 이면 decoder 가 실제 차이를 그린다). cloth_control_curves.npz 가 있으면 Cloth Move 에 InD 만 다시 학습한 모델이 같이 있다.
  acwm_oodmix_amount.png     모델 스텝마다 물체의 양: Real, InD only, InD+OoD. 환경마다 한 줄, 평가 그룹마다 한 칸. 띠 = 그룹 평균의 95 % 구간.

usage (env sourced): python <this> [dir] [png dir]
  dir      rollout_diff.json 이 있는 곳 (기본 $DINO_WORK/results/acwm_phys/ood_mix/rollout_diff)
  png dir  그림을 쓸 곳 (기본 dir). 루트 README 의 그림은 assets 로 쓴다.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = os.environ["DINO_WORK"]
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORK, "results", "acwm_phys", "ood_mix", "rollout_diff")
PNG = sys.argv[2] if len(sys.argv) > 2 else OUT
R = json.load(open(f"{OUT}/rollout_diff.json"))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
ORANGE, PURPLE, GREY, BLUE = "#eb6834", "#4a3aa7", "#8b8a85", "#2a78d6"  # InD only (talk figures' model color), InD+OoD, second InD-only run, Real (talk figures' Real color)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans",
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
TITLE = {"cloth": "Cloth Move", "sand": "Push Sand", "water": "Pour Water", "stack": "Stack Cube"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0, labelsize=8)


def fig_ratio():
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))
    # (group key, label, color, linestyle)
    LINES = {
        "cloth": [("ind_test/all", "InD", INK2, "-"), ("ood_test/smaller", "OoD 0.26x area", PURPLE, "--"), ("ood_test/larger", "OoD 2.5x area", PURPLE, "-")],
        "sand": [("ind_test/all", "InD", INK2, "-"), ("ood_test/all", "OoD 2x particles", PURPLE, "-")],
        "water": [("ind_test/all", "InD", INK2, "-"), ("ood_test/low", "OoD less water", PURPLE, "--"), ("ood_test/high", "OoD more water", PURPLE, "-")],
        "stack": [("ind_test/all", "InD", INK2, "-"), ("ind_test/fail", "InD, failed", INK2, "--"), ("ood_test/all", "OoD", PURPLE, "-"), ("ood_test/fail", "OoD, failed", PURPLE, "--")],
    }
    for ax, env in zip(axes.ravel(), ("cloth", "sand", "water", "stack")):
        for key, lab, col, ls in LINES[env]:
            s = R[env]["groups"][key]["steps"]
            x = np.arange(len(s["ratio"]))
            r, lo, hi = (np.array(s[k], float) for k in ("ratio", "ratio_lo", "ratio_hi"))
            ok = x >= 1
            ax.fill_between(x[ok], lo[ok], hi[ok], color=col, alpha=0.13, linewidth=0, zorder=2)
            ax.plot(x[ok], r[ok], color=col, linestyle=ls, linewidth=1.6, label=lab, zorder=3)
        ax.axhline(1, color=INK, linewidth=0.8, zorder=1)
        ax.set_yscale("log")
        ax.set_yticks([0.25, 0.5, 1, 2])
        ax.set_yticklabels(["0.25", "0.5", "1", "2"])
        ax.set_ylim(0.2, 3.0)
        ax.minorticks_off()
        ax.set_xlim(1, len(s["ratio"]) - 1)
        ax.text(0.0, 1.03, TITLE[env], transform=ax.transAxes, fontsize=9.5, color=INK, va="bottom")
        ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK2, handlelength=2.2)
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel("rollout step (5 frames)")
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE, InD+OoD / InD only")
    fig.tight_layout(h_pad=2.4, w_pad=2.2)
    fig.savefig(f"{PNG}/acwm_oodmix_mse_ratio.png", dpi=200)
    plt.close(fig)


def fig_retention():
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.5), sharey=True)
    ctrl = R["cloth"].get("control")
    panels = [("Cloth Move, OoD 2.5x area", R["cloth"]["retention"]["larger"], ctrl and ctrl["retention"]["larger"]),
              ("Cloth Move, OoD 0.26x area", R["cloth"]["retention"]["smaller"], ctrl and ctrl["retention"]["smaller"]),
              ("Push Sand, OoD 2x particles", R["sand"]["retention"]["all"], None)]
    for ax, (name, rr, cc) in zip(axes, panels):
        lines = [("Real", rr["real"], BLUE, "-"), ("InD only", rr["ind_only"], ORANGE, "-")]
        if cc:
            lines.append(("InD only, 2nd run", cc["control"], GREY, "-"))
        lines.append(("InD+OoD", rr["ind_ood"], PURPLE, "-"))
        for lab, v, col, ls in lines:
            r, lo, hi = (np.array(v[k], float) for k in ("r", "lo", "hi"))
            x = np.arange(len(r))
            ax.fill_between(x[1:], lo[1:], hi[1:], color=col, alpha=0.13, linewidth=0, zorder=2)
            ax.plot(x[1:], r[1:], color=col, linestyle=ls, linewidth=1.6, label=lab, zorder=3)
        ax.axhline(0, color=INK, linewidth=0.8, zorder=1)
        ax.axhline(1, color=GRID, linewidth=1.0, zorder=1)
        ax.set_xlim(1, 12)
        ax.set_ylim(-0.45, 1.6)
        ax.set_xticks([1, 3, 6, 9, 12])
        ax.text(0.0, 1.03, name, transform=ax.transAxes, fontsize=9.5, color=INK, va="bottom")
        ax.set_xlabel("rollout step (5 frames)")
        ax.legend(frameon=False, fontsize=7.5, loc="lower left", labelcolor=INK2, handlelength=2.2)
        style(ax)
    axes[0].set_ylabel("retention r(k)")
    fig.tight_layout(w_pad=2.0)
    fig.savefig(f"{PNG}/acwm_oodmix_retention.png", dpi=200)
    plt.close(fig)


def fig_amount():
    rows = [("cloth", [("ind_test/all", "InD"), ("ood_test/smaller", "OoD 0.26x area"), ("ood_test/larger", "OoD 2.5x area")], "cloth pixels"),
            ("sand", [("ind_test/all", "InD"), ("ood_test/all", "OoD 2x particles")], "sand pixels"),
            ("water", [("ind_test/all", "InD"), ("ood_test/low", "OoD less water"), ("ood_test/high", "OoD more water")], "water pixels in the box"),
            ("stack", [("ind_test/fail", "InD, failed"), ("ood_test/fail", "OoD, failed")], "drawn as stacked")]
    fig, axes = plt.subplots(4, 3, figsize=(10.2, 10.6), sharey="row")
    handles = None
    top = {env: 1.06 * max(np.nanmax(R[env]["groups"][key]["amount_steps"][kind]["hi"]) for key, _ in cols for kind in ("real", "ind_only", "ind_ood")) for env, cols, _ in rows}
    for ri, (env, cols, ylab) in enumerate(rows):
        for ci in range(3):
            ax = axes[ri, ci]
            if ci >= len(cols):
                ax.axis("off")
                continue
            key, lab = cols[ci]
            a = R[env]["groups"][key]["amount_steps"]
            for kind, col, name, lw in (("real", BLUE, "Real", 2.0), ("ind_only", ORANGE, "InD only", 1.6), ("ind_ood", PURPLE, "InD+OoD", 1.6)):
                v = a[kind]
                x = np.arange(len(v["mean"]))
                ax.fill_between(x, v["lo"], v["hi"], color=col, alpha=0.13, linewidth=0, zorder=2)
                ax.plot(x, v["mean"], color=col, linewidth=lw, label=name, zorder=3)
            ax.set_xlim(0, len(x) - 1)
            ax.set_ylim(0, top[env])
            ax.text(0.0, 1.03, lab, transform=ax.transAxes, fontsize=9.5, color=INK, va="bottom")
            ax.set_xlabel("rollout step (5 frames)", fontsize=8)
            if ci == 0:
                ax.set_ylabel(ylab)
                handles = ax.get_legend_handles_labels()
            style(ax)
    fig.legend(*handles, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK2, handlelength=2.4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.2, w_pad=2.0)
    fig.savefig(f"{PNG}/acwm_oodmix_amount.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(PNG, exist_ok=True)
    fig_ratio()
    fig_retention()
    fig_amount()
    print("wrote acwm_oodmix_mse_ratio.png acwm_oodmix_retention.png acwm_oodmix_amount.png in", PNG)
