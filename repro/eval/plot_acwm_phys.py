"""ACWM-Phys Cloth Move·Push Sand 의 물체 면적 곡선: 전체 에피소드 롤아웃에서 정답 / 모델 / 정답을 디코더에 통과시킨 재구성.
테스트 그룹별 패널 5개 (Cloth Move InD·작은 천·큰 천, Push Sand InD·OoD). eval_acwm_phys_quant.py 의 <env>_h1f5_quant.json 만 읽는다.
usage: python repro/eval/plot_acwm_phys.py <dir with the *_quant.json files> <out.png>   (예: $DINO_WORK/results/acwm_phys)"""
import sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, OUT = sys.argv[1], sys.argv[2]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
REAL, MODEL, RECON = "#2a78d6", "#eb6834", "#8a8984"
panels = [("cloth_h1f5", "cloth", "ind_test", "all", "Cloth Move  InD test (training size)"),
          ("cloth_h1f5", "cloth", "ood_test", "smaller", "Cloth Move  OoD smaller cloth"),
          ("cloth_h1f5", "cloth", "ood_test", "larger", "Cloth Move  OoD larger cloth"),
          ("sand_h1f5", "sand", "ind_test", "all", "Push Sand  InD test"),
          ("sand_h1f5", "sand", "ood_test", "all", "Push Sand  OoD (2x particles)")]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.3), facecolor=SURFACE, gridspec_kw={"wspace": 0.16, "width_ratios": [1, 1, 1, 1, 1]})
cache = {}
for ax, (tag, obj, st, g, title) in zip(axes, panels):
    r = cache.setdefault(tag, json.load(open(f"{D}/{tag}_quant.json")))
    d = r["sets"][st]["groups"][g][obj]
    f = r["frameskip"]
    x = [k * f for k in range(len(d["px_curve"]["gt"]))]
    ax.set_facecolor(SURFACE)
    ax.plot(x, d["px_curve"]["recon"], color=RECON, lw=1.2, solid_capstyle="round", label="Decoder on real frames", zorder=2)
    ax.plot(x, d["px_curve"]["gt"], color=REAL, lw=2, solid_capstyle="round", solid_joinstyle="round", label="Real", zorder=3)
    ax.plot(x, d["px_curve"]["model"], color=MODEL, lw=2, solid_capstyle="round", solid_joinstyle="round", label="Model rollout", zorder=4)
    top = 5900 if obj == "cloth" else 4300
    ends = {key: d["px_curve"][key][-1] for key in ("gt", "model")}
    close = abs(ends["gt"] - ends["model"]) < 0.07 * top  # end labels would overlap: put one above and one below its marker
    for key, col in (("gt", REAL), ("model", MODEL)):
        ax.plot(x[-1], ends[key], "o", ms=6, color=col, mec=SURFACE, mew=1.5, zorder=5)
        up = ends[key] >= max(ends.values())
        ax.annotate(f"{ends[key]:.0f}", (x[-1], ends[key]), xytext=(6, (5 if up else -5) if close else 0), textcoords="offset points",
                    va=("bottom" if up else "top") if close else "center", fontsize=8.5, color=INK)
    ax.set_ylim(0, top); ax.set_xlim(0, 76); ax.set_xticks([0, 20, 40, 60])
    if title.split("  ")[1].startswith("OoD"):
        ax.tick_params(labelleft=False)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=8)
    ax.set_xlabel("frame")
axes[0].set_ylabel("cloth pixels (224 x 224 frame)")
axes[3].set_ylabel("sand pixels (224 x 224 frame)")
h, l = axes[0].get_legend_handles_labels()
order = [1, 2, 0]
fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 1.08))
pos = [a.get_position() for a in axes]
for a, p_ in zip(axes[3:], pos[3:]):
    a.set_position([p_.x0 + 0.035, p_.y0, p_.width, p_.height])  # gap between the two environments (their y axes differ)
fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor=SURFACE)
print("wrote", OUT)
