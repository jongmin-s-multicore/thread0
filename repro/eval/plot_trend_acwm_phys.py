"""학습량에 따른 물리량: trend_acwm_phys.py 의 JSON 여러 개(원래 학습의 epoch 들, 이어 학습한 epoch 들)를 모아
마지막 프레임의 물체 픽셀 수(Cloth Move, Push Sand)와 쌓임 비율(Stack Cube)을 학습 step 배수에 대해 그린다.
파란 선은 실제 값, 주황 선은 모델, 회색 선은 같은 모델의 학습 분포 안(InD) 값이다.

usage: python repro/eval/plot_trend_acwm_phys.py <out.png> cloth=<base epoch>:<json>[,<json>...] sand=... stack=...
  예: cloth=15:$R/cloth_h1f5_trend.json,$R/long/cloth_ep30.json,$R/long/cloth_ep60.json,$R/long/cloth_ep120.json
"""
import sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
REAL, MODEL, REF = "#2a78d6", "#eb6834", "#8a8984"


def load(spec):
    base, paths = spec.split(":", 1)
    out = {}
    for p in paths.split(","):
        r = json.load(open(p))
        for st, groups in r["sets"].items():
            for g, by_epoch in groups.items():
                for e, v in by_epoch.items():
                    out.setdefault((st, g), {})[int(e) / int(base)] = v
    return out


specs = dict(a.split("=", 1) for a in sys.argv[2:])
panels = []  # (title, y label, series for the model, real value, InD reference series or None, percent?)
if "cloth" in specs:
    d = load(specs["cloth"])
    for g, title in (("larger", "Cloth Move, OoD larger cloth"), ("smaller", "Cloth Move, OoD smaller cloth")):
        panels.append((title, "cloth pixels, last frame", d[("ood_test", g)], d[("ind_test", "all")], False))
if "sand" in specs:
    d = load(specs["sand"])
    panels.append(("Push Sand, OoD 2x particles", "sand pixels, last frame", d[("ood_test", "all")], d[("ind_test", "all")], False))
if "stack" in specs:
    d = load(specs["stack"])
    panels.append(("Stack Cube, failed OoD episodes", "last frame shows a stacked cube", d[("ood_test", "fail")], None, True))

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.2), facecolor=SURFACE, gridspec_kw={"wspace": 0.42})
for ax, (title, ylab, model, ind, pct) in zip(axes, panels):
    xs = sorted(x for x in model if x >= 1)
    k = 100.0 if pct else 1.0
    real = model[xs[-1]]["real"] * k
    ym = [model[x]["model"] * k for x in xs]
    ax.set_facecolor(SURFACE)
    ax.axhline(real, color=REAL, lw=2, label="Real", zorder=2)
    if ind is not None:
        ax.plot(xs, [ind[x]["model"] for x in xs], color=REF, lw=1.2, marker="o", ms=3.5, label="Model on InD episodes (training size)", zorder=3)
    ax.plot(xs, ym, color=MODEL, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, solid_capstyle="round", label="Model", zorder=4)
    for x, y in zip(xs, ym):
        ax.annotate(f"{y:.0f}{' %' if pct else ''}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.annotate(f"real {real:.0f}{' %' if pct else ''}", (xs[-1] * 1.4, real), xytext=(0, 4), textcoords="offset points", ha="right", va="bottom", fontsize=8.5, color=INK)
    ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}x" for x in xs]); ax.minorticks_off()
    top = max(real, max(ym), max((ind[x]["model"] for x in xs), default=0) if ind else 0) * 1.22
    ax.set_ylim(0, top); ax.set_xlim(xs[0] * 0.8, xs[-1] * 1.45)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=8)
    ax.set_xlabel("optimizer steps, relative to the original run"); ax.set_ylabel(ylab + (" (%)" if pct else ""))
h, l = axes[0].get_legend_handles_labels()
order = [l.index(x) for x in ("Real", "Model", "Model on InD episodes (training size)") if x in l]
fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 1.09))
fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor=SURFACE)
print("wrote", OUT)
