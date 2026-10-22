"""Cloth pixels along the rollout: Real, Enc->Dec and the models of curves.py, mean over the test episodes of each set.
Computer Modern type (the paper's LaTeX font, bundled with matplotlib), muted colors, no title; the caption goes into the document.

usage: python3 -I make_fig.py <small_only.json> <out.png>
"""
import json, sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.unicode_minus": False})
res = json.load(open(sys.argv[1]))
C = res["curves_px_by_model_step"]
F = res["meta"]["frameskip"]

# kind -> (label, color, linestyle, linewidth)
STYLE = {
    "real": ("Real", "#222222", "-", 2.6),
    "recon": ("Enc$\\to$Dec", "#222222", (0, (1, 2)), 2.0),
    "orig": ("InD only", "#8c857b", "-", 4.6),
    "ctrl": ("InD only, same code", "#8c857b", (0, (4, 2.5)), 3.6),
    "small": ("InD + smaller", "#2f7f7b", "-", 2.4),
    "prev": ("InD + smaller + larger", "#c26a2b", "-", 2.8),
}
PANELS = [("OoD larger, odd 25", "OoD larger"), ("OoD smaller, odd 25", "OoD smaller"), ("InD", "InD")]

fig, axes = plt.subplots(1, 3, figsize=(17, 6.0), dpi=100, gridspec_kw={"wspace": 0.24})
for ax, (key, name) in zip(axes, PANELS):
    for kind, (label, color, ls, lw) in STYLE.items():
        if kind in C[key]:
            y = C[key][kind]
            ax.plot([F * i for i in range(len(y))], y, color=color, linestyle=ls, linewidth=lw, label=label)
    ax.set_xlim(0, F * (len(C[key]["real"]) - 1))
    ax.set_ylim(0, 1.22 * max(max(v) for v in C[key].values()))  # room above the curves for the panel name
    ax.set_xlabel("frame", fontsize=22, labelpad=14)
    ax.tick_params(labelsize=19, length=6, width=1.2, pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(1.3)
    ax.text(0.03, 0.96, name, transform=ax.transAxes, fontsize=24, va="top", ha="left")
axes[0].set_ylabel("cloth pixels", fontsize=22, labelpad=14)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=19, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=2.8, columnspacing=3.0, labelspacing=0.7)
fig.subplots_adjust(left=0.07, right=0.985, top=0.96, bottom=0.36)
fig.savefig(sys.argv[2], dpi=100)
print("wrote", sys.argv[2])
