"""2x2 summary table (train camera x eval camera), 5-step rollout latent MSE, as a slide image.

Numbers from <out>/rollout_mse_fixed_model.json (written by rollout_mse_camera.py).

    python scripts/camera_summary_table.py --out $DINO_WORK/results/camera_motion/talk
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
HERE = Path(__file__).resolve().parent
INK, MUTED = "#1a1a19", "#5f5e5a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    res = json.load(open(out / "rollout_mse_fixed_model.json"))["per_episode"]
    h5 = lambda key: np.array(res[key])[:, -1].mean()
    rows = [("Train: single camera", ""), ("Train: moving camera", "(+ camera pose change as action)")]
    keys = [["{o}/eval_fixed", "{o}/eval_move"], ["told/{o}/eval_fixed", "told/{o}/eval_move"]]
    vals = [[[h5(k.format(o=o)) for o in ("rope", "granular")] for k in row] for row in keys]
    best = [int(np.argmin([vals[r][c][0] for r in range(2)])) for c in range(2)]
    assert best == [0, 1], best

    fs = 15
    fig = plt.figure(figsize=(8.6, 2.75), facecolor="white")
    X = [0.03, 0.50, 0.80]          # column anchors (left, center, center)
    def T(x, y, s, **kw):
        kw.setdefault("color", INK)
        fig.text(x, y, s, fontsize=kw.pop("size", fs), va="center", **kw)
    def rule(y, lw):
        fig.add_artist(plt.Line2D([0.02, 0.98], [y, y], lw=lw, color=INK))

    rule(0.95, 1.2)
    T(X[1], 0.84, "Eval: training camera", ha="center", weight="bold")
    T(X[2], 0.84, "Eval: moving camera", ha="center", weight="bold")
    T(X[1], 0.73, "Rope / Granular", ha="center", size=fs - 2, color=MUTED)
    T(X[2], 0.73, "Rope / Granular", ha="center", size=fs - 2, color=MUTED)
    rule(0.655, 0.6)
    for r, (name, note) in enumerate(rows):
        y = 0.53 - r * 0.21
        if note:
            T(X[0], y + 0.04, name, ha="left")
            T(X[0], y - 0.05, note, ha="left", size=fs - 3, color=MUTED)
        else:
            T(X[0], y, name, ha="left")
        for c in range(2):
            v = vals[r][c]
            T(X[c + 1], y, f"{v[0]:.2f} / {v[1]:.2f}", ha="center", weight="bold" if best[c] == r else "normal")
    rule(0.20, 1.2)
    T(0.03, 0.10, "5-step rollout latent MSE over 100 evaluation episodes. Bold: lowest in each column.",
      ha="left", size=fs - 3, color=MUTED)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"summary_table.{ext}", dpi=300 if ext == "png" else None, facecolor="white")
    print("wrote", out / "summary_table.{png,svg,pdf}", vals)


if __name__ == "__main__":
    main()
