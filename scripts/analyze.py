#!/usr/bin/env python
"""Turn the eval jsonl files into tables and figures.

Reads only results/{eval,rollout,latent}/*.jsonl, so it is safe to re-run at any
time (including while an eval is still going) and never needs a GPU.

    python scripts/analyze.py --results_dir $EXP_ROOT/results

Outputs (results/analysis/):
    summary.csv / summary.md    one row per model x condition
    rollout.csv                 mean latent error per horizon
    correlation.csv             latent-cost vs true-distance correlations
    fig_success.png             success rate per condition, model A vs B
    fig_corr.png                latent cost vs fingertip-target distance scatter
    fig_rollout.png             latent error vs prediction horizon
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

CONDITION_ORDER = ["T1_clean", "T2_train_bg", "T3_val_bg", "T4_mismatch"]
MODEL_ORDER = ["A_clean", "B_distract"]


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # truncated tail from an interrupted run
    return rows


def load_dir(results_dir, sub):
    d = Path(results_dir) / sub
    rows = []
    if not d.exists():
        return rows
    for path in sorted(d.glob("*.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def dedupe(rows, keys):
    """Keep the last record per key: a resumed run may re-record an episode."""
    out = {}
    for r in rows:
        out[tuple(r.get(k) for k in keys)] = r
    return list(out.values())


def sort_key(model, cond):
    m = MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
    c = CONDITION_ORDER.index(cond) if cond in CONDITION_ORDER else len(CONDITION_ORDER)
    return (m, c, model, cond)


def wilson_ci(k, n, z=1.96):
    """Wilson score interval: honest about small n, unlike a normal approximation."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")

    def rank(a):
        order = np.argsort(a)
        r = np.empty(len(a), float)
        r[order] = np.arange(len(a), dtype=float)
        # average ties
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        if counts.max() > 1:
            sums = np.zeros(len(counts))
            np.add.at(sums, inv, r)
            r = (sums / counts)[inv]
        return r

    return pearson(rank(x), rank(y))


def quantile(values, q):
    return float(np.quantile(values, q)) if len(values) else float("nan")


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #


def build_summary(plan_rows, rollout_rows, latent_rows, horizons):
    groups = {}
    for r in plan_rows:
        groups.setdefault((r["model"], r["cond"]), []).append(r)

    roll = {}
    for r in rollout_rows:
        roll.setdefault((r["model"], r["cond"], r["horizon"]), []).append(
            r["z_visual_err"]
        )

    lat = {}
    for r in latent_rows:
        lat.setdefault((r["model"], r["cond"]), []).append(r)

    table = []
    for (model, cond) in sorted(groups, key=lambda k: sort_key(*k)):
        rows = groups[(model, cond)]
        n = len(rows)
        succ = sum(1 for r in rows if r["success"])
        lo, hi = wilson_ci(succ, n)
        final = [r["final_fingertip_target_dist"] for r in rows]
        init = [r["init_fingertip_target_dist"] for r in rows]
        entry = {
            "model": model,
            "cond": cond,
            "n": n,
            "success_rate": succ / n if n else float("nan"),
            "success_ci_lo": lo,
            "success_ci_hi": hi,
            "final_dist_mean": float(np.mean(final)) if final else float("nan"),
            "final_dist_median": quantile(final, 0.5),
            "final_dist_p25": quantile(final, 0.25),
            "final_dist_p75": quantile(final, 0.75),
            "init_dist_mean": float(np.mean(init)) if init else float("nan"),
            "dist_improvement": (
                float(np.mean(init) - np.mean(final)) if final else float("nan")
            ),
            "latent_cost_imagined": float(
                np.mean([r["latent_cost_imagined"] for r in rows])
            ),
            "latent_cost_reached": float(
                np.mean([r["latent_cost_reached"] for r in rows])
            ),
        }
        lrows = lat.get((model, cond), [])
        if lrows:
            cost = [r["latent_visual_cost"] for r in lrows]
            true = [r["fingertip_target_dist"] for r in lrows]
            entry["latent_state_pearson"] = pearson(cost, true)
            entry["latent_state_spearman"] = spearman(cost, true)
            per_ep = {}
            for r in lrows:
                per_ep.setdefault(r["episode"], ([], []))
                per_ep[r["episode"]][0].append(r["latent_visual_cost"])
                per_ep[r["episode"]][1].append(r["fingertip_target_dist"])
            within = [spearman(c, t) for c, t in per_ep.values()]
            within = [v for v in within if not math.isnan(v)]
            entry["latent_state_spearman_within_episode"] = (
                float(np.mean(within)) if within else float("nan")
            )
            entry["latent_n_pairs"] = len(lrows)
        for h in horizons:
            vals = roll.get((model, cond, h), [])
            entry[f"z_err_h{h}"] = float(np.mean(vals)) if vals else float("nan")
        table.append(entry)
    return table


def write_csv(path, rows, columns=None):
    import csv

    if not rows:
        return
    columns = columns or list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {path}")


def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        return f"{v:.{nd}f}"
    return str(v)


def write_markdown(path, table, horizons):
    cols = [
        ("model", "model"),
        ("cond", "condition"),
        ("n", "n"),
        ("success_rate", "success"),
        ("success_ci", "95% CI"),
        ("final_dist_mean", "final dist"),
        ("final_dist_median", "median"),
        ("dist_improvement", "improve"),
        ("latent_state_pearson", "r (cost,dist)"),
        ("latent_state_spearman", "rho"),
        ("latent_state_spearman_within_episode", "rho within-ep"),
    ] + [(f"z_err_h{h}", f"z_err h{h}") for h in horizons]

    lines = [
        "# Distracting Reacher: results",
        "",
        "`success` = fingertip within the target radius (0.05) of the true target.",
        "`final dist` = true fingertip-to-target distance at the end of the rollout.",
        "`r (cost,dist)` = correlation between the planner's visual latent cost and",
        "the true fingertip-target distance; it collapsing while `z_err` stays flat",
        "points at the cost function rather than the predictor.",
        "",
        "| " + " | ".join(label for _, label in cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for row in table:
        cells = []
        for key, _ in cols:
            if key == "success_ci":
                cells.append(
                    f"{fmt(row.get('success_ci_lo'), 2)}-{fmt(row.get('success_ci_hi'), 2)}"
                )
            elif key in ("n",):
                cells.append(str(row.get(key, "-")))
            elif key in ("model", "cond"):
                cells.append(str(row.get(key, "-")))
            else:
                cells.append(fmt(row.get(key)))
        lines.append("| " + " | ".join(cells) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"[md ] {path}")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #


def setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3}
    )
    return plt


def fig_success(table, out_path):
    plt = setup_mpl()
    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in table)]
    models += sorted({r["model"] for r in table} - set(models))
    conds = [c for c in CONDITION_ORDER if any(r["cond"] == c for r in table)]
    if not conds:
        return
    x = np.arange(len(conds))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(conds), 3.4))
    for mi, model in enumerate(models):
        vals, los, his = [], [], []
        for cond in conds:
            row = next(
                (r for r in table if r["model"] == model and r["cond"] == cond), None
            )
            vals.append(row["success_rate"] if row else np.nan)
            los.append(row["success_rate"] - row["success_ci_lo"] if row else 0)
            his.append(row["success_ci_hi"] - row["success_rate"] if row else 0)
        ax.bar(
            x + mi * width - 0.4 + width / 2,
            vals,
            width,
            yerr=[los, his],
            capsize=3,
            label=model,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=15)
    ax.set_ylabel("planning success rate")
    ax.set_ylim(0, 1)
    ax.set_title("Success rate by test condition (bars: Wilson 95% CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[fig] {out_path}")


def fig_corr(latent_rows, out_path):
    if not latent_rows:
        return
    plt = setup_mpl()
    models = sorted({r["model"] for r in latent_rows}, key=lambda m: sort_key(m, "")[0])
    conds = [c for c in CONDITION_ORDER if any(r["cond"] == c for r in latent_rows)]
    if not conds:
        return
    fig, axes = plt.subplots(
        len(models),
        len(conds),
        figsize=(2.5 * len(conds), 2.5 * len(models)),
        squeeze=False,
        sharex="row",
        sharey=True,
    )
    for mi, model in enumerate(models):
        for ci, cond in enumerate(conds):
            ax = axes[mi][ci]
            sel = [
                r for r in latent_rows if r["model"] == model and r["cond"] == cond
            ]
            if not sel:
                ax.set_axis_off()
                continue
            cost = [r["latent_visual_cost"] for r in sel]
            true = [r["fingertip_target_dist"] for r in sel]
            ax.scatter(cost, true, s=4, alpha=0.35)
            r_p = pearson(cost, true)
            r_s = spearman(cost, true)
            ax.set_title(f"{model} / {cond}\nr={fmt(r_p,2)}  rho={fmt(r_s,2)}", fontsize=8)
            if ci == 0:
                ax.set_ylabel("true fingertip-target dist")
            if mi == len(models) - 1:
                ax.set_xlabel("visual latent cost")
    fig.suptitle("Planning cost vs ground truth (each point = one sampled rollout)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[fig] {out_path}")


def fig_rollout(rollout_rows, out_path):
    if not rollout_rows:
        return
    plt = setup_mpl()
    agg = {}
    for r in rollout_rows:
        agg.setdefault((r["model"], r["cond"]), {}).setdefault(
            r["horizon"], []
        ).append(r["z_visual_err"])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for (model, cond) in sorted(agg, key=lambda k: sort_key(*k)):
        hs = sorted(agg[(model, cond)])
        means = [float(np.mean(agg[(model, cond)][h])) for h in hs]
        ls = "-" if model == "A_clean" else "--"
        ax.plot(hs, means, ls, marker="o", ms=3, label=f"{model} / {cond}")
    ax.set_xlabel("prediction horizon (world-model steps)")
    ax.set_ylabel("open-loop latent error (MSE)")
    ax.set_yscale("log")
    ax.set_title("Open-loop latent error vs horizon")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[fig] {out_path}")


# --------------------------------------------------------------------------- #


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--out_dir", default=None, help="default: <results_dir>/analysis")
    p.add_argument("--horizons", default="1,3,5,10")
    args = p.parse_args(argv)

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir or results_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    plan_rows = dedupe(load_dir(results_dir, "eval"), ["model", "cond", "episode"])
    rollout_rows = dedupe(
        load_dir(results_dir, "rollout"), ["model", "cond", "episode", "horizon"]
    )
    latent_rows = dedupe(
        load_dir(results_dir, "latent"), ["model", "cond", "episode", "sample"]
    )
    print(
        f"[load] plan={len(plan_rows)} rollout={len(rollout_rows)} "
        f"latent={len(latent_rows)} records"
    )
    if not plan_rows and not rollout_rows and not latent_rows:
        raise SystemExit(f"No records found under {results_dir}. Run scripts/eval.py first.")

    table = build_summary(plan_rows, rollout_rows, latent_rows, horizons)
    write_csv(out_dir / "summary.csv", table)
    write_markdown(out_dir / "summary.md", table, horizons)

    # per-horizon long table
    roll_agg = {}
    for r in rollout_rows:
        roll_agg.setdefault((r["model"], r["cond"], r["horizon"]), []).append(r)
    roll_table = [
        {
            "model": m,
            "cond": c,
            "horizon": h,
            "n": len(v),
            "z_visual_err_mean": float(np.mean([x["z_visual_err"] for x in v])),
            "z_visual_err_std": float(np.std([x["z_visual_err"] for x in v])),
            "z_proprio_err_mean": float(np.mean([x["z_proprio_err"] for x in v])),
        }
        for (m, c, h), v in sorted(roll_agg.items(), key=lambda kv: sort_key(kv[0][0], kv[0][1]) + (kv[0][2],))
    ]
    write_csv(out_dir / "rollout.csv", roll_table)

    corr_table = [
        {
            "model": r["model"],
            "cond": r["cond"],
            "n_pairs": r.get("latent_n_pairs"),
            "pearson": r.get("latent_state_pearson"),
            "spearman": r.get("latent_state_spearman"),
            "spearman_within_episode": r.get("latent_state_spearman_within_episode"),
        }
        for r in table
    ]
    write_csv(out_dir / "correlation.csv", corr_table)

    fig_success(table, out_dir / "fig_success.png")
    fig_corr(latent_rows, out_dir / "fig_corr.png")
    fig_rollout(rollout_rows, out_dir / "fig_rollout.png")

    print("\n" + (out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
