"""ACWM-DiT-S (released checkpoints, run on super3) next to the paper and next to our DINO-WM models.

usage: python repro/acwm_dit/compare.py      (files below are under $DINO_WORK)
reads  results/acwm_dit_baseline/summary/<env>_summary.json   (copied from super3 ~/acwm_run/results/main/)
       results/acwm_phys/{cloth,sand,stack}_h1f5.json, results/pour_water/h1f5_ep6.json   (DINO-WM, "acwm" block)
writes results/acwm_dit_baseline/compare.md

Table 1: official eval.py numbers vs paper Table 1 (all 37 frames of each window).
Table 2: same frames for both models (frameskip-5 frames of each window, see analyze.py), mean over episodes [95 % interval].
"""
import json, os

ROOT = os.path.join(os.environ.get("DINO_WORK", os.path.expanduser("~/dinowm")), "results", "acwm_dit_baseline")
RES = os.path.dirname(ROOT)

# arXiv 2605.08567 Table 1 (ACWM-DiT-S, 100k steps, 50 steps); MSE and M-MSE x1e-3. Cloth Move row is the 3-dim action
# variant; the released checkpoint is the 8-dim one, whose numbers are in Table 7 (MSE and PSNR only).
PAPER = {
    "push_cube": {"ind_test": (2.919, 11.43, 25.35, 0.955), "ood_test": (2.950, 7.47, 25.30, 0.954)},
    "stack_cube": {"ind_test": (5.516, 10.93, 22.58, 0.889), "ood_test": (6.996, 13.40, 21.55, 0.872)},
    "push_rope": {"ind_test": (0.214, 2.61, 36.70, 0.988), "ood_test": (0.329, 12.86, 34.83, 0.985)},
    "clothmove": {"ind_test": (12.861, None, 18.91, None), "ood_test": (6.922, None, 21.60, None)},
    "push_sand": {"ind_test": (0.519, 6.22, 32.85, 0.975), "ood_test": (1.526, 15.54, 28.16, 0.941)},
    "pour_water": {"ind_test": (2.630, 11.81, 25.80, 0.911), "ood_test": (3.488, 14.21, 24.57, 0.874)},
    "robot_arm": {"ind_test": (1.434, 13.45, 28.43, 0.969), "ood_test": (6.559, 53.80, 21.83, 0.902)},
    "reacher": {"ind_test": (0.260, 5.63, 35.85, 0.992), "ood_test": (0.272, 8.89, 35.65, 0.992)},
}
DINO = {
    "pour_water": ("pour_water/h1f5_ep6.json", "DINO-WM h1f5 (6 ep)"),
    "clothmove": ("acwm_phys/cloth_h1f5.json", "DINO-WM h1f5 (15 ep)"),
    "push_sand": ("acwm_phys/sand_h1f5.json", "DINO-WM h1f5 (18 ep)"),
    "stack_cube": ("acwm_phys/stack_h1f5.json", "DINO-WM h1f5 (25 ep)"),
}
ORDER = ["pour_water", "clothmove", "push_sand", "stack_cube", "push_rope", "push_cube", "robot_arm", "reacher"]
NAME = {"pour_water": "Pour Water", "clothmove": "Cloth Move", "push_sand": "Push Sand", "stack_cube": "Stack Cube",
        "push_rope": "Push Rope", "push_cube": "Push Cube", "robot_arm": "Robot Arm", "reacher": "Reacher"}
SPLIT = {"ind_test": "InD", "ood_test": "OoD"}


def f(x, scale=1.0, nd=3):
    return "-" if x is None else f"{x * scale:.{nd}f}"


def ci(v, scale=1.0, nd=3):
    if v is None or v[0] is None:
        return "-"
    return f"{v[0] * scale:.{nd}f} [{v[1] * scale:.{nd}f}, {v[2] * scale:.{nd}f}]"


def main():
    sums = {}
    for env in ORDER:
        p = os.path.join(ROOT, "summary", f"{env}_summary.json")
        if os.path.exists(p):
            sums[env] = json.load(open(p))
    out = ["# ACWM-DiT-S baseline (released checkpoints) vs paper and DINO-WM", ""]
    out += ["MSE and M-MSE are x1e-3. Status per environment: windows done / total.", ""]

    out += ["## 1. Official eval.py vs paper Table 1", "",
            "All 37 frames of each window, batch 2, 50 steps, seed 0. \"all episodes\" uses every test episode (10 windows each);",
            "\"first 50\" is what eval_all.sh evaluates (`--max_trajs 50` takes the first 50 entries of metadata.pt).",
            "PSNR (eval.py) is the mean of per-batch PSNR; the paper's PSNR equals 10 log10(1/MSE), given as \"PSNR from MSE\".", "",
            "| Env | Split | Checkpoint step | Run | Windows | MSE | M-MSE | PSNR (eval.py) | PSNR from MSE | SSIM |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for env in ORDER:
        for split in ("ind_test", "ood_test"):
            pa = PAPER[env][split]
            ref = "paper Table 7 (8-dim actions)" if env == "clothmove" else "paper Table 1"
            out.append(f"| {NAME[env]} | {SPLIT[split]} | 100k | {ref} | - | {f(pa[0])} | {f(pa[1], nd=2)} | - | {f(pa[2], nd=2)} | {f(pa[3])} |")
            s = sums.get(env, {}).get("splits", {}).get(split)
            if not s:
                continue
            for run, m in s["official"].items():
                first50 = run != "all" or (s["n_meta"] > 50 and s["windows_total"] == 500)
                label = "ours, first 50 episodes" if first50 else "ours, all episodes"
                done = f"{2 * m['batches']}/{s['windows_total']}" if run == "all" else f"{2 * m['batches']}"
                out.append(f"| {NAME[env]} | {SPLIT[split]} | {sums[env]['checkpoint_step']} | {label} | {done} | {f(m['mse'], 1e3)} | "
                           f"{f(m['masked_mse'], 1e3, 2)} | {f(m['psnr'], nd=2)} | {f(m['psnr_from_mse'], nd=2)} | {f(m['ssim'])} |")

    out += ["", "## 2. Same frames for both models", "",
            "Frames 5, 10, ..., 35 of every window (Pour Water 15, ..., 35: the DINO-WM evaluation starts there so that the",
            "num_hist 3 model can be compared), all test episodes, 10 windows per episode, ACWM obs_shape, values in [0, 1].",
            "Mean over episodes with a 95 % bootstrap interval. copy_first (repeat window frame 0) is the same for both sides",
            "if the windows and frames line up.", "",
            "| Env | Split | Group | Episodes | Model | MSE | M-MSE | PSNR | SSIM | copy_first MSE |",
            "|---|---|---|---:|---|---|---|---:|---:|---:|"]
    for env in ORDER:
        if env not in DINO:
            continue
        dpath, dname = DINO[env]
        dj = json.load(open(os.path.join(RES, dpath)))
        for split in ("ind_test", "ood_test"):
            dset = dj["sets"][split]
            s = sums.get(env, {}).get("splits", {}).get(split)
            for g in dset["acwm"]:
                if g == "ratio_to_copy_first":
                    continue
                d = dset["acwm"][g]
                n_ep = dset["group_sizes"].get(g, "-")
                out.append(f"| {NAME[env]} | {SPLIT[split]} | {g} | {n_ep} | {dname} | {ci(d['model']['mse'], 1e3)} | {ci(d['model']['masked_mse'], 1e3, 2)} | "
                           f"{f(d['model']['psnr'][0], nd=2)} | {f(d['model']['ssim'][0])} | {f(d['copy_first']['mse'][0], 1e3)} |")
                if s and "dino" in s and g in s["dino"]["model"]:
                    a, c = s["dino"]["model"][g], s["dino"]["copy_first"][g]
                    tag = "" if s["complete"] else f" (partial: {s['windows_done']}/{s['windows_total']} windows)"
                    out.append(f"| {NAME[env]} | {SPLIT[split]} | {g} | {s['group_sizes'][g]} | ACWM-DiT-S released, step {sums[env]['checkpoint_step']}{tag} | "
                               f"{ci(a['mse'], 1e3)} | {ci(a['masked_mse'], 1e3, 2)} | {f(a['psnr'][0], nd=2)} | {f(a['ssim'][0])} | {f(c['mse'][0], 1e3)} |")
    out += ["", "## 3. Amounts at the end of a whole-episode rollout", "",
            "ACWM-DiT: 37-frame windows chained from frame 0 (rollout_chain.py; window 1 is the official window that starts at",
            "frame 0, each next window starts from the last predicted frame). DINO-WM: open-loop rollout of the whole episode at",
            "frameskip 5. Both measured with the same colour masks at 224x224 (amount.py = eval_acwm_phys_quant.py /",
            "eval_pour_water.py). Cloth / sand: pixels at the last DINO-WM step (frame 60). Pour Water: water pixels on the",
            "box floor averaged over frames 250-295. Stack Cube: share of episodes whose last frame shows the yellow cube on",
            "the green one. Each side's ground truth is measured from the real video with the same code.", "",
            "| Env | Split | Group | Episodes | Quantity | Real (ACWM side) | ACWM-DiT-S | Real (DINO side) | DINO-WM h1f5 |",
            "|---|---|---|---:|---|---:|---:|---:|---:|"]
    DQ = {"clothmove": ("acwm_phys/cloth_h1f5_quant.json", "cloth"), "push_sand": ("acwm_phys/sand_h1f5_quant.json", "sand"),
          "stack_cube": ("acwm_phys/stack_h1f5_quant.json", None)}
    for env in ("pour_water", "clothmove", "push_sand", "stack_cube"):
        ap = os.path.join(ROOT, "summary", f"{env}_amount.json")
        if not os.path.exists(ap):
            continue
        aj = json.load(open(ap))
        for split in ("ind_test", "ood_test"):
            sa = aj["sets"].get(split)
            if not sa:
                continue
            for g, r in sa["groups"].items():
                n = sa["group_sizes"][g]
                if env == "pour_water":
                    dj = json.load(open(os.path.join(RES, "pour_water/h1f5_ep6.json")))["sets"][split]["m3_fill"].get(g)
                    a = r["fill"]
                    out.append(f"| {NAME[env]} | {SPLIT[split]} | {g} | {n} | box-floor water px | {a['gt_px'][0]:.0f} | {a['model_px'][0]:.0f} | "
                               f"{dj['gt_px'][0]:.0f} | {dj['model_px'][0]:.0f} |" if dj else f"| {NAME[env]} | {SPLIT[split]} | {g} | {n} | box-floor water px | {a['gt_px'][0]:.0f} | {a['model_px'][0]:.0f} | - | - |")
                    continue
                dpath, obj = DQ[env]
                dq = json.load(open(os.path.join(RES, dpath)))["sets"][split]["groups"].get(g, {})
                if env == "stack_cube":
                    a, d = r["stacked_last_frame"], dq.get("stacked_last_frame")
                    dd = (f"{d['gt']['stacked_rate']:.2f} | {d['model']['stacked_rate']:.2f}") if d else "- | -"
                    out.append(f"| {NAME[env]} | {SPLIT[split]} | {g} | {n} | stacked in last frame (label {a['success_rate_label']:.2f}) | "
                               f"{a['gt']['stacked_rate']:.2f} | {a['model']['stacked_rate']:.2f} | {dd} |")
                    continue
                a, d = r[obj], dq.get(obj)
                dd = f"{d['gt_px_last'][0]:.0f} | {d['model']['px_last'][0]:.0f}" if d else "- | -"
                out.append(f"| {NAME[env]} | {SPLIT[split]} | {g} | {n} | {obj} px, last step | {a['gt_px_last'][0]:.0f} | {a['model']['px_last'][0]:.0f} | {dd} |")

    open(os.path.join(ROOT, "compare.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
