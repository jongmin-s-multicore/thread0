"""ACWM-Phys four-environment results in the layout of the paper's Table 1 (markdown + LaTeX).

usage: python repro/acwm_dit/paper_table.py      (files below are under $DINO_WORK)
writes results/acwm_dit_baseline/paper_table.md, paper_table.tex

Table A  ACWM-DiT-S, official eval.py on all 37 frames: paper vs the released checkpoint (first 50 episodes = eval_all.sh,
         and all test episodes).
Table B  DINO-WM vs ACWM-DiT-S on the same frames: every test episode, 10 windows of 37 frames each, scored on window
         frames 5, 10, ..., 35 (Pour Water 15, ..., 35), the frames a frameskip-5 DINO-WM predicts. MSE / M-MSE / SSIM are
         means over episodes, PSNR = 10 log10(1 / mean MSE) as in the paper. "Copy first frame" repeats window frame 0.
"""
import json, math, os

HERE = os.path.join(os.environ.get("DINO_WORK", os.path.expanduser("~/dinowm")), "results", "acwm_dit_baseline")
RES = os.path.dirname(HERE)
ENVS = [("stack_cube", "Stack Cube", "acwm_phys/stack_h1f5.json", 25), ("push_sand", "Push Sand", "acwm_phys/sand_h1f5.json", 18),
        ("clothmove", "Cloth Move", "acwm_phys/cloth_h1f5.json", 15), ("pour_water", "Pour Water", "pour_water/h1f5_ep6.json", 6)]
SPLITS = ("ind_test", "ood_test")
# arXiv 2605.08567 Table 1 (Cloth Move: Table 7, 8-dim actions, MSE and PSNR only); (MSE, M-MSE, SSIM, PSNR), MSE x1e-3
PAPER = {"stack_cube": [(5.516, 10.93, 0.889, 22.58), (6.996, 13.40, 0.872, 21.55)],
         "push_sand": [(0.519, 6.22, 0.975, 32.85), (1.526, 15.54, 0.941, 28.16)],
         "clothmove": [(12.861, None, None, 18.91), (6.922, None, None, 21.60)],
         "pour_water": [(2.630, 11.81, 0.911, 25.80), (3.488, 14.21, 0.874, 24.57)]}


def row_from(m, mse, mmse, ssim):
    return (1e3 * m[mse], 1e3 * m[mmse], m[ssim], 10 * math.log10(1 / m[mse]))


def main():
    A, B = {}, {}
    for env, name, dpath, ep in ENVS:
        s = json.load(open(os.path.join(HERE, "summary", f"{env}_summary.json")))
        d = json.load(open(os.path.join(RES, dpath)))
        A[env] = {"paper": PAPER[env], "step": s["checkpoint_step"], "first50": [], "all": []}
        B[env] = {"DINO-WM": [], "ACWM-DiT-S": [], "Copy first frame": [], "dino_epochs": ep, "step": s["checkpoint_step"]}
        for sp in SPLITS:
            o = s["splits"][sp]["official"]
            f50 = o.get("shipped_first_250_batches", o["all"])  # splits with 50 episodes: the first 50 are all of them
            for key, m in (("first50", f50), ("all", o["all"])):
                A[env][key].append((1e3 * m["mse"], 1e3 * m["masked_mse"], m["ssim"], m["psnr_from_mse"]))
            dm = d["sets"][sp]["acwm"]["all"]
            B[env]["DINO-WM"].append(row_from({k: v[0] for k, v in dm["model"].items()}, "mse", "masked_mse", "ssim"))
            B[env]["Copy first frame"].append(row_from({k: v[0] for k, v in dm["copy_first"].items()}, "mse", "masked_mse", "ssim"))
            am = s["splits"][sp]["dino"]["model"]["all"]
            B[env]["ACWM-DiT-S"].append(row_from({k: v[0] for k, v in am.items()}, "mse", "masked_mse", "ssim"))

    f = lambda x, nd: "–" if x is None else f"{x:.{nd}f}"
    nds = (3, 2, 3, 2)
    better = (min, min, max, max)
    md = ["# ACWM-Phys, four environments", "",
          "MSE and M-MSE are scaled by 10⁻³. ↓ lower is better, ↑ higher is better. PSNR = 10 log10(1 / mean MSE) (the paper's convention).", ""]
    hdr = "| Env | Model | InD MSE↓ | InD M-MSE↓ | InD SSIM↑ | InD PSNR↑ | OoD MSE↓ | OoD M-MSE↓ | OoD SSIM↑ | OoD PSNR↑ |"
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    md += ["## Table A. ACWM-DiT-S with the official evaluation (all 37 frames of each window)", "",
           "First 50 = what `scripts/eval_all.sh` scores (`--max_trajs 50` takes the first 50 entries of metadata.pt).", "", hdr, sep]
    for env, name, _, _ in ENVS:
        rows = [("Paper", A[env]["paper"]), ("Released ckpt, first 50 episodes", A[env]["first50"]), ("Released ckpt, all episodes", A[env]["all"])]
        for i, (lab, r) in enumerate(rows):
            cells = [f(r[s][k], nds[k]) for s in range(2) for k in range(4)]
            md.append(f"| {name if i == 0 else ''} | {lab} | " + " | ".join(cells) + " |")
    md += ["", "Cloth Move paper row: Table 7, 8-dim actions (the released checkpoint; Table 1 lists the 3-dim variant). "
           f"The released Cloth Move checkpoint is at step {A['clothmove']['step']:,}; the others at 100,000.", ""]

    md += ["## Table B. DINO-WM vs ACWM-DiT-S on the same frames", "",
           "All test episodes, 10 windows of 37 frames per episode, scored on window frames 5, 10, ..., 35 (Pour Water 15, ..., 35). "
           "Bold: the better of the two models.", "", hdr, sep]
    for env, name, _, ep in ENVS:
        for i, lab in enumerate(("DINO-WM", "ACWM-DiT-S", "Copy first frame")):
            cells = []
            for s in range(2):
                for k in range(4):
                    v = B[env][lab][s][k]
                    if lab != "Copy first frame":
                        best = better[k](B[env]["DINO-WM"][s][k], B[env]["ACWM-DiT-S"][s][k])
                        cells.append(f"**{f(v, nds[k])}**" if v == best else f(v, nds[k]))
                    else:
                        cells.append(f(v, nds[k]))
            md.append(f"| {name if i == 0 else ''} | {lab} | " + " | ".join(cells) + " |")
    md += ["", "DINO-WM: frameskip 5, num_hist 1, batch 24, trained "
           + ", ".join(f"{n} {B[e]['dino_epochs']} epochs" for e, n, _, _ in ENVS)
           + " (about 62k–65k optimizer steps each). ACWM-DiT-S: the released checkpoints "
           f"(Cloth Move step {B['clothmove']['step']:,}, the others 100,000). DINO-WM decodes 224×224 and is upsampled to 240×240 "
           "(240×400 for Push Sand) for scoring; ACWM-DiT-S decodes at that size.", ""]
    open(os.path.join(HERE, "paper_table.md"), "w").write("\n".join(md) + "\n")

    # LaTeX (booktabs)
    def tex_cell(v, k, bold=False):
        s = "--" if v is None else f"{v:.{nds[k]}f}"
        return f"\\textbf{{{s}}}" if bold else s
    tx = [r"% generated by results/acwm_dit_baseline/paper_table.py", r"\begin{table*}[t]", r"\centering", r"\small",
          r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llcccccccc}", r"\toprule",
          r" & & \multicolumn{4}{c}{In-Distribution Test} & \multicolumn{4}{c}{Out-of-Distribution Test} \\",
          r"\cmidrule(lr){3-6} \cmidrule(lr){7-10}",
          r"Env & Model & MSE$\downarrow$ & M-MSE$\downarrow$ & SSIM$\uparrow$ & PSNR$\uparrow$ & MSE$\downarrow$ & M-MSE$\downarrow$ & SSIM$\uparrow$ & PSNR$\uparrow$ \\",
          r"\midrule"]
    for j, (env, name, _, _) in enumerate(ENVS):
        labs = ("DINO-WM", "ACWM-DiT-S", "Copy first frame")
        for i, lab in enumerate(labs):
            cells = []
            for s in range(2):
                for k in range(4):
                    v = B[env][lab][s][k]
                    bold = lab != "Copy first frame" and v == better[k](B[env]["DINO-WM"][s][k], B[env]["ACWM-DiT-S"][s][k])
                    cells.append(tex_cell(v, k, bold))
            first = f"\\multirow{{3}}{{*}}{{{name}}}" if i == 0 else ""
            shown = lab if lab != "Copy first frame" else r"\textit{Copy first frame}"
            tx.append(f"{first} & {shown} & " + " & ".join(cells) + r" \\")
        if j < len(ENVS) - 1:
            tx.append(r"\midrule")
    tx += [r"\bottomrule", r"\end{tabular}",
           r"\caption{Open-loop prediction on ACWM-Phys. Every test episode, 10 windows of 37 frames per episode; both models are scored "
           r"on window frames $5,10,\dots,35$ (Pour Water $15,\dots,35$), the frames a frameskip-5 DINO-WM predicts. MSE and M-MSE are "
           r"scaled by $10^{-3}$; PSNR $=10\log_{10}(1/\overline{\mathrm{MSE}})$. ACWM-DiT-S: released checkpoints (Cloth Move at step 11k). "
           r"Bold: better of the two models.}",
           r"\label{tab:acwm_dinowm}", r"\end{table*}"]
    open(os.path.join(HERE, "paper_table.tex"), "w").write("\n".join(tx) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
