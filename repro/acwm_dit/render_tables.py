"""Paper-style tables (PDF + PNG) for slides: DINO-WM vs ACWM-DiT-S on ACWM-Phys (four environments).

usage: python3 repro/acwm_dit/render_tables.py      (files below are under $DINO_WORK)
needs tools/tectonic_env/bin/tectonic and pdftoppm; writes results/acwm_dit_baseline/figs/table{1,2,3}.{tex,pdf,png}

Table 1  DINO-WM vs ACWM-DiT-S on the same frames (all test episodes, window frames 5..35, Pour Water 15..35)
Table 2  ACWM-DiT-S official evaluation: paper vs released checkpoint (first 50 episodes, all episodes)
Table 3  OoD / outcome groups (MSE and M-MSE) for the same comparison as Table 1
"""
import json, math, os, subprocess

HERE = os.path.join(os.environ.get("DINO_WORK", os.path.expanduser("~/dinowm")), "results", "acwm_dit_baseline")
RES = os.path.dirname(HERE)
ROOT = os.path.dirname(RES)
OUT = os.path.join(HERE, "figs")
TECTONIC = os.path.join(ROOT, "tools/tectonic_env/bin/tectonic")
ENVS = [("stack_cube", "Stack Cube", "acwm_phys/stack_h1f5.json", 25), ("push_sand", "Push Sand", "acwm_phys/sand_h1f5.json", 18),
        ("clothmove", "Cloth Move", "acwm_phys/cloth_h1f5.json", 15), ("pour_water", "Pour Water", "pour_water/h1f5_ep6.json", 6)]
SPLITS = ("ind_test", "ood_test")
PAPER = {"stack_cube": [(5.516, 10.93, 0.889, 22.58), (6.996, 13.40, 0.872, 21.55)],
         "push_sand": [(0.519, 6.22, 0.975, 32.85), (1.526, 15.54, 0.941, 28.16)],
         "clothmove": [(12.861, None, None, 18.91), (6.922, None, None, 21.60)],
         "pour_water": [(2.630, 11.81, 0.911, 25.80), (3.488, 14.21, 0.874, 24.57)]}
ND = (3, 2, 3, 2)
BETTER = (min, min, max, max)

PREAMBLE = r"""\documentclass[border=10pt,varwidth=26cm]{standalone}
\usepackage{newtxtext,newtxmath}
\usepackage{booktabs,multirow,xcolor}
\definecolor{base}{gray}{0.45}
\begin{document}
\small
"""


def document(head, rows, cap):
    """the tabular in a save box; the caption below it is set in a minipage of the same width"""
    tab = head + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}"
    return (PREAMBLE + r"\newsavebox{\tbox}" + "\n" + r"\sbox{\tbox}{" + tab + "}" + "\n" + r"\usebox{\tbox}" + "\n\n"
            + r"\vspace{6pt}" + "\n" + r"\begin{minipage}{\wd\tbox}\footnotesize\setlength{\parindent}{0pt}" + cap + r"\end{minipage}"
            + "\n" + r"\end{document}" + "\n")


def metrics(m):
    return (1e3 * m["mse"][0], 1e3 * m["masked_mse"][0], m["ssim"][0], 10 * math.log10(1 / m["mse"][0]))


def load():
    data = {}
    for env, name, dpath, ep in ENVS:
        s = json.load(open(os.path.join(HERE, "summary", f"{env}_summary.json")))
        d = json.load(open(os.path.join(RES, dpath)))
        e = {"name": name, "ep": ep, "step": s["checkpoint_step"], "paper": PAPER[env], "first50": [], "all": [], "groups": {}}
        for sp in SPLITS:
            o = s["splits"][sp]["official"]
            f50 = o.get("shipped_first_250_batches", o["all"])
            e["first50"].append((1e3 * f50["mse"], 1e3 * f50["masked_mse"], f50["ssim"], f50["psnr_from_mse"]))
            e["all"].append((1e3 * o["all"]["mse"], 1e3 * o["all"]["masked_mse"], o["all"]["ssim"], o["all"]["psnr_from_mse"]))
            for g in d["sets"][sp]["acwm"]:
                if g == "ratio_to_copy_first" or g not in s["splits"][sp]["dino"]["model"]:
                    continue
                dm, am = d["sets"][sp]["acwm"][g], s["splits"][sp]["dino"]
                e["groups"][(sp, g)] = {"n": d["sets"][sp]["group_sizes"][g], "DINO-WM": metrics(dm["model"]), "ACWM-DiT-S": metrics(am["model"][g]),
                                        "Copy": metrics(dm["copy_first"])}
        data[env] = e
    return data


def cell(v, k, bold=False, gray=False):
    s = "--" if v is None else f"{v:.{ND[k]}f}"
    if bold:
        s = rf"\textbf{{{s}}}"
    if gray:
        s = rf"\textcolor{{base}}{{{s}}}"
    return s


HEAD = r"""\begin{tabular}{ll*{8}{c}}
\toprule
 & & \multicolumn{4}{c}{In-Distribution Test} & \multicolumn{4}{c}{Out-of-Distribution Test} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
Environment & Model & MSE$\downarrow$ & M-MSE$\downarrow$ & SSIM$\uparrow$ & PSNR$\uparrow$ & MSE$\downarrow$ & M-MSE$\downarrow$ & SSIM$\uparrow$ & PSNR$\uparrow$ \\
\midrule
"""


def table1(data):
    rows = []
    for j, (env, *_rest) in enumerate(ENVS):
        e = data[env]
        all_g = {sp: e["groups"][(sp, "all")] for sp in SPLITS}
        for i, lab in enumerate(("DINO-WM", "ACWM-DiT-S", "Copy")):
            cells = []
            for sp in SPLITS:
                g = all_g[sp]
                for k in range(4):
                    v = g[lab][k]
                    bold = lab != "Copy" and v == BETTER[k](g["DINO-WM"][k], g["ACWM-DiT-S"][k])
                    cells.append(cell(v, k, bold, gray=lab == "Copy"))
            name = rf"\multirow{{3}}{{*}}{{{e['name']}}}" if i == 0 else ""
            shown = {"DINO-WM": "DINO-WM", "ACWM-DiT-S": "ACWM-DiT-S", "Copy": r"\textcolor{base}{\textit{Copy first frame}}"}[lab]
            rows.append(f"{name} & {shown} & " + " & ".join(cells) + r" \\")
        if j < len(ENVS) - 1:
            rows.append(r"\midrule")
    eps = ", ".join(f"{data[e]['name']} {data[e]['ep']}" for e, *_ in ENVS)
    cap = (r"\textbf{Table 1.} Open-loop video prediction on ACWM-Phys: DINO-WM vs.\ ACWM-DiT-S on the same frames. "
           r"Every test episode, 10 windows of 37 frames per episode, conditioned on the first frame and the window's actions. "
           r"Both models are scored on window frames $5, 10, \dots, 35$ (Pour Water $15, \dots, 35$), the frames a frameskip-5 DINO-WM predicts. "
           r"MSE and M-MSE are scaled by $10^{-3}$; PSNR $= 10\log_{10}(1/\overline{\mathrm{MSE}})$. Bold: the better of the two models. "
           rf"DINO-WM: frameskip 5, one context frame, batch 24, trained {eps} epochs (62k--65k steps). "
           r"ACWM-DiT-S: released checkpoints (100k steps; Cloth Move 11k steps). "
           r"\textit{Copy first frame} repeats the conditioning frame.")
    return document(HEAD, rows, cap)


def table2(data):
    rows = []
    for j, (env, *_rest) in enumerate(ENVS):
        e = data[env]
        n_first = "first 50" if env != "pour_water" else "all 50"
        variants = [("Paper" + (r" (Table 7)" if env == "clothmove" else ""), e["paper"]),
                    (f"Released ckpt, {n_first} episodes", e["first50"])]
        if env != "pour_water":
            variants.append(("Released ckpt, all episodes", e["all"]))
        for i, (lab, r) in enumerate(variants):
            cells = [cell(r[s][k], k) for s in range(2) for k in range(4)]
            name = rf"\multirow{{{len(variants)}}}{{*}}{{{e['name']}}}" if i == 0 else ""
            rows.append(f"{name} & {lab} & " + " & ".join(cells) + r" \\")
        if j < len(ENVS) - 1:
            rows.append(r"\midrule")
    cap = (r"\textbf{Table 2.} ACWM-DiT-S with the official \texttt{eval.py} (all 37 frames of each window, 50 denoising steps, batch 2, "
           r"10 windows per episode, seed 0), released checkpoints. MSE and M-MSE are scaled by $10^{-3}$; PSNR $= 10\log_{10}(1/\overline{\mathrm{MSE}})$. "
           r"\textit{First 50} is what \texttt{scripts/eval\_all.sh} scores: \texttt{--max\_trajs 50} takes the first 50 entries of \texttt{metadata.pt}, "
           r"which are only successful stacks for Stack Cube (OoD: only the \textit{down} shift) and only the smaller cloth for Cloth Move OoD. "
           r"Cloth Move: the released checkpoint has 8-dim actions and stopped at 11k steps; the paper row is Table 7 (8-dim), "
           r"Table 1 of the paper reports the 3-dim variant. Pour Water has 50 test episodes per split.")
    head = HEAD.replace("Environment & Model", "Environment & Evaluation")
    return document(head, rows, cap)


def table3(data):
    groups = [("stack_cube", "ind_test", "success", "InD, stacked"), ("stack_cube", "ind_test", "fail", "InD, not stacked"),
              ("stack_cube", "ood_test", "success", "OoD, stacked"), ("stack_cube", "ood_test", "fail", "OoD, not stacked"),
              ("clothmove", "ood_test", "smaller", "OoD, smaller cloth"), ("clothmove", "ood_test", "larger", "OoD, larger cloth"),
              ("pour_water", "ood_test", "low", "OoD, less water"), ("pour_water", "ood_test", "high", "OoD, more water")]
    rows, last = [], None
    for env, sp, g, lab in groups:
        e = data[env]
        G = e["groups"][(sp, g)]
        if last and last != env:
            rows.append(r"\midrule")
        name = e["name"] if last != env else ""
        last = env
        cells = []
        for k in (0, 1):
            for m in ("DINO-WM", "ACWM-DiT-S"):
                v = G[m][k]
                cells.append(cell(v, k, bold=v == BETTER[k](G["DINO-WM"][k], G["ACWM-DiT-S"][k])))
            cells.append(cell(G["Copy"][k], k, gray=True))
        rows.append(f"{name} & {lab} & {G['n']} & " + " & ".join(cells) + r" \\")
    head = r"""\begin{tabular}{llr*{6}{c}}
\toprule
 & & & \multicolumn{3}{c}{MSE$\downarrow$} & \multicolumn{3}{c}{M-MSE$\downarrow$} \\
\cmidrule(lr){4-6}\cmidrule(lr){7-9}
Environment & Group & Episodes & DINO-WM & ACWM-DiT-S & \textcolor{base}{\textit{Copy}} & DINO-WM & ACWM-DiT-S & \textcolor{base}{\textit{Copy}} \\
\midrule
"""
    cap = (r"\textbf{Table 3.} The comparison of Table 1 split into groups. Stack Cube: whether the yellow cube ends on the green one "
           r"(metadata \texttt{success}). Cloth Move OoD: the two cloth sizes (smaller: 0.26$\times$, larger: 2.5$\times$ the training cloth area). "
           r"Pour Water OoD: the two water amounts in the cup at frame 0 (0.45--0.76$\times$ and 1.45--2.27$\times$ the training median). "
           r"MSE and M-MSE scaled by $10^{-3}$; bold: the better model.")
    return document(head, rows, cap)


def table4():
    """amounts at the end of a whole-episode open-loop rollout (both models from frame 0 with all actions)"""
    A = {e: json.load(open(os.path.join(HERE, "summary", f"{e}_amount.json"))) for e in ("pour_water", "clothmove", "push_sand", "stack_cube")}
    pw = json.load(open(os.path.join(RES, "pour_water/h1f5_ep6.json")))
    dq = {"clothmove": json.load(open(os.path.join(RES, "acwm_phys/cloth_h1f5_quant.json"))),
          "push_sand": json.load(open(os.path.join(RES, "acwm_phys/sand_h1f5_quant.json"))),
          "stack_cube": json.load(open(os.path.join(RES, "acwm_phys/stack_h1f5_quant.json")))}
    spec = [("stack_cube", "Stack Cube", "Stacked in last frame (\\%)", [("ind_test", "success", "InD, stacked"), ("ind_test", "fail", "InD, not stacked"),
                                                                        ("ood_test", "success", "OoD, stacked"), ("ood_test", "fail", "OoD, not stacked")]),
            ("push_sand", "Push Sand", "Sand area, last frame (px)", [("ind_test", "all", "InD"), ("ood_test", "all", "OoD, more sand")]),
            ("clothmove", "Cloth Move", "Cloth area, last frame (px)", [("ind_test", "all", "InD"), ("ood_test", "smaller", "OoD, smaller cloth"), ("ood_test", "larger", "OoD, larger cloth")]),
            ("pour_water", "Pour Water", "Water on the box floor (px)", [("ind_test", "all", "InD"), ("ood_test", "low", "OoD, less water"), ("ood_test", "high", "OoD, more water")])]
    rows = []
    for j, (env, name, qty, groups) in enumerate(spec):
        for i, (sp, g, lab) in enumerate(groups):
            a = A[env]["sets"][sp]
            if env == "pour_water":
                fa, fd = a["groups"][g]["fill"], pw["sets"][sp]["m3_fill"][g]
                real, acwm, dino = fa["gt_px"][0], fa["model_px"][0], fd["model_px"][0]
                fmt = lambda v: f"{v:,.0f}"
            elif env == "stack_cube":
                sa, sd = a["groups"][g]["stacked_last_frame"], dq[env]["sets"][sp]["groups"][g]["stacked_last_frame"]
                real, acwm, dino = 100 * sa["gt"]["stacked_rate"], 100 * sa["model"]["stacked_rate"], 100 * sd["model"]["stacked_rate"]
                fmt = lambda v: f"{v:.1f}"
            else:
                obj = "cloth" if env == "clothmove" else "sand"
                real, acwm = a["groups"][g][obj]["gt_px_last"][0], a["groups"][g][obj]["model"]["px_last"][0]
                dino = dq[env]["sets"][sp]["groups"][g][obj]["model"]["px_last"][0]
                fmt = lambda v: f"{v:,.0f}"
            n = a["group_sizes"][g]
            bd, ba = abs(dino - real) < abs(acwm - real), abs(acwm - real) < abs(dino - real)
            c = lambda v, b: (rf"\textbf{{{fmt(v)}}}" if b else fmt(v)).replace(",", "{,}")
            first = rf"\multirow{{{len(groups)}}}{{*}}{{\shortstack[l]{{{name}\\[-1pt]\scriptsize {qty}}}}}" if i == 0 else ""
            rows.append(f"{first} & {lab} & {n} & {c(real, False)} & {c(dino, bd)} & {c(acwm, ba)} \\\\")
        if j < len(spec) - 1:
            rows.append(r"\midrule")
    head = r"""\begin{tabular}{llrccc}
\toprule
Environment & Group & Episodes & Real & DINO-WM & ACWM-DiT-S \\
\midrule
"""
    cap = (r"\textbf{Table 4.} Physical quantities at the end of a whole-episode open-loop rollout. Both models start from the real "
           r"frame 0 and get all actions of the episode. DINO-WM rolls out at frameskip 5; ACWM-DiT-S chains 37-frame windows, each "
           r"conditioned on the last predicted frame of the previous one (Pour Water: 9 windows; the others: 2). All frames are measured "
           r"at $224{\times}224$ with the same colour masks, on frames $0, 5, 10, \dots$: Stack Cube uses the last frame (yellow cube on the "
           r"green one), Push Sand and Cloth Move frame 60, Pour Water the water on the box floor averaged over frames 250--295. "
           r"\textit{Real} is the real video measured the same way. Bold: closer to Real. ACWM-DiT-S never saw a Stack Cube training "
           r"window that starts after frame 21; its second window starts at frame 36.")
    return document(head, rows, cap)


def build(name, tex):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{name}.tex")
    open(p, "w").write(tex)
    r = subprocess.run([TECTONIC, "-X", "compile", "--outdir", OUT, p], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise SystemExit(f"tectonic failed for {name}")
    subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", os.path.join(OUT, f"{name}.pdf"), os.path.join(OUT, name)], check=True)
    print("wrote", os.path.join(OUT, f"{name}.pdf"), os.path.join(OUT, f"{name}.png"))


if __name__ == "__main__":
    data = load()
    build("table1", table1(data))
    build("table2", table2(data))
    build("table3", table3(data))
    build("table4", table4())
