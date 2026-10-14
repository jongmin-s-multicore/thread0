"""4번 실험 표를 논문 형식(LaTeX booktabs)으로 만든다. 값은 talk_render.py 의 <env>_summary.json (양) 과 lpips_table.py 의 <env>_lpips.json (LPIPS).
tectonic 으로 PDF 를 만들고 pdftoppm 으로 PNG 를 만든다.

usage: python make_table.py <render dir> <out stem>   ->  <out stem>.tex / .pdf / .png
"""
import os, sys, json, subprocess

R, STEM = sys.argv[1], sys.argv[2]
TECTONIC = os.path.join(os.environ.get("DINO_WORK", os.path.expanduser("~/dinowm")), "tools/tectonic_env/bin/tectonic")
S = {k: json.load(open(f"{R}/{k}_summary.json"))["groups"] for k in ("cloth", "sand", "water", "stack")}
L = {k: json.load(open(f"{R}/{k}_lpips.json"))["groups"] for k in ("cloth", "sand", "water", "stack")}
ROWS = [  # (environment, split label, key, group, quantity, statistic)
    ("Cloth Move", "InD", "cloth", "ind_test/all", "px", "last"),
    ("", "OoD, 0.26$\\times$ area", "cloth", "ood_test/smaller", "px", "last"),
    ("", "OoD, 2.5$\\times$ area", "cloth", "ood_test/larger", "px", "last"),
    ("Push Sand", "InD", "sand", "ind_test/all", "px", "last"),
    ("", "OoD, 2$\\times$ particles", "sand", "ood_test/all", "px", "last"),
    ("Pour Water", "InD", "water", "ind_test/all", "box", "fill"),
    ("", "OoD, less water", "water", "ood_test/low", "box", "fill"),
    ("", "OoD, more water", "water", "ood_test/high", "box", "fill"),
    ("Stack Cube", "InD, failed", "stack", "ind_test/fail", "stacked", "last"),
    ("", "OoD, failed", "stack", "ood_test/fail", "stacked", "last"),
]


def amount(key, grp, q, st, kd):
    return S[key][grp]["all"][q][kd][st]


lines = []
for i, (env, split, key, grp, q, st) in enumerate(ROWS):
    if i and env:
        lines.append("\\addlinespace[3pt]")
    n = S[key][grp]["all"]["n"]
    real = amount(key, grp, q, st, "real")
    if q == "stacked":
        cells = [f"{100 * real:.0f}\\%"] + [f"\\multicolumn{{2}}{{c}}{{{100 * amount(key, grp, q, st, kd):.0f}\\%}}" for kd in ("recon", "pred", "rollout")]
    else:
        cells = [f"{real:,.0f}"] + [f"{amount(key, grp, q, st, kd):,.0f} & ({100 * amount(key, grp, q, st, kd) / real:.0f}\\%)" for kd in ("recon", "pred", "rollout")]
    lp = [f"{L[key][grp]['from25'][kd]:.3f}" for kd in ("recon", "pred", "rollout")]
    lines.append(f"{env} & {split} & {n} & " + " & ".join(cells) + " & " + " & ".join(lp) + " \\\\")

caption = (
    "Object amount and LPIPS on the ACWM-Phys test episodes. "
    "Object amount is the number of object pixels in the last frame ($224{\\times}224$), counted with HSV color masks: "
    "cloth (orange front and pink back side), sand (light blue) and, for Pour Water, water (blue) inside the floor band of the target box, "
    "averaged over the last 5\\,s (frames 250--295). "
    "For Stack Cube, the amount is the share of episodes with a failed real stack whose last frame shows the yellow cube on the green cube "
    "(both cubes visible, yellow centre less than 11\\,px to the side of and 6.5--35\\,px above the green centre). "
    "The percentage after each value is the ratio to Real in the same row. 100\\% means the same amount as in the real video. "
    "LPIPS (VGG, v0.1) is the perceptual distance between a generated frame and the real frame at the same time step "
    "($224{\\times}224$, 0 for identical images, lower is better). "
    "It is computed on every fifth frame from frame 25 to the end of the episode and averaged over frames and episodes. "
    "$n$ is the number of test episodes."
)
tex = r"""\documentclass[border=8pt,varwidth=30cm]{standalone}
\usepackage{newtxtext,newtxmath}
\usepackage{booktabs}
\usepackage{caption}
\captionsetup{labelfont=bf,font=small,labelsep=period,justification=justified,singlelinecheck=false,skip=6pt}
\begin{document}
\newsavebox{\tbox}
\sbox{\tbox}{\small\setlength{\tabcolsep}{4.2pt}%%
\begin{tabular}{@{}llr r r@{\,}l r@{\,}l r@{\,}l ccc@{}}
\toprule
 & & & \multicolumn{7}{c}{Object amount, last frame} & \multicolumn{3}{c}{LPIPS $\downarrow$, frame 25 onward} \\
\cmidrule(lr){4-10} \cmidrule(l){11-13}
Environment & Split & $n$ & Real & \multicolumn{2}{c}{Enc$\to$Dec} & \multicolumn{2}{c}{Model 25-step} & \multicolumn{2}{c}{Model rollout} & Enc$\to$Dec & 25-step & Rollout \\
\midrule
%s
\bottomrule
\end{tabular}}
\begin{minipage}{\wd\tbox}
\captionof{table}{%s}
\usebox{\tbox}

\end{minipage}
\end{document}
""" % ("\n".join(lines), caption)
open(STEM + ".tex", "w").write(tex)
d = os.path.dirname(os.path.abspath(STEM))
subprocess.run([TECTONIC, "--chatter", "minimal", os.path.basename(STEM) + ".tex"], cwd=d, check=True)
subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", STEM + ".pdf", STEM], check=True)
print("wrote", STEM + ".tex", STEM + ".pdf", STEM + ".png")
