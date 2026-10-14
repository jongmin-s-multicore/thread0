"""2번 실험(InD + OoD 섞어 학습) 의 픽셀 오차 표 (ACWM-Phys 지표: MSE, M-MSE, PSNR, SSIM) 를 논문 형식(LaTeX booktabs)으로 만든다.
값은 results/acwm_phys/ood_mix/acwm_metrics_mix.py 의 <env>_mix_acwm.json. 지표마다 InD only 와 InD + OoD 를 나란히 두고 좋은 쪽을 굵게 한다.
  rollout  첫 프레임에서 에피소드 끝까지 예측 (양 표 table_oodmix 와 같은 예측)
  window   37 프레임 창 10개 (원래 평가의 acwm 블록, table_pixel_metrics 와 같은 방식)
tectonic 으로 PDF 를 만들고 pdftoppm 으로 PNG 를 만든다.

usage: python make_table_oodmix_pixel.py {rollout,window} <out stem>   ->  <out stem>.tex / .pdf / .png
"""
import os, sys, json, subprocess

PROTO, STEM = sys.argv[1], sys.argv[2]
WORK = os.environ.get("DINO_WORK", os.path.expanduser("~/dinowm"))
TECTONIC = os.path.join(WORK, "tools/tectonic_env/bin/tectonic")
R = {k: json.load(open(os.path.join(WORK, f"results/acwm_phys/ood_mix/{k}_mix_acwm.json")))["groups"] for k in ("cloth", "sand", "water", "stack")}
ROWS = [  # (environment, split label, key, group)
    ("Cloth Move", "InD", "cloth", "ind_test/all"),
    ("", "OoD, 0.26$\\times$ area", "cloth", "ood_test/smaller"),
    ("", "OoD, 2.5$\\times$ area", "cloth", "ood_test/larger"),
    ("Push Sand", "InD", "sand", "ind_test/all"),
    ("", "OoD, 2$\\times$ particles", "sand", "ood_test/all"),
    ("Pour Water", "InD", "water", "ind_test/all"),
    ("", "OoD, less water", "water", "ood_test/low"),
    ("", "OoD, more water", "water", "ood_test/high"),
    ("Stack Cube", "InD", "stack", "ind_test/all"),
    ("", "OoD, cube start", "stack", "ood_test/all"),
]
METRICS = [("mse", 1e3, 2, False), ("masked_mse", 1e3, 1, False), ("psnr", 1, 2, True), ("ssim", 1, 3, True)]  # key, scale, digits, higher is better

lines = []
for i, (env, split, key, grp) in enumerate(ROWS):
    if i and env:
        lines.append("\\addlinespace[3pt]")
    g = R[key][grp]
    cells = []
    for k, sc, nd, hi in METRICS:
        a, b = g[PROTO]["ind_only"][k], g[PROTO]["ind_ood"][k]
        sa, sb = f"{a * sc:.{nd}f}", f"{b * sc:.{nd}f}"
        if sa != sb:
            better_b = (b > a) == hi
            sa, sb = (sa, f"\\textbf{{{sb}}}") if better_b else (f"\\textbf{{{sa}}}", sb)
        cells += [sa, sb]
    lines.append(f"{env} & {split} & {g['n']} & " + " & ".join(cells) + " \\\\")

how = {
    "rollout": "The model gets the first real frame and the actions of the whole episode and predicts every fifth frame to the end of the episode "
               "(the same predictions as the object amounts in the rollout table). All predicted frames from frame 5 on are scored. "
               "The motion mask of M-MSE is built from all frames of the episode.",
    "window": "Ten 37-frame windows are taken from each test episode, as in the ACWM-Phys evaluation; "
              "from the first frame of a window and the actions, the model predicts every fifth frame, "
              "and frames 5--35 are scored (Pour Water: frames 15--35). The motion mask of M-MSE is built from the frames of the window. "
              "Values are averaged over windows, then over episodes.",
}[PROTO]
caption = (
    "Pixel error of the original model trained on InD episodes only and of a model trained on InD and some OoD test episodes. "
    + how + " "
    "Frames are compared at the ACWM-Phys resolution ($240{\\times}240$, Push Sand $240{\\times}400$) in $[0, 1]$. "
    "MSE and \\mbox{M-MSE} (MSE weighted toward moving pixels) are $\\times 10^{-3}$. Bold: the better of the two models. "
    "OoD episodes added to training: Cloth Move 50, Push Sand 50, Pour Water 26, Stack Cube 136. "
    "OoD rows use only the OoD test episodes that were not added to training. "
    "Both models use the same decoder, trained on InD frames only. One seed. $n$ is the number of test episodes."
)
tex = r"""\documentclass[border=8pt,varwidth=40cm]{standalone}
\usepackage{newtxtext,newtxmath}
\usepackage{booktabs}
\usepackage{caption}
\captionsetup{labelfont=bf,font=small,labelsep=period,justification=justified,singlelinecheck=false,skip=6pt}
\begin{document}
\newsavebox{\tbox}
\sbox{\tbox}{\small\setlength{\tabcolsep}{4.5pt}\renewcommand{\arraystretch}{1.1}%%
\begin{tabular}{@{}llr cc cc cc cc@{}}
\toprule
 & & & \multicolumn{2}{c}{MSE$\downarrow$} & \multicolumn{2}{c}{M-MSE$\downarrow$} & \multicolumn{2}{c}{PSNR$\uparrow$} & \multicolumn{2}{c}{SSIM$\uparrow$} \\
\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(l){10-11}
Environment & Test set & $n$ & InD only & InD + OoD & InD only & InD + OoD & InD only & InD + OoD & InD only & InD + OoD \\
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
