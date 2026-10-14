"""DINO-WM 의 InD / OoD 예측 오차 표 (ACWM-Phys 지표: MSE, M-MSE, PSNR, SSIM) 를 논문 형식(LaTeX booktabs)으로 만든다.
환경마다 InD 행과 OoD 행. OoD 행의 괄호는 OoD ÷ InD. 표가 좁아서 캡션 폭은 최소 15 cm 로 두고 표를 가운데에 놓는다.
값은 평가 JSON 의 "acwm" 블록 (37프레임 창, 에피소드마다 10개, 프레임 5..35, Pour Water 15..35) 이다. compare.md §2 와 같은 값.
tectonic 으로 PDF 를 만들고 pdftoppm 으로 PNG 를 만든다.

usage: python make_table_pixel.py <out stem>   (cwd 무관)  ->  <out stem>.tex / .pdf / .png
"""
import os, sys, json, subprocess

ROOT = os.environ.get("DINO_WORK", os.path.expanduser("~/dinowm"))
TECTONIC = os.path.join(ROOT, "tools/tectonic_env/bin/tectonic")
STEM = sys.argv[1]
ENVS = [  # (name, eval JSON), in the order the user asked for
    ("Stack Cube", "results/acwm_phys/stack_h1f5.json"),
    ("Push Sand", "results/acwm_phys/sand_h1f5.json"),
    ("Cloth Move", "results/acwm_phys/cloth_h1f5.json"),
    ("Pour Water", "results/pour_water/h1f5_ep6.json"),
]


def val_ratio(x, ref, scale, nd):
    """value cell and ratio cell (ratio empty for the InD row)"""
    v = f"{x * scale:.{nd}f}"
    return [v, "" if ref is None else f"{{\\scriptsize ({x / ref:.2f}$\\times$)}}"]


lines = []
for i, (env, path) in enumerate(ENVS):
    d = json.load(open(os.path.join(ROOT, path)))
    ind = d["sets"]["ind_test"]
    if i:
        lines.append("\\addlinespace[4pt]")
    for split, label in (("ind_test", "InD"), ("ood_test", "OoD")):
        s = d["sets"][split]
        m, mi = s["acwm"]["all"]["model"], ind["acwm"]["all"]["model"]
        first = split == "ood_test"
        model = (val_ratio(m["mse"][0], mi["mse"][0] if first else None, 1e3, 2)
                 + val_ratio(m["masked_mse"][0], mi["masked_mse"][0] if first else None, 1e3, 1)
                 + [f"{m['psnr'][0]:.2f}", f"{m['ssim'][0]:.3f}"])
        row = [env if split == "ind_test" else "", label, str(s["group_sizes"]["all"])] + model
        lines.append(" & ".join(row) + " \\\\")

caption = (
    "Open-loop prediction error of DINO-WM on the in-distribution (InD) and out-of-distribution (OoD) test sets "
    "of four \\mbox{ACWM-Phys} environments. Each model is trained on InD data only. "
    "Ten 37-frame windows are taken from each test episode; from the first frame and the actions, the model predicts every fifth frame. "
    "MSE and \\mbox{M-MSE} (MSE weighted toward moving pixels) are $\\times 10^{-3}$. "
    "In parentheses: OoD value divided by InD value."
)
tex = r"""\documentclass[border=8pt,varwidth=40cm]{standalone}
\usepackage{newtxtext,newtxmath}
\usepackage{booktabs}
\usepackage{caption}
\captionsetup{labelfont=bf,font=small,labelsep=period,justification=justified,singlelinecheck=false,skip=6pt}
\begin{document}
\newsavebox{\tbox}
\sbox{\tbox}{\small\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.12}%%
\begin{tabular}{@{}llr r@{\,}l r@{\,}l cc@{}}
\toprule
Environment & Test set & $n$ & \multicolumn{2}{c}{MSE$\downarrow$} & \multicolumn{2}{c}{M-MSE$\downarrow$} & PSNR$\uparrow$ & SSIM$\uparrow$ \\
\midrule
%s
\bottomrule
\end{tabular}}
\newlength{\capw}\setlength{\capw}{\wd\tbox}\ifdim\capw<15cm\setlength{\capw}{15cm}\fi
\begin{minipage}{\capw}
\captionof{table}{%s}
\centering\usebox{\tbox}

\end{minipage}
\end{document}
""" % ("\n".join(lines), caption)
open(STEM + ".tex", "w").write(tex)
d = os.path.dirname(os.path.abspath(STEM))
subprocess.run([TECTONIC, "--chatter", "minimal", os.path.basename(STEM) + ".tex"], cwd=d, check=True)
subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", STEM + ".pdf", STEM], check=True)
print("wrote", STEM + ".tex", STEM + ".pdf", STEM + ".png")
