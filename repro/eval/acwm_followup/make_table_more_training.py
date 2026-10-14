"""3번 실험 (InD 로 더 학습) 표를 논문 형식(LaTeX booktabs)으로 만든다. table_pixel_metrics 와 같은 모양이다.
환경마다 InD 행, OoD 행(그룹별), InD 검증 손실 행. 열은 실제 값과 학습량 1x, 2x, 4x, 8x 의 open-loop rollout 값.
값: Cloth/Sand/Stack 1x 는 results/acwm_phys/<env>_h1f5_trend.json, 2x-8x 는 results/acwm_phys/long/<env>_ep<E>.json
(trend_acwm_phys.py, 마지막 프레임), Pour Water 는 results/acwm_phys/long/water_ep<E>.json (water_long_eval.py, 마지막 5초 상자 물 픽셀).
검증 손실은 predictor_long.py 로그 (1x 는 불러온 가중치를 다시 잰 값). 없는 값은 "--".
tectonic 으로 PDF 를 만들고 pdftoppm 으로 PNG 를 만든다.

슬라이드용: --slide 는 캡션을 빼고, --ood_only 는 InD 행을 뺀다 (OoD 행과 검증 손실 행만).
--wide 는 슬라이드의 가로로 긴 칸용이다: 행이 Real, 1x-8x 이고 열이 환경별 테스트 세트와 검증 손실이다 (캡션, n 없음).

usage: python make_table_more_training.py <out stem> [--slide] [--ood_only] [--wide]   (cwd 무관)  ->  <out stem>.tex / .pdf / .png
"""
import os, sys, json, subprocess

ROOT = os.environ.get("DINO_WORK", os.path.expanduser("~/dinowm"))
TECTONIC = os.path.join(ROOT, "tools/tectonic_env/bin/tectonic")
STEM = sys.argv[1]
SLIDE, OOD_ONLY, WIDE = "--slide" in sys.argv[2:], "--ood_only" in sys.argv[2:], "--wide" in sys.argv[2:]
R = os.path.join(ROOT, "results/acwm_phys")
# (name, key, epochs 1x/2x/4x/8x, rows (label, set, group), loss log), in the order the user asked for
ENVS = [
    ("Stack Cube", "stack", (25, 50, 100, 200), [("InD, failed", "ind_test", "fail"), ("OoD, failed", "ood_test", "fail")],
     "runs/colab_acwm_long/train_stack_h1f5.jsonl"),
    ("Push Sand", "sand", (18, 36, 72, 144), [("InD", "ind_test", "all"), ("OoD", "ood_test", "all")],
     "runs/colab_acwm_long/train_sand_h1f5.jsonl"),
    ("Cloth Move", "cloth", (15, 30, 60, 120), [("InD", "ind_test", "all"), ("OoD, smaller", "ood_test", "smaller"), ("OoD, larger", "ood_test", "larger")],
     "runs/colab_acwm_long/train_cloth_h1f5.jsonl"),
    ("Pour Water", "water", (6, 12, 24, 48), [("InD", "ind_test", "all"), ("OoD, less water", "ood_test", "low"), ("OoD, more water", "ood_test", "high")],
     "runs/colab_acwm_long_pw/train_pw_h1f5.jsonl"),
]


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def amount(key, ep, sp, g, first):
    """(n, real, model) or None"""
    if key == "water":
        d = load(f"{R}/long/water_ep{ep}.json")
        if d is None:
            return None
        v = d["box_fill"][f"{sp}/{g}"]
        return d["groups"][f"{sp}/{g}"], v["real"], v["rollout"]
    d = load(f"{R}/{key}_h1f5_trend.json" if first else f"{R}/long/{key}_ep{ep}.json")
    if d is None:
        return None
    v = d["sets"][sp][g][str(ep)]
    return v["n"], v["real"], v["model"]


def losses(path):
    """{epoch: val loss}; the first record of a run re-measures the loaded weights"""
    out = {}
    if os.path.exists(os.path.join(ROOT, path)):
        for l in open(os.path.join(ROOT, path)):
            r = json.loads(l)
            if "val_z_visual_loss" in r:
                out.setdefault(r["epoch"], r["val_z_visual_loss"])
    return out


lines = []
for i, (env, key, eps, rows, log) in enumerate(ENVS):
    fmt = (lambda x: f"{100 * x:.0f}\\%") if key == "stack" else (lambda x: f"{x:,.0f}")
    if i:
        lines.append("\\addlinespace[4pt]")
    rows = [r for r in rows if not (OOD_ONLY and r[1] == "ind_test")]
    for j, (label, sp, g) in enumerate(rows):
        vals = [amount(key, e, sp, g, k == 0) for k, e in enumerate(eps)]
        known = [v for v in vals if v is not None]
        n, real = (known[0][0], fmt(known[0][1])) if known else ("", "--")
        cells = [env if j == 0 else "", label, str(n), real] + [fmt(v[2]) if v else "--" for v in vals]
        lines.append(" & ".join(cells) + " \\\\")
    L = losses(log)
    lines.append(" & ".join(["", "InD val.\\ loss", "", ""] + [f"{L[e]:.3f}" if e in L else "--" for e in eps]) + " \\\\")

caption = (
    "DINO-WM trained longer on in-distribution (InD) data only: the original model (1$\\times$) is trained further on the same data "
    "to 2$\\times$, 4$\\times$ and 8$\\times$ its epochs. Values are open-loop rollouts over the whole test episode, averaged over episodes: "
    "object pixels in the last frame (Push Sand, Cloth Move), water pixels in the box over the last 5\\,s (Pour Water), "
    "share of failed stacks drawn as stacked (Stack Cube)."
)
tex = r"""\documentclass[border=8pt,varwidth=40cm]{standalone}
\usepackage{newtxtext,newtxmath}
\usepackage{booktabs}
\usepackage{caption}
\captionsetup{labelfont=bf,font=small,labelsep=period,justification=justified,singlelinecheck=false,skip=6pt}
\begin{document}
\newsavebox{\tbox}
\sbox{\tbox}{\small\setlength{\tabcolsep}{6pt}\renewcommand{\arraystretch}{1.12}%%
\begin{tabular}{@{}llrr rrrr@{}}
\toprule
Environment & Test set & $n$ & Real & 1$\times$ & 2$\times$ & 4$\times$ & 8$\times$ \\
\midrule
%s
\bottomrule
\end{tabular}}
%s
\end{document}
""" % ("\n".join(lines), "\\usebox{\\tbox}" if SLIDE else """\\newlength{\\capw}\\setlength{\\capw}{\\wd\\tbox}\\ifdim\\capw<15cm\\setlength{\\capw}{15cm}\\fi
\\begin{minipage}{\\capw}
\\captionof{table}{%s}
\\centering\\usebox{\\tbox}

\\end{minipage}""" % caption)
# short column names and units for the wide layout
WIDE_HEAD = {"stack": ("Stack Cube (\\% stacked)", ["InD fail", "OoD fail"]), "sand": ("Push Sand (sand px)", ["InD", "OoD"]),
             "cloth": ("Cloth Move (cloth px)", ["InD", "Smaller", "Larger"]), "water": ("Pour Water (water px in box)", ["InD", "Less", "More"])}
if WIDE:
    cols, top, sub, rule, body = [], [], ["Training"], [], [[r] for r in ("Real", "1$\\times$", "2$\\times$", "4$\\times$", "8$\\times$")]
    c = 2
    for env, key, eps, rows, log in ENVS:
        fmt = (lambda x: f"{100 * x:.0f}\\%") if key == "stack" else (lambda x: f"{x:,.0f}")
        name, short = WIDE_HEAD[key]
        rows = [r for r in rows if not (OOD_ONLY and r[1] == "ind_test")]
        short = short[len(short) - len(rows):]  # InD column dropped with --ood_only
        k = len(rows) + 1
        top.append(f"\\multicolumn{{{k}}}{{c}}{{{name}}}"); rule.append(f"\\cmidrule(lr){{{c}-{c + k - 1}}}"); c += k
        sub += short + ["Val.\\ loss"]
        L = losses(log)
        for label, sp, g in rows:
            vals = [amount(key, e, sp, g, j == 0) for j, e in enumerate(eps)]
            known = [v for v in vals if v is not None]
            body[0].append(fmt(known[0][1]) if known else "--")
            for j, v in enumerate(vals):
                body[j + 1].append(fmt(v[2]) if v else "--")
        body[0].append("")
        for j, e in enumerate(eps):
            body[j + 1].append(f"{L[e]:.3f}" if e in L else "--")
        cols.append("r" * k)
    lines = [" & ".join(r) + " \\\\" for r in body]
    lines.insert(1, "\\addlinespace[3pt]")
    tex = r"""\documentclass[border=8pt,varwidth=60cm]{standalone}
\usepackage{newtxtext,newtxmath}
\usepackage{booktabs}
\begin{document}
{\small\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.15}%%
\begin{tabular}{@{}l %s@{}}
\toprule
 & %s \\
%s
%s \\
\midrule
%s
\bottomrule
\end{tabular}}
\end{document}
""" % ("@{\\hspace{10pt}}".join(cols), " & ".join(top), "".join(rule), " & ".join(sub), "\n".join(lines))
open(STEM + ".tex", "w").write(tex)
d = os.path.dirname(os.path.abspath(STEM))
subprocess.run([TECTONIC, "--chatter", "minimal", os.path.basename(STEM) + ".tex"], cwd=d, check=True)
subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", STEM + ".pdf", STEM], check=True)
print("wrote", STEM + ".tex", STEM + ".pdf", STEM + ".png")
