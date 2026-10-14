"""2번 실험(InD + OoD 섞어 학습) 표를 논문 형식(LaTeX booktabs)으로 만든다. 값은 rollout 마지막 프레임의 양이다 (25-step 은 넣지 않는다).
Cloth Move 는 results/acwm_phys/ood_mix/{base_ep15,oodmix}_heldout.json, 나머지는 <env>_mix_eval.json (mix_eval.py). OoD 행은 학습에 넣지 않은 에피소드만.
tectonic 으로 PDF 를 만들고 pdftoppm 으로 PNG 를 만든다 (make_table.py 와 같은 형식).

usage: python make_table_oodmix.py <out stem>   ->  <out stem>.tex / .pdf / .png
"""
import os, sys, json, subprocess

STEM = sys.argv[1]
WORK = os.environ.get("DINO_WORK", os.path.expanduser("~/dinowm"))
TECTONIC = os.path.join(WORK, "tools/tectonic_env/bin/tectonic")
D = os.path.join(WORK, "results/acwm_phys/ood_mix")
CLOTH = {m: json.load(open(f"{D}/{f}"))["sets"] for m, f in (("base", "base_ep15_heldout.json"), ("mix", "oodmix_heldout.json"))}
MIX = {k: json.load(open(f"{D}/{k}_mix_eval.json"))["result"] for k in ("sand", "water", "stack")}
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


def values(key, grp, q, st):
    """n, real, InD-only model, InD+OoD model (rollout)"""
    if key == "cloth":
        s, g = grp.split("/")
        b, m = CLOTH["base"][s][g]["15"], CLOTH["mix"][s][g]["15"]
        assert b["real"] == m["real"] and b["n"] == m["n"]
        return b["n"], b["real"], b["model"], m["model"]
    a, b = MIX[key]["ind_only"][grp]["all"], MIX[key]["ind_ood"][grp]["all"]
    assert a["n"] == b["n"] and a[q]["real"][st] == b[q]["real"][st]
    return a["n"], a[q]["real"][st], a[q]["rollout"][st], b[q]["rollout"][st]


lines = []
for i, (env, split, key, grp, q, st) in enumerate(ROWS):
    if i and env:
        lines.append("\\addlinespace[3pt]")
    n, real, base, mix = values(key, grp, q, st)
    if q == "stacked":
        cells = [f"{100 * real:.0f}\\%"] + [f"\\multicolumn{{2}}{{c}}{{{100 * v:.0f}\\%}}" for v in (base, mix)]
    else:
        cells = [f"{real:,.0f}"] + [f"{v:,.0f} & ({100 * v / real:.0f}\\%)" for v in (base, mix)]
    lines.append(f"{env} & {split} & {n} & " + " & ".join(cells) + " \\\\")

caption = (
    "Object amount at the end of the model rollout, for the original model trained on InD episodes only and for a model trained on InD "
    "and some OoD test episodes. "
    "The rollout starts from the first real frame and predicts the whole episode from the actions. "
    "OoD episodes added to training: Cloth Move 50 (25 per cloth size), Push Sand 50, Pour Water 26, Stack Cube 136, "
    "on top of 1,800, 1,605, 882 and 1,235 InD training episodes. "
    "OoD rows use only the OoD test episodes that were not added to training. "
    "Both models have the same settings and the same decoder, which was trained on InD frames only. "
    "The InD+OoD predictor was trained from scratch for the same number of epochs, with one seed. "
    "Object amount is the number of object pixels in the last frame ($224{\\times}224$), counted with HSV color masks: "
    "cloth (orange front and pink back side), sand (light blue) and, for Pour Water, water (blue) inside the floor band of the target box, "
    "averaged over the last 5\\,s (frames 250--295). "
    "For Stack Cube, the amount is the share of episodes with a failed real stack whose last frame shows the yellow cube on the green cube. "
    "The percentage after each value is the ratio to Real in the same row. 100\\% means the same amount as in the real video. "
    "$n$ is the number of test episodes."
)
tex = r"""\documentclass[border=8pt,varwidth=30cm]{standalone}
\usepackage{newtxtext,newtxmath}
\usepackage{booktabs}
\usepackage{caption}
\captionsetup{labelfont=bf,font=small,labelsep=period,justification=justified,singlelinecheck=false,skip=6pt}
\begin{document}
\newsavebox{\tbox}
\sbox{\tbox}{\small\setlength{\tabcolsep}{5pt}%%
\begin{tabular}{@{}llr r r@{\,}l r@{\,}l@{}}
\toprule
 & & & \multicolumn{5}{c}{Object amount, last frame of the rollout} \\
\cmidrule(l){4-8}
Environment & Split & $n$ & Real & \multicolumn{2}{c}{InD only} & \multicolumn{2}{c}{InD + OoD} \\
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
