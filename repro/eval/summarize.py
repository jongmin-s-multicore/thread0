"""planning·예측 품질·학습 결과를 모아 $DINO_WORK/results/summary.{md,json} 로 쓴다 (레포 밖 — 결과는 이슈로 올린다).
repro/env.sh 를 source 한 셸에서 실행한다: python3 repro/eval/summarize.py"""
import json, os, glob

RUNS = os.environ["DINO_RUNS"]
TRAIN = os.environ["DINO_TRAIN"]
RESULTS = os.path.join(os.environ["DINO_WORK"], "results")

PAPER = {  # arXiv 2411.04983 v2, Table 1 / Table 8 (MPC, CEM, GD rows); Rope/Granular are Chamfer distance
    "pointmaze": {"MPC": 0.98, "CEM": 0.80, "GD": 0.22},
    "pusht": {"MPC": 0.90, "CEM": 0.86, "GD": 0.28},
    "wall": {"MPC": 0.96, "CEM": 0.74, "GD": None},
    "rope": {"MPC": 0.41},
    "granular": {"MPC": 0.26},
}
PAPER_PQ = {  # Table 9 (Ours): LPIPS, SSIM
    "pusht": (0.007, 0.985), "wall": (0.0016, 0.997), "rope": (0.009, 0.985), "granular": (0.035, 0.940),
}


def read_logs(name):
    p = f"{RUNS}/{name}/logs.json"
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def status(name):
    d = f"{RUNS}/{name}"
    if os.path.exists(f"{d}/DONE"):
        return "done"
    if not os.path.exists(f"{d}/run_info.txt"):
        return "pending"
    return "failed" if "END" in open(f"{d}/run_info.txt").read() else "running"


def mpc_curve(name, key):
    return [(l["step"], l[f"mpc/{key}"]) for l in read_logs(name) if f"mpc/{key}" in l]


def final(name, key):
    for l in read_logs(name):
        if f"final_eval/{key}" in l:
            return l[f"final_eval/{key}"]
    return None


out, md = {}, ["# DINO-WM reproduction results (seed 99)", ""]
md += ["## Planning success rate (PointMaze / PushT / Wall, 50 episodes, released checkpoints)", "",
       "| Env | MPC-CEM (ours) | MPC budget used | paper MPC | CEM open-loop = MPC@1 (ours) | paper CEM | GD (ours) | paper GD | CEM 30-step OL (ours) |",
       "|---|---|---|---|---|---|---|---|---|"]
for env, mpc, gd, cem30 in [("pointmaze", "pointmaze_mpc", "pointmaze_gd", "pointmaze_cem30"),
                            ("pusht", "pusht_mpc", "pusht_gd", None), ("wall", "wall_mpc", "wall_gd", "wall_cem30")]:
    curve = mpc_curve(mpc, "success_rate")
    sr = curve[-1][1] if curve else None
    sr1 = curve[0][1] if curve else None
    g = final(gd, "success_rate")
    c30 = final(cem30, "success_rate") if cem30 else None
    out[env] = dict(mpc_curve=curve, mpc_status=status(mpc), gd=g, gd_status=status(gd), cem30=c30, paper=PAPER[env])
    f = lambda v: "-" if v is None else f"{v:.2f}"
    md.append(f"| {env} | {f(sr)} ({status(mpc)}) | {len(curve)} iters | {f(PAPER[env]['MPC'])} | {f(sr1)} | {f(PAPER[env]['CEM'])} | "
              f"{f(g)} ({status(gd)}) | {f(PAPER[env]['GD'])} | {f(c30) if cem30 else 'n/a'} |")
md += ["", "### MPC success rate vs. MPC iteration budget (success is sticky, as in the repo)", ""]
for env in ["pointmaze", "pusht", "wall"]:
    c = out[env]["mpc_curve"]
    md.append(f"- **{env}**: " + (", ".join(f"@{s}: {v:.2f}" for s, v in c) if c else "no MPC iteration finished yet"))

md += ["", "## Deformable (Chamfer distance, 10 instances, our trained world models)", "",
       "| Env | CD after each MPC iteration (ours) | final CD (ours) | paper CD |", "|---|---|---|---|"]
for env in ["rope", "granular"]:
    c = mpc_curve(f"{env}_mpc", "mean_chamfer_distance")
    fin = final(f"{env}_mpc", "mean_chamfer_distance")
    out[env] = dict(cd_curve=c, final_cd=fin, status=status(f"{env}_mpc"), paper=PAPER[env])
    md.append(f"| {env} | {', '.join(f'@{s}: {v:.3f}' for s, v in c) or '-'} | {'-' if fin is None else f'{fin:.3f}'} "
              f"({status(env + '_mpc')}) | {PAPER[env]['MPC']} |")

md += ["", "## Prediction quality (decoded 1-step prediction, full validation split)", "",
       "| Env | LPIPS (ours) | paper LPIPS | SSIM (ours) | paper SSIM | ckpt epoch | #val slices |", "|---|---|---|---|---|---|---|"]
for env in ["pointmaze", "pusht", "wall", "rope", "granular"]:
    p = f"{RESULTS}/pred_quality/{env}.json"
    if os.path.exists(p):
        q = json.load(open(p))
        pl, ps = PAPER_PQ.get(env, (None, None))
        out[f"pq_{env}"] = q
        md.append(f"| {env} | {q['pred_lpips']:.4f} | {pl if pl is not None else 'n/a'} | {q['pred_ssim']:.4f} | "
                  f"{ps if ps is not None else 'n/a'} | {q['ckpt_epoch']} | {q['n_evaluated']} |")
    else:
        md.append(f"| {env} | - | | - | | | |")

md += ["", "## Training (deformable world models)", ""]
for env in ["rope", "granular"]:
    p = f"{TRAIN}/outputs/train_{env}/epoch_logs.jsonl"
    if os.path.exists(p):
        logs = [json.loads(l) for l in open(p) if l.strip()]
        last = logs[-1]
        out[f"train_{env}"] = logs
        md.append(f"- **{env}**: {len(logs)} epochs done; last train_loss {last.get('train_loss', float('nan')):.4f}, "
                  f"val_loss {last.get('val_loss', float('nan')):.4f}, val_img_lpips_pred {last.get('val_img_lpips_pred', float('nan')):.4f}")
    else:
        md.append(f"- **{env}**: no epoch finished yet")

os.makedirs(RESULTS, exist_ok=True)
open(f"{RESULTS}/summary.md", "w").write("\n".join(md) + "\n")
json.dump(out, open(f"{RESULTS}/summary.json", "w"), indent=1)
print("\n".join(md))
