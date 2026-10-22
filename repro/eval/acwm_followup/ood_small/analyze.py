"""curves.py 의 npz 에서 표를 만든다 (Cloth Move, 작은 천 + 중간 천만 학습했을 때 큰 천을 예측하는가).

평가 에피소드는 모델마다 학습에 없던 것만 쓴다.
  InD                  InD test 100
  OoD smaller, odd 25  smaller 홀수 번호 25    (orig, ctrl, small, prev 모두에게 학습에 없던 것)
  OoD larger, odd 25   larger 홀수 번호 25     (네 모델 모두)
  OoD larger, all 50   larger 전부 50          (orig, ctrl, small. prev 는 짝수 25 를 학습에 썼으므로 뺀다)
같은 에피소드를 짝지어 비교하고, 에피소드를 다시 뽑는 bootstrap 으로 95 % 구간을 낸다 (마지막 프레임 px 는 10,000 회, retention 은 2,000 회).
모델마다 학습 실행은 한 번이다.

retention r(k): (OoD 그룹 평균 - InD 평균) 의 예측 천 화소 / 같은 차이의 인코더→디코더 화소.
  rollout r(5), r(12)   첫 실제 프레임에서 끝까지 예측했을 때 model step 5, 12 (step = 5 프레임)
  one step              한 step 앞의 실제 프레임에서 예측한 값, k = 2..12 평균
  25 frames             다섯 step 앞의 실제 프레임에서 예측한 값, k = 5..12 평균
1 이면 OoD 천 크기 차이가 유지되고 0 이면 InD 와 같다.

usage: python3 -I analyze.py <curves.npz> [<out dir>]   ->  small_only.json, tables.md
"""
import json, os, re, sys
import numpy as np

B_PX, B_R = 10000, 2000
MODELS = ["orig", "ctrl", "small", "prev"]
NAME = {"orig": "InD only (train.py, fp32)", "ctrl": "InD only (bf16)", "small": "InD + smaller 25", "prev": "InD + smaller 25 + larger 25"}
SEEN_LARGER = {"prev"}  # models that trained on even-numbered larger episodes


def parse(names):
    grp = np.array([n.split("_episode")[0] for n in names])
    num = np.array([int(re.search(r"(\d+)", n.split("episode")[-1]).group(1)) for n in names])
    return grp, num


def eval_sets(z):
    grp, num = parse(z["ood_test/names"])
    odd = num % 2 == 1
    return {
        "InD": ("ind_test", np.arange(len(z["ind_test/names"]))),
        "OoD smaller, odd 25": ("ood_test", np.where((grp == "smaller") & odd)[0]),
        "OoD larger, odd 25": ("ood_test", np.where((grp == "larger") & odd)[0]),
        "OoD larger, all 50": ("ood_test", np.where(grp == "larger")[0]),
    }


def valid(model, label):
    return not (label == "OoD larger, all 50" and model in SEEN_LARGER)


def last_and_late(px, kf):
    n = len(px)
    return px[np.arange(n), kf - 1], np.nanmean(px[:, 5:], axis=1)


def ci(v, rng):
    return [float(x) for x in np.percentile(v[rng.integers(0, len(v), (B_PX, len(v)))].mean(1), [2.5, 97.5])]


def ratio_ci(a, b, rng):
    i = rng.integers(0, len(a), (B_PX, len(a)))
    return [float(x) for x in np.percentile(100 * a[i].mean(1) / b[i].mean(1), [2.5, 97.5])]


def rk(pred_g, pred_i, rec_g, rec_i):
    """retention at every point k: arrays (n_episodes, K)"""
    with np.errstate(all="ignore"):
        return (np.nanmean(pred_g, 0) - np.nanmean(pred_i, 0)) / (np.nanmean(rec_g, 0) - np.nanmean(rec_i, 0))


def r_summary(P, gi, ii, rec):
    """P: {kind: (n_ood + n_ind rows)}. gi, ii index arrays into the OoD / InD arrays. -> r5, r12 (rollout), one step, 25 frames"""
    (pg, pi) = P
    roll = rk(pg["rollout"][gi], pi["rollout"][ii], rec[0][gi], rec[1][ii])
    a1 = rk(pg["ahead1"][gi], pi["ahead1"][ii], rec[0][gi], rec[1][ii])
    a5 = rk(pg["ahead5"][gi], pi["ahead5"][ii], rec[0][gi], rec[1][ii])
    return {"r5": roll[5], "r12": roll[12], "one_step": np.nanmean(a1[2:13]), "f25": np.nanmean(a5[5:13])}


def main():
    z = np.load(sys.argv[1])
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(sys.argv[1]))
    models = [m for m in MODELS if f"ood_test/px/{m}" in z.files]
    rng = np.random.default_rng(0)
    sets = eval_sets(z)
    res = {"models": models, "meta": json.loads(str(z["meta"])), "sets": {}, "retention": {}}
    curves = {}

    # ---------------- last-frame pixels
    for label, (sp, idx) in sets.items():
        kf = z[f"{sp}/kf"][idx]
        kinds = ["real", "recon"] + [m for m in models if valid(m, label)]
        pe = {k: last_and_late(z[f"{sp}/px/{k}"][idx], kf) for k in kinds}
        iou = {k: z[f"{sp}/iou/{k}"][idx][np.arange(len(idx)), kf - 1] for k in kinds if k != "real"}
        R = res["sets"][label] = {"n": int(len(idx)), "kinds": {}, "diff": {}}
        for k in kinds:
            last, late = pe[k]
            row = {"last": float(last.mean()), "late": float(late.mean()), "last_ci": ci(last, rng)}
            if k != "real":
                row.update(last_pct_of_real=float(100 * last.mean() / pe["real"][0].mean()), last_pct_ci=ratio_ci(last, pe["real"][0], rng),
                           late_pct_of_real=float(100 * late.mean() / pe["real"][1].mean()), iou_last=float(np.nanmean(iou[k])))
            if f"{sp}/mse/{k}" in z.files:
                row["mse_x1e3"] = float(np.nanmean(z[f"{sp}/mse/{k}"][idx]) * 1e3)
                row["mmse_x1e3"] = float(np.nanmean(z[f"{sp}/mmse/{k}"][idx]) * 1e3)
            R["kinds"][k] = row
        for a, b in (("small", "ctrl"), ("small", "orig"), ("ctrl", "orig"), ("prev", "small"), ("prev", "ctrl")):
            if a in pe and b in pe:
                d = pe[a][0] - pe[b][0]
                R["diff"][f"{a}-{b}"] = {"mean": float(d.mean()), "ci": ci(d, rng), "late_mean": float((pe[a][1] - pe[b][1]).mean())}
        curves[label] = {k: [float(x) for x in np.nanmean(z[f"{sp}/px/{k}"][idx], axis=0)] for k in kinds}
    res["curves_px_by_model_step"] = curves

    # ---------------- retention
    ind_idx = sets["InD"][1]
    rec_ood, rec_ind = z["ood_test/px/recon"], z["ind_test/px/recon"]

    def arrays(m):
        g = {"rollout": z[f"ood_test/px/{m}"], "ahead1": z[f"ood_test/ahead1/{m}"], "ahead5": z[f"ood_test/ahead5/{m}"]}
        i = {"rollout": z[f"ind_test/px/{m}"], "ahead1": z[f"ind_test/ahead1/{m}"], "ahead5": z[f"ind_test/ahead5/{m}"]}
        return g, i
    A = {m: arrays(m) for m in models}
    for label in ("OoD larger, odd 25", "OoD larger, all 50", "OoD smaller, odd 25"):
        gi = sets[label][1]
        ms = [m for m in models if valid(m, label)]
        point = {m: r_summary(A[m], gi, ind_idx, (rec_ood, rec_ind)) for m in ms}
        draws = {m: {k: [] for k in ("r5", "r12", "one_step", "f25")} for m in ms}
        for _ in range(B_R):
            g_s, i_s = rng.integers(0, len(gi), len(gi)), rng.integers(0, len(ind_idx), len(ind_idx))
            for m in ms:
                s = r_summary(A[m], gi[g_s], ind_idx[i_s], (rec_ood, rec_ind))
                for k, v in s.items():
                    draws[m][k].append(v)
        out = res["retention"][label] = {"n": int(len(gi)), "models": {}, "diff": {}}
        for m in ms:
            out["models"][m] = {k: float(v) for k, v in point[m].items()}
        for a, b in (("small", "ctrl"), ("small", "orig"), ("ctrl", "orig"), ("prev", "small"), ("prev", "ctrl")):
            if a in ms and b in ms:
                out["diff"][f"{a}-{b}"] = {}
                for k in ("r5", "r12", "one_step", "f25"):
                    d = np.array(draws[a][k]) - np.array(draws[b][k])
                    out["diff"][f"{a}-{b}"][k] = {"mean": float(point[a][k] - point[b][k]), "ci": [float(x) for x in np.nanpercentile(d, [2.5, 97.5])]}
    json.dump(res, open(os.path.join(out_dir, "small_only.json"), "w"), indent=1)

    # ---------------- markdown
    L = []
    f0 = lambda x: f"{x:,.0f}"

    def cell(R, k, key="last"):
        r = R["kinds"].get(k)
        if r is None:
            return "–"
        s = f0(r[key])
        return s if k == "real" else f"{s} ({r['last_pct_of_real' if key == 'last' else 'late_pct_of_real']:.0f} %)"
    for title, key in (("Cloth pixels in the last frame of the rollout, mean over episodes (% of Real)", "last"),
                       ("Cloth pixels averaged over frames 25 to the end of the rollout (% of Real)", "late")):
        L += [f"### {title}", "", "| Test set | n | Real | Enc→Dec | " + " | ".join(NAME[m] for m in models) + " |", "|---|---:|---:|---:|" + "---:|" * len(models)]
        for label, R in res["sets"].items():
            L.append(f"| {label} | {R['n']} | {cell(R, 'real', key)} | {cell(R, 'recon', key)} | " + " | ".join(cell(R, m, key) for m in models) + " |")
        L.append("")
    pairs = ("small-ctrl", "small-orig", "ctrl-orig", "prev-small")
    L += ["### Last-frame pixels, paired difference between two models (px, 95 % bootstrap interval over episodes)", "",
          "| Test set | n | " + " | ".join(p.replace("-", " − ") for p in pairs) + " |", "|---|---:|" + "---:|" * len(pairs)]
    for label, R in res["sets"].items():
        cs = []
        for pair in pairs:
            d = R["diff"].get(pair)
            cs.append("–" if d is None else f"{d['mean']:+,.0f} [{d['ci'][0]:+,.0f}, {d['ci'][1]:+,.0f}]")
        L.append(f"| {label} | {R['n']} | " + " | ".join(cs) + " |")
    L += ["", "### Retention r(k) of the OoD size difference (1 = the OoD size is kept, 0 = pulled to the InD size)", "",
          "| Test set | Model | rollout r(5) | rollout r(12) | one step | 25 frames |", "|---|---|---:|---:|---:|---:|"]
    for label, RR in res["retention"].items():
        for m, v in RR["models"].items():
            L.append(f"| {label} | {NAME[m]} | {v['r5']:.2f} | {v['r12']:.2f} | {v['one_step']:.2f} | {v['f25']:.2f} |")
    L += ["", "### Retention, paired difference between two models (95 % bootstrap interval over episodes)", "",
          "| Test set | Pair | rollout r(5) | rollout r(12) | one step | 25 frames |", "|---|---|---:|---:|---:|---:|"]
    for label, RR in res["retention"].items():
        for pair, v in RR["diff"].items():
            L.append(f"| {label} | {pair.replace('-', ' − ')} | " + " | ".join(f"{v[k]['mean']:+.2f} [{v[k]['ci'][0]:+.2f}, {v[k]['ci'][1]:+.2f}]" for k in ("r5", "r12", "one_step", "f25")) + " |")
    L += ["", "### Mask IoU of the last frame against Real, rollout MSE and motion-masked MSE (x1e-3, mean over model steps)", "",
          "| Test set | Model | IoU last | MSE | M-MSE |", "|---|---|---:|---:|---:|"]
    for label, R in res["sets"].items():
        for m in ["recon"] + models:
            r = R["kinds"].get(m)
            if r is not None:
                L.append(f"| {label} | {'Enc→Dec' if m == 'recon' else NAME[m]} | {r['iou_last']:.3f} | " + (f"{r['mse_x1e3']:.2f} | {r['mmse_x1e3']:.1f} |" if "mse_x1e3" in r else "– | – |"))
    open(os.path.join(out_dir, "tables.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
