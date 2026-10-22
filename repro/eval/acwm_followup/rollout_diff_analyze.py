"""InD 만 학습한 DINO-WM 과 InD+OoD 섞어 학습한 DINO-WM 의 rollout 차이 분석 (ACWM-Phys, 모델마다 한 번 학습한 값).

입력 (<dir>, 기본 $DINO_WORK/results/acwm_phys/ood_mix/rollout_diff):
  <env>_rollout_curves.npz    rollout_curves.py 가 만든 에피소드마다, 모델 스텝마다의 오차와 양
  cloth_control_curves.npz    cloth_control.py 가 만든 Cloth Move 대조군 (InD 만 다시 학습한 모델). 없으면 대조군 절을 뺀다.
  ../<env>_mix_acwm.{json,npz}  acwm_metrics_mix.py 의 PSNR, SSIM, 37 프레임 창 값
OoD test 는 학습에 넣지 않은 에피소드만. 95 % 구간은 에피소드를 다시 뽑는 bootstrap (10,000 회)이고 두 모델은 같은 에피소드에서 짝지어 비교한다.
rollout = 첫 실제 프레임과 action 으로 에피소드 끝까지 예측 (Cloth, Sand, Stack 은 모델 스텝 12 개, Pour Water 는 59 개; 1 스텝 = 5 프레임).

usage (env sourced): python <this> [dir]  ->  <dir>/rollout_diff.json, tables.md (전체 표), readme_tables.md (루트 README 의 두 표)
"""
import json, os, sys
import numpy as np

WORK = os.environ["DINO_WORK"]
D = os.path.join(WORK, "results", "acwm_phys", "ood_mix")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, "rollout_diff")
CTRL = os.path.exists(f"{OUT}/cloth_control_curves.npz")
B = 10000
M0, M1 = "ind_only", "ind_ood"
ENVS = ("cloth", "sand", "water", "stack")
TITLE = {"cloth": "Cloth Move", "sand": "Push Sand", "water": "Pour Water", "stack": "Stack Cube"}
GROUPS = {  # (set, group, label)
    "cloth": [("ind_test", "all", "InD"), ("ood_test", "smaller", "OoD, 0.26x area"), ("ood_test", "larger", "OoD, 2.5x area")],
    "sand": [("ind_test", "all", "InD"), ("ood_test", "all", "OoD, 2x particles")],
    "water": [("ind_test", "all", "InD"), ("ood_test", "low", "OoD, less water"), ("ood_test", "high", "OoD, more water")],
    "stack": [("ind_test", "all", "InD"), ("ind_test", "fail", "InD, failed"), ("ood_test", "all", "OoD"), ("ood_test", "fail", "OoD, failed")],
}
ONSET_PX = 200  # Pour Water: box water pixels that count as "the water has arrived"
FILL_FROM = 50  # Pour Water: step from which the water in the box is averaged (last 5 s), as in talk_render.py
BANDS = {13: [(1, 4), (5, 8), (9, 12)], 12: [(1, 4), (5, 8), (9, 11)], 60: [(1, 15), (16, 40), (41, 59)]}  # early, middle, late model steps
AMOUNT_Q = {"cloth": "px", "sand": "px", "water": "box", "stack": "stacked"}
RETENTION_Q = {"cloth": "px", "sand": "px", "water": "px"}  # Pour Water: all water pixels (the box is empty at the start, so its OoD difference is 0 there)


def load(env):
    z = np.load(f"{OUT}/{env}_rollout_curves.npz")
    st = np.load(f"{D}/{env}_mix_acwm.npz")
    for sp in ("ind_test", "ood_test"):
        assert list(z[f"{sp}/names"]) == list(st[f"{sp}/names"]), (env, sp)
    return z, st


def rel_ci(a, b, rng):
    """change of mean(b) against mean(a) in % of mean(a), 95 % bootstrap interval over episodes (paired)"""
    idx = rng.integers(0, len(a), (B, len(a)))
    d = (b[idx].mean(1) - a[idx].mean(1)) / a[idx].mean(1) * 100
    return [float((b.mean() - a.mean()) / a.mean() * 100), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def abs_ci(a, b, rng):
    idx = rng.integers(0, len(a), (B, len(a)))
    d = b[idx].mean(1) - a[idx].mean(1)
    return [float(b.mean() - a.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def ratio_curve(a, b, rng, n=2000):
    """per step: mean(b)/mean(a) over episodes and its 95 % bootstrap interval; a, b: (episodes, steps) with NaN where there is no value"""
    idx = rng.integers(0, len(a), (n, len(a)))
    with np.errstate(all="ignore"):
        r = np.nanmean(b[idx], 1) / np.nanmean(a[idx], 1)
        return (np.nanmean(b, 0) / np.nanmean(a, 0)).tolist(), np.nanpercentile(r, 2.5, 0).tolist(), np.nanpercentile(r, 97.5, 0).tolist()


def amount(z, env, sp, kind):
    """per-episode object amount at the end of the rollout (Pour Water: water in the box averaged over the last 5 s; Stack Cube: 1 if stacked)"""
    kf = z[f"{sp}/kf"]
    a = z[f"{sp}/{AMOUNT_Q[env]}/{kind}"]
    return np.nanmean(a[:, FILL_FROM:], 1) if env == "water" else a[np.arange(len(kf)), kf - 1]


def retention(z, env, kind, g, rng, n=2000):
    """r(k) = (OoD group mean - InD mean) of the predicted amount / the same for Encoder->Decoder, per model step; with a bootstrap interval"""
    q = RETENTION_Q[env]
    mo = z[f"ood_test/group/{g}"]
    pi, po = z[f"ind_test/{q}/{kind}"], z[f"ood_test/{q}/{kind}"][mo]
    ri, ro = z[f"ind_test/{q}/recon"], z[f"ood_test/{q}/recon"][mo]
    f = lambda pi, po, ri, ro: (np.nanmean(po, 0) - np.nanmean(pi, 0)) / (np.nanmean(ro, 0) - np.nanmean(ri, 0))
    ii, io = rng.integers(0, len(pi), (n, len(pi))), rng.integers(0, len(po), (n, len(po)))
    bs = np.stack([f(pi[ii[i]], po[io[i]], ri[ii[i]], ro[io[i]]) for i in range(n)])
    return f(pi, po, ri, ro).tolist(), np.nanpercentile(bs, 2.5, 0).tolist(), np.nanpercentile(bs, 97.5, 0).tolist(), (pi, po, ri, ro, ii, io)


def slope_ci(x, ys, rng, n=B):
    """OLS slope of each y (dict of arrays) on x with 95 % bootstrap intervals (episodes resampled jointly, so differences between the ys are paired)"""
    sl = lambda x, y: float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
    idx = rng.integers(0, len(x), (n, len(x)))
    xm = x[idx] - x[idx].mean(1, keepdims=True)
    out = {}
    for k, y in ys.items():
        ym = y[idx] - y[idx].mean(1, keepdims=True)
        bs = (xm * ym).sum(1) / (xm ** 2).sum(1)
        out[k] = [sl(x, y), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out, idx, xm, {k: y[idx] - y[idx].mean(1, keepdims=True) for k, y in ys.items()}


def main():
    rng = np.random.default_rng(0)
    rng2 = np.random.default_rng(1)  # for the Real / amount-curve additions, so that the other draws stay the same
    out = {}
    for env in ENVS:
        z, st = load(env)
        K = z["ind_test/mse/ind_only"].shape[1]
        o = out[env] = {"meta": json.loads(str(z["meta"])), "K": K, "groups": {}}
        for sp, g, lab in GROUPS[env]:
            m = z[f"{sp}/group/{g}"]
            r = {"label": lab, "set": sp, "group": g, "n": int(m.sum())}
            for name in ("mse", "mmse"):  # whole rollout: mean over steps, then over episodes
                a, b = (np.nanmean(z[f"{sp}/{name}/{k}"], 1)[m] for k in (M0, M1))
                r[name] = {"ind_only": float(a.mean()), "ind_ood": float(b.mean()), "change_pct": rel_ci(a, b, rng), "win_rate": float((b < a).mean()), "wins": int((b < a).sum())}
            for name in ("psnr", "ssim"):
                a, b = st[f"{sp}/rollout/{M0}/{name}"][m], st[f"{sp}/rollout/{M1}/{name}"][m]
                r[name] = {"ind_only": float(a.mean()), "ind_ood": float(b.mean()), "diff": abs_ci(a, b, rng), "win_rate": float((b > a).mean())}
            real, a, b = (amount(z, env, sp, k)[m] for k in ("real", M0, M1))
            ea, eb = np.abs(a - real), np.abs(b - real)
            r["amount"] = {"real": float(real.mean()), "ind_only": float(a.mean()), "ind_ood": float(b.mean()), "mae_ind_only": float(ea.mean()),
                           "mae_ind_ood": float(eb.mean()), "mae_change": abs_ci(ea, eb, rng), "win_rate_abs_err": float((eb < ea).mean())}
            if env == "stack":
                r["amount"]["agree_ind_only"], r["amount"]["agree_ind_ood"] = float((a == real).mean()), float((b == real).mean())
            else:
                r["amount"]["change_of_mean"] = abs_ci(a, b, rng)
                den = real.mean() - a.mean()
                r["amount"]["gap_closed"] = float((b.mean() - a.mean()) / den) if abs(den) > 0.1 * real.mean() else None
            # per step: error of both models and the distance between the two rollouts
            ma, mb = z[f"{sp}/mse/{M0}"][m], z[f"{sp}/mse/{M1}"][m]
            ratio, lo, hi = ratio_curve(ma, mb, rng)
            r["steps"] = {"mse_ind_only": np.nanmean(ma, 0).tolist(), "mse_ind_ood": np.nanmean(mb, 0).tolist(), "dmse": np.nanmean(z[f"{sp}/dmse"][m], 0).tolist(),
                          "ratio": ratio, "ratio_lo": lo, "ratio_hi": hi}
            r["dmse_mean"] = float(np.nanmean(z[f"{sp}/dmse"][m]))
            r["bands"] = {}
            for s0, s1 in BANDS[K]:
                a_, b_ = np.nanmean(ma[:, s0 : s1 + 1], 1), np.nanmean(mb[:, s0 : s1 + 1], 1)
                r["bands"][f"{s0}-{s1}"] = {"ind_only": float(a_.mean()), "ind_ood": float(b_.mean()), "change_pct": rel_ci(a_, b_, rng), "win_rate": float((b_ < a_).mean())}
            r["amount_steps"] = {}  # group mean of the amount at every model step, with Real
            for kind in ("real", M0, M1):
                a = z[f"{sp}/{AMOUNT_Q[env]}/{kind}"][m]
                bs = np.nanmean(a[rng2.integers(0, len(a), (2000, len(a)))], 1)
                r["amount_steps"][kind] = {"mean": np.nanmean(a, 0).tolist(), "lo": np.nanpercentile(bs, 2.5, 0).tolist(), "hi": np.nanpercentile(bs, 97.5, 0).tolist()}
            o["groups"][f"{sp}/{g}"] = r
        if env in ("cloth", "sand", "water"):  # retention of the OoD difference
            o["retention"] = {}
            for g in sorted({g for sp, g, _ in GROUPS[env] if sp == "ood_test"}):
                if g == "all" and env != "sand":
                    continue
                o["retention"][g] = {}
                for kind in (M0, M1):
                    r_, lo_, hi_, _ = retention(z, env, kind, g, rng)
                    o["retention"][g][kind] = {"r": r_, "lo": lo_, "hi": hi_}
    if CTRL:  # Cloth Move: a second InD-only model as a control
        z = np.load(f"{OUT}/cloth_control_curves.npz")
        c = out["cloth"]["control"] = {"model": os.path.basename(json.loads(str(z["meta"]))["models"]["control"]), "groups": {}}
        for sp, g, lab in GROUPS["cloth"]:
            m = z[f"{sp}/group/{g}"]
            e = {k: np.nanmean(z[f"{sp}/mse/{k}"], 1)[m] for k in (M0, "control", M1)}
            kf = z[f"{sp}/kf"]
            last = {k: z[f"{sp}/px/{k}"][np.arange(len(kf)), kf - 1][m] for k in ("real", "recon", M0, "control", M1)}
            c["groups"][f"{sp}/{g}"] = {
                "label": lab, "n": int(m.sum()), "mse": {k: float(v.mean()) for k, v in e.items()},
                "change_control_vs_orig": rel_ci(e[M0], e["control"], rng), "change_oodmix_vs_orig": rel_ci(e[M0], e[M1], rng),
                "change_oodmix_vs_control": rel_ci(e["control"], e[M1], rng), "last_px": {k: float(v.mean()) for k, v in last.items()},
                "dmse": {k: float(np.nanmean(z[f"{sp}/dmse/{k}"][m])) for k in ("ind_only__ind_ood", "ind_only__control", "ind_ood__control")}}
        c["retention"] = {}
        for g in ("larger", "smaller"):
            c["retention"][g] = {}
            for kind in (M0, "control", M1):
                r_, lo_, hi_, _ = retention(z, "cloth", kind, g, rng)
                c["retention"][g][kind] = {"r": r_, "lo": lo_, "hi": hi_}
    # slope of the predicted amount at the end on the real amount, over InD episodes and held-out OoD episodes together
    out["slope"] = {}
    for env in ("cloth", "sand", "water"):
        z, _ = load(env)
        zc = np.load(f"{OUT}/cloth_control_curves.npz") if env == "cloth" and CTRL else None
        mi, mo = z["ind_test/group/all"], z["ood_test/group/all"]
        cat = lambda zz, k: np.concatenate([amount(zz, env, "ind_test", k)[mi], amount(zz, env, "ood_test", k)[mo]])
        x = cat(z, "real")
        ys = {M0: cat(z, M0), M1: cat(z, M1)}
        if zc is not None:
            ys["control"] = np.concatenate([amount(zc, env, "ind_test", "control")[zc["ind_test/group/all"]], amount(zc, env, "ood_test", "control")[zc["ood_test/group/all"]]])
        sl, idx, xm, ym = slope_ci(x, ys, rng)
        bs = {k: (xm * v).sum(1) / (xm ** 2).sum(1) for k, v in ym.items()}
        d = bs[M1] - bs[M0]
        out["slope"][env] = {"n": int(len(x)), "slope": sl, "ind_ood_minus_ind_only": [sl[M1][0] - sl[M0][0], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        if "control" in sl:
            dc = bs["control"] - bs[M0]
            out["slope"][env]["control_minus_ind_only"] = [sl["control"][0] - sl[M0][0], float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))]
    rng3 = np.random.default_rng(2)
    zw, _ = load("water")  # Pour Water: first step at which the box holds at least ONSET_PX water pixels
    out["water"]["onset"] = {}
    for sp, g, lab in GROUPS["water"]:
        m = zw[f"{sp}/group/{g}"]
        on = {}
        for kind in ("real", M0, M1):
            hit = zw[f"{sp}/box/{kind}"][m] >= ONSET_PX
            on[kind] = np.where(hit.any(1), hit.argmax(1), np.nan)
        res = {"label": lab, "n": int(m.sum()), "mean": {k: float(np.nanmean(v)) for k, v in on.items()}}
        for kind in (M0, M1):
            d = on[kind] - on["real"]
            d = d[~np.isnan(d)]
            bs = d[rng3.integers(0, len(d), (B, len(d)))].mean(1)
            res[kind] = {"minus_real": [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "earlier_share": float((d < 0).mean()),
                         "mean_abs": float(np.abs(d).mean())}
        out["water"]["onset"][f"{sp}/{g}"] = res
    for env in ("cloth", "sand", "water"):  # Real: its own OoD difference over the Encoder->Decoder one (1 = the decoder draws the real difference)
        z, _ = load(env)
        for g in out[env]["retention"]:
            r_, lo_, hi_, _ = retention(z, env, "real", g, rng2)
            out[env]["retention"][g]["real"] = {"r": r_, "lo": lo_, "hi": hi_}
    if CTRL:
        for g in ("larger", "smaller"):
            out["cloth"]["control"]["retention"][g]["real"] = out["cloth"]["retention"][g]["real"]
    json.dump(out, open(f"{OUT}/rollout_diff.json", "w"), indent=1)
    return out


def fmt_tables(out):
    L = []
    k3 = lambda x: f"{1e3 * x:.2f}"
    pct = lambda c: f"{c[0]:+.0f} % [{c[1]:+.0f}, {c[2]:+.0f}]"
    L.append("# Rollout: InD-only DINO-WM vs InD+OoD DINO-WM\n")
    L.append("Rollout = first real frame + actions, predicted to the end of the episode (1 step = 5 frames). OoD rows use only the OoD test episodes that were not trained on. "
             "One training run per model. Intervals: 95 % bootstrap over episodes, both models on the same episodes.\n")
    L.append("## 1. Pixel error over the whole rollout\n")
    L.append("MSE and M-MSE x1e-3 (lower is better); change = InD+OoD vs InD only; win = share of episodes where InD+OoD has the lower MSE.\n")
    L.append("| env | split | n | MSE InD only | MSE InD+OoD | change | win | M-MSE InD only | M-MSE InD+OoD | change | PSNR | SSIM |")
    L.append("|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---|")
    for env in ENVS:
        for i, (k, r) in enumerate(out[env]["groups"].items()):
            L.append(f"| {TITLE[env] if i == 0 else ''} | {r['label']} | {r['n']} | {k3(r['mse']['ind_only'])} | {k3(r['mse']['ind_ood'])} | {pct(r['mse']['change_pct'])} | "
                     f"{100 * r['mse']['win_rate']:.0f} % | {k3(r['mmse']['ind_only'])} | {k3(r['mmse']['ind_ood'])} | {r['mmse']['change_pct'][0]:+.0f} % | "
                     f"{r['psnr']['ind_only']:.2f} -> {r['psnr']['ind_ood']:.2f} | {r['ssim']['ind_only']:.3f} -> {r['ssim']['ind_ood']:.3f} |")
    L.append("\n## 2. Object amount at the end of the rollout\n")
    L.append("Cloth Move, Push Sand: object pixels in the last frame. Pour Water: water pixels in the box, mean over the last 5 s. Stack Cube: share of episodes drawn as stacked in the last frame. "
             "MAE = mean absolute difference to Real per episode. Gap closed = (InD+OoD - InD only) / (Real - InD only), only where the gap is more than 10 % of Real.\n")
    L.append("| env | split | n | Real | InD only | InD+OoD | MAE InD only | MAE InD+OoD | MAE change | gap closed |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    for env in ENVS:
        for i, (k, r) in enumerate(out[env]["groups"].items()):
            a = r["amount"]
            if env == "stack":
                f = lambda x: f"{100 * x:.0f} %"
                mae = f"{100 * a['mae_ind_only']:.0f} %", f"{100 * a['mae_ind_ood']:.0f} %", f"agree {100 * a['agree_ind_only']:.0f} -> {100 * a['agree_ind_ood']:.0f} %"
                gap = "-"
            else:
                f = lambda x: f"{x:,.0f}"
                mae = f"{a['mae_ind_only']:,.0f}", f"{a['mae_ind_ood']:,.0f}", f"{a['mae_change'][0]:+,.0f} [{a['mae_change'][1]:+,.0f}, {a['mae_change'][2]:+,.0f}]"
                gap = "-" if a["gap_closed"] is None else f"{100 * a['gap_closed']:.0f} %"
            L.append(f"| {TITLE[env] if i == 0 else ''} | {r['label']} | {r['n']} | {f(a['real'])} | {f(a['ind_only'])} | {f(a['ind_ood'])} | {mae[0]} | {mae[1]} | {mae[2]} | {gap} |")
    L.append("\n## 3. Error by rollout stage\n")
    L.append("MSE x1e-3 averaged over the steps of each stage (Cloth, Sand: steps 1-4, 5-8, 9-12; Stack: 1-4, 5-8, 9-11; Pour Water: 1-15, 16-40, 41-59). change = InD+OoD vs InD only.\n")
    L.append("| env | split | early InD only | early InD+OoD | change | middle InD only | middle InD+OoD | change | late InD only | late InD+OoD | change |")
    L.append("|---|---|---:|---:|---|---:|---:|---|---:|---:|---|")
    for env in ENVS:
        for i, (k, r) in enumerate(out[env]["groups"].items()):
            cells = []
            for b in r["bands"].values():
                cells += [k3(b["ind_only"]), k3(b["ind_ood"]), pct(b["change_pct"])]
            L.append(f"| {TITLE[env] if i == 0 else ''} | {r['label']} | " + " | ".join(cells) + " |")
    L.append("\n## 4. How far apart the two rollouts are\n")
    L.append("MSE between the frames of the two models, averaged over the rollout (x1e-3), next to each model's error to Real.\n")
    L.append("| env | split | between the two models | error InD only | error InD+OoD |")
    L.append("|---|---|---:|---:|---:|")
    for env in ENVS:
        for i, (k, r) in enumerate(out[env]["groups"].items()):
            L.append(f"| {TITLE[env] if i == 0 else ''} | {r['label']} | {k3(r['dmse_mean'])} | {k3(r['mse']['ind_only'])} | {k3(r['mse']['ind_ood'])} |")
    if CTRL:
        c = out["cloth"]["control"]
        L.append("\n## 5. Cloth Move: how big is the difference between two InD-only runs?\n")
        L.append(f"Control = second InD-only model trained from scratch ({c['model']}), same architecture and data, encoder and decoder of the original. "
                 "The three predictors roll out the same episodes. MSE x1e-3; changes are against the original InD-only model.\n")
        L.append("| split | n | MSE original | MSE control (InD only, 2nd run) | change | MSE InD+OoD | change | InD+OoD vs control | last-frame px: Real / original / control / InD+OoD | MSE between: orig-control / orig-InD+OoD / control-InD+OoD |")
        L.append("|---|---:|---:|---:|---|---:|---|---|---|---|")
        for k, r in c["groups"].items():
            lp, dm = r["last_px"], r["dmse"]
            L.append(f"| {r['label']} | {r['n']} | {k3(r['mse']['ind_only'])} | {k3(r['mse']['control'])} | {pct(r['change_control_vs_orig'])} | {k3(r['mse']['ind_ood'])} | {pct(r['change_oodmix_vs_orig'])} | "
                     f"{pct(r['change_oodmix_vs_control'])} | {lp['real']:,.0f} / {lp['ind_only']:,.0f} / {lp['control']:,.0f} / {lp['ind_ood']:,.0f} | "
                     f"{k3(dm['ind_only__control'])} / {k3(dm['ind_only__ind_ood'])} / {k3(dm['ind_ood__control'])} |")
    L.append("\n## 6. Retention of the OoD difference along the rollout\n")
    L.append("r(k) = (OoD group mean - InD mean) of the predicted amount / the same for Encoder->Decoder, at model step k (Pour Water: all water pixels in the frame). 1 = the model keeps the full OoD difference, 0 = the OoD group looks like InD. "
             "Interval: 95 % bootstrap over episodes.\n")
    ks = {"cloth": [1, 3, 5, 8, 12], "sand": [1, 3, 5, 8, 12], "water": [5, 15, 30, 45, 59]}
    for env in ("cloth", "sand", "water"):
        for g, rr in out[env]["retention"].items():
            kk = ks[env]
            has_ctrl = env == "cloth" and CTRL
            L.append(f"\n{TITLE[env]}, OoD {g}" + (" (+ control: second InD-only run)" if has_ctrl else ""))
            L.append("\n| model | " + " | ".join(f"k={k}" for k in kk) + " |")
            L.append("|---|" + "---:|" * len(kk))
            rows = [("Real", rr["real"]), ("InD only", rr[M0])]
            if has_ctrl:
                rows.append(("InD only, 2nd run", out["cloth"]["control"]["retention"][g]["control"]))
            rows.append(("InD+OoD", rr[M1]))
            for lab, v in rows:
                L.append(f"| {lab} | " + " | ".join(f"{v['r'][k]:.2f} [{v['lo'][k]:.2f}, {v['hi'][k]:.2f}]" for k in kk) + " |")
    L.append("\n## 7. Does the predicted amount at the end follow the real amount?\n")
    L.append("Slope of the predicted end amount on the real end amount over the InD test episodes and the held-out OoD test episodes together (1 = follows the real amount, 0 = ignores it). "
             "Amount as in section 2. Interval: 95 % bootstrap over episodes; the change is paired.\n")
    L.append("| env | n | InD only | InD+OoD | change (InD+OoD - InD only) | InD only, 2nd run | change (2nd run - InD only) |")
    L.append("|---|---:|---|---|---|---|---|")
    fs = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
    for env, r in out["slope"].items():
        c2 = ("-", "-") if "control" not in r["slope"] else (fs(r["slope"]["control"]), f"{r['control_minus_ind_only'][0]:+.2f} [{r['control_minus_ind_only'][1]:+.2f}, {r['control_minus_ind_only'][2]:+.2f}]")
        d = r["ind_ood_minus_ind_only"]
        L.append(f"| {TITLE[env]} | {r['n']} | {fs(r['slope'][M0])} | {fs(r['slope'][M1])} | {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}] | {c2[0]} | {c2[1]} |")
    L.append("\n## 8. Window protocol vs whole-episode rollout\n")
    L.append("MSE x1e-3, InD only -> InD+OoD, from ../<env>_mix_acwm.json. Window = the original evaluation: 10 windows of 37 frames per episode, every fifth frame predicted from the window's first frame. "
             "Rollout = the whole episode from its first frame (this file).\n")
    L.append("| env | split | window | rollout |")
    L.append("|---|---|---|---|")
    for env in ENVS:
        js = json.load(open(f"{D}/{env}_mix_acwm.json"))["groups"]
        for i, (k, r) in enumerate(out[env]["groups"].items()):
            w, ro = js[k]["window"], js[k]["rollout"]
            L.append(f"| {TITLE[env] if i == 0 else ''} | {r['label']} | {k3(w['ind_only']['mse'])} -> {k3(w['ind_ood']['mse'])} | {k3(ro['ind_only']['mse'])} -> {k3(ro['ind_ood']['mse'])} |")
    L.append("\n## 9. Amount along the rollout, with Real\n")
    L.append("Group mean of the object amount at model step k: Real / InD only / InD+OoD. Cloth Move, Push Sand: object pixels. Pour Water: water pixels in the box. "
             "Stack Cube: % of episodes drawn as stacked.\n")
    S9 = {"cloth": [3, 6, 9, 12], "sand": [3, 6, 9, 12], "water": [10, 20, 30, 40, 50, 59], "stack": [3, 6, 9, 11]}
    for env in ENVS:
        ks = S9[env]
        L.append(f"\n{TITLE[env]}\n")
        L.append("| split | n | " + " | ".join(f"k={k}" for k in ks) + " |")
        L.append("|---|---:|" + "---:|" * len(ks))
        f = (lambda x: f"{100 * x:.0f}") if env == "stack" else (lambda x: f"{x:,.0f}")
        for key, r in out[env]["groups"].items():
            a = r["amount_steps"]
            L.append(f"| {r['label']} | {r['n']} | " + " | ".join(" / ".join(f(a[kd]["mean"][k]) for kd in ("real", M0, M1)) for k in ks) + " |")
    L.append("\n## 10. Pour Water: when the water reaches the box\n")
    L.append(f"First model step (1 step = 5 frames) at which the box holds at least {ONSET_PX} water pixels. Model minus Real: mean over episodes, 95 % bootstrap interval, share of episodes where the model is earlier than Real.\n")
    L.append("| split | n | Real | InD only | InD+OoD | InD only - Real | earlier | InD+OoD - Real | earlier |")
    L.append("|---|---:|---:|---:|---:|---|---:|---|---:|")
    for k, r in out["water"]["onset"].items():
        a, b = r[M0], r[M1]
        L.append(f"| {r['label']} | {r['n']} | {r['mean']['real']:.1f} | {r['mean'][M0]:.1f} | {r['mean'][M1]:.1f} | {a['minus_real'][0]:+.1f} [{a['minus_real'][1]:+.1f}, {a['minus_real'][2]:+.1f}] | "
                 f"{100 * a['earlier_share']:.0f} % | {b['minus_real'][0]:+.1f} [{b['minus_real'][1]:+.1f}, {b['minus_real'][2]:+.1f}] | {100 * b['earlier_share']:.0f} % |")
    L.append("\n## Notes\n")
    L.append("- One training run per model. " + ("Cloth Move also has a second InD-only run (the 'absolute' predictor of the pred-param experiment, branch hanbin5/pred-param, trained with a different script in float32), "
                                                "which shows how far two InD-only runs are apart. The other environments have no such run." if CTRL else "There is no second InD-only run."))
    L.append("- OoD episodes added to training: Cloth Move 50 (2.7 % of the training set), Push Sand 50 (3.0 %), Pour Water 26 (2.9 %), Stack Cube 136 (9.9 %). The OoD rows use only the OoD test episodes that were not added.")
    gap = max(o["meta"]["max_abs_recon_difference_between_models"] for o in out.values() if isinstance(o, dict) and "meta" in o and "max_abs_recon_difference_between_models" in o["meta"])
    L.append(f"- Both models use the same encoder (frozen DINOv2) and the same decoder trained on InD frames. The decoded real frames of the two models differ by at most {gap:.1e}. Only the predictor differs.")
    L.append("- Retention for Pour Water uses all water pixels in the frame. Its early values above 1 come from the decoded water pixel count while the water moves. After step 30 both models have r close to 0.")
    L.append("- Per-step values are in rollout_diff.json (steps.*, amount_steps.*). Scripts: rollout_curves.py, cloth_control.py, rollout_diff_analyze.py, rollout_diff_figs.py.")
    open(f"{OUT}/tables.md", "w").write("\n".join(L) + "\n")


README_ROWS = [  # (env, group key, label, unit of the amount in the first table)
    ("cloth", "ood_test/larger", "Cloth Move, OoD larger cloth", "cloth pixels"),
    ("cloth", "ood_test/smaller", "Cloth Move, OoD smaller cloth", "cloth pixels"),
    ("cloth", "ind_test/all", "Cloth Move, InD test", "cloth pixels"),
    ("sand", "ood_test/all", "Push Sand, OoD 2× particles", "sand pixels"),
    ("sand", "ind_test/all", "Push Sand, InD test", "sand pixels"),
    ("water", "ood_test/low", "Pour Water, OoD less water", "water pixels in the box"),
    ("water", "ood_test/high", "Pour Water, OoD more water", "water pixels in the box"),
    ("water", "ind_test/all", "Pour Water, InD test", "water pixels in the box"),
    ("stack", "ood_test/fail", "Stack Cube, OoD failed stacks", "shows the cube stacked"),
    ("stack", "ind_test/fail", "Stack Cube, InD failed stacks", "shows the cube stacked"),
]


def fmt_readme(out):
    """the two tables of the root README (the first for the last frame of the rollout, the second for the rollout error)"""
    L = ["| Last frame of the rollout | Real | InD only | InD + OoD |", "|---|---:|---:|---:|"]
    for env, key, label, unit in README_ROWS:
        r = out[env]["groups"][key]
        a = r["amount"]
        f = (lambda x: f"{100 * x:.0f} %") if env == "stack" else (lambda x: f"{x:,.0f}")
        L.append(f"| {label} ({r['n']}): {unit} | {f(a['real'])} | {f(a['ind_only'])} | {f(a['ind_ood'])} |")
    L += ["", "| Rollout error, MSE × 10⁻³ | InD only | InD + OoD | Change [95 % interval] | Episodes with the lower error: InD + OoD |", "|---|---:|---:|---|---:|"]
    for env, key, label, unit in README_ROWS:
        if env == "stack":
            key = key.replace("/fail", "/all")
            label = label.replace(" failed stacks", "")
        r = out[env]["groups"][key]
        m = r["mse"]
        c = m["change_pct"]
        L.append(f"| {label} ({r['n']}) | {1e3 * m['ind_only']:.2f} | {1e3 * m['ind_ood']:.2f} | {c[0]:+.0f} % [{c[1]:+.0f}, {c[2]:+.0f}] | {m['wins']} of {r['n']} |")
    open(f"{OUT}/readme_tables.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    out = main()
    fmt_tables(out)
    fmt_readme(out)
    print(open(f"{OUT}/tables.md").read())
    print("wrote", f"{OUT}/rollout_diff.json", f"{OUT}/tables.md", f"{OUT}/readme_tables.md")
