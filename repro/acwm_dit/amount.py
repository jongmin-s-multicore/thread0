"""Physical amounts in ACWM-DiT whole-episode rollouts (rollout_chain.py), measured the way our DINO-WM evaluation does.

Same colour masks, frame size and frames as thread0 repro/eval/eval_acwm_phys_quant.py (Cloth Move, Push Sand, Stack Cube)
and the fill level of repro/eval/eval_pour_water.py (Pour Water): every frame is resized to 224x224 (INTER_AREA) and
measured at frames 0, 5, 10, ... (the frames a frameskip-5 DINO-WM rollout produces). The ground truth is the real video
measured with the same code.

usage (cwd = ACWM-Phys code root, see README.md here):  python <this folder>/amount.py <chain dir> <env> [<env> ...]
writes <chain dir>/<env>_amount.json
"""
import json, os, sys
import numpy as np
import cv2
import torch

sys.path.insert(0, os.getcwd())
from acwm.dataset.data_config import get_config_by_name

F = 5
SIZE = 224
MIN_PX = 6
STACK = dict(dx=11.0, dy=(-35.0, -6.5))
BAND = (140, 225, 310, 300)  # Pour Water destination box floor band in 360x360 px
FILL_FROM = 250
W0_SPLIT = 4000
ENV = {"clothmove": "acwm_cloth_move", "push_sand": "acwm_push_sand", "stack_cube": "acwm_stack_cube", "pour_water": "pour_water"}
AREAS = {"acwm_cloth_move": ["cloth"], "acwm_push_sand": ["sand"], "acwm_stack_cube": ["yellow", "green"]}


def boot(vals, n=2000, seed_=0):
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [None, None, None]
    idx = np.random.RandomState(seed_).randint(0, len(x), (n, len(x)))
    b = x[idx].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def hue(hsv, lo, hi, s, v=80):
    h = hsv[..., 0]
    m = (h >= lo) & (h <= hi) if lo <= hi else ((h >= lo) | (h <= hi))
    return m & (hsv[..., 1] > s) & (hsv[..., 2] > v)


def blob(m):
    n = int(m.sum())
    if n < MIN_PX:
        return n, np.nan, np.nan
    ys, xs = np.nonzero(m)
    return n, float(xs.mean()), float(ys.mean())


def largest_dark(hsv, min_px=30):
    m = ((hsv[..., 2] < 90) & (hsv[..., 1] < 90)).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return 0, np.nan, np.nan
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    a = int(st[i, cv2.CC_STAT_AREA])
    return (a, float(cen[i][0]), float(cen[i][1])) if a >= min_px else (a, np.nan, np.nan)


def measure(env, rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    if env == "acwm_cloth_move":
        cloth = hue(hsv, 8, 16, 120) | hue(hsv, 158, 170, 60)
        return {"cloth": blob(cloth), "blue": blob(hue(hsv, 105, 113, 100)), "red": blob(hue(hsv, 175, 4, 90))}, {"cloth": cloth}
    if env == "acwm_push_sand":
        sand = hue(hsv, 90, 102, 100)
        return {"sand": blob(sand), "board": largest_dark(hsv)}, {"sand": sand}
    yellow, green = hue(hsv, 26, 33, 180, 130), hue(hsv, 55, 66, 150, 60)
    return {"yellow": blob(yellow), "green": blob(green)}, {"yellow": yellow, "green": green}


def water_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[..., 0] >= 98) & (hsv[..., 0] <= 110) & (hsv[..., 1] > 70) & (hsv[..., 2] > 100)


def band_count(frames_u8, size):
    s = size / 360.0
    x0, y0, x1, y1 = (int(round(v * s)) for v in BAND)
    return np.array([water_mask(f)[y0:y1, x0:x1].sum() for f in frames_u8], dtype=float)


def read_video(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(out)


def r224(fr):
    return cv2.resize(fr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def stacked(q, e, k):
    if not (q["yellow"][e, k, 0] >= 15 and q["green"][e, k, 0] >= 15):
        return False
    dx, dy = q["yellow"][e, k, 1] - q["green"][e, k, 1], q["yellow"][e, k, 2] - q["green"][e, k, 2]
    return bool(abs(dx) < STACK["dx"] and STACK["dy"][0] < dy < STACK["dy"][1])


def groups_of(env, split, root, names, meta_by):
    g = {"all": np.arange(len(names))}
    if env == "clothmove" and split == "ood_test":
        larger = np.array([n.startswith("larger") for n in names])
        g["smaller"], g["larger"] = np.where(~larger)[0], np.where(larger)[0]
    if env == "stack_cube":
        succ = np.array([bool(meta_by[n]["success"]) for n in names])
        g["success"], g["fail"] = np.where(succ)[0], np.where(~succ)[0]
        if split == "ood_test":
            lab = np.array([str(meta_by[n]["ood_label"]) for n in names])
            for l in sorted(set(lab)):
                g[l] = np.where(lab == l)[0]
    if env == "pour_water" and split == "ood_test":
        w0 = np.array([int(water_mask(read_video(os.path.join(root, n))[0]).sum()) for n in names])
        g["low"], g["high"] = np.where(w0 < W0_SPLIT)[0], np.where(w0 >= W0_SPLIT)[0]
    return g


def run(chain_dir, env):
    qenv = ENV[env]
    cfg = get_config_by_name(env)
    result = {"env": env, "frame_size": SIZE, "frameskip": F, "sets": {}}
    for split in ("ind_test", "ood_test"):
        p = os.path.join(chain_dir, env, f"{split}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        names, lengths, frames = [str(n) for n in z["names"]], z["length"], z["frames"]
        root = os.path.join(cfg.root_dir, split)
        meta = torch.load(os.path.join(root, "metadata.pt"), weights_only=True)
        meta_by = {os.path.basename(e["video_path"]): e for e in meta}
        E = len(names)
        groups = groups_of(env, split, root, names, meta_by)
        res = {"n_episodes": E, "group_sizes": {g: int(len(v)) for g, v in groups.items()}, "checkpoint_step": int(z["checkpoint_step"]), "groups": {}}
        if env == "pour_water":
            fill = {"gt": np.zeros(E), "model": np.zeros(E)}
            for e in range(E):
                gt = read_video(os.path.join(root, names[e]))
                ks = [k for k in range(0, len(gt), F) if k >= FILL_FROM and k < lengths[e]]
                fill["gt"][e] = band_count([r224(gt[k]) for k in ks], SIZE).mean()
                fill["model"][e] = band_count([r224(frames[e, k]) for k in ks], SIZE).mean()
            for g, mem in groups.items():
                gm, mm = fill["gt"][mem], fill["model"][mem]
                res["groups"][g] = {"fill": {"gt_px": boot(gm), "model_px": boot(mm),
                                             "model_rel_err": boot(np.abs(mm - gm) / np.maximum(gm, 1.0)),
                                             "model_signed_rel_err": boot((mm - gm) / np.maximum(gm, 1.0)),
                                             "corr_model_gt": float(np.corrcoef(mm, gm)[0, 1]) if len(mem) > 2 and mm.std() > 0 else None}}
            res["per_episode"] = {"names": names, "fill_gt": fill["gt"].tolist(), "fill_model": fill["model"].tolist()}
            result["sets"][split] = res
            continue
        acts = [meta_by[n]["actions"] for n in names]
        k_full = np.array([min(len(a) // F, (min(int(L), len(a) + 1) - 1) // F) + 1 for a, L in zip(acts, lengths)])
        Kmax = int(k_full.max())
        objs = None
        Q, IOU = None, None
        for e in range(E):
            gt = read_video(os.path.join(root, names[e]))
            for k in range(k_full[e]):
                sg, mg = measure(qenv, r224(gt[F * k]))
                sm, mm = measure(qenv, r224(frames[e, F * k]))
                if objs is None:
                    objs = list(sg)
                    Q = {s: {o: np.full((E, Kmax, 3), np.nan) for o in objs} for s in ("gt", "model")}
                    IOU = {a: np.full((E, Kmax), np.nan) for a in AREAS[qenv]}
                for o in objs:
                    Q["gt"][o][e, k], Q["model"][o][e, k] = sg[o], sm[o]
                for a in AREAS[qenv]:
                    union = (mm[a] | mg[a]).sum()
                    IOU[a][e, k] = (mm[a] & mg[a]).sum() / union if union >= MIN_PX else np.nan
        last = k_full - 1
        pred = np.zeros((E, Kmax), bool)
        for e in range(E):
            pred[e, 1 : k_full[e]] = True
        if qenv == "acwm_stack_cube":
            success = np.array([bool(meta_by[n]["success"]) for n in names])
            flag = {s: np.array([stacked(Q[s], e, last[e]) for e in range(E)]) for s in ("gt", "model")}
        for g, mem in groups.items():
            r = {}
            ar = np.arange(len(mem))
            for o in objs:
                d = {}
                px = {s: Q[s][o][mem][:, :, 0] for s in ("gt", "model")}
                if o in AREAS[qenv]:
                    gl, ml = px["gt"][ar, last[mem]], px["model"][ar, last[mem]]
                    rel = (px["model"] - px["gt"]) / np.maximum(px["gt"], 1.0)
                    d["px_curve"] = {s: np.nanmean(px[s], 0).tolist() for s in px}
                    d["gt_px_last"] = boot(gl)
                    d["model"] = {"px_last": boot(ml), "signed_rel_err_last": boot((ml - gl) / np.maximum(gl, 1.0)),
                                  "abs_rel_err_last": boot(np.abs(ml - gl) / np.maximum(gl, 1.0)),
                                  "abs_rel_err_steps": boot(np.nanmean(np.where(pred[mem], np.abs(rel), np.nan), 1)),
                                  "signed_rel_err_steps": boot(np.nanmean(np.where(pred[mem], rel, np.nan), 1)),
                                  "iou_steps": boot(np.nanmean(np.where(pred[mem], IOU[o][mem], np.nan), 1)),
                                  "iou_last": boot(IOU[o][mem][ar, last[mem]])}
                xy = {s: Q[s][o][mem][:, :, 1:] for s in ("gt", "model")}
                err = np.linalg.norm(xy["model"] - xy["gt"], axis=2)
                d["center"] = {"model": {"err_px_steps": boot(np.nanmean(np.where(pred[mem], err, np.nan), 1)), "err_px_last": boot(err[ar, last[mem]])}}
                r[o] = d
            if qenv == "acwm_stack_cube":
                a, b = flag["model"][mem], flag["gt"][mem]
                r["stacked_last_frame"] = {"success_rate_label": float(success[mem].mean()),
                                           "gt": {"stacked_rate": float(b.mean()), "agree_with_label": float((b == success[mem]).mean())},
                                           "model": {"stacked_rate": float(a.mean()), "agree_with_label": float((a == success[mem]).mean()),
                                                     "agree_with_gt_frame": float((a == b).mean()), "both": int((a & b).sum()),
                                                     "only_model": int((a & ~b).sum()), "only_gt": int((~a & b).sum()), "neither": int((~a & ~b).sum())}}
            res["groups"][g] = r
        res["k_full"] = k_full.tolist()
        result["sets"][split] = res
    json.dump(result, open(os.path.join(chain_dir, f"{env}_amount.json"), "w"), indent=1)
    return result


def show(r):
    print(f"== {r['env']}")
    for split, s in r["sets"].items():
        for g, d in s["groups"].items():
            for o, v in d.items():
                if o == "fill":
                    print(f"  {split}/{g} fill: gt {v['gt_px'][0]:.0f}  model {v['model_px'][0]:.0f}  signed rel err {v['model_signed_rel_err'][0]:+.3f}")
                elif o == "stacked_last_frame":
                    print(f"  {split}/{g} stacked: label {v['success_rate_label']:.3f}  gt frame {v['gt']['stacked_rate']:.3f}  model {v['model']['stacked_rate']:.3f}  agree {v['model']['agree_with_gt_frame']:.3f}  only_model {v['model']['only_model']}  only_gt {v['model']['only_gt']}")
                elif "model" in v:
                    m = v["model"]
                    print(f"  {split}/{g} {o}: px last gt {v['gt_px_last'][0]:.0f}  model {m['px_last'][0]:.0f}  signed rel err {m['signed_rel_err_last'][0]:+.3f}  IoU steps {m['iou_steps'][0]:.3f}")


if __name__ == "__main__":
    for env in sys.argv[2:]:
        show(run(sys.argv[1], env))
