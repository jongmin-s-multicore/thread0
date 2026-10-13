"""ACWM-Phys Cloth Move / Push Sand / Stack Cube 의 물리량 비교. eval_acwm_phys.py 가 저장한 전체 에피소드 롤아웃 프레임
(<prefix>_<set>.npz: 모델 예측, 정답을 인코더·디코더에 통과시킨 재구성)과 정답 프레임을 같은 색 마스크로 재서 비교한다. GPU 를 쓰지 않는다.

  Cloth Move  천(주황 + 뒤집힌 면 분홍) 픽셀 수·마스크 IoU·중심, 그리퍼 공 두 개(파랑, 빨강)의 중심
  Push Sand   모래(하늘색) 픽셀 수·IoU·중심, 밀판(가장 큰 어두운 덩어리)의 중심
  Stack Cube  노란·초록 큐브의 픽셀 수·IoU·중심, 마지막 프레임의 "쌓임" 판정(노란 큐브가 초록 큐브 바로 위)

재구성(recon)에서 잰 값이 정답과 다르면 그 양은 디코더가 못 그리는 것이라 모델 비교에 쓰지 않는다 (JSON 의 recon 항목이 그 기준).

usage (cwd = $DINO_WM, env sourced):
  python repro/eval/eval_acwm_phys_quant.py <ckpt_base_path> <model_name> <out_prefix> [--sets val,ind_test,ood_test]
  -> <out_prefix>_quant.json, <out_prefix>_quant_<set>.npz (정답 프레임 포함)
"""
import os, sys, json, argparse
import numpy as np
import torch
import hydra
import cv2
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
from datasets.acwm_phys_dset import load_acwm_phys_split
from datasets.traj_dset import TrajSubset

MIN_PX = 6  # an object with fewer pixels (224x224) counts as not visible
STACK = dict(dx=11.0, dy=(-35.0, -6.5))  # yellow centre relative to green, 224 px (|dx| < 25, -80 < dy < -15 at 512 px)


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
    """rgb (h, w, 3) uint8 -> {object: (px, x, y)}, {mask name: bool array}"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    if env == "acwm_cloth_move":
        cloth = hue(hsv, 8, 16, 120) | hue(hsv, 158, 170, 60)
        return {"cloth": blob(cloth), "blue": blob(hue(hsv, 105, 113, 100)), "red": blob(hue(hsv, 175, 4, 90))}, {"cloth": cloth}
    if env == "acwm_push_sand":
        sand = hue(hsv, 90, 102, 100)
        return {"sand": blob(sand), "board": largest_dark(hsv)}, {"sand": sand}
    yellow, green = hue(hsv, 26, 33, 180, 130), hue(hsv, 55, 66, 150, 60)
    return {"yellow": blob(yellow), "green": blob(green)}, {"yellow": yellow, "green": green}


AREAS = {"acwm_cloth_move": ["cloth"], "acwm_push_sand": ["sand"], "acwm_stack_cube": ["yellow", "green"]}  # objects with a mask (IoU, pixel count)


def boot(vals, n=2000, seed_=0):
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [None, None, None]
    idx = np.random.RandomState(seed_).randint(0, len(x), (n, len(x)))
    b = x[idx].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def stacked(q, e, k):
    """yellow cube right above the green cube in frame k of episode e (image rule)"""
    if not (q["yellow"][e, k, 0] >= 15 and q["green"][e, k, 0] >= 15):
        return False
    dx, dy = q["yellow"][e, k, 1] - q["green"][e, k, 1], q["yellow"][e, k, 2] - q["green"][e, k, 2]
    return bool(abs(dx) < STACK["dx"] and STACK["dy"][0] < dy < STACK["dy"][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_base"); ap.add_argument("model_name"); ap.add_argument("out_prefix")
    ap.add_argument("--sets", default="val,ind_test,ood_test")
    args = ap.parse_args()
    cfg = OmegaConf.load(Path(args.ckpt_base) / "outputs" / args.model_name / "hydra.yaml")
    env, f, H = cfg.env.name, cfg.frameskip, cfg.num_hist
    ds_cfg = cfg.env.dataset
    ds_kw = dict(n_rollout=ds_cfg.n_rollout, exclude=list(ds_cfg.exclude), normalize_action=cfg.normalize_action,
                 action_dims=None if ds_cfg.action_dims is None else list(ds_cfg.action_dims), action_format=ds_cfg.action_format)
    tf = hydra.utils.instantiate(ds_cfg.transform)
    result = {"model": args.model_name, "env": env, "frameskip": f, "num_hist": H, "min_px": MIN_PX, "sets": {}}
    for name in args.sets.split(","):
        z = np.load(f"{args.out_prefix}_{name}.npz")
        names, k_full = [str(n) for n in z["names"]], z["k_full"]
        if name == "val":
            _, trajs = hydra.utils.call(ds_cfg, num_hist=H, num_pred=cfg.num_pred, frameskip=f)
            S = trajs["valid"]
        else:
            S = load_acwm_phys_split(tf, ds_cfg.data_path, name, **ds_kw)
        base, idxs = (S.dataset, list(S.indices)) if isinstance(S, TrajSubset) else (S, list(range(len(S))))
        idxs = idxs[: len(names)]
        assert [base.video_names[i] for i in idxs] == names, "episode order differs from the evaluation"
        E, Kmax = len(names), int(k_full.max())
        src = {"gt": np.zeros_like(z["frames_model"]), "model": z["frames_model"], "recon": z["frames_recon"]}
        for e in range(E):
            vis = base.get_frames(idxs[e], range(0, f * (k_full[e] - 1) + 1, f))[0]["visual"]
            src["gt"][e, : k_full[e]] = ((vis.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()
        objs = list(measure(env, src["gt"][0, 0])[0])
        Q = {s: {o: np.full((E, Kmax, 3), np.nan) for o in objs} for s in src}  # px, x, y
        IOU = {s: {a: np.full((E, Kmax), np.nan) for a in AREAS[env]} for s in ("model", "recon")}
        for e in range(E):
            for k in range(k_full[e]):
                sc_gt, mk_gt = measure(env, src["gt"][e, k])
                for o in objs:
                    Q["gt"][o][e, k] = sc_gt[o]
                for s in ("model", "recon"):
                    sc, mk = measure(env, src[s][e, k])
                    for o in objs:
                        Q[s][o][e, k] = sc[o]
                    for a in AREAS[env]:
                        union = (mk[a] | mk_gt[a]).sum()
                        IOU[s][a][e, k] = (mk[a] & mk_gt[a]).sum() / union if union >= MIN_PX else np.nan

        last = k_full - 1
        pred = np.zeros((E, Kmax), bool)  # predicted steps
        for e in range(E):
            pred[e, H : k_full[e]] = True
        groups = {k[6:]: z[k] for k in z.files if k.startswith("group_")}
        res = {"n_episodes": E, "group_sizes": {g: int(len(v)) for g, v in groups.items()}, "objects": objs, "groups": {}}
        if env == "acwm_stack_cube":
            meta = {Path(m["video_path"]).name: m for m in torch.load(base.split_path / "metadata.pt", map_location="cpu", weights_only=True)}
            success = np.array([bool(meta[n]["success"]) for n in names])
            flag = {s: np.array([stacked(Q[s], e, last[e]) for e in range(E)]) for s in src}
        for g, mem in groups.items():
            r = {}
            if len(mem) == 0:
                res["groups"][g] = r
                continue
            ar = np.arange(len(mem))
            for o in objs:
                d = {}
                px = {s: Q[s][o][mem][:, :, 0] for s in src}
                if o in AREAS[env]:
                    d["px_curve"] = {s: np.nanmean(np.where(np.isfinite(px[s]), px[s], np.nan), 0).tolist() for s in src}
                    d["px_first"] = boot(px["gt"][:, 0])
                    for s in ("model", "recon"):
                        gl, ml = px["gt"][ar, last[mem]], px[s][ar, last[mem]]
                        rel = (px[s] - px["gt"]) / np.maximum(px["gt"], 1.0)
                        d[s] = {"px_last": boot(ml), "signed_rel_err_last": boot((ml - gl) / np.maximum(gl, 1.0)), "abs_rel_err_last": boot(np.abs(ml - gl) / np.maximum(gl, 1.0)),
                                "abs_rel_err_steps": boot(np.nanmean(np.where(pred[mem], np.abs(rel), np.nan), 1)),
                                "signed_rel_err_steps": boot(np.nanmean(np.where(pred[mem], rel, np.nan), 1)),
                                "corr_last": float(np.corrcoef(ml, gl)[0, 1]) if len(mem) > 2 and np.std(ml) > 0 and np.std(gl) > 0 else None,
                                "iou_steps": boot(np.nanmean(np.where(pred[mem], IOU[s][o][mem], np.nan), 1)), "iou_last": boot(IOU[s][o][mem][ar, last[mem]]),
                                "iou_curve": np.nanmean(IOU[s][o][mem], 0).tolist()}
                    d["gt_px_last"] = boot(px["gt"][ar, last[mem]])
                # centre position: error against the ground truth where both are visible, and the same for "nothing moved" (frame 0)
                xy = {s: Q[s][o][mem][:, :, 1:] for s in src}
                still = np.linalg.norm(xy["gt"] - xy["gt"][:, :1], axis=2)
                d["center"] = {"moved_px_steps": boot(np.nanmean(np.where(pred[mem], still, np.nan), 1)), "moved_px_last": boot(still[ar, last[mem]])}
                for s in ("model", "recon"):
                    err = np.linalg.norm(xy[s] - xy["gt"], axis=2)
                    gt_vis, s_vis = np.isfinite(xy["gt"][:, :, 0]) & pred[mem], np.isfinite(xy[s][:, :, 0]) & pred[mem]
                    d["center"][s] = {"err_px_steps": boot(np.nanmean(np.where(pred[mem], err, np.nan), 1)), "err_px_last": boot(err[ar, last[mem]]),
                                      "err_px_curve": np.nanmean(err, 0).tolist(),
                                      "missing_when_gt_visible": float((gt_vis & ~np.isfinite(xy[s][:, :, 0])).sum() / max(gt_vis.sum(), 1)),
                                      "present_when_gt_absent": float((s_vis & ~np.isfinite(xy["gt"][:, :, 0])).sum() / max((pred[mem] & ~np.isfinite(xy["gt"][:, :, 0])).sum(), 1))}
                r[o] = d
            if env == "acwm_stack_cube":
                st = {"success_rate_label": float(success[mem].mean())}
                for s in src:
                    st[s] = {"stacked_rate": float(flag[s][mem].mean()), "agree_with_label": float((flag[s][mem] == success[mem]).mean())}
                    if s != "gt":
                        a, b = flag[s][mem], flag["gt"][mem]
                        st[s].update({"agree_with_gt_frame": float((a == b).mean()), "both": int((a & b).sum()), "only_" + s: int((a & ~b).sum()),
                                      "only_gt": int((~a & b).sum()), "neither": int((~a & ~b).sum())})
                r["stacked_last_frame"] = st
            res["groups"][g] = r
        result["sets"][name] = res
        np.savez_compressed(f"{args.out_prefix}_quant_{name}.npz", names=np.array(names), k_full=k_full, frames_gt=src["gt"],
                            **{f"{s}_{o}": Q[s][o] for s in src for o in objs}, **{f"iou_{s}_{a}": IOU[s][a] for s in IOU for a in AREAS[env]},
                            **{f"group_{g}": v for g, v in groups.items()})
        print(f"[{name}] {E} episodes", flush=True)
        for g, r in res["groups"].items():
            for o in objs:
                if o not in r:
                    continue
                d = r[o]
                line = f"  {name}/{g} {o}:"
                if o in AREAS[env]:
                    line += f" px last gt {d['gt_px_last'][0]:.0f} model {d['model']['px_last'][0]:.0f} recon {d['recon']['px_last'][0]:.0f} | IoU steps model {d['model']['iou_steps'][0]:.3f} recon {d['recon']['iou_steps'][0]:.3f} |"
                c = d["center"]
                fm = lambda v: "n/a" if v[0] is None else f"{v[0]:.1f}"
                line += f" centre err px (steps) model {fm(c['model']['err_px_steps'])} recon {fm(c['recon']['err_px_steps'])} moved {fm(c['moved_px_steps'])} | missing model {c['model']['missing_when_gt_visible']:.2f} recon {c['recon']['missing_when_gt_visible']:.2f}"
                print(line)
            if "stacked_last_frame" in r:
                print(f"  {name}/{g} stacked: {json.dumps(r['stacked_last_frame'])}")
    json.dump(result, open(f"{args.out_prefix}_quant.json", "w"), indent=1)
    print("wrote", f"{args.out_prefix}_quant.json")


if __name__ == "__main__":
    main()
