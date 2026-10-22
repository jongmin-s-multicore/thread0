"""InD 만 학습한 DINO-WM 과 InD+OoD 섞어 학습한 DINO-WM 의 rollout 을 에피소드마다, 모델 스텝(5 프레임)마다 비교한다.

평가 대상은 acwm_metrics_mix.py 와 같다: InD test 전부, OoD test 중 학습에 넣지 않은 에피소드(<split.json> 의 held_out).
두 모델은 첫 실제 프레임과 action 으로 에피소드 끝까지 예측한다 (acwm_metrics_mix.py 의 rollout). 저장하는 값 (모델 스텝 k 마다):
  mse, mmse   실제 프레임과의 픽셀 오차 (metric 크기, [0, 1]). k 에 대한 평균이 acwm_metrics_mix.py 의 rollout MSE, M-MSE 와 같다.
  dmse        두 모델이 그린 프레임 사이의 MSE (두 rollout 이 서로 얼마나 다른가)
  amount      물체의 양 (cloth, sand: 픽셀 수, water: 상자 안 물 픽셀과 전체 물 픽셀, stack: 쌓임 판정). Real, Encoder->Decoder, 두 모델.
  iou         cloth, sand: 실제 프레임의 물체 마스크와 겹치는 정도
두 모델은 encoder 와 decoder 가 같다 (predictor 만 다르다). recon 의 차이를 같이 적어 확인한다.
모델과 split.json 의 위치는 acwm_metrics_mix.py 의 MIX, SPLIT 과 같다.

usage (cwd = $DINO_WM, env sourced):
  python <this> {cloth,sand,stack,water} <out.npz> [--max_eps N]
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

WORK = os.environ["DINO_WORK"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import talk_render as TR
import acwm_metrics_mix as AM
import eval_pour_water as EPW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key", choices=list(AM.MIX)); ap.add_argument("out")
    ap.add_argument("--max_eps", type=int, default=0)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    key, dev, t_start = args.key, "cuda", time.time()
    spec = TR.SPEC[key]
    cfg = OmegaConf.load(TR.CKPT / spec["base"][0] / "hydra.yaml")
    H, f = cfg.num_hist, cfg.frameskip
    hw = (240, 240) if key == "water" else AM.EAP.OBS_HW[cfg.env.name]
    held = json.load(open(os.path.join(WORK, AM.SPLIT[key])))
    keep = {n for v in held["held_out"].values() for n in v}
    sets = {}
    for sp, (d, groups, tf) in TR.load_sets(key, cfg).items():
        if sp == held["extra_split"]:
            groups = {g: [int(e) for e in m if d.video_names[e] in keep] for g, m in groups.items()}
            if "all" not in groups:
                groups["all"] = sorted({e for m in groups.values() for e in m})
        if args.max_eps:
            groups = {g: list(m)[: args.max_eps] for g, m in groups.items()}
        sets[sp] = (d, {g: [int(x) for x in m] for g, m in groups.items() if len(m)}, tf)
    models = {"ind_only": TR.load(*spec["base"], cfg, dev), "ind_ood": TR.load(*AM.MIX[key], cfg, dev)}
    print({"ind_only": spec["base"], "ind_ood": AM.MIX[key]}, flush=True)
    up = lambda p: F.interpolate((p.clamp(-1, 1) + 1) / 2, size=hw, mode="bilinear", align_corners=False).permute(0, 2, 3, 1).cpu().numpy()

    out = {}
    recon_gap = 0.0
    for sp, (d, groups, tf) in sets.items():
        eps = sorted({e for m in groups.values() for e in m})
        rows = []
        for n_, e in enumerate(eps):
            vis, vis_m, act, ep = AM.episode_frames(key, d, tf, e, hw)
            kf = 300 // f if key == "water" else min(len(act) // f, (ep.n_obs - 1) // f) + 1
            fo = vis[: f * (kf - 1) + 1 : f][None]
            fa = act[: f * (kf - 1)].reshape(1, kf - 1, -1)
            gt_all = vis_m[f * np.arange(kf)]
            gt, w = gt_all[H:], 0.01 + np.abs(gt_all - gt_all[0][None]).max(axis=(0, 3))  # ACWM motion mask over the whole episode
            dec, recs = {}, {}
            for name, m in models.items():
                p, rec = EPW.rollout(m, fo, fa, H, dev)[:2]
                dec[name], recs[name] = p[0], rec[0]
            recon_gap = max(recon_gap, float((recs["ind_only"] - recs["ind_ood"]).abs().max()))
            r = {"kf": kf}
            pm = {name: up(dec[name][H:]) for name in models}
            for name in models:
                err = (pm[name] - gt) ** 2
                r[f"mse/{name}"] = err.mean((1, 2, 3))
                r[f"mmse/{name}"] = (w[None, :, :, None] * err).sum((1, 2, 3)) / (w.sum() * 3)
            r["dmse"] = ((pm["ind_only"] - pm["ind_ood"]) ** 2).mean((1, 2, 3))
            fr = {"real": EPW.to_u8(fo[0]), "recon": EPW.to_u8(recs["ind_only"])}
            fr.update({name: EPW.to_u8(dec[name]) for name in models})
            for q, kinds in TR.measures(key, fr).items():
                for kd, v in kinds.items():
                    r[f"{q}/{kd}"] = np.asarray(v, float)
            rows.append(r)
            if n_ % 25 == 0:
                print(f"[{key} {sp}] {n_}/{len(eps)} {time.time() - t_start:.0f}s", flush=True)
        K = max(r["kf"] for r in rows)
        pad = lambda v, lead: np.concatenate([np.full(lead, np.nan), v, np.full(K - lead - len(v), np.nan)])
        for k in rows[0]:
            if k == "kf":
                out[f"{sp}/kf"] = np.array([r["kf"] for r in rows])
            else:  # pixel errors start at step H (the first H frames are given); amounts and IoU cover all steps
                lead = H if k.split("/")[0] in ("mse", "mmse", "dmse") else 0
                out[f"{sp}/{k}"] = np.stack([pad(r[k], lead) if lead else np.concatenate([r[k], np.full(K - len(r[k]), np.nan)]) for r in rows])
        out[f"{sp}/names"] = np.array([d.video_names[e] for e in eps])
        for g, mem in groups.items():
            out[f"{sp}/group/{g}"] = np.array([e in set(mem) for e in eps])
    out["meta"] = np.array(json.dumps({"key": key, "models": {"ind_only": list(spec["base"]), "ind_ood": list(AM.MIX[key])}, "split": AM.SPLIT[key], "H": int(H),
                                       "frameskip": int(f), "hw": list(hw), "max_abs_recon_difference_between_models": recon_gap,
                                       "elapsed_s": round(time.time() - t_start)}))
    np.savez_compressed(args.out, **out)
    print("max |recon(ind_only) - recon(ind_ood)| =", recon_gap)
    print("wrote", args.out, f"{time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
