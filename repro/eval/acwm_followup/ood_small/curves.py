"""Cloth Move: 작은 천 + 중간 천(InD) 만 학습한 모델이 큰 천을 예측하는지 본다. 모델 네 개를 같은 에피소드에서 rollout 으로 비교한다.

  orig   train_cloth_h1f5 ep15            InD 만 (원래 모델, train.py, fp32)
  ctrl   train_cloth_h1f5_indctl ep15     InD 만 (predictor_long.py, TF32 + bf16)  같은 코드·정밀도의 대조군
  small  train_cloth_h1f5_oodsmall ep15   InD + OoD smaller 25 (짝수 번호). larger 는 학습에 없다
  prev   train_cloth_h1f5_oodmix ep15     InD + OoD smaller 25 + larger 25 (짝수 번호). 이전 실행

모델마다 predictor 만 다르고 encoder 와 decoder 는 같다. 첫 실제 프레임과 action 으로 에피소드 끝까지 예측하고 (cloth_control.py 와 같은 방법),
모델 스텝(5 프레임)마다 천 픽셀 수, 실제 마스크와의 IoU, 픽셀 MSE, 움직임 마스크 MSE 를 저장한다.
retention r(k) 용으로 ahead1 (프레임 t 를 한 step 앞의 실제 프레임에서 예측), ahead5 (다섯 step = 25 프레임 앞의 실제 프레임에서 예측) 의 천 픽셀 수도 저장한다
(retention r(k) 의 정의는 analyze.py 에 있다).
InD test 100 개와 OoD test 100 개(smaller 50, larger 50) 전부를 돌린다. 어느 모델에게 학습에 없던 에피소드인지는 analyze.py 가 가른다.

usage (cwd = 레포 루트, env sourced):
  CUDA_VISIBLE_DEVICES=0 python repro/eval/acwm_followup/ood_small/curves.py <out.npz> [--models orig,ctrl,small,prev] [--max_eps N] [--device cuda]
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repro/eval/acwm_followup: talk_render.py, acwm_metrics_mix.py
import talk_render as TR
import acwm_metrics_mix as AM
import eval_pour_water as EPW

MODELS = {"orig": ("train_cloth_h1f5", 15), "ctrl": ("train_cloth_h1f5_indctl", 15),
          "small": ("train_cloth_h1f5_oodsmall", 15), "prev": ("train_cloth_h1f5_oodmix", 15)}


def px_of(x):
    """cloth pixels of decoded frames (n, 3, h, w) in [-1, 1]"""
    return np.array([TR.measure(TR.SPEC["cloth"]["env"], im)[0]["cloth"][0] for im in EPW.to_u8(x)], float)


@torch.no_grad()
def ahead_frames(model, vis, fa, H, dev, n):
    """frame t predicted from the real frames n model steps earlier (H frames of history). -> index of the first predicted frame, decoded frames (nw, 3, h, w)"""
    nw = len(vis) - H - n + 1
    io = torch.arange(H)[None] + torch.arange(nw)[:, None]
    ia = torch.arange(H + n - 1)[None] + torch.arange(nw)[:, None]
    zk, _ = model.rollout({"visual": vis[io].to(dev), "proprio": torch.zeros(nw, H, 1, device=dev)}, fa[0][ia].to(dev))
    return H + n - 1, model.decode_obs({m: v[:, -1:] for m, v in zk.items()})[0]["visual"][:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--max_eps", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    key, dev, t_start = "cloth", args.device, time.time()
    spec = TR.SPEC[key]
    cfg = OmegaConf.load(TR.CKPT / spec["base"][0] / "hydra.yaml")
    H, f = cfg.num_hist, cfg.frameskip
    hw = AM.EAP.OBS_HW[cfg.env.name]
    sets = {}
    for sp, (d, groups, tf) in TR.load_sets(key, cfg).items():
        if args.max_eps:
            groups = {g: list(m)[: args.max_eps] for g, m in groups.items()}
        sets[sp] = (d, groups, tf)
    models = {n: TR.load(*MODELS[n], cfg, dev) for n in args.models.split(",")}
    print({n: MODELS[n] for n in models}, flush=True)
    up = lambda p: F.interpolate((p.clamp(-1, 1) + 1) / 2, size=hw, mode="bilinear", align_corners=False).permute(0, 2, 3, 1).cpu().numpy()

    out = {}
    recon_gap = 0.0
    for sp, (d, groups, tf) in sets.items():
        eps = sorted({int(e) for m in groups.values() for e in m})
        rows = []
        for n_, e in enumerate(eps):
            vis, vis_m, act, ep = AM.episode_frames(key, d, tf, e, hw)
            kf = min(len(act) // f, (ep.n_obs - 1) // f) + 1
            fo = vis[: f * (kf - 1) + 1 : f][None]
            fa = act[: f * (kf - 1)].reshape(1, kf - 1, -1)
            gt_all = vis_m[f * np.arange(kf)]
            gt, w = gt_all[H:], 0.01 + np.abs(gt_all - gt_all[0][None]).max(axis=(0, 3))  # ACWM motion mask over the whole episode
            dec, rec0 = {}, None
            for name, m in models.items():
                p, rec = EPW.rollout(m, fo, fa, H, dev)[:2]
                dec[name] = p[0]
                if rec0 is None:
                    rec0 = rec[0]
                recon_gap = max(recon_gap, float((rec[0] - rec0).abs().max()))
            pm = {name: up(dec[name][H:]) for name in models}
            r = {"kf": kf}
            for name in models:
                err = (pm[name] - gt) ** 2
                r[f"mse/{name}"] = err.mean((1, 2, 3))
                r[f"mmse/{name}"] = (w[None, :, :, None] * err).sum((1, 2, 3)) / (w.sum() * 3)
            fr = {"real": EPW.to_u8(fo[0]), "recon": EPW.to_u8(rec0)}
            fr.update({name: EPW.to_u8(dec[name]) for name in models})
            for q, kinds in TR.measures(key, fr).items():
                for kd, v in kinds.items():
                    r[f"{q}/{kd}"] = np.asarray(v, float)
            for n in (1, 5):
                for name, m in models.items():
                    s0, x = ahead_frames(m, fo[0], torch.as_tensor(fa), H, dev, n)
                    r[f"ahead{n}/{name}"] = np.concatenate([np.full(s0, np.nan), px_of(x)])
            rows.append(r)
            if n_ % 25 == 0:
                print(f"[{key} {sp}] {n_}/{len(eps)} {time.time() - t_start:.0f}s", flush=True)
        K = max(r["kf"] for r in rows)
        for k in rows[0]:
            if k == "kf":
                out[f"{sp}/kf"] = np.array([r["kf"] for r in rows])
                continue
            lead = H if k.split("/")[0] in ("mse", "mmse") else 0  # pixel errors start at step H (the first H frames are given); ahead<n> carry their own NaN prefix
            out[f"{sp}/{k}"] = np.stack([np.concatenate([np.full(lead, np.nan), r[k], np.full(K - lead - len(r[k]), np.nan)]) for r in rows])
        out[f"{sp}/names"] = np.array([d.video_names[e] for e in eps])
    out["meta"] = np.array(json.dumps({"models": {n: list(MODELS[n]) for n in models}, "H": int(H), "frameskip": int(f), "hw": list(hw),
                                       "max_abs_recon_difference_between_models": recon_gap, "elapsed_s": round(time.time() - t_start)}))
    np.savez_compressed(args.out, **out)
    print("max |recon(model) - recon(first model)| =", recon_gap)
    print("wrote", args.out, f"{time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
