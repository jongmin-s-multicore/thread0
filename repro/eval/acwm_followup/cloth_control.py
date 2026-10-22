"""Cloth Move 대조군: InD 만 학습한 두 번째 모델과 원래 모델의 rollout 이 얼마나 다른가.

두 번째 모델은 pred-param 실험(브랜치 hanbin5/pred-param, 아직 push 하지 않음)의 'absolute' 재학습이다. 같은 구조, 같은 InD 데이터로 처음부터
15 epoch 학습했고 출력 방식은 원래와 같다. 이 브랜치에는 그 체크포인트의 로더(repro/eval/pred_variants_eval.py, models/pred_variants.py)가 없어서
이 스크립트는 그 파일들이 있는 체크아웃에서만 돈다.
encoder 와 decoder 는 원래 모델의 것을 쓰고 predictor, action encoder, proprio encoder 만 바꾼다 (pred_variants_eval.py 와 같은 방식).
rollout_curves.py 와 같은 에피소드(InD test 전부, OoD test 중 학습에 넣지 않은 것), 같은 계산으로 세 모델을 돌린다:
  ind_only  원래 모델 (train_cloth_h1f5 epoch 15)
  ind_ood   InD+OoD 섞어 학습 (train_cloth_h1f5_oodmix epoch 15)
  control   InD 만 다시 학습 (<control ckpt>, variant abs, epoch 15)
저장: mse, mmse, px, iou (모델마다), 모델 쌍마다 dmse.

usage (cwd = $DINO_WM, env sourced):
  python <this> <control ckpt, train_cloth_h1f5_abs_ep15.pt> <out.npz> [--max_eps N]
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

try:
    import pred_variants_eval as PV  # branch hanbin5/pred-param
except ImportError:
    sys.exit("cloth_control.py needs repro/eval/pred_variants_eval.py and models/pred_variants.py of branch hanbin5/pred-param (they load the second InD-only predictor)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("control"); ap.add_argument("out"); ap.add_argument("--max_eps", type=int, default=0)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    key, dev, t_start = "cloth", "cuda", time.time()
    spec = TR.SPEC[key]
    cfg = OmegaConf.load(TR.CKPT / spec["base"][0] / "hydra.yaml")
    H, f = cfg.num_hist, cfg.frameskip
    hw = AM.EAP.OBS_HW[cfg.env.name]
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
    A = sets["ind_test"][0].action_dim * f
    models = {"ind_only": TR.load(*spec["base"], cfg, dev), "ind_ood": TR.load(*AM.MIX[key], cfg, dev)}
    mc = TR.load(*spec["base"], cfg, dev)
    V = PV.load_variant(args.control, cfg, (mc.encoder.emb_dim, A), dev)
    assert V["variant"] == "abs", V["variant"]
    mc.predictor, mc.action_encoder, mc.proprio_encoder = V["predictor"], V["action_encoder"], V["proprio_encoder"]
    models["control"] = mc
    print("control:", V["path"], "variant", V["variant"], "epoch", V["epoch"], flush=True)
    up = lambda p: F.interpolate((p.clamp(-1, 1) + 1) / 2, size=hw, mode="bilinear", align_corners=False).permute(0, 2, 3, 1).cpu().numpy()
    pairs = [("ind_only", "ind_ood"), ("ind_only", "control"), ("ind_ood", "control")]

    out = {}
    for sp, (d, groups, tf) in sets.items():
        eps = sorted({e for m in groups.values() for e in m})
        rows = []
        for n_, e in enumerate(eps):
            vis, vis_m, act, ep = AM.episode_frames(key, d, tf, e, hw)
            kf = min(len(act) // f, (ep.n_obs - 1) // f) + 1
            fo = vis[: f * (kf - 1) + 1 : f][None]
            fa = act[: f * (kf - 1)].reshape(1, kf - 1, -1)
            gt_all = vis_m[f * np.arange(kf)]
            gt, w = gt_all[H:], 0.01 + np.abs(gt_all - gt_all[0][None]).max(axis=(0, 3))
            dec, rec0 = {}, None
            for name, m in models.items():
                p, rec = EPW.rollout(m, fo, fa, H, dev)[:2]
                dec[name] = p[0]
                rec0 = rec[0] if rec0 is None else rec0
            pm = {name: up(dec[name][H:]) for name in models}
            r = {"kf": kf}
            for name in models:
                err = (pm[name] - gt) ** 2
                r[f"mse/{name}"] = err.mean((1, 2, 3))
                r[f"mmse/{name}"] = (w[None, :, :, None] * err).sum((1, 2, 3)) / (w.sum() * 3)
            for a, b in pairs:
                r[f"dmse/{a}__{b}"] = ((pm[a] - pm[b]) ** 2).mean((1, 2, 3))
            fr = {"real": EPW.to_u8(fo[0]), "recon": EPW.to_u8(rec0)}
            fr.update({name: EPW.to_u8(dec[name]) for name in models})
            for q, kinds in TR.measures(key, fr).items():
                for kd, v in kinds.items():
                    r[f"{q}/{kd}"] = np.asarray(v, float)
            rows.append(r)
            if n_ % 25 == 0:
                print(f"[{key} {sp}] {n_}/{len(eps)} {time.time() - t_start:.0f}s", flush=True)
        K = max(r["kf"] for r in rows)
        for k in rows[0]:
            if k == "kf":
                out[f"{sp}/kf"] = np.array([r["kf"] for r in rows])
                continue
            lead = H if k.split("/")[0] in ("mse", "mmse", "dmse") else 0
            out[f"{sp}/{k}"] = np.stack([np.concatenate([np.full(lead, np.nan), r[k], np.full(K - lead - len(r[k]), np.nan)]) for r in rows])
        out[f"{sp}/names"] = np.array([d.video_names[e] for e in eps])
        for g, mem in groups.items():
            out[f"{sp}/group/{g}"] = np.array([e in set(mem) for e in eps])
    out["meta"] = np.array(json.dumps({"models": {"ind_only": list(spec["base"]), "ind_ood": list(AM.MIX[key]), "control": os.path.basename(V["path"])}, "H": int(H),
                                       "frameskip": int(f), "elapsed_s": round(time.time() - t_start)}))
    np.savez_compressed(args.out, **out)
    print("wrote", args.out, f"{time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
