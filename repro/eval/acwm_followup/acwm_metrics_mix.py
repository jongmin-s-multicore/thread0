"""2번 실험 (InD + OoD 섞어 학습): InD 만 학습한 원래 모델과 OoD 를 섞어 학습한 모델의 픽셀 오차 (ACWM-Phys 지표 MSE, M-MSE, PSNR, SSIM).
OoD test 는 학습에 넣지 않은 에피소드(<split.json> 의 held_out)만, InD test 는 전부. 두 모델을 같은 에피소드, 같은 창에서 잰다.

  window  원래 평가(eval_acwm_phys.py, eval_pour_water.py)의 "acwm" 블록과 같다. 에피소드마다 37 프레임 창 10개, 창의 첫 프레임과
          action 으로 5 프레임마다 예측해 프레임 5..35 (Pour Water 15..35) 를 잰다. 움직임 마스크는 창의 프레임들로 만든다.
          창 평균 -> 에피소드 평균.
  rollout 첫 프레임과 에피소드의 action 으로 끝까지 예측하고 프레임 5 부터 끝까지 5 프레임마다 잰다. 같은 지표,
          움직임 마스크는 에피소드 전체(5 프레임 간격)로 만든다.
지표 크기는 ACWM obs_shape (Cloth Move, Stack Cube, Pour Water 240x240, Push Sand 240x400), [0, 1].
InD only 모델의 window 값은 원래 평가 JSON 의 ind_test acwm 값과 같아야 한다 (확인용으로 같이 적는다).

usage (cwd = $DINO_WM, env sourced):
  python <this> {cloth,sand,stack,water} <out.json> [--max_eps N]
  -> <out.json>, <out>.npz (에피소드별 값)
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

WORK = os.environ["DINO_WORK"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import talk_render as TR
import eval_pour_water as EPW
import eval_acwm_phys as EAP

MIX = {"cloth": ("train_cloth_h1f5_oodmix", 15), "sand": ("train_sand_h1f5_oodmix", 18), "water": ("train_pw_h1f5_oodmix", 6),
       "stack": ("train_stack_h1f5_oodmix", 25)}
SPLIT = {"cloth": "runs/colab_acwm_long/ood/train_cloth_h1f5_oodmix_split.json", "sand": "runs/colab_acwm_ood2/sand/train_sand_h1f5_oodmix_split.json",
         "water": "runs/colab_acwm_ood2/pw/train_pw_h1f5_oodmix_split.json", "stack": "runs/colab_acwm_ood2/stack/train_stack_h1f5_oodmix_split.json"}
REF = {"cloth": "results/acwm_phys/cloth_h1f5.json", "sand": "results/acwm_phys/sand_h1f5.json", "stack": "results/acwm_phys/stack_h1f5.json",
       "water": "results/pour_water/h1f5_ep6.json"}
KS = ("mse", "masked_mse", "psnr", "ssim")


def episode_frames(key, d, tf, e, hw):
    """model frames (n, 3, 224, 224) in [-1, 1], metric frames (n, h, w, 3) in [0, 1], normalized actions (n - 1, a)"""
    if key == "water":
        ep = EPW.Episode(d, e, tf)
        return ep.vis, ep.vis240, ep.act, ep
    ep = EAP.Episode(d, e, tf, hw)
    return ep.vis, ep.vis_hw, ep.act, ep


def score(models, key, d, tf, e, H, f, hw, s0):
    vis, vis_m, act, ep = episode_frames(key, d, tf, e, hw)
    up = lambda p: F.interpolate((p.clamp(-1, 1) + 1) / 2, size=hw, mode="bilinear", align_corners=False).permute(0, 2, 3, 1).cpu().numpy()
    out = {}
    # windows
    K = (EAP.WINDOW - 1) // f + 1
    if key == "water":
        cuts = EPW.CUTS
        obs, acts = EPW.make_windows(ep, f, K)
    else:
        cuts = np.linspace(0, ep.n_obs - EAP.WINDOW, EAP.N_CUTS, dtype=int)
        obs, acts = EAP.make_windows(ep, f, K, cuts)
    preds = {name: EPW.rollout(m, obs, acts, H, "cuda")[0] for name, m in models.items()}
    for name in list(models) + ["copy_first"]:
        v = {k: [] for k in KS}
        for w_i, t0 in enumerate(cuts):
            gt_win = vis_m[t0 + f * np.arange(K)]
            gt = gt_win[H + s0 :]
            p = np.repeat(gt_win[:1], len(gt), 0) if name == "copy_first" else up(preds[name][w_i, H + s0 :])
            for k, x in EPW.acwm_metrics(p, gt, gt_win[0], gt_win).items():
                v[k].append(x)
        out[("window", name)] = {k: float(np.mean(x)) for k, x in v.items()}
    # whole-episode rollout from the first H frames
    if key == "water":
        kf = 300 // f
    else:
        kf = min(len(act) // f, (ep.n_obs - 1) // f) + 1
    fo = vis[: f * (kf - 1) + 1 : f][None]
    fa = act[: f * (kf - 1)].reshape(1, kf - 1, -1)
    fid = f * np.arange(kf)
    gt_all = vis_m[fid]
    for name, m in models.items():
        p = up(EPW.rollout(m, fo, fa, H, "cuda")[0][0, H:])
        out[("rollout", name)] = EPW.acwm_metrics(p, gt_all[H:], gt_all[0], gt_all)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key", choices=list(MIX)); ap.add_argument("out")
    ap.add_argument("--max_eps", type=int, default=0)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    t_start = time.time()
    spec = TR.SPEC[args.key]
    cfg = OmegaConf.load(TR.CKPT / spec["base"][0] / "hydra.yaml")
    H, f = cfg.num_hist, cfg.frameskip
    hw = (240, 240) if args.key == "water" else EAP.OBS_HW[cfg.env.name]
    s0 = max(H, EPW.COMMON_FROM) - H if args.key == "water" else 0
    held = json.load(open(os.path.join(WORK, SPLIT[args.key])))
    keep = {n for v in held["held_out"].values() for n in v}
    sets = {}
    for sp, (d, groups, tf) in TR.load_sets(args.key, cfg).items():
        if sp == held["extra_split"]:
            groups = {g: [int(e) for e in m if d.video_names[e] in keep] for g, m in groups.items()}
            if "all" not in groups:
                groups["all"] = sorted({e for m in groups.values() for e in m})
        if args.max_eps:
            groups = {g: list(m)[: args.max_eps] for g, m in groups.items()}
        sets[sp] = (d, {g: [int(x) for x in m] for g, m in groups.items() if len(m)}, tf)
    models = {"ind_only": TR.load(*spec["base"], cfg, "cuda"), "ind_ood": TR.load(*MIX[args.key], cfg, "cuda")}
    names = {"ind_only": list(spec["base"]), "ind_ood": list(MIX[args.key])}

    per_ep, res = {}, {}
    for sp, (d, groups, tf) in sets.items():
        eps = sorted({e for m in groups.values() for e in m})
        S = {}
        for n_, e in enumerate(eps):
            S[e] = score(models, args.key, d, tf, e, H, f, hw, s0)
            if n_ % 25 == 0:
                print(f"[{args.key} {sp}] {n_}/{len(eps)} {time.time() - t_start:.0f}s", flush=True)
        for (proto, name) in S[eps[0]]:
            for k in KS:
                per_ep[f"{sp}/{proto}/{name}/{k}"] = np.array([S[e][(proto, name)][k] for e in eps])
        per_ep[f"{sp}/names"] = np.array([d.video_names[e] for e in eps])
        for g, mem in groups.items():
            res[f"{sp}/{g}"] = {"n": len(mem), **{proto: {name: {k: float(np.mean([S[e][(proto, name)][k] for e in mem])) for k in KS}
                                                        for (pr, name) in S[eps[0]] if pr == proto} for proto in ("window", "rollout")}}
    ref = json.load(open(os.path.join(WORK, REF[args.key])))["sets"]["ind_test"]["acwm"]["all"]["model"]
    info = {"env": cfg.env.name, "models": names, "split": SPLIT[args.key], "obs_hw": list(hw), "window_first_scored_frame": f * (H + s0),
            "check_ind_only_window_vs_original_eval": {k: [res["ind_test/all"]["window"]["ind_only"][k], ref[k][0]] for k in KS},
            "groups": res, "elapsed_s": round(time.time() - t_start)}
    json.dump(info, open(args.out, "w"), indent=1)
    np.savez_compressed(os.path.splitext(args.out)[0] + ".npz", **per_ep)
    for g, r in res.items():
        print(f"{g:18s} n={r['n']:3d}  " + "  ".join(f"{proto} " + " / ".join(f"{1e3 * r[proto][m]['mse']:.2f}" for m in ("ind_only", "ind_ood")) for proto in ("window", "rollout")))
    print("check (ours, original):", info["check_ind_only_window_vs_original_eval"])
    print("wrote", args.out, f"{info['elapsed_s']}s")


if __name__ == "__main__":
    main()
