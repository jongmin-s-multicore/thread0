"""Pour Water 전체 에피소드 롤아웃 프레임을 render_acwm_phys.py 가 읽는 형식으로 저장한다 (모델 하나).
ind_test, ood_test 의 모든 에피소드를 처음 num_hist 프레임 + 모든 action 으로 롤아웃하고 정답·모델·재구성 프레임과 스텝별 LPIPS 를 남긴다.
그룹: ind_test 는 all, ood_test 는 all / low / high (첫 프레임 물 양 W0 기준, eval_pour_water.py 와 같다).

usage (cwd = $DINO_WM, env sourced):
  python repro/eval/frames_pour_water.py <ckpt_base_path> <out_prefix> [--model train_pw_h1f5] [--epoch 6]
  -> <out_prefix>.json, <out_prefix>_<set>.npz, <out_prefix>_quant_<set>.npz
  python repro/eval/render_acwm_phys.py <out_prefix> <out_dir> --fps 6 [--per_episode 10]
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_pour_water as E
from render_pour_water import load
from utils import seed
from datasets.acwm_pour_water_dset import load_acwm_pour_water_split
from datasets.img_transforms import default_transform
from metrics.lpipsPyTorch.modules.lpips import LPIPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_base"); ap.add_argument("out_prefix")
    ap.add_argument("--model", default="train_pw_h1f5")
    ap.add_argument("--epoch", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    torch.set_grad_enabled(False)
    cfg, model = load(args.ckpt_base, args.model, args.epoch)
    ds_cfg, f, H = cfg.env.dataset, cfg.frameskip, cfg.num_hist
    K = 300 // f
    tf = default_transform(cfg.img_size)
    crit = LPIPS("vgg", "0.1").to("cuda").eval()
    seed(cfg.training.seed)
    for sp in ("ind_test", "ood_test"):
        d = load_acwm_pour_water_split(tf, ds_cfg.data_path, sp, n_rollout=ds_cfg.n_rollout, exclude=ds_cfg.exclude, normalize_action=cfg.normalize_action)
        n = len(d)
        w0 = np.array([E.water_px_frame0(d, i) for i in range(n)])
        groups = {"all": np.arange(n)}
        if sp == "ood_test":
            groups["low"], groups["high"] = np.where(w0 < E.W0_SPLIT)[0], np.where(w0 >= E.W0_SPLIT)[0]
        fr = {k: np.zeros((n, K, cfg.img_size, cfg.img_size, 3), np.uint8) for k in ("gt", "model", "recon")}
        lp = np.full((n, K - H), np.nan)
        for e in range(n):
            ep = E.Episode(d, e, tf)
            fo = ep.vis[::f][:K][None]
            fa = ep.act[: f * (K - 1)].reshape(1, K - 1, -1)
            pred, rec, _, _, obs = E.rollout(model, fo, fa, H, "cuda")
            lp[e] = E.pix(crit, pred[:, H:], obs[:, H:])["lpips"][0]
            fr["gt"][e], fr["model"][e], fr["recon"][e] = E.to_u8(obs[0]), E.to_u8(pred[0]), E.to_u8(rec[0])
            if e % 10 == 0:
                print(f"[{sp}] episode {e}/{n}  lpips {np.nanmean(lp[e]):.3f}", flush=True)
        names = np.array([d.video_names[i] for i in range(n)])
        k_full = np.full(n, K)
        g = {f"group_{k}": v for k, v in groups.items()}
        np.savez_compressed(f"{args.out_prefix}_{sp}.npz", names=names, k_full=k_full, frameskip=f, num_hist=H, w0=w0,
                            frames_model=fr["model"], frames_recon=fr["recon"], m3_model_lpips=lp, **g)
        np.savez_compressed(f"{args.out_prefix}_quant_{sp}.npz", names=names, k_full=k_full, frames_gt=fr["gt"], **g)
        print(f"[{sp}] {n} episodes, groups " + ", ".join(f"{k} {len(v)}" for k, v in groups.items()) + f", whole-episode LPIPS {np.nanmean(lp):.3f}", flush=True)
    json.dump({"env": "acwm_pour_water", "model": args.model, "ckpt_epoch": args.epoch, "frameskip": int(f), "num_hist": int(H)},
              open(f"{args.out_prefix}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
