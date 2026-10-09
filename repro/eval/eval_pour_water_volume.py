"""Pour Water 물 양 분석 (eval_pour_water.py 의 보조).

전체 에피소드 롤아웃(처음 num_hist 프레임 + 모든 action)에서 물 색 픽셀 수의 시간 곡선을 정답·모델·정답 재구성(인코더-디코더)으로 비교한다:
화면 전체와 상자 바닥 띠(eval_pour_water.py 의 BAND). 에피소드별 값이라 산점도와 곡선을 그릴 수 있다. 같이 action 민감도를 창 시작 t0 별로
낸다 (정규화한 action 0 = 학습 평균 action, 다음 에피소드의 action 으로 교체).

usage (cwd = $DINO_WM, env sourced):
  python repro/eval/eval_pour_water_volume.py <ckpt_base_path> <model_name> <out_npz> [--epoch N] [--sets val,ind_test,ood_test] [--max_eps 0]
"""
import os, sys, argparse, time
import numpy as np
import torch
import hydra
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_pour_water as E
from plan import load_model
from utils import seed
from datasets.acwm_pour_water_dset import load_acwm_pour_water_split
from datasets.img_transforms import default_transform
from metrics.lpipsPyTorch.modules.lpips import LPIPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_base"); ap.add_argument("model_name"); ap.add_argument("out_npz")
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--sets", default="val,ind_test,ood_test")
    ap.add_argument("--max_eps", type=int, default=0)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    model_path = Path(args.ckpt_base) / "outputs" / args.model_name
    cfg = OmegaConf.load(model_path / "hydra.yaml")
    ck = model_path / "checkpoints" / (f"model_{args.epoch}.pth" if args.epoch else "model_latest.pth")
    dev = "cuda"
    H, f = cfg.num_hist, cfg.frameskip
    K, K_full = 36 // f + 1, 300 // f
    s0 = max(H, E.COMMON_FROM) - H
    seed(cfg.training.seed)
    ds_cfg = cfg.env.dataset
    tf = default_transform(cfg.img_size)
    model = load_model(ck, cfg, cfg.num_action_repeat, dev)
    model.eval()
    crit = LPIPS("vgg", "0.1").to(dev).eval()

    wanted = args.sets.split(",")
    sets = {}
    if "val" in wanted:
        sets["val"] = hydra.utils.call(ds_cfg, num_hist=H, num_pred=cfg.num_pred, frameskip=f)[1]["valid"]
    for sp in ("ind_test", "ood_test"):
        if sp in wanted:
            sets[sp] = load_acwm_pour_water_split(tf, ds_cfg.data_path, sp, n_rollout=ds_cfg.n_rollout, exclude=ds_cfg.exclude, normalize_action=cfg.normalize_action)
    zero_act = torch.zeros(2 * f)  # normalized units: the training-mean action (0.09 std from "no motion" in the move dim)

    out = {"cuts": E.CUTS, "num_hist": np.array(H), "frameskip": np.array(f), "aggregated_from_window_frame": np.array(max(H, E.COMMON_FROM))}
    for name, Sset in sets.items():
        t_set = time.time()
        base, idxs = E.unwrap(Sset)
        n_ep = len(idxs) if not args.max_eps else min(args.max_eps, len(idxs))
        W = len(E.CUTS)
        out[f"{name}/w0"] = np.array([E.water_px_frame0(base, idxs[e]) for e in range(n_ep)])
        for m in ("gt", "pred", "rec"):
            out[f"{name}/tot_{m}"] = np.zeros((n_ep, K_full))
            out[f"{name}/band_{m}"] = np.zeros((n_ep, K_full))
        for v in ("true", "shuffled", "zero", "recon_copy", "copy_last"):
            for k in ("lpips", "mse"):
                out[f"{name}/t0_{v}_{k}"] = np.zeros((n_ep, W))
        for e in range(n_ep):
            ep = E.Episode(base, idxs[e], tf)
            fo = ep.vis[::f][:K_full][None]
            fa = ep.act[: f * (K_full - 1)].reshape(1, K_full - 1, -1)
            fp, fr, _, _, fobs = E.rollout(model, fo, fa, H, dev)
            for m, src in (("gt", fobs), ("pred", fp), ("rec", fr)):
                u8 = E.to_u8(src[0].cpu())
                out[f"{name}/tot_{m}"][e] = [E.water_mask(x).sum() for x in u8]
                out[f"{name}/band_{m}"][e] = E.band_count(u8, 224)
            obs, acts = E.make_windows(ep, f, K)
            pred, rec, _, _, obs_d = E.rollout(model, obs, acts, H, dev)
            tgt = obs_d[:, H + s0 :]
            other = base.actions[idxs[(e + 1) % n_ep]]
            variants = {"true": pred[:, H + s0 :], "recon_copy": rec[:, H - 1 : H].expand_as(tgt), "copy_last": obs_d[:, H - 1 : H].expand_as(tgt)}
            for v, a_ in (("shuffled", E.make_windows(ep, f, K, act=other)[1]), ("zero", zero_act.expand(W, K - 1, -1))):
                variants[v] = E.rollout(model, obs, a_, H, dev)[0][:, H + s0 :]
            for v, p in variants.items():
                r = E.pix(crit, p, tgt)
                out[f"{name}/t0_{v}_lpips"][e] = r["lpips"].mean(1)
                out[f"{name}/t0_{v}_mse"][e] = r["mse"].mean(1)
            if e % 10 == 0:
                print(f"[{name}] episode {e}/{n_ep} {time.time() - t_set:.0f}s", flush=True)
        print(f"[{name}] done {time.time() - t_set:.0f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
    np.savez_compressed(args.out_npz, **out)
    print("wrote", args.out_npz)


if __name__ == "__main__":
    main()
