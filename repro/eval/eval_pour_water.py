"""ACWM-Phys Pour Water 예측 품질 평가 (DINO-WM). 모델 없이 계산하는 기준선과 함께 낸다.

세트: val (학습 데이터에서 train.py 와 같은 분할로 뗀 검증 에피소드), ind_test, ood_test.
ood_test 는 첫 프레임 물 색 픽셀 수(W0)로 낮음(<4000) / 높음으로 나눈다.

  M1  한 스텝: 문맥 num_hist 프레임으로 다음 프레임 예측 (eval_pred_quality.py 와 같은 계산: LPIPS(VGG)·SSIM·MSE, 224, [-1, 1]).
      기준선 copy_last(마지막 문맥 프레임), recon_copy(그 프레임을 인코더·디코더에 통과), recon_ceiling(정답 프레임을 인코더·디코더에 통과)
  M2  열린 루프 롤아웃: ACWM-Phys eval.py 와 같은 37프레임 창, 시작 10곳. 프레임 t0 + frameskip·k 의 k = num_hist.. 를 예측.
      같은 지표 + 잠재(DINOv2 패치 임베딩) MSE·코사인 + ACWM 방식 지표(240×240, [0, 1], MSE·M-MSE·PSNR·SSIM)와 정적 기준선.
      표는 창의 프레임 3 이상(max(num_hist, 3))에서만 모아 num_hist 가 다른 모델을 같은 프레임으로 비교하고, 곡선(m2_curve)은 예측 프레임 전부.
  M3  전체 에피소드 롤아웃 (처음 num_hist 프레임 + 모든 action): 스텝별 곡선, 끝 구간 상자 바닥 물 양
  M5  action 민감도: 같은 창을 다른 에피소드의 action / action 0 으로 롤아웃

usage (cwd = $DINO_WM, env sourced):
  python repro/eval/eval_pour_water.py <ckpt_base_path> <model_name> <out_json> [--epoch N] [--sets val,ind_test,ood_test]
         [--max_slices 5000] [--max_eps 0] [--no_full]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import hydra
import cv2
import decord
from pathlib import Path
from omegaconf import OmegaConf
from einops import rearrange
from skimage.metrics import structural_similarity

sys.path.insert(0, os.getcwd())
from plan import load_model
from utils import seed
from datasets.acwm_pour_water_dset import ACWMPourWaterDataset, LazyTrajSlicerDataset, load_acwm_pour_water_split
from datasets.img_transforms import default_transform
from datasets.traj_dset import TrajSubset
from metrics.image_metrics import ssim
from metrics.lpipsPyTorch.modules.lpips import LPIPS

CUTS = np.linspace(0, 300 - 37, 10, dtype=int)  # ACWM-Phys eval.py: 37-frame windows, test_cuts 10
W0_SPLIT = 4000  # ood_test first-frame water pixels (360x360): low 1675-2795, high 5360-8366
BAND = (140, 225, 310, 300)  # destination box floor band x0, y0, x1, y1 in 360x360 px
FILL_FROM = 250  # frames from this index on are used for the final fill level
COMMON_FROM = 3  # window frame index from which models with different num_hist are compared (a window has 8 frames at stride 5)
THIRDS = {"all": None, "early": (0, 80), "mid": (80, 200), "late": (200, 300)}  # by window start t0


def water_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[..., 0] >= 98) & (hsv[..., 0] <= 110) & (hsv[..., 1] > 70) & (hsv[..., 2] > 100)


def band_count(frames_u8, size):
    s = size / 360.0
    x0, y0, x1, y1 = (int(round(v * s)) for v in BAND)
    return np.array([water_mask(f)[y0:y1, x0:x1].sum() for f in frames_u8], dtype=float)


def as_np(x):  # decord returns NDArray, or a torch tensor when another dataset module set the torch bridge
    return x.asnumpy() if hasattr(x, "asnumpy") else x.numpy()


def to_u8(x):  # (n, 3, h, w) in [-1, 1] -> (n, h, w, 3) uint8
    return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def lpips_each(crit, x, y):  # LPIPS.forward without the final batch mean
    fx, fy = crit.net(x), crit.net(y)
    res = [l((a - b) ** 2).mean((2, 3), True) for a, b, l in zip(fx, fy, crit.lin)]
    return torch.sum(torch.cat(res, 1), dim=1).flatten()


def pix(crit, x, y):
    """x, y: (b, s, 3, h, w) in [-1, 1] -> dict of (b, s) arrays (repo conventions: SSIM and MSE on the [-1, 1] images)"""
    b, s = x.shape[:2]
    xf, yf = x.flatten(0, 1), y.flatten(0, 1)
    parts = []
    for i in range(0, len(xf), 64):  # chunks keep the VGG feature maps small next to a running training job
        a, c = xf[i : i + 64], yf[i : i + 64]
        parts.append({"lpips": lpips_each(crit, a, c), "ssim": ssim(a, c, size_average=False), "mse": ((a - c) ** 2).flatten(1).mean(1)})
    return {k: torch.cat([p[k] for p in parts]).reshape(b, s).float().cpu().numpy() for k in parts[0]}


def boot(vals, n=2000, seed_=0):
    """mean and 95% bootstrap interval over episodes"""
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [None, None, None]
    idx = np.random.RandomState(seed_).randint(0, len(x), (n, len(x)))
    b = x[idx].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def boot_ratio(num, den, n=2000, seed_=0):
    """bootstrap of mean(num) / mean(den) over episodes"""
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = np.random.RandomState(seed_).randint(0, len(num), (n, len(num)))
    b = num[idx].mean(1) / den[idx].mean(1)
    return [float(num.mean() / den.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


class Episode:
    """One episode of a set: 224px frames for the model, 240px [0, 1] frames for the ACWM-style metrics, normalized actions."""

    def __init__(self, base, real, tf=None):
        self.path = base.split_path / base.video_names[real]
        self.act = base.actions[real]
        vr = decord.VideoReader(str(self.path), num_threads=1)
        raw = rearrange(torch.as_tensor(as_np(vr.get_batch(list(range(len(vr)))))), "T H W C -> T C H W")
        if tf is not None:
            self.vis = torch.cat([tf(raw[s : s + 60].float() / 255.0) for s in range(0, len(raw), 60)])  # (T, 3, 224, 224) in [-1, 1]
        self.vis240 = np.stack([cv2.resize(f, (240, 240), interpolation=cv2.INTER_AREA) for f in raw.permute(0, 2, 3, 1).numpy()]).astype(np.float32) / 255.0


def water_px_frame0(base, real):
    vr = decord.VideoReader(str(base.split_path / base.video_names[real]), num_threads=1)
    return int(water_mask(as_np(vr[0])).sum())


def unwrap(S):
    return (S.dataset, list(S.indices)) if isinstance(S, TrajSubset) else (S, list(range(len(S))))


def acwm_metrics(pred, gt, first, gt_win):
    """ACWM-Phys eval.py compute_metrics for one window. pred, gt: (T, 240, 240, 3) in [0, 1] (the predicted frames);
    first: window frame 0; gt_win: all frames of the window (the motion mask is the max |frame - first| over them)."""
    mse = float(((pred - gt) ** 2).mean())
    mask = np.abs(gt_win - first[None]).max(axis=(0, 3))  # (240, 240)
    w = (0.01 + mask)[None, :, :, None]
    mmse = float((w * (pred - gt) ** 2).sum() / (w.sum() * pred.shape[0] * pred.shape[3] + 1e-8))
    psnr = float(10 * np.log10(1.0 / (mse + 1e-8)))
    ss = float(np.mean([structural_similarity(p, g, data_range=1.0, channel_axis=-1) for p, g in zip(pred, gt)]))
    return {"mse": mse, "masked_mse": mmse, "psnr": psnr, "ssim": ss}


@torch.no_grad()
def rollout(model, obs, acts, H, dev):
    """obs (b, K, 3, h, w) GT frames, acts (b, K-1, a) macro actions -> decoded rollout, decoded GT reconstruction, visual latents"""
    b, K = obs.shape[:2]
    obs = obs.to(dev)
    zeros = lambda n: torch.zeros(b, n, 1, device=dev)
    z_obses, _ = model.rollout({"visual": obs[:, :H], "proprio": zeros(H)}, acts.to(dev))
    pred = model.decode_obs(z_obses)[0]["visual"]
    z_gt = model.encode_obs({"visual": obs, "proprio": zeros(K)})
    rec = model.decode_obs(z_gt)[0]["visual"]
    return pred, rec, z_obses["visual"], z_gt["visual"], obs


def make_windows(ep, f, K, act=None):
    act = ep.act if act is None else act
    obs = torch.stack([ep.vis[t0 + f * np.arange(K)] for t0 in CUTS])
    acts = torch.stack([act[t0 : t0 + f * (K - 1)].reshape(K - 1, -1) for t0 in CUTS])
    return obs, acts


def fmt(x, nd=4):
    return "n/a" if x is None or x[0] is None else f"{x[0]:.{nd}f} [{x[1]:.{nd}f}, {x[2]:.{nd}f}]"


def print_summary(result):
    for name, r in result["sets"].items():
        for g in r["m1"]:
            m1, m2, ac = r["m1"][g], r["m2"][g]["all"], r["acwm"][g]
            print(f"== {name}/{g}  episodes {r['group_sizes'][g]}  one-step slices {m1['n_slices']}")
            print("  M1 LPIPS  " + "  ".join(f"{k} {fmt(m1[k]['lpips'], 3)}" for k in ("model", "copy_last", "recon_copy", "recon_ceiling")))
            print("  M1 SSIM   " + "  ".join(f"{k} {fmt(m1[k]['ssim'], 3)}" for k in ("model", "copy_last", "recon_copy", "recon_ceiling")))
            print("  M1 latent MSE  " + "  ".join(f"{k} {fmt(v, 3)}" for k, v in m1["latent_mse"].items()))
            print("  M2 LPIPS  " + "  ".join(f"{k} {fmt(m2[k]['lpips'], 3)}" for k in ("model", "copy_last", "recon_copy", "recon_ceiling", "copy_first")))
            print("  M2 skill vs recon_copy  LPIPS " + fmt(r["m2"][g]["skill_vs_recon_copy"]["lpips"], 3) + "  MSE " + fmt(r["m2"][g]["skill_vs_recon_copy"]["mse"], 3))
            print("  ACWM-style MSE x1e3  " + "  ".join(f"{k} {1e3 * ac[k]['mse'][0]:.2f}" for k in ("model", "copy_first", "copy_last", "time_mean")) +
                  "   ratio to copy_first: model " + fmt(ac["ratio_to_copy_first"]["model"]["mse"], 3))
            m5 = r["m5"][g]["late"]
            print("  M5 late-window LPIPS  " + "  ".join(f"{k} {fmt(m5[k]['lpips'], 3)}" for k in ("true", "shuffled", "zero")))
            if r["m3_fill"]:
                fl = r["m3_fill"][g]
                print(f"  M3 fill-level rel. error: model {fmt(fl['model_rel_err'], 3)}  recon {fmt(fl['recon_rel_err'], 3)}  gate(recon<=0.10) {fl['valid_gate_recon_rel_err_le_0.10']}  corr(model, gt) {fl['corr_model_gt']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_base"); ap.add_argument("model_name"); ap.add_argument("out_json")
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--sets", default="val,ind_test,ood_test")
    ap.add_argument("--max_slices", type=int, default=5000)
    ap.add_argument("--max_eps", type=int, default=0)
    ap.add_argument("--no_full", action="store_true")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    model_path = Path(args.ckpt_base) / "outputs" / args.model_name
    cfg = OmegaConf.load(model_path / "hydra.yaml")
    ck = model_path / "checkpoints" / (f"model_{args.epoch}.pth" if args.epoch else "model_latest.pth")
    dev = "cuda"
    H, f, P = cfg.num_hist, cfg.frameskip, cfg.num_pred
    K = 36 // f + 1  # frames of a 37-frame window at stride f
    S = K - H  # predicted frames per window
    s0 = max(H, COMMON_FROM) - H  # steps skipped in the aggregated tables so every model is scored on frames >= COMMON_FROM (curves keep all steps)
    K_full = 300 // f
    seed(cfg.training.seed)
    ds_cfg = cfg.env.dataset
    tf = default_transform(cfg.img_size)
    model = load_model(ck, cfg, cfg.num_action_repeat, dev)
    model.eval()
    epoch = int(torch.load(ck, map_location="cpu")["epoch"])
    crit = LPIPS("vgg", "0.1").to(dev).eval()

    wanted = args.sets.split(",")
    datasets, trajs = hydra.utils.call(ds_cfg, num_hist=H, num_pred=P, frameskip=f)
    sets, slicers = {}, {}
    if "val" in wanted:
        sets["val"], slicers["val"] = trajs["valid"], datasets["valid"]
    for sp in ("ind_test", "ood_test"):
        if sp in wanted:
            d = load_acwm_pour_water_split(tf, ds_cfg.data_path, sp, n_rollout=ds_cfg.n_rollout, exclude=ds_cfg.exclude, normalize_action=cfg.normalize_action)
            sets[sp] = d
            slicers[sp] = LazyTrajSlicerDataset(d, H + P, f)
    train_base = ACWMPourWaterDataset(ds_cfg.data_path, "ind_train", n_rollout=ds_cfg.n_rollout, exclude=ds_cfg.exclude)
    t0 = time.time()
    # time-of-episode mean video (first 100 training episodes), the ACWM-style "time_mean" baseline
    acc = np.zeros((300, 240, 240, 3), np.float64)
    nt = min(100, len(train_base))
    for i in range(nt):
        acc += Episode(train_base, i).vis240
    time_mean = (acc / nt).astype(np.float32)
    print(f"time mean built from {nt} training episodes in {time.time() - t0:.0f}s", flush=True)

    result = {"model": args.model_name, "ckpt_epoch": epoch, "num_hist": H, "frameskip": f, "window_frames": K, "predicted_per_window": S,
              "aggregated_from_window_frame": max(H, COMMON_FROM), "curve_first_window_frame": H,
              "cuts": CUTS.tolist(), "sets": {}}
    zero_act = (-train_base.action_mean / train_base.action_std).repeat(f)  # "no motion" in normalized units, one macro action
    for name, Sset in sets.items():
        t_set = time.time()
        base, idxs = unwrap(Sset)
        n_ep = len(idxs) if not args.max_eps else min(args.max_eps, len(idxs))
        w0 = np.array([water_px_frame0(base, idxs[e]) for e in range(n_ep)])
        groups = {"all": np.arange(n_ep)}
        if name == "ood_test":
            groups["low"], groups["high"] = np.where(w0 < W0_SPLIT)[0], np.where(w0 >= W0_SPLIT)[0]
        res = {"n_episodes": int(n_ep), "group_sizes": {g: int(len(v)) for g, v in groups.items()}, "w0": w0.tolist()}

        # ---------- M1: one step on slices ----------
        sl = slicers[name]
        pick = np.linspace(0, len(sl) - 1, min(args.max_slices, len(sl))).round().astype(int)
        pick = pick[sl.slices[pick, 0] < n_ep]
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(sl, pick.tolist()), batch_size=32, shuffle=False, num_workers=8)
        m1 = {k: [] for k in ("model", "copy_last", "recon_copy", "recon_ceiling")}
        m1z = {"model": [], "copy_latent": []}
        with torch.no_grad():
            for obs, act, _ in loader:
                obs = {k: v.to(dev) for k, v in obs.items()}
                z_out, vis_out, vis_rec, _, _ = model(obs, act.to(dev))
                x = obs["visual"]
                tgt = x[:, H : H + 1]
                for k, pred in (("model", vis_out[:, -1:]), ("copy_last", x[:, H - 1 : H]), ("recon_copy", vis_rec[:, H - 1 : H]), ("recon_ceiling", vis_rec[:, H : H + 1])):
                    m1[k].append(pix(crit, pred, tgt))
                zg = model.encode_obs(obs)["visual"]
                zp = z_out[:, -1, :, : zg.shape[-1]]
                m1z["model"].append(((zp - zg[:, H]) ** 2).mean((1, 2)).cpu().numpy())
                m1z["copy_latent"].append(((zg[:, H - 1] - zg[:, H]) ** 2).mean((1, 2)).cpu().numpy())
        ep_of = sl.slices[pick, 0]
        m1 = {k: {m: np.concatenate([b[m][:, 0] for b in v]) for m in ("lpips", "ssim", "mse")} for k, v in m1.items()}
        m1z = {k: np.concatenate(v) for k, v in m1z.items()}

        def per_episode(vals, eps):
            return np.array([vals[eps == e].mean() for e in np.unique(eps)])

        res["m1"] = {}
        for g, members in groups.items():
            sel = np.isin(ep_of, members)
            r = {k: {m: boot(per_episode(v[m][sel], ep_of[sel])) for m in v} for k, v in m1.items()}
            r["latent_mse"] = {k: boot(per_episode(v[sel], ep_of[sel])) for k, v in m1z.items()}
            r["n_slices"] = int(sel.sum())
            res["m1"][g] = r
        print(f"[{name}] M1 done ({len(pick)} slices) {time.time() - t_set:.0f}s", flush=True)

        # ---------- M2 / M5 / M3 per episode ----------
        E = n_ep
        methods = ("model", "copy_last", "recon_copy", "recon_ceiling", "copy_first")
        W = len(CUTS)
        A = {m: {k: np.zeros((E, W, S)) for k in ("lpips", "ssim", "mse")} for m in methods}
        A5 = {m: {k: np.zeros((E, W, S)) for k in ("lpips", "ssim", "mse")} for m in ("shuffled", "zero")}
        LZ = {m: {k: np.zeros((E, W, S)) for k in ("mse", "cos")} for m in ("model", "copy_latent")}
        AC = {m: {k: np.zeros((E, W)) for k in ("mse", "masked_mse", "psnr", "ssim")} for m in ("model", "copy_first", "copy_last", "time_mean")}
        full_k = list(range(H, K_full))
        FC = {m: {k: np.zeros((E, len(full_k))) for k in ("lpips", "ssim", "mse")} for m in ("model", "copy_last", "recon_ceiling")}
        fill_k = [k for k in full_k if k * f >= FILL_FROM]
        FILL = {m: np.zeros(E) for m in ("gt", "model", "recon")}
        for e in range(E):
            ep = Episode(base, idxs[e], tf)
            other = base.actions[idxs[(e + 1) % n_ep]]  # actions of the next episode (normalized)
            obs, acts = make_windows(ep, f, K)
            pred, rec, zp, zg, obs_d = rollout(model, obs, acts, H, dev)
            tgt = obs_d[:, H:]
            for m, p in (("model", pred[:, H:]), ("copy_last", obs_d[:, H - 1 : H].expand_as(tgt)), ("recon_copy", rec[:, H - 1 : H].expand_as(tgt)),
                         ("recon_ceiling", rec[:, H:]), ("copy_first", obs_d[:, :1].expand_as(tgt))):
                for k, v in pix(crit, p, tgt).items():
                    A[m][k][e] = v
            for m, z in (("model", zp[:, H:]), ("copy_latent", zg[:, H - 1 : H].expand_as(zg[:, H:]))):
                LZ[m]["mse"][e] = ((z - zg[:, H:]) ** 2).mean((2, 3)).cpu().numpy()
                LZ[m]["cos"][e] = F.cosine_similarity(z, zg[:, H:], dim=-1).mean(-1).cpu().numpy()
            # ACWM-style metrics at 240x240 on the same predicted frames
            for w_i, t0_ in enumerate(CUTS):
                fid = t0_ + f * np.arange(K)
                gt_win = ep.vis240[fid]
                gt = gt_win[H + s0 :]
                up = lambda p: F.interpolate((p.clamp(-1, 1) + 1) / 2, size=240, mode="bilinear", align_corners=False).permute(0, 2, 3, 1).cpu().numpy()
                preds = {"model": up(pred[w_i, H + s0 :]), "copy_first": np.repeat(gt_win[:1], S - s0, 0), "copy_last": np.repeat(gt_win[H - 1 : H], S - s0, 0), "time_mean": time_mean[fid[H + s0 :]]}
                for m, p in preds.items():
                    for k, v in acwm_metrics(p, gt, gt_win[0], gt_win).items():
                        AC[m][k][e, w_i] = v
            # action sensitivity
            _, acts_sh = make_windows(ep, f, K, act=other)
            acts_zero = zero_act.expand(W, K - 1, -1)
            for m, a_ in (("shuffled", acts_sh), ("zero", acts_zero)):
                p5 = rollout(model, obs, a_, H, dev)[0]
                for k, v in pix(crit, p5[:, H:], tgt).items():
                    A5[m][k][e] = v
            # M3: whole episode from the first H frames
            if not args.no_full:
                fo = ep.vis[::f][:K_full][None]
                fa = ep.act[: f * (K_full - 1)].reshape(1, K_full - 1, -1)
                fp, fr, _, _, fobs = rollout(model, fo, fa, H, dev)
                for m, p in (("model", fp[:, H:]), ("copy_last", fobs[:, H - 1 : H].expand_as(fobs[:, H:])), ("recon_ceiling", fr[:, H:])):
                    for k, v in pix(crit, p, fobs[:, H:]).items():
                        FC[m][k][e] = v[0]
                for m, src in (("gt", fobs), ("model", fp), ("recon", fr)):
                    FILL[m][e] = band_count(to_u8(src[0, fill_k].cpu()), 224).mean()
            if e % 10 == 0:
                print(f"[{name}] episode {e}/{E} {time.time() - t_set:.0f}s", flush=True)

        # ---------- summaries ----------
        t0s = CUTS
        wsel = {t: np.arange(W) if r is None else np.where((t0s >= r[0]) & (t0s < r[1]))[0] for t, r in THIRDS.items()}
        res["m2"], res["m2_curve"], res["acwm"], res["m5"], res["m2_latent"], res["m3_curve"], res["m3_fill"] = {}, {}, {}, {}, {}, {}, {}
        for g, members in groups.items():
            res["m2"][g], res["m5"][g], res["m2_latent"][g] = {}, {}, {}
            for t, ws in wsel.items():
                res["m2"][g][t] = {m: {k: boot(A[m][k][members][:, ws][:, :, s0:].mean((1, 2))) for k in A[m]} for m in methods}
                res["m5"][g][t] = {"true": {k: boot(A["model"][k][members][:, ws][:, :, s0:].mean((1, 2))) for k in A["model"]}}
                for m in A5:
                    res["m5"][g][t][m] = {k: boot(A5[m][k][members][:, ws][:, :, s0:].mean((1, 2))) for k in A5[m]}
                res["m2_latent"][g][t] = {m: {k: boot(LZ[m][k][members][:, ws][:, :, s0:].mean((1, 2))) for k in LZ[m]} for m in LZ}
            res["m2_curve"][g] = {m: {k: A[m][k][members].mean((0, 1)).tolist() for k in A[m]} for m in methods}
            res["acwm"][g] = {m: {k: boot(AC[m][k][members].mean(1)) for k in AC[m]} for m in AC}
            res["acwm"][g]["ratio_to_copy_first"] = {m: {k: boot_ratio(AC[m][k][members].mean(1), AC["copy_first"][k][members].mean(1)) for k in ("mse", "masked_mse")} for m in ("model", "copy_last", "time_mean")}
            # skill: how much of the gap between recon_copy and the recon ceiling the model closes (LPIPS, MSE)
            res["m2"][g]["skill_vs_recon_copy"] = {}
            for k in ("lpips", "mse"):
                num = (A["recon_copy"][k][members][:, :, s0:].mean((1, 2)) - A["model"][k][members][:, :, s0:].mean((1, 2)))
                den = (A["recon_copy"][k][members][:, :, s0:].mean((1, 2)) - A["recon_ceiling"][k][members][:, :, s0:].mean((1, 2)))
                res["m2"][g]["skill_vs_recon_copy"][k] = boot_ratio(num, den)
            if not args.no_full:
                res["m3_curve"][g] = {m: {k: FC[m][k][members].mean(0).tolist() for k in FC[m]} for m in FC}
                rel = lambda m: np.abs(FILL[m][members] - FILL["gt"][members]) / np.maximum(FILL["gt"][members], 1.0)
                fr_ = {"gt_px": boot(FILL["gt"][members]), "model_px": boot(FILL["model"][members]), "recon_px": boot(FILL["recon"][members]),
                       "model_rel_err": boot(rel("model")), "recon_rel_err": boot(rel("recon")),
                       "model_signed_rel_err": boot((FILL["model"][members] - FILL["gt"][members]) / np.maximum(FILL["gt"][members], 1.0)),
                       "corr_model_gt": float(np.corrcoef(FILL["model"][members], FILL["gt"][members])[0, 1]) if len(members) > 2 else None,
                       "corr_recon_gt": float(np.corrcoef(FILL["recon"][members], FILL["gt"][members])[0, 1]) if len(members) > 2 else None}
                fr_["valid_gate_recon_rel_err_le_0.10"] = bool(fr_["recon_rel_err"][0] is not None and fr_["recon_rel_err"][0] <= 0.10)
                res["m3_fill"][g] = fr_
        res["m3_steps"] = full_k
        res["seconds"] = round(time.time() - t_set, 1)
        result["sets"][name] = res
        print(f"[{name}] done {res['seconds']}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    json.dump(result, open(args.out_json, "w"), indent=1)
    print_summary(result)
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
