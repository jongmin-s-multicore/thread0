"""ACWM-Phys Cloth Move / Push Sand / Stack Cube / Push Cube / Push Rope / Robot Arm / Reacher 예측 품질 평가 (DINO-WM).
지표와 기준선은 eval_pour_water.py 와 같다.

세트: val (학습 데이터에서 train.py 와 같은 분할로 뗀 검증 에피소드), ind_test, ood_test.
그룹: Cloth Move ood_test 는 smaller / larger (파일 이름), Stack Cube 는 success / fail (metadata) 와 ood_test 의 up·down·left·right.
      [repro] 이슈 #4: 테스트 세트마다 first50 (metadata 순서 앞 50개, ACWM eval.py --max_trajs 50 이 보는 에피소드).
      Push Cube ood_test 는 cubes1 (episode 0-19) / cubes5 (20-49), Robot Arm ood_test 는 inside / outside
      (누적 action 이 모든 관절에서 학습 에피소드의 누적 action 범위 안인가). Reacher 는 에피소드가 37프레임이라 시작 1곳만 잰다.

  M1  한 스텝: 문맥 num_hist 프레임으로 다음 프레임 예측 (LPIPS(VGG)·SSIM·MSE, 224, [-1, 1]), 잠재 MSE.
      기준선 copy_last, recon_copy(마지막 문맥 프레임을 인코더·디코더에 통과), recon_ceiling(정답 프레임을 인코더·디코더에 통과)
  M2  열린 루프 롤아웃: ACWM-Phys eval.py 와 같은 37프레임 창, 에피소드마다 시작 10곳 (linspace(0, 프레임 수 - 37, 10)).
      같은 지표 + 잠재 MSE·코사인 + ACWM 방식 지표(환경의 obs_shape, [0, 1], MSE·M-MSE·PSNR·SSIM)와 정적 기준선.
  M3  전체 에피소드 롤아웃 (처음 num_hist 프레임 + 모든 action): 스텝별 곡선. 디코딩한 프레임(모델·재구성)을 npz 로 저장한다
      (환경별 물리량은 eval_acwm_phys_quant.py 가 이 프레임으로 계산한다).
  M5  action 민감도: 같은 창을 다른 에피소드의 action / 정규화한 action 0 으로 롤아웃

usage (cwd = $DINO_WM, env sourced):
  python repro/eval/eval_acwm_phys.py <ckpt_base_path> <model_name> <out_prefix> [--epoch N] [--sets val,ind_test,ood_test]
         [--max_slices 4000] [--max_eps 0]
  -> <out_prefix>.json, <out_prefix>_<set>.npz
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

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import load_model
from utils import seed
from datasets.acwm_phys_dset import ACWMPhysDataset, load_acwm_phys_split
from datasets.acwm_pour_water_dset import LazyTrajSlicerDataset
from metrics.lpipsPyTorch.modules.lpips import LPIPS
from eval_pour_water import pix, boot, boot_ratio, acwm_metrics, rollout, to_u8, as_np, unwrap, fmt

WINDOW = 37  # ACWM-Phys eval.py seq_len
N_CUTS = 10  # eval.py default test_cuts (the configs say 5; the paper does not say which)
OBS_HW = {"acwm_cloth_move": (240, 240), "acwm_push_sand": (240, 400), "acwm_stack_cube": (240, 240),  # ACWM data_config obs_shape
          "acwm_push_cube": (240, 240), "acwm_push_rope": (240, 240), "acwm_robot_arm": (240, 240), "acwm_reacher": (240, 240)}
CUTS_ENV = {"acwm_reacher": 1}  # [repro] 37-frame episodes: the 10 starts would all be the same window
PIX = ("lpips", "ssim", "mse")


class Episode:
    """One episode: frames for the model (dataset transform), frames at the ACWM obs_shape in [0, 1], normalized actions."""

    def __init__(self, base, real, tf, hw):
        n = base.n_obs[real]
        vr = decord.VideoReader(str(base.split_path / base.video_names[real]), num_threads=1)
        raw = torch.as_tensor(as_np(vr.get_batch(list(range(n)))))
        self.vis = tf(rearrange(raw, "T H W C -> T C H W").float() / 255.0) if tf is not None else None
        self.vis_hw = np.stack([cv2.resize(fr, (hw[1], hw[0]), interpolation=cv2.INTER_AREA) for fr in raw.numpy()]).astype(np.float32) / 255.0
        self.act = base.actions[real]
        self.n_obs = n


def cumulative_inside(base, names, data_path, ds_kw):
    """[repro] Robot Arm: is the cumulative action (joint angles from the start pose) of each episode inside the per-joint
    range of the cumulative actions of the training episodes"""
    def cums(split_path, keep):
        meta = {Path(e["video_path"]).name: e for e in torch.load(split_path / "metadata.pt", map_location="cpu", weights_only=True)}
        return [torch.cumsum(meta[n]["actions"].double(), 0) for n in keep]
    train = ACWMPhysDataset(data_path, "ind_train", **ds_kw)
    c = torch.cat(cums(train.split_path, train.video_names))
    lo, hi = c.min(0).values, c.max(0).values
    return np.array([bool(((x >= lo) & (x <= hi)).all()) for x in cums(base.split_path, names)])


def episode_groups(env, set_name, base, idxs, data_path=None, ds_kw=None):
    names = [base.video_names[i] for i in idxs]
    groups = {"all": np.arange(len(idxs))}
    if set_name in ("ind_test", "ood_test") and len(idxs) > 50:  # [repro]
        groups["first50"] = np.arange(50)
    if env == "acwm_push_cube" and set_name == "ood_test":  # [repro]
        num = np.array([int(n.split("episode_")[-1].split(".")[0]) for n in names])
        groups["cubes1"], groups["cubes5"] = np.where(num < 20)[0], np.where(num >= 20)[0]
    if env == "acwm_robot_arm" and set_name == "ood_test":  # [repro]
        inside = cumulative_inside(base, names, data_path, ds_kw)
        groups["inside"], groups["outside"] = np.where(inside)[0], np.where(~inside)[0]
    if env == "acwm_cloth_move" and set_name == "ood_test":
        larger = np.array([n.startswith("larger") for n in names])
        groups["smaller"], groups["larger"] = np.where(~larger)[0], np.where(larger)[0]
    if env == "acwm_stack_cube":
        meta = {Path(e["video_path"]).name: e for e in torch.load(base.split_path / "metadata.pt", map_location="cpu", weights_only=True)}
        success = np.array([bool(meta[n]["success"]) for n in names])
        groups["success"], groups["fail"] = np.where(success)[0], np.where(~success)[0]
        if set_name == "ood_test":
            label = np.array([str(meta[n]["ood_label"]) for n in names])
            for lab in sorted(set(label)):
                groups[lab] = np.where(label == lab)[0]
    return groups, names


def make_windows(ep, f, K, cuts, act=None):
    act = ep.act if act is None else act
    span = f * (K - 1)
    obs = torch.stack([ep.vis[t0 + f * np.arange(K)] for t0 in cuts])
    acts = torch.stack([act[min(t0, len(act) - span) :][:span].reshape(K - 1, -1) for t0 in cuts])  # another episode may be shorter
    return obs, acts


def print_summary(result):
    for name, r in result["sets"].items():
        for g in r["m1"]:
            m1, m2, ac, m3 = r["m1"][g], r["m2"][g], r["acwm"][g], r["m3"][g]
            print(f"== {name}/{g}  episodes {r['group_sizes'][g]}  one-step slices {m1['n_slices']}")
            if r["group_sizes"][g] == 0:
                continue
            print("  M1 LPIPS  " + "  ".join(f"{k} {fmt(m1[k]['lpips'], 3)}" for k in ("model", "copy_last", "recon_copy", "recon_ceiling")))
            print("  M1 latent MSE  " + "  ".join(f"{k} {fmt(v, 3)}" for k, v in m1["latent_mse"].items()))
            print("  M2 LPIPS  " + "  ".join(f"{k} {fmt(m2[k]['lpips'], 3)}" for k in ("model", "copy_last", "recon_copy", "recon_ceiling")))
            print("  M2 skill vs recon_copy  LPIPS " + fmt(m2["skill_vs_recon_copy"]["lpips"], 3) + "  MSE " + fmt(m2["skill_vs_recon_copy"]["mse"], 3)
                  + "   latent MSE model " + fmt(r["m2_latent"][g]["model"]["mse"], 3) + " copy " + fmt(r["m2_latent"][g]["copy_latent"]["mse"], 3))
            print("  ACWM-style MSE x1e3  " + "  ".join(f"{k} {1e3 * ac[k]['mse'][0]:.2f}" for k in ("model", "copy_first", "time_mean")) +
                  "   M-MSE x1e3 " + "  ".join(f"{k} {1e3 * ac[k]['masked_mse'][0]:.2f}" for k in ("model", "copy_first", "time_mean")) +
                  f"   SSIM model {ac['model']['ssim'][0]:.3f}  PSNR model {ac['model']['psnr'][0]:.2f}")
            m5 = r["m5"][g]
            print("  M5 LPIPS  " + "  ".join(f"{k} {fmt(m5[k]['lpips'], 3)}" for k in ("true", "shuffled", "zero")))
            print("  M3 whole episode LPIPS  " + "  ".join(f"{k} {fmt(m3[k]['lpips'], 3)}" for k in ("model", "copy_first", "recon_ceiling")) +
                  "   last step: " + "  ".join(f"{k} {fmt(m3['last_step'][k]['lpips'], 3)}" for k in ("model", "copy_first", "recon_ceiling")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_base"); ap.add_argument("model_name"); ap.add_argument("out_prefix")
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--sets", default="val,ind_test,ood_test")
    ap.add_argument("--max_slices", type=int, default=4000)
    ap.add_argument("--max_eps", type=int, default=0)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    model_path = Path(args.ckpt_base) / "outputs" / args.model_name
    cfg = OmegaConf.load(model_path / "hydra.yaml")
    ck = model_path / "checkpoints" / (f"model_{args.epoch}.pth" if args.epoch else "model_latest.pth")
    dev = "cuda"
    env = cfg.env.name
    hw = OBS_HW[env]
    H, f, P = cfg.num_hist, cfg.frameskip, cfg.num_pred
    K = (WINDOW - 1) // f + 1  # frames of a 37-frame window at stride f
    S = K - H  # predicted frames per window
    seed(cfg.training.seed)
    ds_cfg = cfg.env.dataset
    ds_kw = dict(n_rollout=ds_cfg.n_rollout, exclude=list(ds_cfg.exclude), normalize_action=cfg.normalize_action,
                 action_dims=None if ds_cfg.action_dims is None else list(ds_cfg.action_dims), action_format=ds_cfg.action_format,
                 exclude_by_split=OmegaConf.to_container(ds_cfg.exclude_by_split) if ds_cfg.get("exclude_by_split") else None)  # [repro]
    n_cuts = CUTS_ENV.get(env, N_CUTS)
    tf = hydra.utils.instantiate(ds_cfg.transform)
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
            d = load_acwm_phys_split(tf, ds_cfg.data_path, sp, **ds_kw)
            sets[sp] = d
            slicers[sp] = LazyTrajSlicerDataset(d, H + P, f)
    # time-of-episode mean video (first 100 training episodes), the ACWM-style "time_mean" baseline
    train_base = ACWMPhysDataset(ds_cfg.data_path, "ind_train", **ds_kw)
    t0 = time.time()
    nt = min(100, len(train_base))
    t_max = max(train_base.n_obs[:nt])
    acc, cnt = np.zeros((t_max, hw[0], hw[1], 3), np.float64), np.zeros(t_max)
    for i in range(nt):
        v = Episode(train_base, i, None, hw).vis_hw
        acc[: len(v)] += v
        cnt[: len(v)] += 1
    time_mean = (acc / cnt[:, None, None, None]).astype(np.float32)
    print(f"time mean built from {nt} training episodes in {time.time() - t0:.0f}s", flush=True)

    result = {"model": args.model_name, "env": env, "ckpt_epoch": epoch, "num_hist": H, "frameskip": f, "window_frames": K,
              "predicted_per_window": S, "n_cuts": n_cuts, "obs_hw": list(hw), "sets": {}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    for name, Sset in sets.items():
        t_set = time.time()
        base, idxs = unwrap(Sset)
        n_ep = len(idxs) if not args.max_eps else min(args.max_eps, len(idxs))
        idxs = idxs[:n_ep]
        groups, names = episode_groups(env, name, base, idxs, ds_cfg.data_path, ds_kw)
        res = {"n_episodes": int(n_ep), "group_sizes": {g: int(len(v)) for g, v in groups.items()}}

        # ---------- M1: one step on slices ----------
        sl = slicers[name]
        pick = np.linspace(0, len(sl) - 1, min(args.max_slices, len(sl))).round().astype(int)
        pick = pick[sl.slices[pick, 0] < n_ep]
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(sl, pick.tolist()), batch_size=32, shuffle=False, num_workers=8)
        m1 = {k: [] for k in ("model", "copy_last", "recon_copy", "recon_ceiling")}
        m1z = {"model": [], "copy_latent": []}
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
        m1 = {k: {m: np.concatenate([b[m][:, 0] for b in v]) for m in PIX} for k, v in m1.items()}
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
        E, W = n_ep, n_cuts
        k_full = [min(len(base.actions[i]) // f, (base.n_obs[i] - 1) // f) + 1 for i in idxs]  # frames 0, f, 2f, ... of the whole episode
        Kmax = max(k_full)
        methods = ("model", "copy_last", "recon_copy", "recon_ceiling", "copy_first")
        A = {m: {k: np.zeros((E, W, S)) for k in PIX} for m in methods}
        A5 = {m: {k: np.zeros((E, W, S)) for k in PIX} for m in ("shuffled", "zero")}
        LZ = {m: {k: np.zeros((E, W, S)) for k in ("mse", "cos")} for m in ("model", "copy_latent")}
        AC = {m: {k: np.zeros((E, W)) for k in ("mse", "masked_mse", "psnr", "ssim")} for m in ("model", "copy_first", "time_mean")}
        FC = {m: {k: np.full((E, Kmax - H), np.nan) for k in PIX} for m in ("model", "copy_first", "recon_ceiling")}
        FZ = {m: np.full((E, Kmax - H), np.nan) for m in ("model", "copy_latent")}
        frames = {m: np.zeros((E, Kmax, cfg.img_size, cfg.img_size, 3), np.uint8) for m in ("model", "recon")}
        zero_act = torch.zeros(base.action_dim * f)  # a macro action of normalized zeros = the training-mean action
        up = lambda p: F.interpolate((p.clamp(-1, 1) + 1) / 2, size=hw, mode="bilinear", align_corners=False).permute(0, 2, 3, 1).cpu().numpy()
        for e in range(E):
            ep = Episode(base, idxs[e], tf, hw)
            cuts = np.linspace(0, ep.n_obs - WINDOW, n_cuts, dtype=int)
            obs, acts = make_windows(ep, f, K, cuts)
            pred, rec, zp, zg, obs_d = rollout(model, obs, acts, H, dev)
            tgt = obs_d[:, H:]
            for m, p in (("model", pred[:, H:]), ("copy_last", obs_d[:, H - 1 : H].expand_as(tgt)), ("recon_copy", rec[:, H - 1 : H].expand_as(tgt)),
                         ("recon_ceiling", rec[:, H:]), ("copy_first", obs_d[:, :1].expand_as(tgt))):
                for k, v in pix(crit, p, tgt).items():
                    A[m][k][e] = v
            for m, z in (("model", zp[:, H:]), ("copy_latent", zg[:, H - 1 : H].expand_as(zg[:, H:]))):
                LZ[m]["mse"][e] = ((z - zg[:, H:]) ** 2).mean((2, 3)).cpu().numpy()
                LZ[m]["cos"][e] = F.cosine_similarity(z, zg[:, H:], dim=-1).mean(-1).cpu().numpy()
            # ACWM-style metrics at the ACWM obs_shape on the same predicted frames
            for w_i, t0_ in enumerate(cuts):
                fid = t0_ + f * np.arange(K)
                gt_win = ep.vis_hw[fid]
                gt = gt_win[H:]
                preds = {"model": up(pred[w_i, H:]), "copy_first": np.repeat(gt_win[:1], S, 0), "time_mean": time_mean[np.minimum(fid[H:], len(time_mean) - 1)]}
                for m, p in preds.items():
                    for k, v in acwm_metrics(p, gt, gt_win[0], gt_win).items():
                        AC[m][k][e, w_i] = v
            # action sensitivity
            _, acts_sh = make_windows(ep, f, K, cuts, act=base.actions[idxs[(e + 1) % n_ep]])
            for m, a_ in (("shuffled", acts_sh), ("zero", zero_act.expand(W, K - 1, -1))):
                p5 = rollout(model, obs, a_, H, dev)[0]
                for k, v in pix(crit, p5[:, H:], tgt).items():
                    A5[m][k][e] = v
            # M3: whole episode from the first H frames
            kf = k_full[e]
            fo = ep.vis[: f * (kf - 1) + 1 : f][None]
            fa = ep.act[: f * (kf - 1)].reshape(1, kf - 1, -1)
            fp, fr, fzp, fzg, fobs = rollout(model, fo, fa, H, dev)
            for m, p in (("model", fp[:, H:]), ("copy_first", fobs[:, :1].expand_as(fobs[:, H:])), ("recon_ceiling", fr[:, H:])):
                for k, v in pix(crit, p, fobs[:, H:]).items():
                    FC[m][k][e, : kf - H] = v[0]
            FZ["model"][e, : kf - H] = ((fzp[:, H:] - fzg[:, H:]) ** 2).mean((2, 3))[0].cpu().numpy()
            FZ["copy_latent"][e, : kf - H] = ((fzg[:, H - 1 : H] - fzg[:, H:]) ** 2).mean((2, 3))[0].cpu().numpy()
            frames["model"][e, :kf] = to_u8(fp[0].cpu())
            frames["recon"][e, :kf] = to_u8(fr[0].cpu())
            if e % 20 == 0:
                print(f"[{name}] episode {e}/{E} {time.time() - t_set:.0f}s", flush=True)

        # ---------- summaries ----------
        k_full = np.array(k_full)
        last = k_full - H - 1  # index of the last predicted step of each episode
        res["m2"], res["m2_curve"], res["acwm"], res["m5"], res["m2_latent"], res["m3"], res["m3_curve"] = {}, {}, {}, {}, {}, {}, {}
        for g, mem in groups.items():
            res["m2"][g] = {m: {k: boot(A[m][k][mem].mean((1, 2))) for k in PIX} for m in methods}
            res["m2"][g]["skill_vs_recon_copy"] = {}
            for k in ("lpips", "mse"):  # how much of the gap between recon_copy and the recon ceiling the model closes
                num = A["recon_copy"][k][mem].mean((1, 2)) - A["model"][k][mem].mean((1, 2))
                den = A["recon_copy"][k][mem].mean((1, 2)) - A["recon_ceiling"][k][mem].mean((1, 2))
                res["m2"][g]["skill_vs_recon_copy"][k] = boot_ratio(num, den) if len(mem) else [None, None, None]
            res["m2_curve"][g] = {m: {k: A[m][k][mem].mean((0, 1)).tolist() for k in PIX} for m in methods} if len(mem) else {}
            res["m2_latent"][g] = {m: {k: boot(LZ[m][k][mem].mean((1, 2))) for k in LZ[m]} for m in LZ}
            res["m5"][g] = {"true": {k: boot(A["model"][k][mem].mean((1, 2))) for k in PIX}}
            for m in A5:
                res["m5"][g][m] = {k: boot(A5[m][k][mem].mean((1, 2))) for k in PIX}
            res["acwm"][g] = {m: {k: boot(AC[m][k][mem].mean(1)) for k in AC[m]} for m in AC}
            res["acwm"][g]["ratio_to_copy_first"] = {m: {k: boot_ratio(AC[m][k][mem].mean(1), AC["copy_first"][k][mem].mean(1)) if len(mem) else [None, None, None]
                                                         for k in ("mse", "masked_mse")} for m in ("model", "time_mean")}
            res["m3"][g] = {m: {k: boot(np.nanmean(FC[m][k][mem], 1)) for k in PIX} for m in FC}
            res["m3"][g]["latent_mse"] = {m: boot(np.nanmean(FZ[m][mem], 1)) for m in FZ}
            res["m3"][g]["last_step"] = {m: {k: boot(FC[m][k][mem, last[mem]]) for k in PIX} for m in FC}
            res["m3"][g]["last_step"]["latent_mse"] = {m: boot(FZ[m][mem, last[mem]]) for m in FZ}
            res["m3_curve"][g] = {m: {k: np.nanmean(FC[m][k][mem], 0).tolist() for k in PIX} for m in FC} if len(mem) else {}
            if len(mem):
                res["m3_curve"][g]["latent_mse"] = {m: np.nanmean(FZ[m][mem], 0).tolist() for m in FZ}
                res["m3_curve"][g]["n"] = np.isfinite(FZ["model"][mem]).sum(0).tolist()
        res["seconds"] = round(time.time() - t_set, 1)
        result["sets"][name] = res
        np.savez_compressed(f"{args.out_prefix}_{name}.npz", names=np.array(names), k_full=k_full, frameskip=f, num_hist=H,
                            frames_model=frames["model"], frames_recon=frames["recon"],
                            **{f"group_{g}": v for g, v in groups.items()},
                            **{f"m3_{m}_{k}": FC[m][k] for m in FC for k in PIX}, **{f"m3_latent_{m}": FZ[m] for m in FZ},
                            **{f"m2_{m}_{k}": A[m][k] for m in methods for k in PIX}, **{f"m5_{m}_{k}": A5[m][k] for m in A5 for k in PIX},
                            **{f"acwm_{m}_{k}": AC[m][k] for m in AC for k in AC[m]})
        print(f"[{name}] done {res['seconds']}s", flush=True)

    json.dump(result, open(f"{args.out_prefix}.json", "w"), indent=1)
    print_summary(result)
    print("wrote", f"{args.out_prefix}.json")


if __name__ == "__main__":
    main()
