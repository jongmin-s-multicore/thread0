"""Pour Water 정성 비교: 정답 / A / B 의 전체 에피소드 롤아웃 (처음 num_hist 프레임 + 모든 action).
세트(ind_test, ood_test 낮음·높음)마다 W0 가 그 그룹 중앙값에 가장 가까운 에피소드 하나를 골라 격자 그림 1장과 나란히 보는 mp4 를 만든다.

usage (cwd = $DINO_WM, env sourced):
  python repro/eval/render_pour_water.py <ckpt_base_path> <out_dir> [--model_a train_pw_h3f5] [--model_b train_pw_h1f5] [--epoch 6]
         [--epoch_a N] [--epoch_b N] [--label_a "A  num_hist 3"] [--label_b "B  num_hist 1"]
  epoch 0 은 checkpoints/model_latest.pth 를 읽는다.
"""
import os, sys, argparse, subprocess
from pathlib import Path
import numpy as np
import torch
import cv2
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_pour_water as E
from plan import load_model
from utils import seed
from datasets.acwm_pour_water_dset import load_acwm_pour_water_split
from datasets.img_transforms import default_transform

TIMES = [3, 11, 19, 27, 35, 43, 51, 59]  # rollout step (k) of the grid columns, 0.5 s per step


def load(ckpt_base, name, epoch):
    mp = Path(ckpt_base) / "outputs" / name
    cfg = OmegaConf.load(mp / "hydra.yaml")
    model = load_model(mp / "checkpoints" / (f"model_{epoch}.pth" if epoch else "model_latest.pth"), cfg, cfg.num_action_repeat, "cuda")
    model.eval()
    return cfg, model


@torch.no_grad()
def rollout_frames(model, cfg, ep):
    f, H, K = cfg.frameskip, cfg.num_hist, 300 // cfg.frameskip
    fo = ep.vis[::f][:K][None]
    fa = ep.act[: f * (K - 1)].reshape(1, K - 1, -1)
    pred, _, _, _, obs = E.rollout(model, fo, fa, H, "cuda")
    return E.to_u8(obs[0].cpu()), E.to_u8(pred[0].cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_base"); ap.add_argument("out_dir")
    ap.add_argument("--model_a", default="train_pw_h3f5"); ap.add_argument("--model_b", default="train_pw_h1f5")
    ap.add_argument("--epoch", type=int, default=6)
    ap.add_argument("--epoch_a", type=int, default=None); ap.add_argument("--epoch_b", type=int, default=None)
    ap.add_argument("--label_a", default="A  num_hist 3"); ap.add_argument("--label_b", default="B  num_hist 1")
    args = ap.parse_args()
    ep_a = args.epoch if args.epoch_a is None else args.epoch_a
    ep_b = args.epoch if args.epoch_b is None else args.epoch_b
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    cfg_a, model_a = load(args.ckpt_base, args.model_a, ep_a)
    ds_cfg = cfg_a.env.dataset
    tf = default_transform(cfg_a.img_size)
    seed(cfg_a.training.seed)
    picks = []  # (label, set, episode index, W0)
    for sp in ("ind_test", "ood_test"):
        d = load_acwm_pour_water_split(tf, ds_cfg.data_path, sp, n_rollout=ds_cfg.n_rollout, exclude=ds_cfg.exclude, normalize_action=cfg_a.normalize_action)
        w0 = np.array([E.water_px_frame0(d, i) for i in range(len(d))])
        groups = {"ind_test": np.arange(len(d))} if sp == "ind_test" else {"ood_low": np.where(w0 < E.W0_SPLIT)[0], "ood_high": np.where(w0 >= E.W0_SPLIT)[0]}
        for g, members in groups.items():
            i = members[np.argmin(np.abs(w0[members] - np.median(w0[members])))]
            picks.append((g, sp, int(i), int(w0[i]), d))
    gt, pa, pb = {}, {}, {}
    for g, sp, i, w0, d in picks:
        ep = E.Episode(d, i, tf)
        gt[g], pa[g] = rollout_frames(model_a, cfg_a, ep)
    del model_a
    cfg_b, model_b = load(args.ckpt_base, args.model_b, ep_b)
    for g, sp, i, w0, d in picks:
        pb[g] = rollout_frames(model_b, cfg_b, E.Episode(d, i, tf))[1]

    # grid: for each picked episode three rows (ground truth, A, B), one column per time
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [(g, w0, nm, src) for g, sp, i, w0, d in picks for nm, src in (("ground truth", gt[g]), (args.label_a, pa[g]), (args.label_b, pb[g]))]
    fig, axes = plt.subplots(len(rows), len(TIMES), figsize=(len(TIMES) * 1.55 + 1.3, len(rows) * 1.55 + 0.5), gridspec_kw={"wspace": 0.03, "hspace": 0.03})
    for r, (g, w0, nm, src) in enumerate(rows):
        for c, k in enumerate(TIMES):
            ax = axes[r, c]; ax.imshow(src[k]); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(f"{k * 0.5:.1f} s", fontsize=9, color="#52514e")
        names = {"ind_test": "InD test", "ood_low": "OoD low volume", "ood_high": "OoD high volume"}
        label = f"{names[g]}\n(first-frame water {w0} px)\n{nm}" if nm == "ground truth" else nm
        axes[r, 0].set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8, color="#52514e", labelpad=6)
    fig.savefig(out / "qual_grid.png", dpi=130, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)

    # side by side videos: ground truth | A | B, 2 frames per rollout step per second would be real time; play 6 steps per second
    for g, sp, i, w0, d in picks:
        h, w = gt[g].shape[1:3]
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{3 * w * 2}x{h * 2}", "-r", "6", "-i", "-",
               "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / f"qual_{g}.mp4")]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for k in range(gt[g].shape[0]):
            panels = []
            for lab, src in (("ground truth", gt[g]), (args.label_a, pa[g]), (args.label_b, pb[g])):
                im = cv2.resize(np.ascontiguousarray(src[k]), (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
                cv2.rectangle(im, (0, 0), (w * 2, 26), (252, 252, 251), -1)
                cv2.putText(im, f"{lab}   {k * 0.5:4.1f} s", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (11, 11, 11), 1, cv2.LINE_AA)
                panels.append(im)
            p.stdin.write(np.concatenate(panels, 1).tobytes())
        p.stdin.close(); p.wait()
    print("wrote", out)


if __name__ == "__main__":
    main()
