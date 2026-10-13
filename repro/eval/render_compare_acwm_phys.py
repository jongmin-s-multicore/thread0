"""여러 모델의 전체 에피소드 롤아웃을 같은 에피소드에서 나란히 보여 준다: Real | 모델 1 | 모델 2 ...
OoD 에피소드를 학습에 넣은 모델(predictor_long.py --extra_split)과 원래 모델을, 학습에 쓰지 않은 OoD 에피소드에서 비교하는 용도다.
CPU 로도 돈다 (--device cpu). 물체 픽셀 수(Cloth Move 의 천, Push Sand 의 모래)를 각 칸의 라벨에 적는다.

--ahead K 를 주면 모델마다 칸이 둘이 된다. "<모델> <K*frameskip>-step" 은 매 시점마다 K 모델 스텝 전의 실제 프레임을 주고
K 스텝을 이어 예측한 것이다 (frameskip 5, K 5 면 25 프레임 앞 = DINO-WM 의 계획 길이). 앞쪽 프레임은 그만큼 이전의 실제 프레임이 없어서
첫 프레임에서 시작한 예측을 보여 준다. "<모델> rollout" 은 처음 num_hist 프레임만 주고 끝까지 모델의 예측을 이어 간 것이다.
--ahead 를 주지 않으면 rollout 만 그린다.

그룹마다 에피소드 N 개를 하나씩 따로 <out_dir>/episodes/<set>_<group>_<순위>_<에피소드>.mp4 로 쓴다.
순위는 마지막 열 모델의 픽셀 수 오차(스텝 평균)가 작은 쪽이 01 이고, 그룹 안에서 분위가 고르게 퍼지도록 고른다.
그룹마다 분위 10 %, 50 %, 90 % 에피소드 3개를 세로로 붙인 overview_<set>_<group>.mp4 도 쓴다 (--gif 를 주면 GIF 로도).

usage (cwd = $DINO_WM, env sourced):
  python repro/eval/render_compare_acwm_phys.py <ckpt_base_path> <out_dir> --models "Original=train_cloth_h1f5:15,With OoD=train_cloth_h1f5_oodmix:15"
         [--ahead 5] [--sets ind_test,ood_test] [--held_out <model>_split.json] [--per_episode 10] [--scale 2] [--fps 3] [--gif] [--device cuda]
"""
import os, sys, json, argparse
import numpy as np
import torch
import hydra
import imageio
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import load_model
from utils import seed
from datasets.acwm_phys_dset import load_acwm_phys_split
from eval_pour_water import to_u8
from eval_acwm_phys import episode_groups
from eval_acwm_phys_quant import measure, AREAS
from render_acwm_phys import show, write_gif, spread, QUANTILES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_base"); ap.add_argument("out_dir")
    ap.add_argument("--models", required=True, help="comma-separated <label>=<model_name>:<epoch>; all models must share the env and frameskip")
    ap.add_argument("--sets", default="ind_test,ood_test")
    ap.add_argument("--held_out", default="", help="<model>_split.json of predictor_long.py: in its extra split, use only the episodes that were not used for training")
    ap.add_argument("--ahead", type=int, default=0, help="model steps of the fixed-horizon column (0 = no such column)")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--per_episode", type=int, default=10)
    ap.add_argument("--max_eps", type=int, default=0)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    models = [(m.split("=")[0], m.split("=")[1].split(":")[0], int(m.split(":")[1])) for m in args.models.split(",")]
    base = Path(args.ckpt_base) / "outputs"
    cfg = OmegaConf.load(base / models[0][1] / "hydra.yaml")
    env, H, f = cfg.env.name, cfg.num_hist, cfg.frameskip
    obj = AREAS[env][0] if env != "acwm_stack_cube" else None
    ds_cfg = cfg.env.dataset
    ds_kw = dict(n_rollout=ds_cfg.n_rollout, exclude=list(ds_cfg.exclude), normalize_action=cfg.normalize_action,
                 action_dims=None if ds_cfg.action_dims is None else list(ds_cfg.action_dims), action_format=ds_cfg.action_format)
    tf = hydra.utils.instantiate(ds_cfg.transform)

    data = {}
    for name in args.sets.split(","):
        d = load_acwm_phys_split(tf, ds_cfg.data_path, name, **ds_kw)
        n = len(d) if not args.max_eps else min(args.max_eps, len(d))
        groups, _ = episode_groups(env, name, d, list(range(n)))
        if len(groups) > 1 and name == "ood_test":
            groups.pop("all")  # the OoD groups are shown separately
        if args.held_out:
            sp = json.load(open(args.held_out))
            if name == sp["extra_split"]:
                ok = {x for v in sp["held_out"].values() for x in v}
                groups = {g: np.array([i for i in mem if d.video_names[i] in ok], dtype=int) for g, mem in groups.items()}
        data[name] = (d, {g: mem for g, mem in groups.items() if len(mem)})

    K = args.ahead
    frames = {}  # (set, episode, column label) -> (k, h, w, 3) uint8
    for label, run, epoch in models:
        seed(cfg.training.seed)
        model = load_model(base / run / "checkpoints" / f"model_{epoch}.pth", cfg, cfg.num_action_repeat, args.device)
        model.eval()
        for name, (d, groups) in data.items():
            for e in sorted({int(i) for mem in groups.values() for i in mem}):  # one episode in memory at a time
                k = min(len(d.actions[e]) // f, (d.n_obs[e] - 1) // f) + 1
                vis = d.get_frames(e, range(0, f * (k - 1) + 1, f))[0]["visual"]
                fa = d.actions[e][: f * (k - 1)].reshape(1, k - 1, -1)
                z_obses, _ = model.rollout({"visual": vis[None, :H].to(args.device), "proprio": torch.zeros(1, H, 1, device=args.device)}, fa.to(args.device))
                full = model.decode_obs(z_obses)[0]["visual"][0]
                frames[(name, e, f"{label} rollout" if K else label)] = to_u8(full.cpu())
                if K:  # frame t from the real frames t-K-H+1 .. t-K; the first H+K-1 frames have no such start and come from the rollout above
                    nw = k - H - K + 1
                    io, ia = torch.arange(H)[None] + torch.arange(nw)[:, None], torch.arange(H + K - 1)[None] + torch.arange(nw)[:, None]
                    zk, _ = model.rollout({"visual": vis[io].to(args.device), "proprio": torch.zeros(nw, H, 1, device=args.device)}, fa[0][ia].to(args.device))
                    pred = model.decode_obs({k_: v[:, -1:] for k_, v in zk.items()})[0]["visual"][:, 0]
                    frames[(name, e, f"{label} {K * f}-step")] = to_u8(torch.cat([full[: H + K - 1], pred]).cpu())
                frames.setdefault((name, e, "Real"), to_u8(vis))
        print(f"rolled out {label} ({run} epoch {epoch})", flush=True)

    cols = ["Real"] + [f"{m[0]} {kind}" for m in models for kind in (f"{K * f}-step", "rollout")] if K else ["Real"] + [m[0] for m in models]
    px = {key: np.array([measure(env, x)[0][obj][0] for x in v]) for key, v in frames.items()} if obj else {}

    def tile(name, e, k, labelled=True):
        cells = []
        for c in cols:
            text = (f"{c}  frame {k * f}" if c == "Real" else c) + (f"  {int(px[(name, e, c)][k]):,} px" if obj else "")
            cells.append(show(frames[(name, e, c)][k], env, text if labelled else None, args.scale))
        return np.concatenate(cells, 1)

    def clip(rows):
        return [rows[0]] * 2 + rows + [rows[-1]] * (2 * args.fps)  # hold the first and the last frame

    out = os.path.join(args.out_dir, "episodes")
    os.makedirs(out, exist_ok=True)
    picks, summary = {}, {}  # summary: group mean of the object pixels over all episodes of the group, not only the rendered ones
    for name, (d, groups) in data.items():
        for g, mem in groups.items():
            if obj:
                t0 = H + K - 1 if K else H  # frames before t0 of the fixed-horizon column are not K steps ahead
                summary[f"{name}/{g}"] = {"n": int(len(mem)), "last_frame": {c: float(np.mean([px[(name, int(e), c)][-1] for e in mem])) for c in cols},
                                          f"frames_from_{t0 * f}": {c: float(np.mean([px[(name, int(e), c)][t0:].mean() for e in mem])) for c in cols}}
            err = np.array([np.abs(px[(name, int(e), cols[-1])] - px[(name, int(e), "Real")]).mean() if obj else 0.0 for e in mem])
            order = mem[np.argsort(err, kind="stable")]
            eps = spread(order, args.per_episode)
            picks[f"{name}/{g}"] = [{"rank": r + 1, "episode": d.video_names[e], **({f"{obj}_px_last": {c: int(px[(name, e, c)][-1]) for c in cols}} if obj else {})} for r, e in enumerate(eps)]
            for r, e in enumerate(eps):
                video = clip([tile(name, e, k) for k in range(len(frames[(name, e, "Real")]))])
                stem = os.path.splitext(d.video_names[e])[0]
                imageio.mimsave(os.path.join(out, f"{name}_{g}_{r + 1:02d}_{stem}.mp4"), video, fps=args.fps, macro_block_size=2)
            if len(order) >= 3:
                three = [int(order[int(round(qt * (len(order) - 1)))]) for qt in QUANTILES]
                n_fr = min(len(frames[(name, e, "Real")]) for e in three)
                video = clip([np.concatenate([tile(name, e, k) for e in three], 0) for k in range(n_fr)])
                stem = os.path.join(args.out_dir, f"overview_{name}_{g}")
                imageio.mimsave(stem + ".mp4", video, fps=args.fps, macro_block_size=2)
                if args.gif:
                    write_gif(video, stem + ".gif", args.fps)
    json.dump({"columns": cols, "models": {m[0]: f"{m[1]} epoch {m[2]}" for m in models}, "ahead_model_steps": K, "summary": summary, "picks": picks}, open(os.path.join(out, "picks.json"), "w"), indent=1)
    print("wrote", args.out_dir, "\n", json.dumps({"summary": summary, "picks": picks}, indent=1))


if __name__ == "__main__":
    main()
