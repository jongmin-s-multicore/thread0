"""학습이 진행되면서 물리량이 어떻게 변하는지: 저장된 epoch 체크포인트마다 전체 에피소드 롤아웃을 돌려
마지막 프레임의 물체 픽셀 수(Cloth Move 의 천, Push Sand 의 모래)나 쌓임 판정(Stack Cube)을 정답과 비교한다.
"더 학습하면 OoD 에서 양이 맞아지는가"를 이미 있는 체크포인트로 보는 용도다. CPU 로도 돈다 (--device cpu).

usage (cwd = $DINO_WM, env sourced):
  python repro/eval/trend_acwm_phys.py <ckpt_base_path> <model_name> <out_json> [--epochs 5,10,15] [--sets ind_test,ood_test]
         [--device cuda] [--max_eps 0] [--held_out <model>_split.json]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import hydra
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import load_model
from utils import seed
from datasets.acwm_phys_dset import load_acwm_phys_split
from eval_pour_water import to_u8
from eval_acwm_phys import episode_groups
from eval_acwm_phys_quant import measure, AREAS, MIN_PX, STACK


def is_stacked(sc):
    y, g = sc["yellow"], sc["green"]
    if not (y[0] >= 15 and g[0] >= 15):
        return False
    return bool(abs(y[1] - g[1]) < STACK["dx"] and STACK["dy"][0] < y[2] - g[2] < STACK["dy"][1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_base"); ap.add_argument("model_name"); ap.add_argument("out_json")
    ap.add_argument("--epochs", default="")
    ap.add_argument("--sets", default="ind_test,ood_test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_eps", type=int, default=0)
    ap.add_argument("--held_out", default="", help="<model>_split.json of predictor_long.py: in its extra split, score only the episodes that were not used for training")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    model_path = Path(args.ckpt_base) / "outputs" / args.model_name
    cfg = OmegaConf.load(model_path / "hydra.yaml")
    env, H, f = cfg.env.name, cfg.num_hist, cfg.frameskip
    ds_cfg = cfg.env.dataset
    ds_kw = dict(n_rollout=ds_cfg.n_rollout, exclude=list(ds_cfg.exclude), normalize_action=cfg.normalize_action,
                 action_dims=None if ds_cfg.action_dims is None else list(ds_cfg.action_dims), action_format=ds_cfg.action_format)
    tf = hydra.utils.instantiate(ds_cfg.transform)
    epochs = [int(e) for e in args.epochs.split(",")] if args.epochs else sorted(int(p.stem.split("_")[1]) for p in (model_path / "checkpoints").glob("model_[0-9]*.pth"))
    obj = AREAS[env][0] if env != "acwm_stack_cube" else None
    result = {"model": args.model_name, "env": env, "epochs": epochs, "quantity": f"{obj} pixels in the last frame" if obj else "last frame shows the cube stacked", "sets": {}}

    data = {}
    for name in args.sets.split(","):
        d = load_acwm_phys_split(tf, ds_cfg.data_path, name, **ds_kw)
        n = len(d) if not args.max_eps else min(args.max_eps, len(d))
        groups, _ = episode_groups(env, name, d, list(range(n)))
        keep = list(range(n))
        if args.held_out:
            sp = json.load(open(args.held_out))
            if name == sp["extra_split"]:
                ok = {x for v in sp["held_out"].values() for x in v}
                keep = [i for i in range(n) if d.video_names[i] in ok]
                groups = {g: np.array([i for i in mem if i in set(keep)], dtype=int) for g, mem in groups.items()}
        kf = [min(len(d.actions[i]) // f, (d.n_obs[i] - 1) // f) + 1 for i in range(n)]
        data[name] = (d, n, kf, groups, {}, keep)  # the dict caches the measurement of the real last frame

    for epoch in epochs:
        t0 = time.time()
        seed(cfg.training.seed)
        model = load_model(model_path / "checkpoints" / f"model_{epoch}.pth", cfg, cfg.num_action_repeat, args.device)
        model.eval()
        for name, (d, n, kf, groups, gt, keep) in data.items():
            val_gt, val_m, iou = np.zeros(n), np.zeros(n), np.full(n, np.nan)
            for e in keep:  # one episode in memory at a time
                k = kf[e]
                vis = d.get_frames(e, range(0, f * (k - 1) + 1, f))[0]["visual"]
                if e not in gt:
                    gt[e] = measure(env, to_u8(vis[-1:])[0])
                fa = d.actions[e][: f * (k - 1)].reshape(1, k - 1, -1)
                z_obses, _ = model.rollout({"visual": vis[None, :H].to(args.device), "proprio": torch.zeros(1, H, 1, device=args.device)}, fa.to(args.device))
                last = model.decode_obs({k_: v[:, -1:] for k_, v in z_obses.items()})[0]["visual"]
                sc, mk = measure(env, to_u8(last[0].cpu())[0])
                if obj:
                    val_gt[e], val_m[e] = gt[e][0][obj][0], sc[obj][0]
                    union = (mk[obj] | gt[e][1][obj]).sum()
                    iou[e] = (mk[obj] & gt[e][1][obj]).sum() / union if union >= MIN_PX else np.nan
                else:
                    val_gt[e], val_m[e] = is_stacked(gt[e][0]), is_stacked(sc)
            res = result["sets"].setdefault(name, {})
            for g, mem in groups.items():
                if len(mem) == 0:
                    continue
                r = {"n": int(len(mem)), "real": float(val_gt[mem].mean()), "model": float(val_m[mem].mean())}
                if obj:
                    r["iou_last"] = float(np.nanmean(iou[mem]))
                else:
                    r["agree"] = float((val_gt[mem] == val_m[mem]).mean())
                res.setdefault(g, {})[str(epoch)] = r
        print(f"epoch {epoch} done in {time.time() - t0:.0f}s: " + "  ".join(
            f"{name}/{g} real {v[str(epoch)]['real']:.3g} model {v[str(epoch)]['model']:.3g}" for name, gs in result["sets"].items() for g, v in gs.items() if str(epoch) in v), flush=True)
        json.dump(result, open(args.out_json, "w"), indent=1)
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
