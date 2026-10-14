"""2번 실험 (InD + OoD 섞어 학습) 의 Push Sand, Pour Water, Stack Cube: InD 만 학습한 원래 모델과 OoD 를 섞어 학습한 모델을 같은 에피소드에서 잰다.
OoD test 는 학습에 넣지 않은 에피소드(<split.json> 의 held_out)만, InD test 는 전부. 양은 발표 도구 talk_render.py 와 같은 방법이다
(Real / Model 25-step / Model rollout. 모래는 모래 픽셀, 물은 받는 상자 안의 물 픽셀, Stack 은 쌓인 것으로 판정된 비율).

usage (cwd = $DINO_WM, env sourced):
  python <this> {sand,water,stack} <split.json> <out.json> [--mix_epoch N] [--max_eps N]
"""
import os, sys, json, argparse, time
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import talk_render as TR

MIX = {"sand": ("train_sand_h1f5_oodmix", 18), "water": ("train_pw_h1f5_oodmix", 6), "stack": ("train_stack_h1f5_oodmix", 25)}
QTY = {"sand": ("px", "7.0f"), "water": ("box", "7.0f"), "stack": ("stacked", "7.2f")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key", choices=list(MIX)); ap.add_argument("split"); ap.add_argument("out")
    ap.add_argument("--mix_epoch", type=int, default=0)
    ap.add_argument("--max_eps", type=int, default=0)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = "cuda"
    spec = TR.SPEC[args.key]
    cfg = OmegaConf.load(TR.CKPT / spec["base"][0] / "hydra.yaml")
    H, f = cfg.num_hist, cfg.frameskip
    held = json.load(open(args.split))
    keep = {n for v in held["held_out"].values() for n in v}
    sets = {}
    for sp, (d, groups, tf) in TR.load_sets(args.key, cfg).items():
        if sp == held["extra_split"]:
            groups = {g: [int(e) for e in m if d.video_names[e] in keep] for g, m in groups.items()}
        if args.max_eps:
            groups = {g: list(m)[: args.max_eps] for g, m in groups.items()}
        sets[sp] = (d, {g: m for g, m in groups.items() if len(m)}, tf)
    mix = (MIX[args.key][0], args.mix_epoch or MIX[args.key][1])
    res = {"env": spec["env"], "split": args.split, "models": {"ind_only": list(spec["base"]), "ind_ood": list(mix)},
           "groups": {f"{sp}/{g}": len(m) for sp, (_, gs, _) in sets.items() for g, m in gs.items()}, "result": {}}
    for name, (run, ep) in res["models"].items():
        t0 = time.time()
        model = TR.load(run, ep, cfg, dev)
        M = {}
        for sp, (d, groups, tf) in sets.items():
            for e in sorted({int(x) for m in groups.values() for x in m}):
                vis, fa = TR.episode(args.key, d, tf, e, f)
                M[(sp, e)] = TR.measures(args.key, TR.frames_of(model, vis, fa, H, dev, recon=False))
        res["result"][name] = TR.summarize(args.key, M, sets, H, None)
        print(f"{name} ({run} epoch {ep}) {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    q, fm = QTY[args.key]
    print(f"{q}, last frame (mean over episodes)    real | ind_only 25-step rollout | ind_ood 25-step rollout")
    for grp, n in res["groups"].items():
        a, b = res["result"]["ind_only"][grp]["all"][q], res["result"]["ind_ood"][grp]["all"][q]
        print(f"  {grp:16s} n={n:3d}  {a['real']['last']:{fm}} | {a['pred']['last']:{fm}} {a['rollout']['last']:{fm}} | {b['pred']['last']:{fm}} {b['rollout']['last']:{fm}}")
    json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
