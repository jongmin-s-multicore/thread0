"""3번 실험 (InD 로 더 학습) 의 Pour Water: 원래 모델(6 epoch)과 InD 로 이어 학습한 모델(12, 24, 48 epoch = 2x, 4x, 8x)을
InD test, OoD test 의 모든 에피소드에서 잰다. 양은 발표 도구 talk_render.py 와 같은 방법이다: 받는 상자 안의 물 픽셀을
마지막 5초(프레임 250-295) 동안 평균한 값 (Real / Model 25-step / Model rollout). Cloth/Sand/Stack 의 long 평가
(trend_acwm_phys.py) 처럼 epoch 마다 JSON 하나를 쓴다: <out_dir>/water_ep<E>.json.

usage (cwd = $DINO_WM, env sourced):
  python <this> <out_dir> [--runs train_pw_h1f5:6,train_pw_h1f5_long:12,...] [--max_eps N]
"""
import os, sys, json, argparse, time
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import talk_render as TR

RUNS = "train_pw_h1f5:6,train_pw_h1f5_long:12,train_pw_h1f5_long:24,train_pw_h1f5_long:48"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--max_eps", type=int, default=0)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = "cuda"
    spec = TR.SPEC["water"]
    cfg = OmegaConf.load(TR.CKPT / spec["base"][0] / "hydra.yaml")
    H, f = cfg.num_hist, cfg.frameskip
    sets = {}
    for sp, (d, groups, tf) in TR.load_sets("water", cfg).items():
        if args.max_eps:
            groups = {g: list(m)[: args.max_eps] for g, m in groups.items()}
        sets[sp] = (d, groups, tf)
    os.makedirs(args.out_dir, exist_ok=True)
    for item in args.runs.split(","):
        run, ep = item.split(":")[0], int(item.split(":")[1])
        t0 = time.time()
        model = TR.load(run, ep, cfg, dev)
        M = {}
        for sp, (d, groups, tf) in sets.items():
            for e in sorted({int(x) for m in groups.values() for x in m}):
                vis, fa = TR.episode("water", d, tf, e, f)
                M[(sp, e)] = TR.measures("water", TR.frames_of(model, vis, fa, H, dev, recon=False))
        r = TR.summarize("water", M, sets, H, None)
        res = {"model": run, "env": spec["env"], "epochs": [ep], "quantity": "water pixels in the receiving box, mean over frames 250-295",
               "groups": {g: v["all"]["n"] for g, v in r.items()},
               "box_fill": {g: {kd: v["all"]["box"][kd]["fill"] for kd in ("real", "pred", "rollout")} for g, v in r.items()},
               "summary": r}
        out = os.path.join(args.out_dir, f"water_ep{ep}.json")
        json.dump(res, open(out, "w"), indent=1)
        print(f"{run} epoch {ep} {time.time() - t0:.0f}s -> {out}", flush=True)
        for g, v in res["box_fill"].items():
            print(f"  {g:14s} n={res['groups'][g]:3d}  real {v['real']:7.0f}  25-step {v['pred']:7.0f}  rollout {v['rollout']:7.0f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
