"""Push Sand 디코더를 더 학습했을 때: 디코더만 바꾼 체크포인트들로 Real / Encoder->Decoder / Model 25-step / Model rollout 의 모래 픽셀을 잰다.
InD test, OoD test 전부. OoD test 는 <split.json> 의 held-out 에피소드(OoD 섞어 학습할 때 쓰지 않은 것)만 따로도 낸다.

usage (cwd = $DINO_WM, env sourced):
  python <this> <split.json> <out.json> <label>=<run>:<epoch> [...]
"""
import os, sys, json, time
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import talk_render as TR


def main():
    split, out, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
    torch.set_grad_enabled(False)
    key, dev = "sand", "cuda"
    cfg = OmegaConf.load(TR.CKPT / TR.SPEC[key]["base"][0] / "hydra.yaml")
    H, f = cfg.num_hist, cfg.frameskip
    held = json.load(open(split))
    sets = TR.load_sets(key, cfg)
    res = {"split": split, "models": {}, "result": {}}
    for s in specs:
        label, rest = s.split("=")
        run, ep = rest.split(":")
        res["models"][label] = [run, int(ep)]
        t0 = time.time()
        model = TR.load(run, int(ep), cfg, dev)
        M = {}
        for sp, (d, groups, tf) in sets.items():
            for e in sorted({int(x) for m in groups.values() for x in m}):
                vis, fa = TR.episode(key, d, tf, e, f)
                M[(sp, e)] = TR.measures(key, TR.frames_of(model, vis, fa, H, dev, recon=True))
        r = TR.summarize(key, M, sets, H, held)
        res["result"][label] = r
        print(f"{label} ({run} epoch {ep}) {time.time() - t0:.0f}s  sand px, last frame: real | recon | 25-step | rollout", flush=True)
        for grp, subs in r.items():
            for sub, v in subs.items():
                p = v["px"]
                print(f"  {grp:14s} {sub:8s} n={v['n']:3d}  {p['real']['last']:6.0f} | {p['recon']['last']:6.0f} | {p['pred']['last']:6.0f} | {p['rollout']['last']:6.0f}", flush=True)
        del model
        torch.cuda.empty_cache()
        json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
