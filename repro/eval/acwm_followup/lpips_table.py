"""4번 실험 표에 쓸 LPIPS: Encoder->Decoder, Model 25-step, Model rollout 을 실제 프레임과 비교한다 (talk_render.py 와 같은 에피소드·같은 프레임).
LPIPS(VGG, v0.1) 는 eval_acwm_phys.py 와 같은 설정이다. 224×224, [-1, 1]. 프레임은 talk_render.frames_of 의 uint8 출력을 쓴다.

에피소드마다 프레임별 값을 평균한다: from5 = 프레임 5 이후 (예측한 프레임 전체, 원래 평가의 m3 와 같은 범위), from25 = 프레임 25 이후 (25-step 예측이 있는 범위).
-> <render dir>/<env>_lpips.json (그룹 평균), <env>_lpips.npz (에피소드 × 프레임)

usage (cwd = $DINO_WM, env sourced):
  python <this> <render dir> [--keys cloth,sand,water,stack]
"""
import os, sys, json, argparse, time
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import talk_render as T
from metrics.lpipsPyTorch.modules.lpips import LPIPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("render_dir")
    ap.add_argument("--keys", default="cloth,sand,water,stack")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    crit = LPIPS("vgg", "0.1").to("cuda").eval()
    to_t = lambda u8: torch.from_numpy(u8).permute(0, 3, 1, 2).float().div(127.5).sub(1).to("cuda")
    for key in args.keys.split(","):
        t0 = time.time()
        run, epoch = T.SPEC[key]["base"]
        cfg = OmegaConf.load(T.CKPT / run / "hydra.yaml")
        f, H = cfg.frameskip, cfg.num_hist
        sets = T.load_sets(key, cfg)
        model = T.load(run, epoch, cfg, "cuda")
        per = {}  # (set, episode) -> {kind: (k,) LPIPS per frame}
        for sp, (d, groups, tf) in sets.items():
            for e in sorted({int(i) for m in groups.values() for i in m}):
                vis, fa = T.episode(key, d, tf, e, f)
                fr = T.frames_of(model, vis, fa, H, "cuda")
                real = to_t(fr["real"])
                per[(sp, e)] = {kd: T.EPW.lpips_each(crit, to_t(fr[kd]), real).cpu().numpy() for kd in ("recon", "pred", "rollout")}
        t25 = H + T.AHEAD - 1
        out = {"model": f"{run} epoch {epoch}", "lpips": "VGG v0.1, 224 px, [-1, 1]", "from5": "frames 5.. (all predicted frames)",
               "from25": "frames 25.. (frames with a 25-step prediction)", "groups": {}}
        for sp, (d, groups, _) in sets.items():
            for g, mem in groups.items():
                mem = [int(e) for e in mem]
                out["groups"][f"{sp}/{g}"] = {"n": len(mem), **{rng: {kd: float(np.mean([per[(sp, e)][kd][k0:].mean() for e in mem])) for kd in ("recon", "pred", "rollout")}
                                                              for rng, k0 in (("from5", H), ("from25", t25))}}
        json.dump(out, open(os.path.join(args.render_dir, f"{key}_lpips.json"), "w"), indent=1)
        kmax = max(len(v["recon"]) for v in per.values())
        pad = lambda a: np.concatenate([a, np.full(kmax - len(a), np.nan)])
        np.savez_compressed(os.path.join(args.render_dir, f"{key}_lpips.npz"),
                            **{f"{sp}/{kd}": np.stack([pad(per[(s, e)][kd]) for (s, e) in per if s == sp]) for sp in sets for kd in ("recon", "pred", "rollout")},
                            **{f"{sp}/names": np.array([sets[sp][0].video_names[e] for (s, e) in per if s == sp]) for sp in sets})
        print(f"[{key}] {len(per)} episodes in {time.time() - t0:.0f}s", flush=True)
        for g, r in out["groups"].items():
            print(f"  {g:18s} n={r['n']:3d} from5: " + " ".join(f"{kd} {r['from5'][kd]:.3f}" for kd in r["from5"]) + " | from25: " + " ".join(f"{kd} {r['from25'][kd]:.3f}" for kd in r["from25"]), flush=True)


if __name__ == "__main__":
    main()
