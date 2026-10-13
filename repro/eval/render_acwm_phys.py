"""ACWM-Phys Cloth Move / Push Sand / Stack Cube 정성 비교: 정답과 모델의 전체 에피소드 롤아웃 (처음 num_hist 프레임 + 모든 action).
eval_acwm_phys.py 와 eval_acwm_phys_quant.py 가 저장한 프레임만 읽는다 (모델·GPU 를 쓰지 않는다).

세트·그룹마다 전체 에피소드 LPIPS 의 10 %, 50 %, 90 % 분위 에피소드를 골라
  qual_<set>_<group>.gif / .mp4   행 = 에피소드 3개, 열 = Real | Model
  grid_<set>_<group>.png          중앙값 에피소드의 Real / Model / Recon 을 스텝 순서로
를 만든다. Push Sand 는 학습 입력(224x224 로 눌린 프레임)을 원래 가로세로비로 되돌려 보여 준다.

usage: python repro/eval/render_acwm_phys.py <out_prefix> <out_dir> [--sets ind_test,ood_test] [--fps 3]
"""
import os, json, argparse
import numpy as np
import cv2
import imageio

ASPECT = {"acwm_push_sand": 512 / 304}  # width / height of the source frames (the others are square)
QUANTILES = (0.1, 0.5, 0.9)


def show(frame, env, label=None, scale=1):
    h = frame.shape[0] * scale
    w = int(round(h * ASPECT.get(env, 1.0) / 2)) * 2
    im = cv2.resize(np.ascontiguousarray(frame), (w, h), interpolation=cv2.INTER_CUBIC)
    if label:
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(im, (0, 0), (tw + 8, th + base + 6), (0, 0, 0), -1)
        cv2.putText(im, label, (4, th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_prefix"); ap.add_argument("out_dir")
    ap.add_argument("--sets", default="ind_test,ood_test")
    ap.add_argument("--fps", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    info = json.load(open(f"{args.out_prefix}.json"))
    env, f = info["env"], info["frameskip"]
    picks = {}
    for name in args.sets.split(","):
        z, q = np.load(f"{args.out_prefix}_{name}.npz"), np.load(f"{args.out_prefix}_quant_{name}.npz")
        src = {"Real": q["frames_gt"], "Model": z["frames_model"], "Recon": z["frames_recon"]}
        k_full, score = z["k_full"], np.nanmean(z["m3_model_lpips"], 1)
        groups = {k[6:]: z[k] for k in z.files if k.startswith("group_")}
        if len(groups) > 1 and name == "ood_test":
            groups.pop("all")  # the OoD groups are shown separately
        for g, mem in groups.items():
            if len(mem) < 3:
                continue
            order = mem[np.argsort(score[mem])]
            eps = [int(order[int(round(qt * (len(order) - 1)))]) for qt in QUANTILES]
            picks[f"{name}/{g}"] = [{"episode": str(z["names"][e]), "lpips": float(score[e])} for e in eps]
            K = int(min(k_full[e] for e in eps))
            video = []
            for k in range(K):
                rows = [np.concatenate([show(src[s][e, k], env, f"{s}  frame {k * f}" if r == 0 else None) for s in ("Real", "Model")], 1) for r, e in enumerate(eps)]
                video.append(np.concatenate(rows, 0))
            video = [video[0]] * 2 + video + [video[-1]] * (2 * args.fps)  # hold the first and the last frame
            stem = os.path.join(args.out_dir, f"qual_{name}_{g}")
            imageio.mimsave(stem + ".gif", video, duration=1000 / args.fps, loop=0)
            imageio.mimsave(stem + ".mp4", video, fps=args.fps, macro_block_size=2)
            e = eps[1]
            steps = list(range(0, int(k_full[e]), 2))
            if steps[-1] != k_full[e] - 1:
                steps.append(int(k_full[e]) - 1)
            grid = np.concatenate([np.concatenate([show(src[s][e, k], env, f"{s}  frame {k * f}" if c == 0 else f"{k * f}") for c, k in enumerate(steps)], 1) for s in ("Real", "Model", "Recon")], 0)
            imageio.imwrite(os.path.join(args.out_dir, f"grid_{name}_{g}.png"), grid)
    json.dump(picks, open(os.path.join(args.out_dir, "picks.json"), "w"), indent=1)
    print("wrote", args.out_dir, "\n", json.dumps(picks, indent=1))


if __name__ == "__main__":
    main()
