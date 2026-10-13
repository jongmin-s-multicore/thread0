"""루트 README 의 ACWM-Phys 그림을 만든다: 환경마다 Real | Model 전체 에피소드 롤아웃 GIF (224 px 칸) 를 <assets dir>/acwm_*.gif 로.
Cloth Move·Push Sand·Stack Cube 는 eval_acwm_phys.py / eval_acwm_phys_quant.py 가 저장한 프레임에서,
Pour Water 는 render_pour_water.py 가 만든 영상(정답 | A | B)의 정답·B 칸에서 만든다. 모델·GPU 를 쓰지 않는다.
면적 그림은 따로: python repro/eval/plot_acwm_phys.py <results>/acwm_phys <assets dir>/acwm_area_cloth_sand.png

usage: python repro/eval/readme_figures_acwm_phys.py $DINO_WORK/results assets
"""
import os, sys
import numpy as np
import cv2
import imageio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_acwm_phys import show, write_gif

# 그림 이름 -> (결과 prefix, 세트, 에피소드). 그 그룹에서 전체 에피소드 LPIPS 가 가운데인 에피소드 (render_acwm_phys.py --per_episode 10 의 05번).
# Stack Cube 실패 예만 08번이다: 실제로는 큐브를 놓쳤는데 모델은 쌓는 경우를 보이려고 골랐다.
EPISODES = {
    "acwm_cloth_move_ind": ("cloth_h1f5", "ind_test", "episode_00044.mp4"),
    "acwm_cloth_move_ood_larger": ("cloth_h1f5", "ood_test", "larger_episode_00029.mp4"),
    "acwm_cloth_move_ood_smaller": ("cloth_h1f5", "ood_test", "smaller_episode_00024.mp4"),
    "acwm_push_sand_ind": ("sand_h1f5", "ind_test", "episode_34.mp4"),
    "acwm_push_sand_ood": ("sand_h1f5", "ood_test", "episode_47.mp4"),
    "acwm_stack_cube_ind": ("stack_h1f5", "ind_test", "episode_11.mp4"),
    "acwm_stack_cube_ood_fail": ("stack_h1f5", "ood_test", "episode_120.mp4"),
}
POUR_WATER = {"acwm_pour_water_ind": "qual_ind_test.mp4", "acwm_pour_water_ood_low": "qual_ood_low.mp4"}
FPS = 3


def hold(frames):
    return [frames[0]] * 2 + frames + [frames[-1]] * (2 * FPS)


def main():
    res, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for name, (tag, st, episode) in EPISODES.items():
        prefix = os.path.join(res, "acwm_phys", tag)
        z, q = np.load(f"{prefix}_{st}.npz"), np.load(f"{prefix}_quant_{st}.npz")
        env = {"cloth_h1f5": "acwm_cloth_move", "sand_h1f5": "acwm_push_sand", "stack_h1f5": "acwm_stack_cube"}[tag]
        e = [str(n) for n in z["names"]].index(episode)
        f = int(z["frameskip"])
        src = {"Real": q["frames_gt"], "Model": z["frames_model"]}
        frames = [np.concatenate([show(src[s][e, k], env, f"{s}  frame {k * f}") for s in ("Real", "Model")], 1) for k in range(int(z["k_full"][e]))]
        write_gif(hold(frames), os.path.join(out, name + ".gif"), FPS)
    for name, video in POUR_WATER.items():  # panels: ground truth | A | B, each with a caption bar on top
        path = os.path.join(res, "pour_water", "figs", video)
        if not os.path.exists(path):
            print("skipped", name, "(no", path + ")")
            continue
        frames = []
        for k, fr in enumerate(imageio.get_reader(path)):
            if k % 3:  # every 3rd rollout step
                continue
            w = fr.shape[1] // 3
            panels = []
            for j, text in ((0, "Real"), (2, "Model")):
                p = cv2.resize(np.ascontiguousarray(fr[:, j * w:(j + 1) * w]), (224, 224), interpolation=cv2.INTER_AREA)
                p[:14] = 0
                cv2.putText(p, f"{text}  frame {k * 5}", (4, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                panels.append(p)
            frames.append(np.concatenate(panels, 1))
        write_gif(hold(frames), os.path.join(out, name + ".gif"), FPS)
    for fn in sorted(os.listdir(out)):
        if fn.startswith("acwm_") and fn.endswith(".gif"):
            print(f"{fn}  {os.path.getsize(os.path.join(out, fn)) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
