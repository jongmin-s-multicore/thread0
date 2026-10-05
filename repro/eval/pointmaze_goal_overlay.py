"""PointMaze 평가 영상에 목표 위치(state_g[:2])를 겹쳐 그린다 (이미 저장된 영상용, 다시 평가하지 않는다).

PointMaze 환경은 에이전트(초록 점)만 그리고 목표 마커는 화면 밖에 둔다 (with_target=False). 평가 영상
(planning/evaluator.py)은 [실제 환경 | 목표] / [월드모델 예측 | 목표] 2x2 배치이고, 목표는 오른쪽 칸(목표 상태를 렌더한
이미지)에서만 보인다. 여기서는 실행 디렉토리의 plan_targets.pkl 에서 state_g 를 읽어(영상 번호 i = 평가 i 번째) 왼쪽 두 칸에
목표 위치를 중심으로 성공 반경(0.5) 빨간 원과 안쪽으로 짧은 눈금 넷을 그린다 (env/pointmaze/goal_marker.py, 새로 평가하면
plan.py 가 같은 표시를 그린다). env 는 물리 하위 스텝(0.01 초) 하나 전 위치를 그리므로, 성공 판정 경계(0.5) 가까이에서 끝난
에피소드는 마지막 프레임의 초록 점이 원 바로 안팎에 보일 수 있다 (판정은 상태로 한다).
목표 칸의 초록 점과 첫 프레임 실제 환경 칸의 초록 점을 찾아 state_g·state_0 의 예상 픽셀과 맞는지 확인하고, 어긋나면 멈춘다.
예전 영상에서는 MPC 에서 성공한 에피소드가 그 뒤 프레임이 가려져(회색) 배치의 가장 긴 길이까지 이어지므로, 가려지기 전 마지막 프레임에서
--hold-sec 초 멈춘 뒤 끝낸다. 칸마다 라벨(Real = 실제 환경, Model = 월드모델 예측, Goal = 목표)을 단다
(현재 evaluator.py 가 새로 저장하는 영상과 같은 배치·표시·선 굵기 비율이고, 크기만 2배다).

usage (cwd = $DINO_WM, env sourced): python repro/eval/pointmaze_goal_overlay.py <run_dir> ... [--glob 'output_final_*.mp4'] [--out-subdir goal_overlay] [--hold-sec 3]
출력: <실행 디렉토리>/<out-subdir>/<영상 이름>.mp4 (2배 확대), checks.json (영상별 목표·시작 위치, 초록 점과의 픽셀 오차, 프레임 수.
다시 돌리면 항목을 더하거나 갱신한다)
"""
import os, re, sys, json, glob, pickle, argparse
import numpy as np
import cv2
import imageio

sys.path.insert(0, os.getcwd())
from env.pointmaze.goal_marker import state_to_px, goal_marker_alpha, blend_marker
from planning.evaluator import label_video_panels, VIDEO_HOLD_SEC
from video_format import is_labeled, count_valid_frames  # repro/eval/video_format.py

MAX_PX_ERR = 2.0  # 초록 점 무게중심과 예상 픽셀의 허용 오차 (224 칸 기준. 렌더 대조 최대 0.26 px + h264 잡음)


def agent_centroid(panel):
    """패널 (H, W, 3) uint8 에서 초록 점(에이전트 site, rgba 0.3 0.6 0.3)의 무게중심 (x, y), 픽셀 수.
    점은 G - max(R, B) 가 대부분 55 이상(중심 80-100, evaluator 가 38 어둡게 한 칸에서도)이고 바닥(파랑)·벽(주황)은 10 아래라
    임계값 25 로 가른다. h264 가 벽 모서리에 남기는 작은 초록 얼룩이 섞이지 않게 가장 큰 연결 덩어리만 쓴다."""
    im = panel.astype(int)
    mask = (im[..., 1] - np.maximum(im[..., 0], im[..., 2]) > 25).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return None, 0
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    n_px = int(stats[k, cv2.CC_STAT_AREA])
    if n_px < 20:
        return None, n_px
    return (float(cents[k, 0]), float(cents[k, 1])), n_px






def overlay_video(src, dst, goal_xy, init_xy, up=2, color=(255, 0, 0), hold_sec=VIDEO_HOLD_SEC):
    with imageio.get_reader(src) as reader:
        fps = reader.get_meta_data().get("fps", 12)
        first = reader.get_data(0)
    h, w = first.shape[:2]
    assert h == w and h % 2 == 0, f"2x2 배치(정사각) 영상이 아니다: {first.shape}"
    p = h // 2
    if is_labeled(first):
        return {"skipped": "already labeled (new evaluator format; plan.py draws the goal marker there)"}

    # 영상 번호와 plan_targets 의 순서가 맞는지: 목표 칸(아래 오른쪽)의 점 = state_g, 첫 프레임 실제 환경 칸의 점 = state_0
    checks = {}
    for name, sub, xy in [("goal", first[p:, p:], goal_xy), ("init", first[:p, :p], init_xy)]:
        c, n_px = agent_centroid(sub)
        if c is None:
            raise ValueError(f"{src}: {name} 칸에서 초록 점을 찾지 못했다 (픽셀 {n_px})")
        err = float(np.hypot(*(np.array(c) - np.array(state_to_px(xy, p)))))
        if err > MAX_PX_ERR * p / 224:
            raise ValueError(f"{src}: {name} 칸의 초록 점 {np.round(c, 1)} 이 state 의 예상 픽셀 "
                             f"{np.round(state_to_px(xy, p), 1)} 과 {err:.1f} px 어긋난다 (영상과 plan_targets 순서가 다르다?)")
        checks[f"{name}_px_err"] = round(err, 3)
    n_valid, n_src = count_valid_frames(src, p)
    alpha = goal_marker_alpha(p, goal_xy, thickness=up, upsample=up)  # 2배 확대 칸 크기, 선 굵기 = 원래 1 px

    writer = imageio.get_writer(dst, fps=fps, macro_block_size=1)
    with imageio.get_reader(src) as reader:
        for i, frame in enumerate(reader):
            if i >= n_valid:
                break
            big = cv2.resize(frame, (w * up, h * up), interpolation=cv2.INTER_LINEAR)
            for oy, ox in [(0, 0), (p, 0)]:  # 왼쪽 위: 실제 환경, 왼쪽 아래: 월드모델 예측
                sub = big[oy * up:(oy + p) * up, ox * up:(ox + p) * up]
                blend_marker(sub, alpha, color)
            big = label_video_panels(big)  # 칸마다 Real / Model / Goal
            writer.append_data(big)
    n_hold = int(round(hold_sec * fps))
    for _ in range(n_hold):  # 마지막 프레임(성공한 장면 또는 실패로 끝난 장면)에서 멈춘다
        writer.append_data(big)
    writer.close()
    return {"goal_xy": [round(float(v), 4) for v in goal_xy], "init_xy": [round(float(v), 4) for v in init_xy],
            **checks, "frames_src": n_src, "frames_valid": n_valid, "frames_hold": n_hold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="실행 디렉토리 (plan_targets.pkl 이 있는 곳)")
    ap.add_argument("--glob", default="output_final_*.mp4", help="실행 디렉토리 안에서 고를 영상 (plan<i>_*.mp4 도 된다)")
    ap.add_argument("--out-subdir", default="goal_overlay")
    ap.add_argument("--hold-sec", type=float, default=VIDEO_HOLD_SEC, help="마지막 프레임에서 멈추는 시간 (초)")
    args = ap.parse_args()

    for run in args.paths:
        targets = os.path.join(run, "plan_targets.pkl")
        if not os.path.exists(targets):
            sys.exit(f"{run}: plan_targets.pkl 이 없다 (실행 디렉토리를 준다)")
        with open(targets, "rb") as f:
            t = pickle.load(f)
        state_0, state_g = np.asarray(t["state_0"]), np.asarray(t["state_g"])
        videos = sorted(glob.glob(os.path.join(run, args.glob)))
        if not videos:
            print(f"{run}: 영상 없음 ({args.glob})")
            continue
        out_dir = os.path.join(run, args.out_subdir)
        if os.path.realpath(out_dir) == os.path.realpath(run):
            sys.exit(f"{run}: 출력이 원본과 같은 디렉토리다 (--out-subdir 를 하위 디렉토리로)")
        os.makedirs(out_dir, exist_ok=True)
        checks_path = os.path.join(out_dir, "checks.json")
        checks = json.load(open(checks_path)) if os.path.exists(checks_path) else {}
        for v in videos:
            m = re.search(r"_(\d+)_(success|failure)\.mp4$", os.path.basename(v))
            if m is None:
                print(f"{v}: 이름에서 평가 번호를 읽지 못해 건너뜀")
                continue
            idx = int(m.group(1))
            if idx >= len(state_g):
                sys.exit(f"{v}: 평가 번호 {idx} 가 plan_targets.pkl 의 목표 수 {len(state_g)} 를 넘는다 (다른 실행의 영상?)")
            dst = os.path.join(out_dir, os.path.basename(v))
            if os.path.realpath(dst) == os.path.realpath(v):
                sys.exit(f"{v}: 출력이 원본과 같은 파일이다 (--glob 이 출력 디렉토리 안 영상을 골랐다?)")
            res = overlay_video(v, dst, state_g[idx, :2], state_0[idx, :2], hold_sec=args.hold_sec)
            if "skipped" in res:
                print(f"{v}: 건너뜀 ({res['skipped']})")
                continue
            checks[os.path.basename(v)] = res
            print(f"{v}: goal={res['goal_xy']} px_err goal {res['goal_px_err']} init {res['init_px_err']} "
                  f"frames {res['frames_valid']}/{res['frames_src']} + hold {res['frames_hold']}")
            with open(checks_path, "w") as f:
                json.dump(checks, f, indent=1)
        print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
