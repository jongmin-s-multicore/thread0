"""PushT 평가 영상에 실제 목표(state_g)의 블록 윤곽을 겹쳐 그린다 (이미 저장된 영상용, 다시 평가하지 않는다).

PushT 환경은 원래 PushT 의 고정 목표(연두색 T, (256, 256, pi/4))를 항상 그리고, 학습 데이터에도 그대로 들어 있다.
planning 의 목표는 goal_source 로 뽑은 임의의 상태라서 연두색 T 와 관계없다. 평가 영상(planning/evaluator.py)은
[실제 환경 | 목표] / [월드모델 예측 | 목표] 2x2 배치이고 목표 칸은 목표 상태를 렌더한 이미지다.
여기서는 첫 프레임의 목표 칸(아래 오른쪽)에서 블록 자세 (x, y, angle) 를 맞추고, 왼쪽 두 칸에 그 윤곽을 그린다.
예전 영상에서는 MPC 에서 성공한 에피소드가 그 뒤 프레임이 가려져(회색) 배치의 가장 긴 길이까지 이어지므로, 가려지기 전 마지막 프레임에서
--hold-sec 초 멈춘 뒤 끝낸다. 칸마다 라벨(Real = 실제 환경, Model = 월드모델 예측, Goal = 목표)을 단다
(현재 evaluator.py 가 새로 저장하는 영상과 같은 형식).

usage (cwd = $DINO_WM, env sourced): python repro/eval/pusht_goal_overlay.py <run_dir|video.mp4> ... [--glob 'output_final_*.mp4'] [--out-subdir goal_overlay] [--hold-sec 3]
출력: <영상 디렉토리>/<out-subdir>/<영상 이름>.mp4 (2배 확대, 빨간 윤곽 = 목표 자세), fits.json (영상별 추정 자세, 맞춤 IoU. 다시 돌리면 항목을 더하거나 갱신한다)
"""
import os, sys, json, glob, argparse
import numpy as np
import cv2
import imageio
import shapely.geometry
import shapely.ops

sys.path.insert(0, os.getcwd())
from env.pusht.goal_outline import block_local_polygons, block_outline_px, draw_block_outline, WINDOW_SIZE
from planning.evaluator import label_video_panels, VIDEO_HOLD_SEC
from video_format import is_labeled, count_valid_frames  # repro/eval/video_format.py

# evaluator._plot_rollout_compare: [-1, 1] 공간에서 correction 0.3 을 빼고 ((x + 1) / 2) * 255 로 저장 -> 픽셀값 38.25 만큼 어둡다.
# 아래 오른쪽(목표) 칸은 한 번, 위 오른쪽 칸은 두 번 빠진다.
CORRECTION_PX = 0.3 * 127.5
# PushTEnv 렌더 색 (pygame DrawOptions 가 채움색을 밝게 그리고 테두리를 원래 색으로 그린다)
PALETTE = {
    "bg": (255, 255, 255), "wall": (211, 211, 211), "wall2": (233, 233, 233),
    "target": (144, 238, 144),                      # 고정 연두색 T (목표가 아님)
    "block": (143, 163, 184), "block_edge": (119, 136, 153),
    "agent": (65, 105, 225), "agent2": (78, 126, 255),
}
SS = 4  # 윤곽 래스터화 초과 표본 배율


def classify(panel):
    """패널 (H, W, 3) uint8 (어둡게 하기 전 색) -> 픽셀별 가장 가까운 팔레트 이름 인덱스."""
    names = list(PALETTE)
    cols = np.array([PALETTE[n] for n in names], dtype=np.float32)
    d = ((panel[:, :, None, :].astype(np.float32) - cols[None, None]) ** 2).sum(-1)
    return names, d.argmin(-1)


def coverage(pose, size):
    """자세 pose 의 블록이 size x size 픽셀마다 덮는 비율 (0..1)."""
    big = np.zeros((size * SS, size * SS), np.uint8)
    for p in block_outline_px(pose, size):
        pts = np.round(((p + 0.5) * SS - 0.5) * 16).astype(np.int32)
        cv2.fillPoly(big, [pts], 1, cv2.LINE_8, 4)
    return big.reshape(size, SS, size, SS).mean((1, 3))


def soft_iou(pose, mask, valid):
    cov = coverage(pose, mask.shape[0])[valid]
    m = mask[valid].astype(np.float32)
    return np.minimum(cov, m).sum() / max(np.maximum(cov, m).sum(), 1e-6)


def fit_block_pose(panel):
    """목표 칸 이미지에서 블록 자세 (x, y, angle) [캔버스 512 좌표] 를 맞춘다. 반환: pose, IoU, 블록 픽셀 수."""
    size = panel.shape[0]
    names, lab = classify(panel)
    mask = np.isin(lab, [names.index("block"), names.index("block_edge")])
    agent = np.isin(lab, [names.index("agent"), names.index("agent2")])
    # 에이전트(원)가 블록 위에 그려져 가린 부분은 판정에서 뺀다
    valid = ~cv2.dilate(agent.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        raise ValueError(f"목표 칸에서 블록을 찾지 못했다 (블록 픽셀 {len(xs)})")
    # 픽셀 -> 캔버스 좌표 (cv2.INTER_LINEAR 축소의 역)
    to_canvas = lambda v: (v + 0.5) * WINDOW_SIZE / size - 0.5
    cx, cy = to_canvas(xs.mean()), to_canvas(ys.mean())
    local_c = np.array(shapely.ops.unary_union(
        [shapely.geometry.Polygon(p) for p in block_local_polygons()]).centroid.coords[0])

    def pose_from(theta, dx=0.0, dy=0.0):
        c, s = np.cos(theta), np.sin(theta)
        off = np.array([[c, -s], [s, c]]) @ local_c
        return np.array([cx - off[0] + dx, cy - off[1] + dy, theta])

    # 1) 각도 전역 탐색 (위치는 무게중심을 맞춘다)
    thetas = np.deg2rad(np.arange(0, 360, 2.0))
    scores = [soft_iou(pose_from(t), mask, valid) for t in thetas]
    best = pose_from(thetas[int(np.argmax(scores))])
    best_s = max(scores)
    # 2) 좌표 하강으로 다듬기
    for step_px, step_deg in [(4, 2), (2, 1), (1, 0.5), (0.5, 0.25), (0.25, 0.1)]:
        improved = True
        while improved:
            improved = False
            for d in ([step_px, 0, 0], [-step_px, 0, 0], [0, step_px, 0], [0, -step_px, 0],
                      [0, 0, np.deg2rad(step_deg)], [0, 0, -np.deg2rad(step_deg)]):
                cand = best + np.array(d)
                sc = soft_iou(cand, mask, valid)
                if sc > best_s + 1e-9:
                    best, best_s, improved = cand, sc, True
    best[2] = best[2] % (2 * np.pi)
    return best, float(best_s), int(mask.sum())




def overlay_video(src, dst, up=2, color=(255, 0, 0), hold_sec=VIDEO_HOLD_SEC):
    with imageio.get_reader(src) as reader:
        fps = reader.get_meta_data().get("fps", 12)
        first = reader.get_data(0)
    h, w = first.shape[:2]
    assert h == w and h % 2 == 0, f"2x2 배치(정사각) 영상이 아니다: {first.shape}"
    p = h // 2
    if is_labeled(first):
        return {"skipped": "already labeled (new evaluator format; plan.py draws the goal outline there)"}
    goal_panel = np.clip(first[p:, p:].astype(np.float32) + CORRECTION_PX, 0, 255).astype(np.uint8)
    pose, iou, n_px = fit_block_pose(goal_panel)
    n_valid, n_src = count_valid_frames(src, p)

    writer = imageio.get_writer(dst, fps=fps, macro_block_size=1)
    with imageio.get_reader(src) as reader:
        for i, frame in enumerate(reader):
            if i >= n_valid:
                break
            big = cv2.resize(frame, (w * up, h * up), interpolation=cv2.INTER_LINEAR)
            for oy, ox in [(0, 0), (p, 0)]:  # 왼쪽 위: 실제 환경, 왼쪽 아래: 월드모델 예측
                sub = big[oy * up:(oy + p) * up, ox * up:(ox + p) * up]
                draw_block_outline(sub, pose, color, thickness=up, upsample=up)
            big = label_video_panels(big)  # 칸마다 Real / Model / Goal
            writer.append_data(big)
    n_hold = int(round(hold_sec * fps))
    for _ in range(n_hold):  # 마지막 프레임(성공한 장면 또는 실패로 끝난 장면)에서 멈춘다
        writer.append_data(big)
    writer.close()
    return {"goal_pose_est": [round(float(v), 3) for v in pose], "fit_iou": round(iou, 4),
            "block_px": n_px, "frames_src": n_src, "frames_valid": n_valid, "frames_hold": n_hold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="실행 디렉토리 또는 mp4 파일")
    ap.add_argument("--glob", default="output_final_*.mp4", help="실행 디렉토리 안에서 고를 영상")
    ap.add_argument("--out-subdir", default="goal_overlay")
    ap.add_argument("--hold-sec", type=float, default=VIDEO_HOLD_SEC, help="마지막 프레임에서 멈추는 시간 (초)")
    args = ap.parse_args()

    fits_by_dir = {}  # out_dir -> {영상 이름: 맞춤 결과}. 이미 있는 fits.json 에 더해 쓴다
    for path in args.paths:
        videos = [path] if path.endswith(".mp4") else sorted(glob.glob(os.path.join(path, args.glob)))
        if not videos:
            print(f"{path}: 영상 없음 ({args.glob})")
            continue
        for v in videos:
            out_dir = os.path.join(os.path.dirname(v), args.out_subdir)
            dst = os.path.join(out_dir, os.path.basename(v))
            if os.path.realpath(dst) == os.path.realpath(v):
                sys.exit(f"{v}: 출력이 원본과 같은 파일이다 (--out-subdir 를 하위 디렉토리로)")
            if out_dir not in fits_by_dir:
                os.makedirs(out_dir, exist_ok=True)
                fits_path = os.path.join(out_dir, "fits.json")
                fits_by_dir[out_dir] = json.load(open(fits_path)) if os.path.exists(fits_path) else {}
            fit = overlay_video(v, dst, hold_sec=args.hold_sec)
            if "skipped" in fit:
                print(f"{v}: 건너뜀 ({fit['skipped']})")
                continue
            fits_by_dir[out_dir][os.path.basename(v)] = fit
            print(f"{v}: goal_pose_est={fit['goal_pose_est']} fit_iou={fit['fit_iou']} "
                  f"frames {fit['frames_valid']}/{fit['frames_src']} + hold {fit['frames_hold']}")
            with open(os.path.join(out_dir, "fits.json"), "w") as f:
                json.dump(fits_by_dir[out_dir], f, indent=1)
    for out_dir in fits_by_dir:
        print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
