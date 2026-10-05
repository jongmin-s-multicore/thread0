"""평가 영상 변환 도구(pusht_goal_overlay.py, pointmaze_goal_overlay.py, reformat_eval_videos.py)가 같이 쓰는 함수.

평가 영상은 2x2 배치다: [실제 환경 | 목표] / [월드모델 예측 | 목표] (planning/evaluator.py _plot_rollout_compare).
예전 evaluator 는 배치에서 가장 긴 롤아웃 길이로 저장해, MPC 에서 일찍 성공한 에피소드는 성공 뒤가 가려진 회색 프레임
(evaluator._mask_traj 가 0 으로 채운 프레임)으로 이어진다. 지금 evaluator 는 칸 라벨(label_video_panels)을 달고,
실행한 마지막 프레임에서 끊고 그 프레임을 VIDEO_HOLD_SEC 초 멈춘다.
"""
import cv2
import imageio

from planning.evaluator import VIDEO_PANEL_LABELS


def label_boxes(frame_size):
    """label_video_panels 가 왼쪽 두 칸(Real, Model)에 그리는 검은 상자 [(y0, y1, x0, x1)] (같은 계산)."""
    ph = frame_size // 2
    scale, thick, pad = ph / 500, max(1, round(ph / 224)), max(2, round(ph / 112))
    boxes = []
    for r, row in enumerate(VIDEO_PANEL_LABELS):
        (tw, th), base = cv2.getTextSize(row[0], cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        boxes.append((r * ph, r * ph + th + base + 2 * pad, 0, tw + 2 * pad))
    return boxes


def is_labeled(frame):
    """이미 라벨이 들어간 새 형식 영상인지: 왼쪽 두 칸의 라벨 자리가 모두 검은 상자(어두운 픽셀이 대부분) + 흰 글자다.
    PointMaze 의 어두운 바닥처럼 한 칸만 어두운 장면은 두 칸이 모두 맞아야 해서 걸리지 않는다."""
    for y0, y1, x0, x1 in label_boxes(frame.shape[0]):
        box = frame[y0:y1, x0:x1].astype(int)
        if not ((box.max(-1) <= 40).mean() > 0.6 and (box.min(-1) >= 200).sum() > 3):
            return False
    return True


def count_valid_frames(src, panel):
    """가려지지 않은 프레임 수와 전체 프레임 수. 가려진 프레임은 왼쪽 위 칸이 한 가지 회색이고(h264 잡음만 있어 std < 1)
    항상 끝에 몰려 있다. 렌더된 장면은 std 20 이상이다 (PushT 가 가장 낮다)."""
    n_valid, n = 0, 0
    with imageio.get_reader(src) as reader:
        for i, frame in enumerate(reader):
            n += 1
            if frame[:panel, :panel].std() > 5:  # 렌더된 장면
                n_valid = i + 1
    return n_valid, n
