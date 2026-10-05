"""예전 평가 영상을 지금 evaluator 형식으로 다시 쓴다 (목표 표시 없이, 다시 평가하지 않는다). Wall 처럼 목표 표시가 필요 없는 환경용.

예전 planning/evaluator.py 는 배치에서 가장 긴 롤아웃 길이로 영상을 저장해서, MPC 에서 일찍 성공한 에피소드는 성공 뒤가
가려진 회색 프레임(evaluator._mask_traj 가 0 으로 채운 프레임)으로 이어진다. 여기서는 가려지기 전 마지막 프레임까지만 쓰고
그 프레임을 --hold-sec 초 더 보여 준 뒤 끝낸다. 칸마다 라벨(Real = 실제 환경, Model = 월드모델 예측, Goal = 목표)을 단다.
크기·배치·라벨은 지금 evaluator.py 가 새로 저장하는 영상과 같다 (확대하지 않는다). 프레임 내용은 다시 인코딩하는 것 말고 그대로다.
목표 표시도 그리는 환경은 pusht_goal_overlay.py, pointmaze_goal_overlay.py 를 쓴다.

usage (cwd = $DINO_WM, env sourced): python repro/eval/reformat_eval_videos.py <run_dir|video.mp4> ... [--glob 'output_final_*.mp4'] [--out-subdir relabeled] [--hold-sec 3]
출력: <영상 디렉토리>/<out-subdir>/<영상 이름>.mp4, frames.json (영상별 원본·유효·멈춤 프레임 수. 다시 돌리면 항목을 더하거나 갱신한다)
"""
import os, sys, json, glob, argparse
import imageio

sys.path.insert(0, os.getcwd())
from planning.evaluator import label_video_panels, VIDEO_HOLD_SEC
from video_format import is_labeled, count_valid_frames  # repro/eval/video_format.py








def reformat_video(src, dst, hold_sec=VIDEO_HOLD_SEC):
    with imageio.get_reader(src) as reader:
        fps = reader.get_meta_data().get("fps", 12)
        first = reader.get_data(0)
    h, w = first.shape[:2]
    assert h == w and h % 2 == 0, f"2x2 배치(정사각) 영상이 아니다: {first.shape}"
    if is_labeled(first):
        return {"skipped": "already labeled (new evaluator format)"}
    n_valid, n_src = count_valid_frames(src, h // 2)
    if n_valid == 0:
        raise ValueError(f"{src}: 렌더된 프레임이 없다")

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    writer = imageio.get_writer(dst, fps=fps, macro_block_size=1)
    with imageio.get_reader(src) as reader:
        for i, frame in enumerate(reader):
            if i >= n_valid:
                break
            last = label_video_panels(frame)
            writer.append_data(last)
    n_hold = int(round(hold_sec * fps))
    for _ in range(n_hold):  # 마지막 프레임(성공한 장면 또는 실패로 끝난 장면)에서 멈춘다
        writer.append_data(last)
    writer.close()
    return {"frames_src": n_src, "frames_valid": n_valid, "frames_hold": n_hold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="실행 디렉토리 또는 mp4 파일")
    ap.add_argument("--glob", default="output_final_*.mp4", help="실행 디렉토리 안에서 고를 영상 (plan<i>_*.mp4 도 된다)")
    ap.add_argument("--out-subdir", default="relabeled")
    ap.add_argument("--hold-sec", type=float, default=VIDEO_HOLD_SEC, help="마지막 프레임에서 멈추는 시간 (초)")
    args = ap.parse_args()

    info_by_dir = {}  # out_dir -> {영상 이름: 프레임 수}. 이미 있는 frames.json 에 더해 쓴다
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
            res = reformat_video(v, dst, hold_sec=args.hold_sec)
            if "skipped" in res:
                print(f"{v}: 건너뜀 ({res['skipped']})")
                continue
            if out_dir not in info_by_dir:  # 실제로 쓴 디렉토리만 (모두 건너뛰면 빈 출력 디렉토리를 만들지 않는다)
                info_path = os.path.join(out_dir, "frames.json")
                info_by_dir[out_dir] = json.load(open(info_path)) if os.path.exists(info_path) else {}
            info_by_dir[out_dir][os.path.basename(v)] = res
            print(f"{v}: frames {res['frames_valid']}/{res['frames_src']} + hold {res['frames_hold']}")
            with open(os.path.join(out_dir, "frames.json"), "w") as f:
                json.dump(info_by_dir[out_dir], f, indent=1)
    for out_dir in info_by_dir:
        print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
