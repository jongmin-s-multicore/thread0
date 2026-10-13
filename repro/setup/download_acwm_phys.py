"""ACWM-Phys (Hugging Face dataset t1an/ACWM-Phys) 의 실험에 쓰는 환경을 고정 커밋으로 $DATASET_DIR/acwm_phys 에 받는다.

받는 것: Pour Water, Cloth Move, Push Sand, Stack Cube 의 ind_train / ind_test / ood_test (mp4 + metadata.pt), 합쳐 2.6 GB.
conf/env/acwm_*.yaml 의 data_path 가 이 위치(HF 저장소와 같은 폴더 구조)를 읽는다. 이미 받은 파일은 다시 받지 않는다.
받은 뒤 split 마다 metadata 항목 수와 mp4 수를 세어 이 커밋에서 확인한 수와 맞는지 본다 (다르면 종료 코드 1).

usage: source repro/env.sh && python repro/setup/download_acwm_phys.py [--envs pour_water,cloth_move,push_sand,stack_cube]
       [--splits ind_train,ind_test,ood_test] [--dest DIR]
"""
import argparse, os, sys, time, traceback
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

REPO = "t1an/ACWM-Phys"
REV = "4017e5f5900a7a8590d4e944b6a84f55df75a0c1"  # main on 2026-10-08 (last modified 2026-05-08)
ENVS = {  # name -> (folder in the dataset repo, episodes in ind_train / ind_test / ood_test at REV)
    "pour_water": ("particle/pour_water", {"ind_train": 1000, "ind_test": 50, "ood_test": 50}),
    "cloth_move": ("deformable/clothmove", {"ind_train": 2000, "ind_test": 100, "ood_test": 100}),
    "push_sand": ("particle/push_sand", {"ind_train": 1784, "ind_test": 100, "ood_test": 100}),
    "stack_cube": ("rigid_dynamics/stack_cube", {"ind_train": 1373, "ind_test": 67, "ood_test": 271}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default=",".join(ENVS))
    ap.add_argument("--splits", default="ind_train,ind_test,ood_test")
    ap.add_argument("--dest", default=os.path.join(os.environ.get("DATASET_DIR", "."), "acwm_phys"))
    args = ap.parse_args()
    envs, splits = args.envs.split(","), args.splits.split(",")
    patterns = [f"{ENVS[e][0]}/{s}/*" for e in envs for s in splits]
    for attempt in range(1, 41):  # the Hub rate-limits anonymous downloads of many small files; files on disk are skipped
        try:
            snapshot_download(repo_id=REPO, repo_type="dataset", revision=REV, local_dir=args.dest, allow_patterns=patterns, max_workers=8)
            break
        except Exception:
            print(f"[{time.strftime('%F %T')}] attempt {attempt} failed, retrying in 120 s", flush=True)
            traceback.print_exc()
            time.sleep(120)
    else:
        sys.exit("gave up after 40 attempts")

    ok = True
    for e in envs:
        sub, expect = ENVS[e]
        for s in splits:
            d = Path(args.dest) / sub / s
            meta = torch.load(d / "metadata.pt", map_location="cpu", weights_only=True)  # a pickle: never load it without weights_only
            have = sum((d / Path(m["video_path"]).name).exists() for m in meta)
            good = len(meta) == expect[s] == have
            ok &= good
            print(f"{'ok ' if good else 'BAD'} {sub}/{s}: metadata {len(meta)}, videos {have}, expected {expect[s]}")
    print("dataset at", args.dest, "(revision", REV[:7] + ")")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
