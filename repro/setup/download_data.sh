#!/bin/bash
# DINO-WM 데이터셋과 공개 체크포인트를 OSF 에서 받아 푼다.
#
#   bash repro/setup/download_data.sh                      # core + checkpoints + deformable (zip 21 GB, 풀면 약 236 GB)
#   bash repro/setup/download_data.sh core checkpoints     # PointMaze·PushT·Wall 과 체크포인트만 (zip 6.1 GB)
#
# 결과: $DATASET_DIR/{point_maze,pusht_noise,wall_single,deformable/{rope,granular}}, $DINO_CKPT/outputs/{point_maze,pusht,wall_single}
# 풀린 크기: point_maze 29 GB, pusht_noise 7 GB, wall_single 55 GB, deformable 144 GB (rope 69, granular 75 — 프레임이 float32),
# 체크포인트 1.1 GB. 받는 동안 zip 원본 21 GB + deformable 합본 16 GB 가 더 필요하다.
# 데이터는 OSF(https://osf.io/bmw48)에서 받는다. 이 레포에 넣거나 재배포하지 않는다.
set -euo pipefail
cd "$(dirname "$0")/../.."
source repro/env.sh
GROUPS_=("$@"); [ ${#GROUPS_[@]} -eq 0 ] && GROUPS_=(core checkpoints deformable)

python3 repro/setup/osf_download.py "$DINO_DOWNLOADS" --only "${GROUPS_[@]}"
mkdir -p "$DATASET_DIR" "$DINO_CKPT"
for g in "${GROUPS_[@]}"; do
  case $g in
    core)
      for f in point_maze pusht_noise wall_single; do unzip -q -o "$DINO_DOWNLOADS/$f.zip" -d "$DATASET_DIR"; echo "unzipped $f"; done ;;
    checkpoints)
      unzip -q -o "$DINO_DOWNLOADS/outputs.zip" -d "$DINO_CKPT" -x '__MACOSX/*'; echo "unzipped checkpoints" ;;
    deformable)
      # 분할 zip: 하나로 합친 뒤 푼다 (dino_wm README 와 같은 방법). 합친 사본(16 GB)은 풀고 나서 지운다
      zip -q -s- "$DINO_DOWNLOADS/deformable.zip" -O "$DINO_DOWNLOADS/deformable_full.zip"
      unzip -q -o "$DINO_DOWNLOADS/deformable_full.zip" -d "$DATASET_DIR"
      rm -f "$DINO_DOWNLOADS/deformable_full.zip"; echo "unzipped deformable" ;;
  esac
done
du -sh "$DATASET_DIR"/* "$DINO_CKPT"/outputs/* 2>/dev/null || true
