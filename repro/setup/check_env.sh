#!/bin/bash
# 설치 점검.
#   bash repro/setup/check_env.sh            # torch/CUDA, mujoco-py, DINOv2 hub, 공개 체크포인트로 PointMaze·PushT·Wall 렌더·동역학 대조
#   bash repro/setup/check_env.sh --pyflex   # + PyFleX import, Rope·Granular 환경 구성/롤아웃, 데이터셋 재생 대조
# 그림은 $DINO_WORK/setup_check/ 에 쓴다 (왼쪽 env, 오른쪽 데이터셋).
set -uo pipefail
cd "$(dirname "$0")/../.."
source repro/env.sh
CHK=$THREAD0/repro/setup/checks
PYFLEX=0; [ "${1:-}" == "--pyflex" ] && PYFLEX=1
fail=0; step() { echo; echo "== $*"; }; bad() { echo "FAIL: $*"; fail=1; }

step "python / torch / CUDA"
python -c "
import sys, torch
print(sys.version.split()[0], torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    a = torch.randn(512, 512, device=f'cuda:{i}'); print(i, torch.cuda.get_device_name(i), float((a @ a).abs().mean()))
" || bad torch

step "mujoco-py"
python -c "import mujoco_py; print('mujoco_py', mujoco_py.__version__)" 2>&1 | grep -v -i warn || bad mujoco_py

step "DINOv2 (torch.hub 고정 커밋, Python 3.9)"
(cd "$DINO_WM" && python -c "
import torch
m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda().eval()
with torch.no_grad(): o = m.forward_features(torch.randn(1, 3, 196, 196).cuda())
print('patch tokens', tuple(o['x_norm_patchtokens'].shape))
" 2>&1 | tail -1) || bad dinov2

step "공개 체크포인트로 env 렌더·동역학 대조 (데이터셋 프레임과 픽셀 MAE, 0~255)"
if [ -d "$DINO_CKPT/outputs/point_maze" ] && [ -d "$DATASET_DIR/point_maze" ]; then
  (cd "$DINO_WM" && python "$CHK/render_check.py" point_maze pusht wall_single 2>&1 | grep '^\[') || bad render_check
else
  echo "skip: $DINO_CKPT/outputs 또는 $DATASET_DIR 이 없다 (repro/setup/download_data.sh)"
fi

if [ $PYFLEX == 1 ]; then
  step "PyFleX"
  python -c "import pyflex; print('pyflex OK')" || bad pyflex
  for obj in rope granular; do
    (cd "$DINO_WM" && python "$CHK/pyflex_env_check.py" $obj 3 2>&1 | grep -E "constructed|rollout done|eval_state|total") || bad "pyflex env $obj"
    if [ -d "$DATASET_DIR/deformable/$obj" ]; then
      (cd "$DINO_WM" && python "$CHK/pyflex_dset_replay.py" $obj 3 2>&1 | grep -E "^t=|saved") || bad "pyflex replay $obj"
    fi
  done
fi
echo; [ $fail == 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
exit $fail
