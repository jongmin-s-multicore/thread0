#!/bin/bash
# PyFleX 빌드 (Rope·Granular planning 용. 학습에는 필요 없다).
# dino_wm README 는 AdaptiGraph 의 install_pyflex.sh 를 가리키지만 PyFleX 소스는 dino_wm 레포에 없다.
# AdaptiGraph(고정 커밋)의 PyFleX/ 를 복사해 xingyu/softgym 도커 이미지(CUDA 9.2, cmake 3.10, gcc 7.5) 안에서
# 이 env 의 Python 3.9 / pybind11 로 빌드한다. 도커는 sudo 없이(docker 그룹) + NVIDIA container runtime 이 필요하다.
#
#   bash repro/setup/install_pyflex.sh [--dry-run]
#
# 결과: $PYFLEXROOT/bindings/build/pyflex.cpython-39-x86_64-linux-gnu.so (repro/env.sh 가 PYTHONPATH 에 넣는다)
set -euo pipefail
cd "$(dirname "$0")/../.."
source repro/env.sh
case "${1:-}" in --dry-run) DRY=1 ;; "") DRY=0 ;; *) echo "usage: $0 [--dry-run]"; exit 2 ;; esac
run() { echo "+ $*"; if [ $DRY == 0 ]; then bash -c "$*"; fi; }

AG=$DINO_WORK/tools/AdaptiGraph
if [ ! -d "$AG/.git" ]; then
  run "git clone -q https://github.com/Boey-li/AdaptiGraph.git '$AG'"
fi
run "git -C '$AG' checkout -q $ADAPTIGRAPH_COMMIT"
if [ ! -d "$PYFLEXROOT" ]; then
  # 레포에 커밋된 bindings/build/ 는 원저자 경로의 CMakeCache 라 지우고 새로 빌드한다
  run "cp -r '$AG/PyFleX' '$PYFLEXROOT' && rm -rf '$PYFLEXROOT/bindings/build'"
fi
run "docker pull xingyu/softgym:latest"
run "docker run --rm --user \$(id -u):\$(id -g) --gpus all \
  -v '$PYFLEXROOT':/workspace/PyFleX -v '$DINO_ENV':/workspace/anaconda \
  xingyu/softgym:latest bash -c '
set -e
export PATH=/workspace/anaconda/bin:\$PATH
export PYFLEXROOT=/workspace/PyFleX
export LD_LIBRARY_PATH=/workspace/PyFleX/external/SDL2-2.0.4/lib/x64:\$LD_LIBRARY_PATH
cd /workspace/PyFleX/bindings && mkdir -p build && cd build
/usr/bin/cmake .. -DPYTHON_EXECUTABLE=/workspace/anaconda/bin/python3 -Dpybind11_DIR=\$(python3 -m pybind11 --cmakedir)
make -j\$(nproc)'"
run "'$DINO_ENV/bin/python' -c 'import pyflex; print(\"pyflex OK\")'"
echo "done. 점검: bash repro/setup/check_env.sh --pyflex"
