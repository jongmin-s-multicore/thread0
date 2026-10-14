#!/bin/bash
# DINO-WM 실행 환경 설치: micromamba env(Python 3.9, 레포 루트 environment.yaml 의 고정 버전) + MuJoCo 2.1.0 + mujoco-py 빌드.
# sudo 를 쓰지 않는다. 코드는 이 레포 자체다 (gaoyuezhou/dino_wm fork). DINOv2 는 models/dino.py 가 고정한다.
#
#   bash repro/setup/install_env.sh --dry-run   # 실행할 명령만 출력
#   bash repro/setup/install_env.sh
#
# 경로는 repro/env.sh (DINO_WORK 등). 디스크: env 9.0 GB (tensorflow·jax 포함, 설치 후 du), pip 다운로드 약 4 GB.
set -euo pipefail
cd "$(dirname "$0")/../.."
source repro/env.sh
case "${1:-}" in --dry-run) DRY=1 ;; "") DRY=0 ;; *) echo "usage: $0 [--dry-run]"; exit 2 ;; esac
run() { echo "+ $*"; if [ $DRY == 0 ]; then bash -c "$*"; fi; }

echo "DINO_WORK=$DINO_WORK DINO_ENV=$DINO_ENV DINO_WM=$DINO_WM MUJOCO_DIR=$MUJOCO_DIR TORCH_HOME=$TORCH_HOME"
run "mkdir -p '$DINO_WORK/tools' '$TORCH_HOME'"

# 1. micromamba (정적 바이너리, $DINO_WORK/tools/micromamba). 다른 경로의 micromamba 를 쓰려면 MICROMAMBA 에 그 경로를 준다
if [ ! -x "$MICROMAMBA" ]; then
  run "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C '$DINO_WORK/tools' bin/micromamba && mv '$DINO_WORK/tools/bin/micromamba' '$MICROMAMBA' && rmdir '$DINO_WORK/tools/bin'"
fi

# 2. conda 부분: environment.yaml 에서 pip 절을 뺀 것 (python 3.9.19, ffmpeg 4.3.2 등 conda-forge 고정 빌드)
run "sed '/^  - pip:/,\$d' '$DINO_WM/environment.yaml' > '$DINO_WORK/tools/env_conda.yaml'"
# [repro] environment.yaml 에 없는 빌드 도구·헤더를 conda 쪽에 더한다. cmake: egl-probe 빌드, 나머지: mujoco-py 빌드 (시스템에 X11/GL 개발 헤더가 없어도 된다)
run "printf '%s\\n' '  - cmake' '  - xorg-libx11' '  - xorg-xorgproto' '  - glew' '  - libglvnd-devel' '  - libegl-devel' '  - libgl-devel' '  - mesalib' >> '$DINO_WORK/tools/env_conda.yaml'"
run "sed -n '/^  - pip:/,\$p' '$DINO_WM/environment.yaml' | sed -n 's/^      - //p' > '$DINO_WORK/tools/requirements_full.txt'"
if [ ! -x "$DINO_ENV/bin/python" ]; then
  run "'$MICROMAMBA' create -y -p '$DINO_ENV' -f '$DINO_WORK/tools/env_conda.yaml'"
fi

# 3. pip 부분: environment.yaml 의 고정 버전 그대로 (torch 2.3.0+cu121 등 217개) + mujoco-py 빌드에 필요한 patchelf.
#    [repro] 목록이 전부 == 로 고정되어 있으므로 pip 에는 --no-deps 를 준다. 의존성 해석을 켜면 pip 가 되감기만 하며 25분 넘게 진전이 없었다.
#    PIP_CONFIG_FILE 은 기본값 /dev/null: 머신의 전역 pip 설정(extra-index-url 이 접근 불가 주소면 패키지마다 재시도하며 멈춘다)을 읽지 않는다.
#    내려받는 도중 연결이 끊기면 pip 자체 재시도로는 복구되지 않는다 (ReadTimeoutError). 받은 파일은 pip 캐시에 남으므로 5회까지 다시 실행한다.
#    uv 가 있으면 uv 로 (해석·다운로드가 빠르다), 없으면 env 의 pip 로. 실제 설치는 uv 0.11.8 로 했다. pip 경로는 같은 인자(--no-deps, 전역 설정 무시)를
#    conda 로 만든 같은 env 에서 직접 실행해 217개 설치까지 확인했다. 이 스크립트 자체를 pip 경로로 끝까지 돌려 보지는 않았다.
if command -v uv >/dev/null 2>&1; then
  run "uv pip install --python '$DINO_ENV/bin/python' -r '$DINO_WORK/tools/requirements_full.txt' patchelf"
else
  run "export PATH='$DINO_ENV/bin':\$PATH PIP_CONFIG_FILE=\${PIP_CONFIG_FILE:-/dev/null}; for i in 1 2 3 4 5; do '$DINO_ENV/bin/python' -m pip install --no-deps --timeout 60 --retries 10 -r '$DINO_WORK/tools/requirements_full.txt' patchelf && break; echo \"pip install 실패 (\$i/5), 다시 시도한다\"; [ \$i == 5 ] && exit 1; done"
fi

# 4. MuJoCo 2.1.0 (mujoco-py 2.1.2.14 용, 기본 위치는 dino_wm README 와 같은 ~/.mujoco/mujoco210). 압축을 임시 디렉토리에 풀고 옮긴다
if [ ! -d "$MUJOCO_DIR/bin" ]; then
  run "T=\$(mktemp -d) && curl -L https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz | tar -xz -C \$T && mkdir -p '$(dirname "$MUJOCO_DIR")' && mv \$T/mujoco210 '$MUJOCO_DIR' && rmdir \$T"
fi

# 5. mujoco-py: 처음 import 할 때 Cython 확장을 컴파일한다. /usr/lib/nvidia 가 있으면 EGL(GPU) 빌더,
#    없으면 CPU 빌더(GL/osmesa.h 필요 — 이 경로는 돌려 보지 않았다). libglew-dev, libgl1-mesa-dev 가 필요하다
#    [repro] 헤더·라이브러리는 위에서 conda 로 받은 $DINO_ENV/include, lib 에서 찾는다
run "CPATH='$DINO_ENV/include' LIBRARY_PATH='$DINO_ENV/lib' LD_LIBRARY_PATH='$DINO_ENV/lib':\$LD_LIBRARY_PATH '$DINO_ENV/bin/python' -c 'import mujoco_py; print(\"mujoco_py OK\")'"

# DINOv2 는 models/dino.py 가 torch.hub 로 85a2460 을 받는다 (첫 실행 때 $TORCH_HOME 에 코드·가중치 84 MB)

echo "done. 점검: source repro/env.sh && bash repro/setup/check_env.sh"
