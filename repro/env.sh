# 경로·환경변수의 단일 출처. 셸마다 레포 루트에서 `source repro/env.sh` 로 쓴다 (bash, zsh).
# 아래 값은 모두 source 전에 export 하면 덮어쓴다. 이 파일에는 일반 기본값만 둔다 — 머신마다 다른 값은 .claude/env.local.sh (gitignore).

if [ -n "${BASH_VERSION:-}" ]; then _env_sh=${BASH_SOURCE[0]}
elif [ -n "${ZSH_VERSION:-}" ]; then _env_sh=${(%):-%x}
else echo "repro/env.sh: bash 또는 zsh 에서 source 한다" >&2; return 1 2>/dev/null || exit 1; fi
THREAD0=$(cd "$(dirname "$_env_sh")/.." && pwd); unset _env_sh
export THREAD0

# 머신별 값(작업 루트, GPU·메모리 조건 등)은 레포에 올리지 않고 .claude/env.local.sh 에 둔다 (gitignore).
# 있으면 아래 기본값보다 먼저 읽는다. 그 파일에서도 ${VAR:-값} 형식으로 써서 이미 export 한 값을 존중한다.
if [ -f "$THREAD0/.claude/env.local.sh" ]; then . "$THREAD0/.claude/env.local.sh"; fi

# 작업 루트: 환경, 데이터, 체크포인트, 실행 산출물이 모두 이 아래에 생긴다. 레포 밖에 둔다 (코드는 이 레포)
export DINO_WORK=${DINO_WORK:-$HOME/dinowm}
case "$DINO_WORK/" in "$THREAD0"/*) echo "repro/env.sh: DINO_WORK($DINO_WORK) 를 레포($THREAD0) 밖으로 정한다" >&2 ;; esac
export DINO_ENV=${DINO_ENV:-$DINO_WORK/envs/dino_wm}           # micromamba env prefix (Python 3.9)
export DINO_WM=${DINO_WM:-$THREAD0}                            # dino_wm 코드 = 이 레포 루트 (gaoyuezhou/dino_wm fork)
export DATASET_DIR=${DATASET_DIR:-$DINO_WORK/data}             # point_maze, pusht_noise, wall_single, deformable/{rope,granular}
export DINO_CKPT=${DINO_CKPT:-$DINO_WORK/checkpoints}          # 공개 체크포인트: $DINO_CKPT/outputs/{point_maze,pusht,wall_single}
export DINO_TRAIN=${DINO_TRAIN:-$DINO_WORK/train_runs}         # 직접 학습한 world model: $DINO_TRAIN/outputs/<run>
export DINO_RUNS=${DINO_RUNS:-$DINO_WORK/runs}                 # planning·평가 산출물
export DINO_DOWNLOADS=${DINO_DOWNLOADS:-$DINO_WORK/downloads}  # OSF zip 원본
export MUJOCO_DIR=${MUJOCO_DIR:-$HOME/.mujoco/mujoco210}       # mujoco-py 가 기본으로 찾는 위치
export MUJOCO_PY_MUJOCO_PATH=$MUJOCO_DIR                       # mujoco-py 는 이 변수로 위치를 바꾼다
export PYFLEXROOT=${PYFLEXROOT:-$DINO_WORK/PyFleX}             # deformable 환경용 (repro/setup/install_pyflex.sh)
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$DINO_WORK/tools/mamba}
export MICROMAMBA=${MICROMAMBA:-$DINO_WORK/tools/micromamba}
# torch.hub 캐시(DINOv2 코드·가중치)를 프로젝트 전용으로 둔다. 전역 TORCH_HOME 과 섞이지 않게 DINO_TORCH_HOME 으로만 덮어쓴다.
export TORCH_HOME=${DINO_TORCH_HOME:-$DINO_WORK/torch_home}

# 고정 버전 (setup 스크립트가 쓴다). DINOv2 hub 코드는 models/dino.py 에서 85a2460 으로 고정한다
export ADAPTIGRAPH_COMMIT=a7c75357793422e9108c36f58c6928f3f8a123ad # Boey-li/AdaptiGraph "[sim] update PyFleX."

# 환경 활성화 (conda activate 대신 PATH 만 앞에 붙인다)
export PATH=$DINO_ENV/bin:$PATH
export CONDA_PREFIX=$DINO_ENV
# /usr/lib/nvidia 가 있으면 mujoco-py 가 EGL(GPU) 빌더를 고른다 (dino_wm README 와 같은 설정)
export LD_LIBRARY_PATH=$MUJOCO_DIR/bin:/usr/lib/nvidia:$PYFLEXROOT/external/SDL2-2.0.4/lib/x64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$PYFLEXROOT/bindings/build${PYTHONPATH:+:$PYTHONPATH}

# 헤드리스 실행
export WANDB_MODE=${WANDB_MODE:-disabled}      # dino_wm 은 wandb 를 부른다. 결과는 로컬 파일로 남긴다
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}  # pusht (pygame)
export EGL_GPU=${EGL_GPU:-0}                   # PyFleX: FleX 를 렌더링할 EGL 장치 번호 (device platform 이라 DISPLAY 가 있어도 X 를 거치지 않는다)
export HYDRA_FULL_ERROR=1
