# thread0 — DINO-WM 재현

[gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm)([arXiv:2411.04983](https://arxiv.org/abs/2411.04983))의 fork 입니다. upstream `0a9492f` 위에 재현에 필요한 수정을 커밋으로 더하고, 벤치마크를 24 GB GPU 2장에서 다시 돌리는 도구를 `repro/` 에 둡니다. 결과와 해석은 [이슈](https://github.com/jongmin-s-multicore/thread0/issues)로 올립니다. 규약은 [AGENTS.md](AGENTS.md), upstream README 는 [README_upstream.md](README_upstream.md).

- 실험 개요·구성·실행 방법: **[repro/README.md](repro/README.md)**
- 모델·데이터·하이퍼파라미터·평가 프로토콜·upstream 과 다른 점: **[repro/SETTINGS.md](repro/SETTINGS.md)**
- upstream 대비 코드 수정: `git diff 0a9492f -- . ':!repro' ':!*.md' ':!.gitignore'` (9개 파일, 고친 곳마다 `[repro]` 주석)

| 벤치마크 | world model | planner | 논문 |
|---|---|---|---|
| PointMaze · PushT · Wall 성공률 (50 인스턴스) | 공개 체크포인트 | MPC-CEM, 오픈루프 CEM(= MPC 1회차), 오픈루프 GD | Table 1, 8 |
| Rope · Granular Chamfer distance (10 인스턴스) | 직접 학습 (H=1, frameskip=1, 100 epoch) | MPC-CEM | Table 1, 8 |
| 예측 품질 LPIPS · SSIM | 위와 같음 | — | Table 4, 9 |

## upstream 에 더한 수정

| 파일 | 내용 | 결과에 주는 영향 |
|---|---|---|
| `models/vit.py` | 추론 때 fp32 `scaled_dot_product_attention` (`DINO_WM_SDPA=1`) | latent 상대오차 ≤ 1.3e-6 |
| `models/visual_world_model.py`, `planning/gd.py` | 롤아웃·GD 역전파를 배치 조각으로 (`DINO_WM_ROLLOUT_CHUNK`, `DINO_WM_GD_CHUNK`) | 같은 계산 |
| `planning/evaluator.py` | 그림에 쓰는 것만 하나씩 디코딩, 평가 뒤 캐시 해제 (upstream issue #23) | 지표 같음 |
| `planning/cem.py`, `planning/mpc.py` | MPC 에서 이미 성공한 에피소드의 CEM 생략 (`DINO_WM_SKIP_SOLVED=1`) | 1회차 같음, 이후 난수 순서만 다름 |
| `train.py` | 검증을 `no_grad` 로, epoch 지표를 `epoch_logs.jsonl` 에도 | 학습 같음 |
| `env/deformable_env/.../flex_env.py` | `pyflex.init()` 을 프로세스당 한 번 (여러 env 에서 segfault) | — |
| `models/dino.py` | DINOv2 hub 코드를 `85a2460` 으로 고정 (Python 3.9, upstream issue #25) | 특징 같음 (max diff 0) |

자세한 근거는 [repro/SETTINGS.md §6](repro/SETTINGS.md#6-upstream-코드논문과-다른-점).

## 빠른 시작

```bash
# 0. 사전 준비: NVIDIA 드라이버, git curl unzip zip, libglew-dev libgl1-mesa-dev (mujoco-py 빌드),
#    Rope·Granular planning 을 하려면 docker + NVIDIA container runtime (sudo 없이 docker 그룹)
git clone https://github.com/jongmin-s-multicore/thread0.git ~/thread0 && cd ~/thread0
export DINO_WORK=~/dinowm      # 작업 루트 (기본값, 레포 밖). 머신별 값은 .claude/env.local.sh 에 둘 수 있다 (gitignore)

# 1. 설치: micromamba env (Python 3.9, environment.yaml 고정 버전) + MuJoCo 2.1.0 + mujoco-py 빌드
bash repro/setup/install_env.sh --dry-run    # 실행할 명령만 확인
bash repro/setup/install_env.sh
bash repro/setup/install_pyflex.sh           # Rope·Granular planning 에만 필요 (도커 빌드, 약 1분)

# 2. 데이터와 공개 체크포인트 (OSF). 전부 받으면 zip 21 GB, 풀면 약 236 GB
bash repro/setup/download_data.sh core checkpoints   # PointMaze·PushT·Wall + 체크포인트만
bash repro/setup/download_data.sh deformable         # Rope·Granular

# 3. 점검 (레포 루트에서, bash 또는 zsh, 새 셸마다 source)
source repro/env.sh
bash repro/setup/check_env.sh            # torch/CUDA, mujoco-py, DINOv2, 공개 체크포인트로 렌더·동역학 대조
bash repro/setup/check_env.sh --pyflex   # + PyFleX, Rope·Granular

# 4. 공개 체크포인트로 planning 한 번 (PointMaze, 에피소드 2개, 1분 안쪽)
python plan.py --config-name plan_point_maze.yaml model_name=point_maze ckpt_base_path=$DINO_CKPT \
  n_evals=2 planner.sub_planner.opt_steps=2 planner.max_iter=1 hydra.run.dir=$DINO_RUNS/smoke/point_maze

# 5. 벤치마크 전체 (작업 큐, 24 GB GPU 2장 기준 하루 이상). 24 GB 이하 GPU 는 먼저 repro/README.md "머신별 설정"
bash repro/queue/start.sh
```

## 구성

```
(upstream dino_wm)       train.py plan.py conf/ models/ planning/ env/ datasets/ metrics/ environment.yaml …
repro/
  README.md SETTINGS.md  실험 개요·실행 방법, 실험 세팅
  env.sh                 경로·환경변수의 단일 출처 (DINO_WORK 등). source 로 실행
  setup/                 env 설치, PyFleX 빌드, OSF 다운로드, 점검 (checks/)
  queue/                 파일 상태 기반 GPU 작업 큐와 러너
  jobs/benchmark.txt     벤치마크 작업 목록
  eval/                  예측 품질(LPIPS·SSIM), 결과 요약
  make_run_yaml.py       runs/*/run.yaml 생성
  runs/{train,eval}/<run-id>/run.yaml   실행별 설정 기록 (결과 수치는 이슈에)
  requirements/          환경 패키지 목록 (pip freeze, conda list)
AGENTS.md                저장소 운영 규약
README_upstream.md       upstream README
```

설치 후 디렉터리 (`$DINO_WORK`, 기본 `~/dinowm`):

| 경로 | 내용 |
|---|---|
| `envs/dino_wm` | micromamba env (Python 3.9.19, torch 2.3.0+cu121), 9.0 GB |
| `data/` | `point_maze` `pusht_noise` `wall_single` `deformable/{rope,granular}` (`DATASET_DIR`) |
| `checkpoints/` | `DINO_CKPT`. 아래 `outputs/{point_maze,pusht,wall_single}` 가 공개 체크포인트 |
| `torch_home/` | torch.hub 캐시: DINOv2 코드(`85a2460`)와 가중치, LPIPS VGG (`TORCH_HOME`) |
| `PyFleX/` | AdaptiGraph `a7c7535` 의 PyFleX 와 빌드 결과 (`PYFLEXROOT`) |
| `train_runs/` `runs/` `results/` | 학습한 world model, planning·평가 산출물, 요약 (`DINO_TRAIN`, `DINO_RUNS`) |
| `tools/` | micromamba, 설치 중 만든 파일, AdaptiGraph clone |
| `downloads/` | OSF zip 원본 21 GB. 압축을 푼 뒤에는 지워도 된다 (`DINO_DOWNLOADS`) |
| `~/.mujoco/mujoco210` | MuJoCo 2.1.0 (`MUJOCO_DIR`) |

## 검증 환경

| 항목 | 값 |
|---|---|
| OS | Ubuntu 24.04, NVIDIA driver 580.178.04 |
| 하드웨어 | 32 코어 CPU, RAM 48 GB + 스왑 64 GB, RTX 3090 Ti 24 GB + RTX 3090 24 GB |
| 소프트웨어 | Python 3.9.19 (conda-forge), torch 2.3.0+cu121, mujoco-py 2.1.2.14 (EGL), gym 0.23.1, hydra-core 1.2.0, PyFleX (CUDA 9.2 도커 빌드) |

`repro/setup/` 스크립트는 위 머신에서 손으로 실행한 명령을 옮긴 것입니다. 각 명령은 실행해 봤고(pip 설치는 uv 로), 스크립트를 새 머신에서 처음부터 끝까지 돌려 보지는 않았습니다.

## 라이선스

- 코드는 upstream 과 같은 MIT ([LICENSE](LICENSE), © gaoyuezhou). `repro/` 와 수정 부분도 MIT 로 둡니다.
- 데이터셋과 체크포인트는 DINO-WM 저자의 [OSF 프로젝트](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28)에서 받습니다. 이 레포에 넣거나 재배포하지 않습니다.
- AdaptiGraph (MIT, PyFleX 는 NVIDIA FleX 기반), DINOv2 (Apache-2.0), MuJoCo 2.1.0 (Apache-2.0) 은 각 저장소에서 받습니다. 학습한 world model 가중치는 이 레포에 넣지 않습니다.
