# repro — DINO-WM 재현 도구와 실행 방법

이 레포(gaoyuezhou/dino_wm 의 fork)로 DINO-WM 의 planning 벤치마크를 다시 돌리는 방법이다. GPU 환경과 무관한 공용 도구(환경 구축·작업 큐·평가)와 프로토콜을 둔다. GPU 환경에 맞춘 코드 수정과 그 환경의 실행 기록은 환경별 브랜치에 있다 ([루트 README](../README.md#branches-and-changes-to-upstream)).

- 모델·데이터·하이퍼파라미터·평가 프로토콜·upstream 과 다른 점: **[SETTINGS.md](SETTINGS.md)**
- 결과와 해석: [이슈](https://github.com/jongmin-s-multicore/thread0/issues)

Reacher, 일반화 환경(WallRandom·PushObj·GranularRandom), 다른 모델 베이스라인은 데이터나 구현이 공개되지 않았거나 코드가 더 필요해서 돌리지 않았다. 인코더 비교(Table 2)도 이번에는 돌리지 않았다 ([SETTINGS.md §1](SETTINGS.md#1-구성)).

---

## 구성

경로는 모두 `repro/env.sh` 에서 온다. 산출물(`$DINO_RUNS`, `$DINO_TRAIN`, `$DINO_WORK/results`)은 레포 밖에 생기고 커밋하지 않는다.

```
repro/
├── env.sh                     경로·환경변수의 단일 출처. 레포 루트에서 source (bash, zsh). .claude/env.local.sh 가 있으면 먼저 읽는다
├── setup/
│   ├── install_env.sh         micromamba env + environment.yaml 고정 버전 + MuJoCo 2.1.0 + mujoco-py 빌드
│   ├── install_pyflex.sh      AdaptiGraph 의 PyFleX 를 xingyu/softgym 도커에서 빌드 (Rope·Granular)
│   ├── download_data.sh osf_download.py   OSF 데이터셋·체크포인트 (병렬 Range, md5 대조), 압축 해제
│   ├── check_env.sh           설치 점검 (--pyflex: deformable 까지)
│   └── checks/                렌더·동역학 대조, PyFleX 환경 구성·데이터셋 재생 대조
├── queue/
│   ├── start.sh               작업 큐를 백그라운드로 띄운다 (다시 띄워도 실행 중인 작업은 그대로)
│   ├── scheduler.py           GPU 슬롯·여유 메모리·선행 작업을 보고 작업을 띄운다. 상태는 $DINO_RUNS/<job>/ 의 파일, OOM 이면 다시 넣는다
│   ├── run_plan.sh            plan.py 실행 (환경 브랜치는 여기서 자기 옵션을 export 한다)
│   ├── run_train.sh run_cmd.sh   train.py / 임의 명령 실행
│   └── kill_job.sh wait_event.sh   작업 끄기 (PID 기준), 다음 이벤트까지 기다리며 요약 보기
├── jobs/benchmark.txt         벤치마크 작업 목록 (job 이름, 종류, 선행 작업, hydra 인자. GPU·메모리 조건은 머신별 overlay)
├── eval/
│   ├── eval_pred_quality.py   검증 분할 전체의 1-step 예측 LPIPS·SSIM
│   ├── pusht_goal_overlay.py  예전 PushT 평가 영상을 지금 형식으로 (2배 확대): 목표 칸 이미지에서 맞춘 목표 블록 윤곽, Real/Model/Goal 라벨, 회색 프레임 대신 3초 멈춤
│   ├── pointmaze_goal_overlay.py  예전 PointMaze 평가 영상을 지금 형식으로 (2배 확대): 목표 위치(plan_targets.pkl 의 state_g) 원, Real/Model/Goal 라벨, 회색 프레임 대신 3초 멈춤
│   ├── reformat_eval_videos.py    예전 평가 영상을 지금 형식으로, 목표 표시 없이 (Wall 등): Real/Model/Goal 라벨, 회색 프레임 대신 3초 멈춤
│   ├── video_format.py        위 세 도구가 같이 쓰는 함수 (가려진 회색 프레임 세기, 라벨이 이미 있는지)
│   ├── deform_arm_video.py    Rope·Granular planning 작업을 다시 돌려, 팔(xArm6)이 미는 중간 프레임까지 담은 평가 영상도 쓴다 (<out_dir>/arm_video/)
│   └── summarize.py           logs.json·epoch_logs·예측 품질을 $DINO_WORK/results/summary.{md,json} 으로
└── requirements/
    ├── requirements-dinowm.txt    pip 패키지 목록 (uv pip freeze)
    └── conda-dinowm.txt           conda 패키지 목록 (micromamba list)
```

job 은 `jobs/benchmark.txt` 와 `$DINO_RUNS/<job>/` 의 이름이다. 실행 기록(`repro/runs/{train,eval}/<run-id>/run.yaml`, run-id 형식은 [AGENTS.md §2](../AGENTS.md#2-재현-디렉토리-구조))은 그 실행을 돌린 환경 브랜치에 둔다.

---

## 환경

### 설치

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

# 5. 벤치마크 전체 (작업 큐). GPU 메모리가 부족하면 환경 브랜치(루트 README "Branches and changes to upstream")를 쓴다. 큐 설정은 아래 "머신별 설정"
bash repro/queue/start.sh
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

설치 스크립트를 확인한 환경:

| 항목 | 값 |
|---|---|
| OS | Ubuntu 24.04, NVIDIA driver 580.178.04 |
| 소프트웨어 | Python 3.9.19 (conda-forge), torch 2.3.0+cu121, mujoco-py 2.1.2.14 (EGL), gym 0.23.1, hydra-core 1.2.0, PyFleX (CUDA 9.2 도커 빌드) |

`repro/setup/` 스크립트는 위 환경에서 손으로 실행한 명령을 옮긴 것이다. 각 명령은 실행해 봤고(pip 설치는 uv 로), 스크립트를 새 머신에서 처음부터 끝까지 돌려 보지는 않았다.

### 설치된 구성

| 항목 | 값 |
|---|---|
| Python env | micromamba prefix env, Python 3.9.19 (conda-forge), `environment.yaml` 의 pip 고정 버전 217개 + patchelf. 목록은 [requirements/](requirements/) |
| 주요 패키지 | torch 2.3.0+cu121, torchvision 0.18.0, gym 0.23.1, mujoco-py 2.1.2.14, d4rl 1.1, hydra-core 1.2.0, pymunk 6.8.0 |
| MuJoCo | 2.1.0 (`~/.mujoco/mujoco210`). mujoco-py 는 EGL(GPU) 빌더로 컴파일됐다 |
| DINOv2 | `models/dino.py` 가 `facebookresearch/dinov2:85a2460` 을 torch.hub 로 받는다 (현재 main 은 Python 3.10 문법) |
| PyFleX | AdaptiGraph `a7c7535` 의 PyFleX, `xingyu/softgym` 도커(CUDA 9.2)에서 빌드. 헤드리스 EGL — `EGL_GPU` 는 렌더링할 EGL 장치 번호 (기본 0, 작업의 GPU 와 따로 정해진다. 바꾸려면 overlay 에 `env:EGL_GPU=N`) |

## 데이터·체크포인트

OSF 프로젝트 [bmw48](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28) 에서 `repro/setup/download_data.sh` 로 받는다 (zip 21 GB, 풀면 약 236 GB). 데이터셋 내용은 [SETTINGS.md §3](SETTINGS.md#3-데이터).
공개 체크포인트는 `point_maze`, `pusht`, `wall_single` 셋이다 (upstream README 의 `model_name=wall` 은 `wall_single` 이 맞다). Rope·Granular 체크포인트는 공개되지 않아 직접 학습한다.

---

## 실행

레포 루트에서 실행한다. 셸마다 `source repro/env.sh` 를 먼저 한다 (bash 또는 zsh).

```bash
# 벤치마크 전체: 작업 큐. 위에서부터 GPU 슬롯이 나는 대로 띄운다 (기본: GPU 당 1개 — 머신별 조건은 아래)
bash repro/queue/start.sh                   # 로그 $DINO_RUNS/scheduler.log
bash repro/queue/wait_event.sh 3300 major   # 작업이 끝나거나 실패할 때까지 기다렸다가 요약 출력
python3 repro/eval/summarize.py             # $DINO_WORK/results/summary.md
bash repro/queue/kill_job.sh pusht_mpc      # 작업 하나 끄기 (다시 넣으려면 $DINO_RUNS/pusht_mpc 를 옮기고 큐를 다시 띄운다)
```

작업 하나만 손으로 돌릴 때 (큐가 쓰는 것과 같은 명령, 두 번째 인자가 GPU 번호).

```bash
# 공개 체크포인트 MPC-CEM
bash repro/queue/run_plan.sh pusht_mpc <gpu> --config-name plan_pusht.yaml model_name=pusht planner.max_iter=10

# Rope 학습 → planning
bash repro/queue/run_train.sh train_rope <gpu> env=deformable_env env.dataset.object_name=rope env.kwargs.object_name=rope \
  frameskip=1 num_hist=1 training.save_every_x_epoch=10
CKPT_BASE=$DINO_TRAIN bash repro/queue/run_plan.sh rope_mpc <gpu> --config-name plan.yaml model_name=train_rope \
  planner=mpc_cem planner.max_iter=5 planner.sub_planner.eval_every=1000 n_evals=10 goal_source=random_state goal_H=5

# 예측 품질
python repro/eval/eval_pred_quality.py $DINO_CKPT pusht $DINO_WORK/results/pred_quality/pusht.json 5000

# 목표 윤곽이 없는 예전 PushT 영상을 지금 형식으로 다시 쓴다 -> $DINO_RUNS/<job>/goal_overlay/ (원본은 그대로)
python repro/eval/pusht_goal_overlay.py $DINO_RUNS/pusht_mpc $DINO_RUNS/pusht_gd
# PointMaze 도 같은 형식으로 (목표는 실행 디렉토리의 plan_targets.pkl 에서 읽는다)
python repro/eval/pointmaze_goal_overlay.py $DINO_RUNS/pointmaze_mpc $DINO_RUNS/pointmaze_gd
# 목표 표시가 필요 없는 환경(Wall)은 라벨·3초 멈춤만 -> $DINO_RUNS/<job>/relabeled/
python repro/eval/reformat_eval_videos.py $DINO_RUNS/wall_mpc $DINO_RUNS/wall_cem30 $DINO_RUNS/wall_gd
# Rope·Granular 평가 영상에서 팔이 움직이게: 원래 작업과 같은 인자·DINO_WM_* 환경으로 다시 돌린다 (run_plan.sh 가 붙이는 ckpt_base_path 를 직접 준다).
# 같은 시작·목표의 재실행이지만 FleX 결과가 프로세스마다 비트 단위로 같지는 않아 에피소드별 값이 원래 실행과 다를 수 있다
CUDA_VISIBLE_DEVICES=<gpu> python repro/eval/deform_arm_video.py $DINO_RUNS/rope_mpc_arm --config-name plan.yaml model_name=train_rope \
  planner=mpc_cem planner.max_iter=5 planner.sub_planner.eval_every=1000 n_evals=10 goal_source=random_state goal_H=5 ckpt_base_path=$DINO_TRAIN
```

- 산출물: planning 은 `$DINO_RUNS/<job>/` 에 `logs.json`(MPC 반복별 `mpc/*`, 마지막 `final_eval/*`), `plan_targets.pkl`(시작·목표, `goal_source=file` 로 재사용 가능), 그림·영상. 학습은 `$DINO_TRAIN/outputs/<job>/` 에 `hydra.yaml`, `checkpoints/model_<epoch>.pth`, `epoch_logs.jsonl`.
- 같은 job 을 다시 돌리려면 `$DINO_RUNS/<job>/` 를 지우거나 옮긴다 (`DONE` 이 있으면 건너뛴다).
- GPU 메모리가 부족하면(예: 24 GB) 환경 브랜치의 메모리 수정을 쓴다 — [`hanbin5/local`](https://github.com/jongmin-s-multicore/thread0/tree/hanbin5/local) 의 `repro/LOCAL.md`.
- PointMaze 작업은 env 워커 50개가 각자 mujoco-py EGL 컨텍스트를 `CUDA_VISIBLE_DEVICES` 의 첫 GPU 에 열어 GPU 메모리를 수 GB 더 쓴다 (upstream `SubprocVectorEnv` 동작). GPU 하나에 작업 여러 개를 돌린다면 overlay 로 `need=` 를 준다.

### 머신별 설정 (레포에 올리지 않는다)

GPU 개수·메모리에 따라 정하는 값은 레포 밖에 둔다. `repro/env.sh` 는 `.claude/env.local.sh` 가 있으면 먼저 읽는다 (`.claude/` 는 gitignore). 그 파일에서도 `${VAR:-값}` 으로 써서 이미 export 한 값을 존중한다.

| 변수 | 뜻 | 기본 |
|---|---|---|
| `DINO_WORK` | 작업 루트 | `~/dinowm` |
| `DINO_GPUS`, `DINO_SLOTS_PER_GPU` | 큐가 쓰는 GPU, GPU 당 동시 작업 수 | nvidia-smi 의 GPU 전부, 1 |
| `DINO_NEED_MB_{PLAN,TRAIN,CMD}` | 작업 종류별로 시작에 필요한 GPU 여유 메모리 (MB) | 0 |
| `DINO_QUEUE_COOLDOWN` | 같은 GPU 에 연달아 띄우는 간격 (초) | 60 |
| `DINO_JOBS_LOCAL` | 작업별 조건 overlay 파일. 한 줄: `<job> [gpu=N] [need=MB] [env:K=V ...]` | 없음 |

```bash
# .claude/env.local.sh 예 — 값은 머신에 맞춰 정한다
export DINO_WORK=${DINO_WORK:-/path/to/dinowm}
export DINO_SLOTS_PER_GPU=${DINO_SLOTS_PER_GPU:-<N>}
export DINO_NEED_MB_PLAN=${DINO_NEED_MB_PLAN:-<MB>} DINO_NEED_MB_TRAIN=${DINO_NEED_MB_TRAIN:-<MB>}
export DINO_JOBS_LOCAL=${DINO_JOBS_LOCAL:-$THREAD0/.claude/jobs.local.txt}
# .claude/jobs.local.txt 예:  pointmaze_gd need=<MB>
```

---

## 결과와 코드 버전

결과는 GitHub 이슈로 올리고, 그 결과를 낸 브랜치와 커밋 해시를 적는다. 러너(`run_plan.sh`·`run_train.sh`·`run_cmd.sh`)는 `run_info.txt` 에 브랜치·커밋·`DINO_WM_*` 환경변수를 기록한다 (이 형식 전에 시작한 실행의 run_info 에는 브랜치·커밋이 없다 — 그 경우 환경 브랜치 문서에 대응을 적는다).

## 라이선스 주의

- 데이터·공개 체크포인트는 OSF 에서 받고 재배포하지 않는다. 직접 학습한 가중치도 이 레포에 넣지 않는다.
- AdaptiGraph (MIT; PyFleX 는 NVIDIA FleX 기반), DINOv2 (Apache-2.0) 는 각 저장소에서 받는다.
