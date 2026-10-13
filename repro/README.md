# repro: DINO-WM 재현 도구와 실행 방법

이 문서는 이 레포(gaoyuezhou/dino_wm 의 fork)로 DINO-WM 의 planning 벤치마크를 다시 실행하는 방법을 설명한다. 여기에는 GPU 환경과 무관한 공용 도구와 프로토콜을 둔다. 공용 도구는 환경 구축, 작업 큐, 평가 도구다. GPU 환경에 맞춘 코드 수정과 그 환경의 실행 기록은 환경별 브랜치에 있다 ([루트 README](../README.md#branches-and-changes-to-upstream)).

- 모델·데이터·하이퍼파라미터·평가 프로토콜·upstream 과 다른 점: [SETTINGS.md](SETTINGS.md)
- 결과와 해석: [이슈](https://github.com/jongmin-s-multicore/thread0/issues)

Reacher, 일반화 환경(WallRandom·PushObj·GranularRandom), 다른 모델 베이스라인은 실행하지 않았다. 데이터나 구현이 공개되지 않았거나 코드가 더 필요하기 때문이다. 인코더 비교(Table 2)도 이번에는 실행하지 않았다 ([SETTINGS.md §1](SETTINGS.md#1-구성)).

---

## 구성

경로는 모두 `repro/env.sh` 에서 정한다. 산출물(`$DINO_RUNS`, `$DINO_TRAIN`, `$DINO_WORK/results`)은 레포 밖에 생긴다. 산출물은 커밋하지 않는다.

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
├── jobs/pusht_long_horizon.txt   PushT 목표 거리별 planning (d = 25·50·75·100, 계획 길이 둘) — SETTINGS.md §5.5
├── eval/
│   ├── eval_pred_quality.py   검증 분할 전체의 1-step 예측 LPIPS·SSIM
│   ├── pusht_goal_overlay.py  예전 PushT 평가 영상을 지금 형식으로 (2배 확대): 목표 칸 이미지에서 맞춘 목표 블록 윤곽, Real/Model/Goal 라벨, 회색 프레임 대신 3초 멈춤
│   ├── pointmaze_goal_overlay.py  예전 PointMaze 평가 영상을 지금 형식으로 (2배 확대): 목표 위치(plan_targets.pkl 의 state_g) 원, Real/Model/Goal 라벨, 회색 프레임 대신 3초 멈춤
│   ├── reformat_eval_videos.py    예전 평가 영상을 지금 형식으로, 목표 표시 없이 (Wall 등): Real/Model/Goal 라벨, 회색 프레임 대신 3초 멈춤
│   ├── video_format.py        위 세 도구가 같이 쓰는 함수 (가려진 회색 프레임 세기, 라벨이 이미 있는지)
│   ├── deform_arm_video.py    Rope·Granular planning 작업을 다시 돌려, 팔(xArm6)이 미는 중간 프레임까지 담은 평가 영상도 쓴다 (<out_dir>/arm_video/)
│   ├── plan_dump_actions.py   plan.py 를 감싸 실행하고 계획한 행동(diag_actions.pkl)도 남긴다
│   ├── pusht_fulldemo_targets.py  PushT 데모 전체 목표 파일 (검증 데모의 첫 상태 → 끝 상태, goal_source=file 용) — SETTINGS.md §5.5
│   ├── pusht_segment_targets.py   PushT 목표 파일: 검증 데모에서 d 스텝 떨어진 시작–목표 쌍 (goal_source=dset 과 같은 방식, 파일로 고정)
│   ├── pusht_fulldemo_mpc.sh      그 목표로 PushT MPC 실행 (pusht_mpc 와 같은 설정, plan_dump_actions.py 경유)
│   ├── pusht_fulldemo_analyze.py  실행한 행동을 재생해 에피소드별 성공·블록 거리·coverage
│   ├── pusht_openloop_error.py    데모의 행동으로 world model 을 끝까지 굴렸을 때의 스텝별 오차
│   ├── pusht_cost_landscape.py    planning 비용을 실제 프레임에서: 데모 경로 대 MPC 경로
│   └── summarize.py           logs.json·epoch_logs·예측 품질을 $DINO_WORK/results/summary.{md,json} 으로
└── requirements/
    ├── requirements-dinowm.txt    pip 패키지 목록 (uv pip freeze)
    └── conda-dinowm.txt           conda 패키지 목록 (micromamba list)
```

job 은 `jobs/benchmark.txt` 와 `$DINO_RUNS/<job>/` 의 이름이다. 실행 기록(`repro/runs/{train,eval}/<run-id>/run.yaml`)은 그 실행을 한 환경 브랜치에 둔다. run-id 형식은 [AGENTS.md §2](../AGENTS.md#2-재현-디렉토리-구조) 에 있다.

---

## 환경

### 설치

한 셸(bash 또는 zsh)에서 레포 루트로 차례로 실행한다. 아래 명령 블록에는 `#` 주석을 넣지 않는다. zsh 는 `setopt interactivecomments` 가 설정되어 있지 않으면 줄 끝 `# ...` 을 명령의 인자로 넘기기 때문이다. oh-my-zsh 등이 이 옵션을 설정하지 않은 zsh 의 기본 상태가 여기에 해당한다.

0. 사전 준비: NVIDIA 드라이버, `git curl bzip2 unzip zip gcc`, mujoco-py 빌드용 `libglew-dev libgl1-mesa-dev` 가 필요하다. `/usr/lib/nvidia` 는 우분투 드라이버 패키지가 만든다. 이 경로가 없으면 mujoco-py 가 CPU 렌더러로 빌드되고 `libosmesa6-dev` 도 필요하다. Rope·Granular planning 에는 docker 와 NVIDIA container runtime 이 필요하다. docker 는 sudo 없이 docker 그룹으로 쓴다. 디스크는 `$DINO_WORK` 에 PointMaze·PushT·Wall 용으로 약 110 GB 가 필요하다. Rope·Granular 에는 약 160 GB 가 더 필요하다. 압축을 푸는 동안에는 15 GB 가 더 필요하다. 홈이 작으면 큰 디스크를 작업 루트로 정한다.
1. 레포와 작업 루트. 홈 디렉토리를 가정하지 않는다. 아래 명령은 지금 디렉토리에 clone 한다. 레포 위치는 어디든 된다. `DINO_WORK` 는 레포 밖이면 어디든 된다. 셋째 줄의 `~/dinowm` 은 기본값이다. 홈이 작으면 큰 디스크 경로로 바꾼다. 다른 경로를 쓰면 새 셸마다 다시 export 하거나 `.claude/env.local.sh` 에 적는다. 이 파일은 gitignore 대상이고 아래 "머신별 설정"에서 설명한다. 홈 아래로 고정되는 것은 MuJoCo 하나다. 위치는 mujoco-py 의 기본 위치인 `~/.mujoco/mujoco210` 이고 `MUJOCO_DIR` 로 바꿀 수 있다.

   ```bash
   git clone https://github.com/jongmin-s-multicore/thread0.git
   cd thread0
   export DINO_WORK=~/dinowm
   ```

2. 설치: micromamba env (Python 3.9, `environment.yaml` 고정 버전)를 만들고 MuJoCo 2.1.0 을 설치하고 mujoco-py 를 빌드한다. `uv` 가 PATH 에 있으면 uv 로, 없으면 pip 로 설치한다. `--dry-run` 은 실행할 명령만 출력한다. `install_pyflex.sh` 는 Rope·Granular planning 에만 필요하다. 도커에서 빌드하고 약 1분 걸린다.

   ```bash
   bash repro/setup/install_env.sh --dry-run
   bash repro/setup/install_env.sh
   bash repro/setup/install_pyflex.sh
   ```

3. 데이터와 공개 체크포인트 (OSF). `core checkpoints` 는 PointMaze·PushT·Wall 과 체크포인트를 받는다. `deformable` 은 Rope·Granular 를 받는다. 모두 받으면 zip 이 21 GB 이고 풀면 약 236 GB 다. 다운로드가 끊기면 같은 명령을 다시 실행한다. 그러면 이어 받는다.

   ```bash
   bash repro/setup/download_data.sh core checkpoints
   bash repro/setup/download_data.sh deformable
   ```

4. 점검. `source repro/env.sh` 는 새 셸마다 레포 루트에서 실행한다. `check_env.sh` 는 torch/CUDA, mujoco-py, DINOv2 를 점검하고 공개 체크포인트로 렌더·동역학을 대조한다. `--pyflex` 를 주면 PyFleX 와 Rope·Granular 도 점검한다.

   ```bash
   source repro/env.sh
   bash repro/setup/check_env.sh
   bash repro/setup/check_env.sh --pyflex
   ```

5. 공개 체크포인트로 planning 을 한 번 실행한다 (PointMaze, 에피소드 2개, 1분 이내).

   ```bash
   python plan.py --config-name plan_point_maze.yaml model_name=point_maze ckpt_base_path=$DINO_CKPT \
     n_evals=2 planner.sub_planner.opt_steps=2 planner.max_iter=1 hydra.run.dir=$DINO_RUNS/smoke/point_maze
   ```

6. 벤치마크 전체를 작업 큐로 실행한다. GPU 메모리가 부족하면 환경 브랜치를 쓴다. 환경 브랜치는 루트 README 의 "Branches and changes to upstream" 에 있다. 큐 설정은 아래 "머신별 설정"에 있다.

   ```bash
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

`repro/setup/` 스크립트는 위 환경에서 손으로 실행한 명령을 스크립트로 옮겨 만들었다. 2026-10-09 에 같은 머신에서 빈 홈 디렉토리와 빈 환경변수(`env -i`)로 위 1–5 를 처음부터 끝까지 실행했다. uv 없이 pip 를 썼고 셸은 bash 였다. `check_env.sh --pyflex` 를 포함해 모든 단계가 성공했다. 그 실행에서 `core`·`deformable` 다운로드는 기존 데이터의 링크로 대신했다. `checkpoints` 다운로드는 같은 스크립트로 받았다. 작업 큐(6)는 실행하지 않았다. `/usr/lib/nvidia` 가 없는 Ubuntu 24.04 컨테이너에서는 mujoco-py 빌드가 `libosmesa6-dev` 가 없으면 실패했고 있으면 성공했다. CPU 렌더러의 PointMaze 프레임은 데이터셋과 0~255 픽셀 평균 0.03 차이가 났고 EGL 은 0.00 이었다. 다른 배포판·GPU 는 확인하지 않았다.

### 설치된 구성

| 항목 | 값 |
|---|---|
| Python env | micromamba prefix env, Python 3.9.19 (conda-forge), `environment.yaml` 의 pip 고정 버전 217개와 patchelf. 목록은 [requirements/](requirements/) 에 있다 |
| 주요 패키지 | torch 2.3.0+cu121, torchvision 0.18.0, gym 0.23.1, mujoco-py 2.1.2.14, d4rl 1.1, hydra-core 1.2.0, pymunk 6.8.0 |
| MuJoCo | 2.1.0 (`~/.mujoco/mujoco210`). mujoco-py 는 EGL(GPU) 빌더로 컴파일됐다 |
| DINOv2 | `models/dino.py` 가 `facebookresearch/dinov2:85a2460` 을 torch.hub 로 받는다. 현재 main 은 Python 3.10 문법을 쓴다 |
| PyFleX | AdaptiGraph `a7c7535` 의 PyFleX, `xingyu/softgym` 도커(CUDA 9.2)에서 빌드. 헤드리스 EGL 을 쓴다. `EGL_GPU` 는 렌더링할 EGL 장치 번호다. 기본은 0 이고 작업의 GPU 와 따로 정해진다. 바꾸려면 overlay 에 `env:EGL_GPU=N` 을 적는다 |

## 데이터·체크포인트

OSF 프로젝트 [bmw48](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28) 에서 `repro/setup/download_data.sh` 로 받는다 (zip 21 GB, 풀면 약 236 GB). 데이터셋 내용은 [SETTINGS.md §3](SETTINGS.md#3-데이터) 에 있다.
공개 체크포인트는 `point_maze`, `pusht`, `wall_single` 셋이다. upstream README 는 `model_name=wall` 이라고 쓰지만 맞는 이름은 `wall_single` 이다. Rope·Granular 체크포인트는 공개되지 않아 학습해서 쓴다.

---

## 실행

레포 루트에서 실행한다. 셸마다 `source repro/env.sh` 를 먼저 실행한다 (bash 또는 zsh).

| 명령 | 하는 일 |
|---|---|
| `bash repro/queue/start.sh` | 벤치마크 전체를 작업 큐로 실행한다. 위에서부터 GPU 슬롯이 비는 대로 시작한다. 기본은 GPU 당 1개다. 머신별 조건은 아래에 있다. 로그는 `$DINO_RUNS/scheduler.log` 에 쓴다 |
| `bash repro/queue/wait_event.sh 3300 major` | 작업이 끝나거나 실패할 때까지 기다린 뒤 요약을 출력한다 |
| `python3 repro/eval/summarize.py` | `$DINO_WORK/results/summary.md` 에 요약을 쓴다 |
| `bash repro/queue/kill_job.sh pusht_mpc` | 작업 하나를 종료한다. 다시 넣으려면 `$DINO_RUNS/pusht_mpc` 를 옮기고 큐를 다시 시작한다 |

작업 하나를 손으로 실행할 때는 아래 명령을 쓴다. 큐가 쓰는 명령과 같다. 두 번째 인자가 GPU 번호다.

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

PushT 목표 거리별 planning (SETTINGS.md §5.5). 목표 파일을 만들고 큐에 작업 목록을 준다.

```bash
for d in 25 50 75 100; do python repro/eval/pusht_segment_targets.py $DINO_WORK/results/pusht_long_horizon/targets_d${d}_n20.pkl $d 20; done
bash repro/queue/start.sh repro/jobs/pusht_long_horizon.txt
```

PushT 데모 전체 목표 (SETTINGS.md §5.5). 큐를 거치지 않고 손으로 돌린다.

```bash
# 목표 파일: targets_all.pkl 과, GPU 2장에 나눠 돌릴 targets_p0.pkl · targets_p1.pkl
python repro/eval/pusht_fulldemo_targets.py $DINO_WORK/results/pusht_full_demo 2
# MPC (인자: 이름, GPU, 목표 파일). 산출물은 $DINO_RUNS/<이름>/
bash repro/eval/pusht_fulldemo_mpc.sh pusht_fulldemo_mpc_p0 <gpu0> $DINO_WORK/results/pusht_full_demo/targets_p0.pkl
bash repro/eval/pusht_fulldemo_mpc.sh pusht_fulldemo_mpc_p1 <gpu1> $DINO_WORK/results/pusht_full_demo/targets_p1.pkl
# 분석
python repro/eval/pusht_fulldemo_analyze.py $DINO_WORK/results/pusht_full_demo/mpc_s99.json \
  $DINO_RUNS/pusht_fulldemo_mpc_p0 $DINO_RUNS/pusht_fulldemo_mpc_p1
CUDA_VISIBLE_DEVICES=<gpu> python repro/eval/pusht_openloop_error.py $DINO_WORK/results/pusht_full_demo
CUDA_VISIBLE_DEVICES=<gpu> python repro/eval/pusht_cost_landscape.py $DINO_WORK/results/pusht_full_demo \
  $DINO_RUNS/pusht_fulldemo_mpc_p0 $DINO_RUNS/pusht_fulldemo_mpc_p1
```

- 산출물: planning 은 `$DINO_RUNS/<job>/` 에 `logs.json`(MPC 반복별 `mpc/*`, 마지막 `final_eval/*`), `plan_targets.pkl`(시작·목표, `goal_source=file` 로 재사용 가능), 그림·영상을 쓴다. 학습은 `$DINO_TRAIN/outputs/<job>/` 에 `hydra.yaml`, `checkpoints/model_<epoch>.pth`, `epoch_logs.jsonl` 을 쓴다.
- 같은 job 을 다시 실행하려면 `$DINO_RUNS/<job>/` 를 지우거나 옮긴다. `DONE` 이 있으면 그 job 은 건너뛴다.
- GPU 메모리가 부족하면(예: 24 GB) 환경 브랜치의 메모리 수정을 쓴다. [`hanbin5/local`](https://github.com/jongmin-s-multicore/thread0/tree/hanbin5/local) 의 `repro/LOCAL.md` 에 설명이 있다.
- PointMaze 작업은 env 워커 50개가 각자 mujoco-py EGL 컨텍스트를 `CUDA_VISIBLE_DEVICES` 의 첫 GPU 에 열어 GPU 메모리를 수 GB 더 쓴다. upstream `SubprocVectorEnv` 의 동작이다. GPU 하나에 작업 여러 개를 실행한다면 overlay 로 `need=` 를 준다.

### 머신별 설정 (레포에 올리지 않는다)

GPU 개수·메모리에 따라 정하는 값은 레포 밖에 둔다. `repro/env.sh` 는 `.claude/env.local.sh` 가 있으면 먼저 읽는다. `.claude/` 는 gitignore 대상이다. 그 파일에서도 `${VAR:-값}` 형식으로 쓴다. 이 형식은 셸에서 먼저 export 한 값을 유지한다.

| 변수 | 뜻 | 기본 |
|---|---|---|
| `DINO_WORK` | 작업 루트 | `~/dinowm` |
| `DINO_GPUS`, `DINO_SLOTS_PER_GPU` | 큐가 쓰는 GPU, GPU 당 동시 작업 수 | nvidia-smi 의 모든 GPU, 1 |
| `DINO_NEED_MB_{PLAN,TRAIN,CMD}` | 작업 종류별로 시작에 필요한 GPU 여유 메모리 (MB) | 0 |
| `DINO_QUEUE_COOLDOWN` | 같은 GPU 에 작업을 연달아 시작하는 간격 (초) | 60 |
| `DINO_JOBS_LOCAL` | 작업별 조건 overlay 파일. 한 줄: `<job> [gpu=N] [need=MB] [env:K=V ...]` | 없음 |

```bash
# .claude/env.local.sh 예. 값은 머신에 맞춰 정한다
export DINO_WORK=${DINO_WORK:-/path/to/dinowm}
export DINO_SLOTS_PER_GPU=${DINO_SLOTS_PER_GPU:-<N>}
export DINO_NEED_MB_PLAN=${DINO_NEED_MB_PLAN:-<MB>} DINO_NEED_MB_TRAIN=${DINO_NEED_MB_TRAIN:-<MB>}
export DINO_JOBS_LOCAL=${DINO_JOBS_LOCAL:-$THREAD0/.claude/jobs.local.txt}
# .claude/jobs.local.txt 예:  pointmaze_gd need=<MB>
```

---

## 결과와 코드 버전

결과는 GitHub 이슈로 올린다. 이슈에는 그 결과를 낸 브랜치와 커밋 해시를 적는다. 러너(`run_plan.sh`·`run_train.sh`·`run_cmd.sh`)는 `run_info.txt` 에 브랜치·커밋·`DINO_WM_*` 환경변수를 기록한다. 이 형식 전에 시작한 실행의 run_info 에는 브랜치·커밋이 없다. 그 경우 환경 브랜치 문서에 대응을 적는다.

## 라이선스 주의

- 데이터·공개 체크포인트는 OSF 에서 받는다. 재배포하지 않는다. 학습해서 만든 가중치도 이 레포에 넣지 않는다.
- AdaptiGraph (MIT), DINOv2 (Apache-2.0) 는 각 저장소에서 받는다. AdaptiGraph 의 PyFleX 는 NVIDIA FleX 기반이다.
