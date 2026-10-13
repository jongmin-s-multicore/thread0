# DINO-WM 재현: 실험 세팅

DINO-WM([arXiv:2411.04983](https://arxiv.org/abs/2411.04983) v2, 코드 [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) `0a9492f`)의 벤치마크를 다시 실행하는 실험의 공용 설정이다. GPU 환경과 무관한 설정만 적는다. 환경별 수정과 측정은 환경 브랜치에 둔다. 결과와 해석은 [이슈](https://github.com/jongmin-s-multicore/thread0/issues)로 따로 올린다. 코드·환경·실행 방법은 [README](README.md) 에 있다. 이 문서의 `queue/`·`jobs/`·`eval/`·`setup/` 경로는 `repro/` 기준이다. 그 밖의 파일 경로는 upstream 코드가 있는 레포 루트 기준이다.
수치는 upstream 설정 파일(`conf/`), 공개 체크포인트의 `hydra.yaml`, 환경 브랜치의 `repro/runs/*/*/run.yaml`(예: `hanbin5/local`), 실행 로그에서 가져왔다. 논문 수치는 v2(2025-02) 기준이다.

## 1. 구성

| 논문 표 | 환경 | world model | planner | 지표 | 인스턴스 | job |
|---|---|---|---|---|---|---|
| Table 1, 8 "MPC" | PointMaze, PushT, Wall | 공개 체크포인트 | MPC-CEM | 성공률 | 50 | `pointmaze_mpc`, `pusht_mpc`, `wall_mpc` |
| Table 8 "CEM" | PointMaze, PushT, Wall | 공개 체크포인트 | 오픈루프 CEM = MPC 1회차 (§5.3) | 성공률 | 50 | 위 MPC 실행의 step 1 |
| Table 8 "GD" | PointMaze, PushT, Wall | 공개 체크포인트 | 오픈루프 GD | 성공률 | 50 | `<env>_gd` |
| 추가 | PointMaze, Wall | 공개 체크포인트 | 오픈루프 CEM, planner 그룹 기본값 30 opt steps | 성공률 | 50 | `<env>_cem30` |
| Table 1, 8 Rope·Granular | Rope, Granular | 이 실험에서 학습 (§4.2) | MPC-CEM | Chamfer distance | 10 | `train_<obj>`, `<obj>_mpc` |
| Table 4, 9 | PushT, Wall, Rope, Granular (+PointMaze) | 위와 같음 | – | 디코딩한 1-step 예측의 LPIPS·SSIM | 검증 분할 | `pq_<env>` |

job 이름은 `repro/jobs/benchmark.txt` 에 적힌 이름이다. 각 실행의 설정 기록(run-id 와 run.yaml)은 그 실행을 수행한 환경 브랜치의 `repro/runs/` 에 있다.

재현하지 않은 것과 이유:

| 대상 | 이유 |
|---|---|
| Reacher (Table 1, 2) | 데이터셋·환경 코드·체크포인트가 공개되지 않았다 (dino_wm issue #22) |
| IRIS, DreamerV3, TD-MPC2, AVDC (Table 1, 3, 4, 9) | dino_wm 에 구현이 없다. 논문은 외부 코드베이스를 썼고 설정을 공개하지 않았다 |
| WallRandom, PushObj (Table 3) | 일반화 실험용 데이터셋과 생성 스크립트가 공개되지 않았다 |
| GranularRandom (Table 3) | 같은 Granular 모델을 입자 수가 다른 환경에서 평가하는 실험이다. 데이터는 필요 없다. 환경 설정(`obj_params.area`)과 시작 상태 샘플링을 바꾸는 코드가 필요하다. 실행하지 않았다 |
| 인코더 비교 R3M·ResNet·DINO-CLS (Table 2) | 저장소에 설정(`encoder=r3m|resnet|dino_cls decoder=transposed_conv`)은 있다. 환경 5개 × 인코더 3개 = world model 15개를 새로 학습해야 한다. 이번에는 실행하지 않았다 |
| 데이터 크기·causal mask·decoder loss ablation (Table 5–7) | 새 학습이 필요하다. mask·decoder loss 는 코드 수정도 필요하다. 실행하지 않았다 |
| Table 10 (A6000 시간 측정) | 하드웨어가 다르다. 환경별 측정은 환경 브랜치 문서에 둔다 (예: `hanbin5/local` 의 `repro/LOCAL.md`) |

## 2. 모델

공개 체크포인트와 이 실험에서 학습한 모델은 모두 upstream `conf/train.yaml` 의 구조다.

| 항목 | 값 |
|---|---|
| 인코더 | DINOv2 ViT-S/14 (`torch.hub` `dinov2_vits14`, hub 코드는 `85a2460` 고정), `x_norm_patchtokens`. 224×224 입력을 196×196 으로 줄여 14×14 = 196 패치 × 384 차원. 동결 (`model.train_encoder=False`) |
| action / proprio 인코더 | Conv1d(k=1) 임베딩 10차원씩, 패치마다 이어 붙인다 (`concat_dim=1`). 토큰은 404 차원이다 |
| predictor | ViT, depth 6, heads 16, dim_head 64, MLP 2048, dropout 0.1, emb_dropout 0 (Wall 공개 체크포인트는 0.1). 프레임 단위 causal mask. 파라미터 20.12M (h=3), 19.96M (h=1) |
| decoder | VQ-VAE 디코더 (`quantize=False`), 10.14M. 입력은 `z.detach()` 다. 디코더 손실은 predictor 로 전파되지 않는다 |
| planning 시 모드 | 체크포인트의 predictor·decoder 는 eval 모드로 저장돼 있어서 planning 중 dropout 이 적용되지 않는다. `plan.py` 는 `model.eval()` 을 호출하지 않는다. 이 동작은 upstream 과 같다 |

## 3. 데이터

데이터는 [OSF](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28) `datasets/` 에서 받는다 (`repro/setup/download_data.sh`). 궤적 수와 길이는 파일에서 셌다.

| 데이터셋 | 궤적 × 길이 | action | 분할 | 풀린 크기 |
|---|---|---|---|---|
| `point_maze` | 2000 × 100 | 2 | 무작위 0.9 / 0.1 (seed 42) | 29 GB |
| `pusht_noise` | train 18685, val 21 (길이 49–246, 평균 125) | 2 | 폴더 (`train/` `val/`) | 7 GB |
| `wall_single` | 1920 × 50 | 2 | 무작위 0.9 / 0.1 (seed 42) | 55 GB |
| `deformable/rope` | 1000 × 20 프레임 (action 20 × 4) | 4 (push 시작·끝 xz) | 무작위 0.9 / 0.1 (seed 42) | 69 GB |
| `deformable/granular` | 1000 × 20 프레임 (action 20 × 4) | 4 | 무작위 0.9 / 0.1 (seed 42) | 75 GB |

- 논문 Table 11 은 Rope·Granular 궤적 길이를 5 로 적었다. 부록 A.1 본문과 실제 데이터는 20 이다.
- deformable 프레임은 float32 `obses.pth` (20, 224, 224, 3) 로 저장돼 있다.

## 4. 학습

### 4.1 공개 체크포인트 (OSF `checkpoints/outputs.zip`, `hydra.yaml` 에서)

| 항목 | point_maze | pusht | wall_single |
|---|---|---|---|
| 저장된 epoch / 설정 epochs | 10 / 500 | 2 / 100 | 65 / 1000 |
| num_hist / frameskip | 3 / 5 | 3 / 5 | 1 / 5 |
| batch (전체 / GPU 당) | 32 / 2 | 32 / 1 | 128 / 4 |
| lr (encoder / decoder / predictor / action enc) | 1e-6 / 3e-4 / 5e-4 / 5e-4 | 같음 | 같음 |
| img_size | 224 | 224 | 224 |

predictor lr 는 논문 Table 12 에 5e-5 로 적혀 있다. upstream 기본값과 공개 체크포인트는 5e-4 다.

### 4.2 Rope·Granular (직접 학습)

공개 체크포인트가 없어서 upstream `train.py` 로 학습한다. 논문 Table 11·12 의 값과 upstream 기본값을 따랐다.

| 항목 | 값 |
|---|---|
| 명령 | `python train.py --config-name train.yaml env=deformable_env env.dataset.object_name=<obj> env.kwargs.object_name=<obj> frameskip=1 num_hist=1 training.save_every_x_epoch=10` |
| num_hist / num_pred / frameskip | 1 / 1 / 1 |
| epochs / batch | 100 / 32 (GPU 1장) |
| optimizer | predictor·action encoder AdamW, decoder Adam. upstream 과 같다 |
| lr | decoder 3e-4, predictor 5e-4, action encoder 5e-4. upstream 기본값이고 공개 체크포인트와 같다 |
| 학습 표본 | 궤적 900 × 슬라이스 19 = 17,100 → 535 iteration/epoch |
| 체크포인트 | 10 epoch 마다 (`model_<ep>.pth`, `model_latest.pth`) |
| seed | 0 (`training.seed`) |

Rope 를 이 설정으로 100 epoch 학습해도 논문 CD 0.41 이 나오지 않았다는 보고가 있다 (dino_wm issue #24). 저자는 답하지 않았다.

## 5. 평가 프로토콜

### 5.1 planning 설정

upstream 설정 파일의 값을 쓴다. 바꾼 값은 굵게 표시했다.

| 항목 | PointMaze (`plan_point_maze.yaml`) | PushT (`plan_pusht.yaml`) | Wall (`plan_wall.yaml`) | Rope·Granular (`plan.yaml` + `planner=mpc_cem`) |
|---|---|---|---|---|
| 인스턴스 (`n_evals`) | 50 | 50 | 50 | 10 |
| 목표 (`goal_source`) | `random_state` | `dset` | `random_state` | **`random_state`** (`plan.yaml` 기본값은 `dset`) |
| `goal_H` (= CEM horizon = MPC 실행 길이) | 5 | 5 | 5 | 5 |
| 목적 함수 | 마지막 예측 프레임의 latent MSE, proprio 가중치 alpha 0 | alpha 1 | alpha 1 | alpha 1. proprio 는 0 으로 고정된 더미다 |
| CEM samples / topk / var_scale | 300 / 30 / 1 | 300 / 30 / 1 | 300 / 30 / 1 | 300 / 30 / 1 |
| CEM opt steps | 10 | 30 | 10 | 30 |
| MPC 최대 반복 (`max_iter`) | **20** (원본 null) | **10** (원본 null) | **20** (원본 null) | **5** (원본 null) |
| 내부 CEM 시뮬 평가 (`eval_every`) | 1 | 1 | 1 | **1000** (§6) |
| seed | 99 → 에피소드 seed `99·n+1` | 같음 | 같음 | 같음 |

- `goal_H=5` 는 macro action 5개다. frameskip 5 인 환경에서는 env 25 스텝이다. Rope·Granular 에서는 push 5번이다.
- PushT 의 `dset` 목표는 검증 궤적의 구간을 env 에서 재생해 얻은 25 스텝 뒤 상태다. 이 목표는 25 스텝 안에 도달 가능하다.
- PushT 화면의 연두색 T 는 원래 PushT 의 고정 목표 (256, 256, π/4) 다. env 가 항상 이 T 를 그리며, 학습 데이터에도 들어 있다. 연두색 T 는 planning 목표가 아니다. planning 목표는 평가 영상·그림의 오른쪽 칸에 있다. 오른쪽 칸은 목표 상태를 렌더한 이미지다. `plan.py` 가 왼쪽 칸에 목표 블록 자세를 빨간 윤곽으로 그린다. 윤곽은 그림·영상에만 그리고, 관측과 지표는 바뀌지 않는다. 평가 영상의 칸 라벨은 `Real` = 실제 env, `Model` = 월드모델 예측, `Goal` = 목표다. 이 윤곽이 없는 예전 영상에는 `repro/eval/pusht_goal_overlay.py` 로 윤곽을 그린다.
- PointMaze 화면에는 초록 점으로 표시된 에이전트만 있다. env 는 목표 마커를 화면 밖에 둔다 (`with_target=False`). `plan.py` 가 평가 영상·그림의 왼쪽 칸에 목표 위치(state_g[:2])를 중심으로 빨간 원(반지름 = 성공 반경 0.5)과 안쪽으로 짧은 눈금 넷을 그린다. 이 표시는 그림·영상에만 그린다. 성공 판정은 상태로 한다. env 는 물리 하위 스텝(0.01 초) 하나 전 위치를 그린다. 이 때문에 경계 가까이에서 끝난 에피소드는 마지막 프레임의 초록 점이 원 경계 가까이의 안쪽 또는 바깥쪽에 보일 수 있다. 지연은 속도 × 0.01 이고, 최고 속도에서 약 0.07 이다. 이 표시가 없는 예전 영상에는 `repro/eval/pointmaze_goal_overlay.py` 로 표시를 그린다. 목표는 실행 디렉토리의 `plan_targets.pkl` 에 있다.
- PointMaze `random_state`: U-maze 빈 공간에서 시작·목표를 독립으로 뽑는다. Wall 에서는 시작과 목표를 벽 반대편 방에서 뽑는다. 벽·문 위치는 검증 궤적에서 가져온다.
- Rope·Granular `random_state`: 시작은 검증 분할의 무작위 궤적에서 무작위 시점(0–18)의 입자 상태다. 목표는 reset 한 모양을 평행이동·회전(Rope) 또는 평행이동·축소(Granular)한 입자 배치다 (`plan.py` `prepare_targets`, `FlexEnvWrapper.sample_random_init_goal_states`).

### 5.2 성공 판정과 지표 (env wrapper 의 `eval_state`)

| 환경 | 성공 | 기록 |
|---|---|---|
| PointMaze | 목표와 위치(x, y) 거리 < 0.5 | 성공률, 상태 거리 |
| PushT | agent·블록 위치 4차원 거리 < 20 그리고 블록 각도 차 < π/9 | 성공률, 상태 거리 |
| Wall | 위치 거리 < 4.5 (env 픽셀 단위) | 성공률, 상태 거리 |
| Rope·Granular | 성공 조건이 CD < 0 이라서 항상 실패한다. upstream 과 같다. 지표는 CD 만 쓴다 | 최종 입자와 목표 입자의 Chamfer distance (xyz, 양방향 평균 거리의 합) |

### 5.3 planner 별 의미

- MPC-CEM (`planning/mpc.py`): CEM 으로 `goal_H` 길이를 계획하고 계획한 행동 모두(`n_taken_actions = goal_H`)를 env 에서 실행한 뒤, 도달한 관측에서 다시 계획한다. 매 반복 끝에 처음부터 지금까지의 행동 전체를 env 에서 다시 실행해 성공을 판정한다. 한 번 성공한 에피소드는 이후에도 성공으로 센다. 성공한 뒤의 행동은 0 이다. `logs.json` 의 `mpc/success_rate`(step = 반복 횟수)가 예산별 성공률이다.
- 오픈루프 CEM = MPC 1회차: `n_taken_actions = horizon` 이라서 MPC 의 첫 반복은 같은 CEM 설정으로 계획한 행동열을 피드백 없이 끝까지 실행하는 것과 같은 계산이다. seed·목표·난수 순서도 같다. Table 8 "CEM" 과의 비교에는 MPC step 1 값을 쓴다. 논문은 오픈루프 CEM 의 opt steps 를 적지 않았으므로 planner 그룹 기본값(30)을 쓰는 변형 `cem30` 을 PointMaze·Wall 에서 따로 실행한다. PushT 는 MPC 설정의 opt steps 가 30 이다.
- 오픈루프 GD (`conf/planner/gd.yaml`): 행동을 표준정규 초기화, SGD lr 1, 1000 스텝, 매 스텝 잡음 0.003. 10 스텝마다 env 에서 평가하고 모두 성공하면 멈춘다.
- CEM·GD 모두 env 의 실제 성공을 조기 종료에만 쓴다. upstream 과 같다 (dino_wm issue #26).

### 5.4 예측 품질 (Table 4, 9)

upstream `train.py` 의 검증과 같은 계산을 검증 분할 전체에 대해 한다 (`eval/eval_pred_quality.py`). 이 계산은 `model.eval()`, `model(obs, act)`, `metrics/image_metrics.eval_images` 로 이루어진다. 이미지 지표는 VGG LPIPS 와 11×11 가우시안 SSIM 이고, 정규화된 [-1, 1] 이미지에서 계산한다. 슬라이스가 5000 개를 넘으면 5000 개만 쓴다. `train.py` 처럼 `seed(training.seed)` 뒤에 데이터셋을 만들어 무작위 순열인 슬라이스 순서를 학습 때와 같게 하고, 그 순서에서 균등 간격으로 고른다. upstream 검증은 첫 배치(32개)에서만 이미지 지표를 계산한다. 논문은 예측 길이와 표본 수를 적지 않았다.

## 6. upstream 코드·논문과 다른 점

main 의 코드 변경은 upstream `0a9492f` 위의 커밋이다. 7개 파일을 바꿨고 수정 블록마다 `[repro]` 주석이 있다. 변경 내용은 `git diff 0a9492f -- . ':!repro' ':!*.md' ':!.gitignore'` 로 확인한다. `max_iter`·`eval_every` 는 실행 인자(`jobs/benchmark.txt`)다.
GPU 환경에 맞춘 메모리·속도 수정(예: 24 GB GPU 용 attention·청크·평가 디코딩)은 환경별 브랜치에 둔다. main 에는 두지 않는다. 예는 [`hanbin5/local`](https://github.com/jongmin-s-multicore/thread0/tree/hanbin5/local) 의 `repro/LOCAL.md` 다.

| 변경 | 이유 | 결과에 주는 영향과 확인 방법 |
|---|---|---|
| MPC `max_iter` 상한 (위 표) | 원본 null 은 50개 중 하나라도 실패하면 끝나지 않는다 (issue #29). 논문은 반복 예산을 적지 않았다 | 한 번 성공한 에피소드는 이후에도 성공으로 세므로 반복별 성공률이 모두 기록된다. 예산별 곡선으로 보고한다 |
| `train.py` 가 epoch 지표를 `epoch_logs.jsonl` 에도 쓴다 | wandb 를 비활성화하고 실행한다 (`WANDB_MODE=disabled`) | 없음 |
| `flex_env.py`: `pyflex.init()` 을 프로세스당 한 번만 호출한다 | `plan.py` 가 한 프로세스에 FlexEnv 10개를 만들면 4–7번째에서 segfault 가 난다 | env 마다 `set_scene` 으로 장면을 다시 만들고 롤아웃은 env 단위로 순차 실행된다. 같은 초기 상태·행동에서 10개 env 결과가 같았다 |
| Rope·Granular 내부 CEM 의 시뮬 평가를 opt step 0 에서만 한다 (`planner.sub_planner.eval_every=1000`) | FleX 롤아웃이 push 한 번에 수 초 걸린다 | deformable 은 success 가 항상 False 라서 조기 종료가 일어나지 않는다. 계획 결과는 같다. 중간 로그는 줄어든다 |
| PushT 평가 그림·영상의 왼쪽 칸(실제 env, 월드모델 예측)에 목표 블록 자세를 빨간 윤곽으로 그린다 (`plan.py`, `planning/evaluator.py`, `env/pusht/goal_outline.py`) | env 가 항상 그리는 연두색 T 는 고정 목표 (256, 256, π/4) 이고 planning 목표(state_g)와 관계없다. 이 때문에 영상을 잘못 읽는다 (§5.1) | 그림·영상만 바뀐다. 성공 판정·지표는 그리기 전에 계산하고 관측은 바뀌지 않는다. 같은 설정에서 윤곽을 그린 실행과 그리지 않은 실행의 `logs.json`·`plan_targets.pkl` 내용이 같았다 |
| PointMaze 평가 그림·영상의 왼쪽 칸에 목표 위치를 빨간 원(성공 반경 0.5)과 안쪽 눈금으로 그린다 (`plan.py`, `planning/evaluator.py`, `env/pointmaze/goal_marker.py`) | env 는 목표를 그리지 않는다. 목표 마커는 화면 밖에 있다. 에이전트가 가야 할 위치는 오른쪽 칸과 비교해야 보인다 | 그림·영상만 바뀐다. 성공 판정·지표는 그리기 전에 계산하고 관측은 바뀌지 않는다. 같은 설정으로 표시를 그리는 실행과 그리기 함수를 항등 함수로 바꾼 실행에서 `logs.json`·`plan_targets.pkl` 이 바이트 단위로 같았다 (PointMaze MPC 에피소드 10개) |
| 평가 영상(모든 환경·planner, `planning/evaluator.py`): 칸마다 한 단어 라벨을 붙인다. 왼쪽 위 `Real` 은 실제 env 에서 행동을 실행한 화면이다. 왼쪽 아래 `Model` 은 같은 행동을 월드모델로 롤아웃해 decoder 로 그린 예측이다. 오른쪽은 `Goal` 이다. 샘플마다 실행한 마지막 프레임(`action_len·frameskip+1`)에서 자르고 그 프레임을 3초(36 프레임) 더 보여 준다 | upstream 은 배치에서 가장 긴 롤아웃 길이로 저장하므로 MPC 에서 일찍 성공한 에피소드는 성공 뒤가 가려진 회색 프레임으로 채워진다. 위아래 줄이 무엇인지 화면에 표시되지 않는다 | 영상만 바뀐다 (프레임 수 = 실행 길이 + 36). png 그리드·성공 판정·지표는 바뀌지 않는다. 같은 설정의 실행에서 `logs.json` 이 같았다 |
| `models/dino.py`: DINOv2 hub 코드를 `facebookresearch/dinov2:85a2460` 으로 고정했다 | 현재 main 은 Python 3.10 문법을 쓴다 (issue #25) | 가중치(`dinov2_vits14_pretrain.pth`)는 같다. 고정 커밋과 이전에 쓴 hub 캐시(같은 커밋)의 patch 특징 차이는 0 이다 |

시뮬레이터 대조 (`repro/setup/check_env.sh`): PointMaze 는 데이터셋 프레임·상태와 완전히 같다. PushT 는 상태가 같고 픽셀 MAE 는 0.3/255 다. Wall 은 프레임이 같다. Wall 데이터셋의 상태 배열은 env 상태와 1스텝부터 다르다. 기록 방식 차이로 보인다. 성공 판정은 env 상태끼리 비교하므로 영향이 없다. Granular 는 3스텝까지 완전히 같다. Rope 는 2스텝까지 입자 위치 오차가 3e-4 이고, 3–4스텝에서는 평균 0.013–0.019 다. 이동량은 약 1.0 이다. 재실행해도 같은 값이 나온다.

## 7. 추가 실험: ACWM-Phys Pour Water (논문 밖)

DINO-WM 논문의 벤치마크가 아니다. 물리 영상과 action 이 같이 있는 공개 데이터 [ACWM-Phys](https://arxiv.org/abs/2605.08567) (HF `t1an/ACWM-Phys`, 커밋 `4017e5f`)의 Pour Water 로 DINO-WM 의 예측 품질을 잰다. 시뮬레이터가 공개되지 않아 planning 은 돌릴 수 없다 (목표·보상·상태도 없다). 결과는 이 문서가 아니라 이슈로 올린다 (AGENTS.md §1).

### 7.1 데이터

- 컵을 기울여 옆 상자에 물을 붓는 PyFleX 장면. 에피소드마다 300프레임(360×360, 10 fps)과 action [300, 4]. action 차원 0·2 는 모든 에피소드에서 0 이라 빼고 2차원으로 쓴다. `metadata.pt` 는 pickle 이라 `torch.load(..., weights_only=True)` 로만 읽는다.
- 쓰는 split: `ind_train` 1000개에서 `episode_60`–`episode_79` 20개를 뺀 980개 (train/val 90/10, seed 42 = 882/98), `ind_test` 50, `ood_test` 50. 뺀 20개는 첫 프레임에서 물이 컵 바닥에 얇게 있고 입자가 흩어져 있으며(최대 물덩어리가 물 픽셀의 0.25–0.89, 나머지 980개는 0.94 이상) 끝 프레임에 상자 바닥 물이 거의 없다(0–71 px, 나머지 중앙값 4,944). 붓는 장면이 없다. 해시·디코딩·길이는 정상이라 파일 검사로는 안 잡히고 영상을 봐야 한다. 제외 목록은 `conf/env/acwm_pour_water.yaml`.
- `ood_test` 는 물 양이 다르다. 첫 프레임의 물 색 픽셀 수(W0, 360×360, HSV hue 98–110·S>70·V>100)가 정상 학습 980개는 2,351–6,202 인데 `ood_test` 는 낮음 26개 1,675–2,795, 높음 24개 5,360–8,366 으로 갈려 있다 (평가는 W0 4,000 으로 나눈다).
- 정렬: a[t] 가 프레임 t→t+1 전이를 만든다 (DINO-WM 슬라이서 규약과 같다). 주황색 팔 중심 속도를 a[t+j] (j=−4..+4)에 회귀하면 j=0 의 주 계수가 0.43–0.88 이고 j=±1..3 은 모두 0.07 이하다 (에피소드 160개). 팔만 본 것이고 물은 확인하지 않았다.
- 로더 `datasets/acwm_pour_water_dset.py`, 설정 `conf/env/acwm_pour_water.yaml` (`env=acwm_pour_water`). 슬라이스가 쓰는 프레임만 decord 로 디코딩한다 (`DeformDataset`·`TrajSlicerDataset` 는 샘플마다 궤적 전체를 읽는다). action 은 분할 전 980개 통계로 z-score, proprio·state 는 Rope 처럼 더미.

### 7.2 학습

| 항목 | 값 |
|---|---|
| 모델·옵티마이저 | §2 와 같다 (DINOv2 ViT-S/14 동결, ViT predictor, VQ-VAE 디코더), predictor lr 5e-5 · decoder 3e-4 · action encoder 5e-4 (논문 Table 12) |
| 입력·frameskip | 224, frameskip 5 (300프레임 → 60 스텝) |
| 모델 A / B | num_hist 3 (DINO-WM 기본) / num_hist 1 (논문 Rope·Granular 설정, ACWM-DiT 가 첫 프레임 하나로 예측하는 조건과 같다) |
| epoch | 6 (epoch = 슬라이스 전부 한 번: A 247,842개, B 256,662개) |
| 배치 | 논문 32. num_hist 3 은 24 GB GPU 에서 OOM 이라 24 로 낮췄다 (이 실험의 브랜치 `hanbin5/water`) |
| 작업 | `repro/jobs/pour_water_pilot.txt` (에피소드 60개·3 epoch 파일럿), `repro/jobs/pour_water.txt` (본 학습 A·B 와 평가). 둘 다 배치 24 |

### 7.3 평가 (`repro/eval/eval_pour_water.py`)

세트는 검증 98개, `ind_test`, `ood_test` (낮음·높음). 지표마다 에피소드 단위 bootstrap 95 % 구간. 기준선은 모델 없이 계산한다: copy_last(마지막 문맥 프레임), recon_copy(그 프레임을 인코더·디코더에 통과), recon_ceiling(정답 프레임을 인코더·디코더에 통과), copy_first(창의 첫 프레임), time_mean(학습 100개의 시각별 평균 영상). 디코더 오차를 모델 오차와 분리하려고 recon 두 개를 둔다 (Rope·Granular 는 recon LPIPS 가 예측 LPIPS 의 91·96 % 였다).

- **한 스텝**: `eval_pred_quality.py` 와 같은 계산 (LPIPS-VGG, SSIM, MSE, 224, [−1, 1]). 잠재 MSE 와 copy-latent 기준선도.
- **열린 루프 롤아웃**: ACWM-Phys `eval.py` 와 같은 37프레임 창, 시작 `linspace(0, 263, 10)`. 프레임 간격 frameskip 로 창의 프레임 k=num_hist.. 를 예측. 같은 지표, 잠재 MSE·코사인, ACWM 방식 지표(240×240, [0, 1], MSE·M-MSE·PSNR·skimage SSIM, `compute_metrics` 와 같은 식)를 구한다. 표는 창의 프레임 3 이상에서만 모아 num_hist 가 다른 모델을 같은 프레임으로 비교한다. 창 시작 시각(<80 / 80–200 / ≥200)으로도 나눈다.
- **전체 에피소드**: 처음 num_hist 프레임과 모든 action 으로 60 스텝을 굴린 오차 곡선, 끝 구간(프레임 250 이상) 상자 바닥(360px 기준 x 140–310, y 225–300)의 물 색 픽셀 수를 정답과 비교. 정답을 재구성한 영상에서 같은 양의 상대 오차가 10 % 를 넘으면 신뢰하지 않는다.
- **action 민감도**: 같은 창을 다음 에피소드의 action / 정규화한 action 0(= 학습 평균 action, 이동 차원에서 "움직임 없음"과 0.09 표준편차 차이)으로 롤아웃해 악화를 본다. 기울기 action 은 처음 약 80 스텝 동안 에피소드 간 거의 같고(이동 action 은 처음부터 다르다) 그래서 창 시작 시각(t0 < 80 / 80–200 / ≥ 200)으로 나눠 본다.

물 양은 `eval_pour_water_volume.py` 가 에피소드별로 따로 낸다 (화면 전체와 상자 바닥의 물 색 픽셀 수의 시간 곡선, 창 시작 t0 별 action 민감도). 그림은 `plot_pour_water.py`, 정성 비교는 `render_pour_water.py`.

SSIM 은 repo 의 `eval_images`(입력 [−1, 1]) 와 ACWM 의 skimage([0, 1])가 정의가 달라 섞지 않는다. ACWM-DiT 논문 수치와는 프레임 부분집합(frameskip 5), 해상도, 모델이 달라 직접 비교하지 않고 같은 코드로 계산한 기준선 대비 비율로 본다. 평가 스크립트는 파일럿 체크포인트로만 돌려 봤고(롤아웃 첫 예측이 `train.py` 의 forward 와 잠재 6e-5 이내로 같다), 본 학습 결과는 이 문서 작성 시점에 없다.

## 8. 추가 실험: ACWM-Phys Cloth Move · Push Sand · Stack Cube (논문 밖)

§7 과 같은 데이터([ACWM-Phys](https://arxiv.org/abs/2605.08567), HF `t1an/ACWM-Phys`, 커밋 `4017e5f`)의 다른 세 환경으로 같은 질문을 본다: DINO-WM 이 학습 분포 안과 밖(OoD)에서 얼마나 예측하는가. 시뮬레이터가 없어 planning 은 돌리지 않는다. 결과 요약은 루트 `README.md`, 수치 JSON 과 영상은 거기 적은 Drive 폴더에 있다.

### 8.1 데이터

`python repro/setup/download_acwm_phys.py` 가 네 환경(Pour Water 포함, 2.6 GB)을 고정 커밋으로 `$DATASET_DIR/acwm_phys` 에 받고 split 별 에피소드 수를 확인한다.

| 환경 | HF 폴더 | 학습 → 학습 / 검증 | ind_test | ood_test | 프레임 | 원본 | action (모델 입력) | OoD |
|---|---|---|---|---|---|---|---|---|
| Cloth Move | `deformable/clothmove` | 2000 → 1800 / 200 | 100 | 100 | 65 (action 64) | 512×512, 16 fps | 8 → 6차원 (상수 3·7 제외) | 천 크기: `smaller_*` 50 (면적 0.26배), `larger_*` 50 (2.5배) |
| Push Sand | `particle/push_sand` | 1784 → 1605 / 179 | 100 | 100 | 64 | 304×512, 10 fps | 7 → 4차원 (다음 프레임 자세의 x, z, cos yaw, sin yaw) | 입자 약 2배 (첫 프레임 모래 픽셀 2.16배) |
| Stack Cube | `rigid_dynamics/stack_cube` | 1373 → 1235 / 138 | 67 | 271 | 53–58 | 512×512, 20 fps (프레임 간격은 시뮬레이션 0.4 s) | 7차원 (말단 delta 6 + 그리퍼) | 노란 큐브 시작 위치가 학습 영역 밖 (up 68 / down 67 / left 69 / right 67) |

학습 / 검증은 `ind_train` 을 무작위 0.9 / 0.1 (seed 42)로 나눈 것이다. 제외한 에피소드는 없다 (Pour Water 같은 깨진 에피소드를 찾지 못했다). 아래 확인은 `repro/setup/checks/acwm_phys/` 의 스크립트로 했다 (실행 순서는 그 폴더의 README.md).

- **Cloth Move.** 그리퍼 공 2개가 천의 두 모서리를 잡고 구 위로 옮긴다. a[t] 가 프레임 t→t+1 을 만든다: 파란 공 중심 속도를 a[4:7][t+j] (j=−3..+3)에 회귀하면 j=0 계수 0.76–0.83, 나머지 0.05 이하, R² 0.98 (학습 400 에피소드). 모든 에피소드에서 처음 3개 action 이 0 이고 그동안 천이 자리를 잡는다. 학습·InD 는 천 크기가 하나뿐이고(프레임 3 의 천 픽셀 13,980–14,330 @512²) metadata 의 `cloth_dimx`·`cloth_dimy` 는 보이는 크기와 상관이 없다 (r ≈ 0.02).
- **Push Sand.** 판으로 입자 더미를 세 번 민다. metadata 의 action 은 "프레임 t 의 밀판 자세" (x, 0.9, z, 쿼터니언 (0, qy, 0, qw)) 다: 밀판 막대의 화면 속도 c[t+1]−c[t] 는 a[t+1]−a[t] 와 맞는다 (j=0 계수 0.65, 나머지 0.04 이하, R² 0.88 / 0.85). 그래서 action t 로 **다음 프레임의 자세 a[t+1]** 를 쓴다 (에피소드당 action 63개). 쿼터니언 부호가 에피소드 안에서 뒤집히므로(에피소드의 28 %) 회전은 yaw = 2·atan2(qy, qw) 의 (cos, sin) 으로 준다. 304×512 프레임은 자르지 않고 224×224 로 줄인다 (`resize_transform`; 밀판이 화면 가장자리까지 간다).
- **Stack Cube.** Franka 가 노란 큐브를 집어 초록 큐브 위에 쌓는다. a[t] 가 t→t+1 을 만든다: 팔 관절 변화 q[t+1]−q[t] 를 a[0:6][t+j] 에 회귀하면 j=0 이 가장 크고(단일 lag R² 0.78–0.86) 제어기 지연으로 a[t−1] 도 영향이 있다. `actions`(delta)가 `actions_abs` 차분보다 팔 움직임을 잘 설명해서(한 프레임 R² 0.85 대 0.50) `actions` 를 쓴다. 학습 에피소드의 27 % (373 / 1373)는 쌓기에 실패한 궤적이고 그대로 쓴다 (`success` 필드). `states` (45차원)에 큐브 위치가 있지만 모델 입력에는 쓰지 않는다. `video_path` 는 원 저장소의 절대경로라 파일 이름만 쓴다.

### 8.2 로더와 학습

로더 `datasets/acwm_phys_dset.py`, 설정 `conf/env/acwm_{cloth_move,push_sand,stack_cube}.yaml`. Pour Water 로더처럼 슬라이스가 쓰는 프레임만 decord 로 디코딩한다. action 은 분할 전 `ind_train` 전체 통계로 z-score, proprio·state 는 더미다.

| 항목 | 값 |
|---|---|
| 모델·옵티마이저 | §7.2 의 모델 B 와 같다: num_hist 1, frameskip 5, 224, predictor lr 5e-5 · decoder 3e-4 · action encoder 5e-4, 배치 24, seed 0 |
| epoch | 환경마다 슬라이스 수가 달라서 optimizer step 이 Pour Water B (64,170)와 비슷하도록 정했다: Cloth Move 4,125 it × 15 = 61,875, Push Sand 3,611 × 18 = 64,998, Stack Cube 2,485 × 25 = 62,125 |
| 작업 | `repro/jobs/acwm_phys_pilot.txt` (에피소드 60개·3 epoch), `repro/jobs/acwm_phys.txt` (본 학습과 평가·물리량·영상) |

- **frameskip 5**: 224 px 에서 k 프레임 뒤를 그대로 복사한 LPIPS 가 k=1 / k=5 에서 Cloth Move 0.017 / 0.058, Push Sand 0.0066 / 0.027, Stack Cube 0.074 / 0.164 다 (학습 40 에피소드; Pour Water 는 0.021 / 0.073). Stack Cube 는 한 프레임 변화가 이미 크지만 설정을 하나로 맞췄다.
- **num_hist 1**: Pour Water 에서 num_hist 3 과 1 이 같은 스텝 수에서 차이가 없었다 (§7). 스캔 특징의 한 스텝 변화를 선형 회귀로 맞추면 frameskip 5 에서 이전 스텝을 더해도 R² 가 Cloth Move 0.582 → 0.592, Stack Cube 0.934 → 0.950 로 거의 같다 (Push Sand 0.084 → 0.140; 선형 대리 지표라 참고용). 과거 프레임이 필요한지를 직접 본 실행은 없다.

### 8.3 평가

`repro/eval/eval_acwm_phys.py` 는 §7.3 의 지표·기준선을 그대로 쓴다 (한 스텝, 37프레임 창의 열린 루프, 전체 에피소드, action 민감도, ACWM 방식 지표). 다른 점:

- 창 시작은 에피소드마다 `linspace(0, 프레임 수 − 37, 10)`. 표는 예측 프레임 전부(창마다 7개)를 모은다 (Pour Water 표는 창 프레임 3 이상).
- 전체 에피소드 롤아웃은 프레임 0, 5, 10, … 를 끝까지 (Cloth Move·Push Sand 12 스텝, Stack Cube 10–11 스텝). 디코딩한 프레임(모델 예측, 정답을 인코더·디코더에 통과시킨 재구성)을 세트별 npz 로 저장한다.
- 그룹: Cloth Move `ood_test` 는 smaller / larger, Stack Cube 는 success / fail 과 `ood_test` 의 방향 4개.
- ACWM 방식 지표의 해상도는 ACWM `data_config.py` 의 obs_shape (240×240, Push Sand 240×400).

`repro/eval/eval_acwm_phys_quant.py` 는 저장한 프레임과 정답 프레임(224×224)을 같은 HSV 색 마스크로 재서 비교한다. 6 px 미만이면 "안 보임"으로 친다.

| 환경 | 재는 것 |
|---|---|
| Cloth Move | 천 = 주황(hue 8–16, S>120) + 분홍 뒷면(hue 158–170, S>60) 의 픽셀 수·IoU·중심. 그리퍼 공: 파랑(hue 105–113), 빨강(hue 175–4) 의 중심 |
| Push Sand | 모래 = 하늘색(hue 90–102, S>100) 의 픽셀 수·IoU·중심. 밀판 = 어두운 픽셀(V<90, S<90)의 가장 큰 덩어리(30 px 이상)의 중심 |
| Stack Cube | 노랑(hue 26–33, S>180, V>130)·초록(hue 55–66, S>150) 큐브의 픽셀 수·IoU·중심. 마지막 프레임의 "쌓임": 두 큐브가 15 px 이상 보이고 노란 큐브 중심이 초록 큐브 중심 기준 \|dx\| < 11, −35 < dy < −6.5 px. 원본 프레임에 같은 규칙(512 px 환산)을 쓰면 `success` 라벨과 98.8 % 일치한다 (학습) |

정답을 재구성한 프레임에서 잰 값(JSON 의 `recon`)이 정답과 다르면 그 양은 디코더가 못 그리는 것이다. InD 에서 재구성의 마스크 IoU 는 Cloth Move 0.92, Push Sand 0.95, 면적 오차 2–3 % 다. Push Sand `ood_test` 는 재구성 면적이 정답보다 17 % 작다 (디코더도 큰 더미를 다 그리지 못한다).

그림은 `repro/eval/plot_acwm_phys.py` (면적 곡선), 영상은 `repro/eval/render_acwm_phys.py` (그룹별 세 에피소드 묶음, `--per_episode N` 으로 에피소드별 영상). 모델을 다시 돌리지 않고 저장한 프레임만 읽는다.

한계: 시드 하나·설정 하나다. 검증 잠재 손실이 마지막 epoch 에도 epoch 당 0.4–0.8 % 씩 내려가고 있었다. 논문의 ACWM-DiT 수치와는 프레임 부분집합(frameskip 5)·split 크기·모델이 달라 직접 비교하지 않는다.
