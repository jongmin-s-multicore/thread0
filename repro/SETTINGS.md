# DINO-WM 재현 — 실험 세팅

DINO-WM([arXiv:2411.04983](https://arxiv.org/abs/2411.04983) v2, 코드 [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) `0a9492f`)의 벤치마크를 다시 돌리는 실험의 공용 설정이다 (GPU 환경과 무관한 것만. 환경별 수정·측정은 환경 브랜치에). 결과와 해석은 [이슈](https://github.com/jongmin-s-multicore/thread0/issues)로 따로 올린다. 코드·환경·실행 방법은 [README](README.md) 에 있고, 이 문서의 `queue/`·`jobs/`·`eval/`·`setup/` 경로는 `repro/` 기준, 그 밖의 파일 경로는 레포 루트(upstream 코드) 기준이다.
수치는 upstream 설정 파일(`conf/`), 공개 체크포인트의 `hydra.yaml`, 환경 브랜치의 `repro/runs/*/*/run.yaml`(예: `hanbin5/local`), 실행 로그에서 가져왔다. 논문 수치는 v2(2025-02) 기준이다.

## 1. 구성

| 논문 표 | 환경 | world model | planner | 지표 | 인스턴스 | job |
|---|---|---|---|---|---|---|
| Table 1, 8 "MPC" | PointMaze, PushT, Wall | 공개 체크포인트 | MPC-CEM | 성공률 | 50 | `pointmaze_mpc`, `pusht_mpc`, `wall_mpc` |
| Table 8 "CEM" | PointMaze, PushT, Wall | 공개 체크포인트 | 오픈루프 CEM = MPC 1회차 (§5.3) | 성공률 | 50 | 위 MPC 실행의 step 1 |
| Table 8 "GD" | PointMaze, PushT, Wall | 공개 체크포인트 | 오픈루프 GD | 성공률 | 50 | `<env>_gd` |
| (추가) | PointMaze, Wall | 공개 체크포인트 | 오픈루프 CEM, planner 그룹 기본값 30 opt steps | 성공률 | 50 | `<env>_cem30` |
| Table 1, 8 Rope·Granular | Rope, Granular | 직접 학습 (§4.2) | MPC-CEM | Chamfer distance | 10 | `train_<obj>`, `<obj>_mpc` |
| Table 4, 9 | PushT, Wall, Rope, Granular (+PointMaze) | 위와 같음 | — | 디코딩한 1-step 예측의 LPIPS·SSIM | 검증 분할 | `pq_<env>` |

job 이름은 `repro/jobs/benchmark.txt` 의 것이다. 각 실행의 설정 기록(run-id 와 run.yaml)은 그 실행을 돌린 환경 브랜치의 `repro/runs/` 에 있다.

재현하지 않은 것과 이유:

| 대상 | 이유 |
|---|---|
| Reacher (Table 1, 2) | 데이터셋·환경 코드·체크포인트가 공개되지 않았다 (dino_wm issue #22) |
| IRIS, DreamerV3, TD-MPC2, AVDC (Table 1, 3, 4, 9) | dino_wm 에 구현이 없다. 논문은 외부 코드베이스를 썼고 설정을 공개하지 않았다 |
| WallRandom, PushObj (Table 3) | 일반화 실험용 데이터셋과 생성 스크립트가 공개되지 않았다 |
| GranularRandom (Table 3) | 같은 Granular 모델을 입자 수가 다른 환경에서 평가하는 것이라 데이터는 필요 없지만, 환경 설정(`obj_params.area`)과 시작 상태 샘플링을 바꾸는 코드가 필요하다. 돌리지 않았다 |
| 인코더 비교 R3M·ResNet·DINO-CLS (Table 2) | 저장소에 설정(`encoder=r3m|resnet|dino_cls decoder=transposed_conv`)은 있지만 환경 5개 × 인코더 3개 = world model 15개를 새로 학습해야 한다. 이번에는 돌리지 않았다 |
| 데이터 크기·causal mask·decoder loss ablation (Table 5–7) | 새 학습이 필요하고, mask·decoder loss 는 코드 수정이 필요하다. 돌리지 않았다 |
| Table 10 (A6000 시간 측정) | 하드웨어가 다르다. 환경별 측정은 환경 브랜치 문서에 둔다 (예: `hanbin5/local` 의 `repro/LOCAL.md`) |

## 2. 모델

공개 체크포인트와 직접 학습한 모델 모두 upstream `conf/train.yaml` 의 구조다.

| 항목 | 값 |
|---|---|
| 인코더 | DINOv2 ViT-S/14 (`torch.hub` `dinov2_vits14`, hub 코드는 `85a2460` 고정), `x_norm_patchtokens`. 224×224 입력을 196×196 으로 줄여 14×14 = 196 패치 × 384 차원. 동결 (`model.train_encoder=False`) |
| action / proprio 인코더 | Conv1d(k=1) 임베딩 10차원씩, 패치마다 이어 붙인다 (`concat_dim=1`) → 토큰 404 차원 |
| predictor | ViT, depth 6, heads 16, dim_head 64, MLP 2048, dropout 0.1, emb_dropout 0 (Wall 공개 체크포인트는 0.1). 프레임 단위 causal mask. 파라미터 20.12M (h=3), 19.96M (h=1) |
| decoder | VQ-VAE 디코더 (`quantize=False`), 10.14M. 입력은 `z.detach()` — 디코더 손실은 predictor 로 흐르지 않는다 |
| planning 시 모드 | 체크포인트의 predictor·decoder 는 eval 모드로 저장돼 있어 planning 중 dropout 이 꺼져 있다. `plan.py` 는 `model.eval()` 을 부르지 않는다 (upstream 그대로) |

## 3. 데이터

[OSF](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28) `datasets/` (`repro/setup/download_data.sh`). 궤적 수와 길이는 파일에서 직접 셌다.

| 데이터셋 | 궤적 × 길이 | action | 분할 | 풀린 크기 |
|---|---|---|---|---|
| `point_maze` | 2000 × 100 | 2 | 무작위 0.9 / 0.1 (seed 42) | 29 GB |
| `pusht_noise` | train 18685, val 21 (길이 49–246, 평균 125) | 2 | 폴더 (`train/` `val/`) | 7 GB |
| `wall_single` | 1920 × 50 | 2 | 무작위 0.9 / 0.1 (seed 42) | 55 GB |
| `deformable/rope` | 1000 × 20 프레임 (action 20 × 4) | 4 (push 시작·끝 xz) | 무작위 0.9 / 0.1 (seed 42) | 69 GB |
| `deformable/granular` | 1000 × 20 프레임 (action 20 × 4) | 4 | 무작위 0.9 / 0.1 (seed 42) | 75 GB |

- 논문 Table 11 은 Rope·Granular 궤적 길이를 5 로 적었지만, 부록 A.1 본문과 실제 데이터는 20 이다.
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

predictor lr 는 논문 Table 12 에 5e-5 로 적혀 있지만 upstream 기본값과 공개 체크포인트는 5e-4 다.

### 4.2 Rope·Granular (직접 학습)

공개 체크포인트가 없어서 upstream `train.py` 로 학습한다. 논문 Table 11·12 의 값과 upstream 기본값을 따랐다.

| 항목 | 값 |
|---|---|
| 명령 | `python train.py --config-name train.yaml env=deformable_env env.dataset.object_name=<obj> env.kwargs.object_name=<obj> frameskip=1 num_hist=1 training.save_every_x_epoch=10` |
| num_hist / num_pred / frameskip | 1 / 1 / 1 |
| epochs / batch | 100 / 32 (GPU 1장) |
| optimizer | predictor·action encoder AdamW, decoder Adam (upstream 그대로) |
| lr | decoder 3e-4, predictor 5e-4, action encoder 5e-4 (upstream 기본값 = 공개 체크포인트와 같다) |
| 학습 표본 | 궤적 900 × 슬라이스 19 = 17,100 → 535 iteration/epoch |
| 체크포인트 | 10 epoch 마다 (`model_<ep>.pth`, `model_latest.pth`) |
| seed | 0 (`training.seed`) |

Rope 를 이 설정으로 100 epoch 학습해도 논문 CD 0.41 이 나오지 않았다는 보고가 있다 (dino_wm issue #24, 저자 답 없음).

## 5. 평가 프로토콜

### 5.1 planning 설정

upstream 설정 파일 그대로이고, 바꾼 값은 굵게 표시했다.

| 항목 | PointMaze (`plan_point_maze.yaml`) | PushT (`plan_pusht.yaml`) | Wall (`plan_wall.yaml`) | Rope·Granular (`plan.yaml` + `planner=mpc_cem`) |
|---|---|---|---|---|
| 인스턴스 (`n_evals`) | 50 | 50 | 50 | 10 |
| 목표 (`goal_source`) | `random_state` | `dset` | `random_state` | **`random_state`** (`plan.yaml` 기본값은 `dset`) |
| `goal_H` (= CEM horizon = MPC 실행 길이) | 5 | 5 | 5 | 5 |
| 목적 함수 | 마지막 예측 프레임의 latent MSE, proprio 가중치 alpha 0 | alpha 1 | alpha 1 | alpha 1 (proprio 는 0 으로 고정된 더미) |
| CEM samples / topk / var_scale | 300 / 30 / 1 | 300 / 30 / 1 | 300 / 30 / 1 | 300 / 30 / 1 |
| CEM opt steps | 10 | 30 | 10 | 30 |
| MPC 최대 반복 (`max_iter`) | **20** (원본 null) | **10** (원본 null) | **20** (원본 null) | **5** (원본 null) |
| 내부 CEM 시뮬 평가 (`eval_every`) | 1 | 1 | 1 | **1000** (§6) |
| seed | 99 → 에피소드 seed `99·n+1` | 같음 | 같음 | 같음 |

- `goal_H=5` 는 macro action 5개다. frameskip 5 인 환경에서는 env 25 스텝, Rope·Granular 는 push 5번이다.
- PushT 의 `dset` 목표는 검증 궤적의 구간을 env 에서 재생해 25 스텝 뒤 상태를 목표로 삼는다 (25 스텝 안에 도달 가능).
- PushT 화면의 연두색 T 는 원래 PushT 의 고정 목표 (256, 256, π/4) 로, env 가 항상 그리고 학습 데이터에도 들어 있다. planning 목표가 아니다. 목표는 평가 영상·그림의 오른쪽 칸(목표 상태를 렌더한 이미지)이고, `plan.py` 가 왼쪽 칸에 목표 블록 자세를 빨간 윤곽으로 그린다 (그림·영상에만, 관측과 지표는 그대로). 평가 영상의 칸 라벨: `Real` = 실제 env, `Model` = 월드모델 예측, `Goal` = 목표. 이 윤곽이 없는 예전 영상은 `repro/eval/pusht_goal_overlay.py` 로 그린다.
- PointMaze 화면에는 에이전트(초록 점)만 있고, env 는 목표 마커를 화면 밖에 둔다 (`with_target=False`). `plan.py` 가 평가 영상·그림의 왼쪽 칸에 목표 위치(state_g[:2])를 중심으로 빨간 원(반지름 = 성공 반경 0.5)과 안쪽으로 짧은 눈금 넷을 그린다 (그림·영상에만). 성공 판정은 상태로 하고 env 는 물리 하위 스텝(0.01 초) 하나 전 위치를 그리므로, 경계 가까이에서 끝난 에피소드는 마지막 프레임의 초록 점이 원 바로 안팎에 보일 수 있다 (지연은 속도 × 0.01, 최고 속도에서 약 0.07). 이 표시가 없는 예전 영상은 `repro/eval/pointmaze_goal_overlay.py` 로 그린다 (목표는 실행 디렉토리의 `plan_targets.pkl`).
- PointMaze `random_state`: U-maze 빈 공간에서 시작·목표를 독립으로 뽑는다. Wall: 시작과 목표를 벽 반대편 방에서 뽑고, 벽·문 위치는 검증 궤적에서 가져온다.
- Rope·Granular `random_state`: 시작은 검증 분할의 무작위 궤적에서 무작위 시점(0–18)의 입자 상태, 목표는 reset 한 모양을 평행이동·회전(Rope) 또는 평행이동·축소(Granular)한 입자 배치다 (`plan.py` `prepare_targets`, `FlexEnvWrapper.sample_random_init_goal_states`).

### 5.2 성공 판정과 지표 (env wrapper 의 `eval_state`)

| 환경 | 성공 | 기록 |
|---|---|---|
| PointMaze | 목표와 위치(x, y) 거리 < 0.5 | 성공률, 상태 거리 |
| PushT | agent·블록 위치 4차원 거리 < 20 그리고 블록 각도 차 < π/9 | 성공률, 상태 거리 |
| Wall | 위치 거리 < 4.5 (env 픽셀 단위) | 성공률, 상태 거리 |
| Rope·Granular | CD < 0 → 항상 실패 (upstream 그대로). 지표는 CD 만 쓴다 | 최종 입자와 목표 입자의 Chamfer distance (xyz, 양방향 평균 거리의 합) |

### 5.3 planner 별 의미

- **MPC-CEM** (`planning/mpc.py`): CEM 으로 `goal_H` 길이를 계획하고 그 전부(`n_taken_actions = goal_H`)를 env 에서 실행한 뒤, 도달한 관측에서 다시 계획한다. 매 반복 끝에 처음부터 지금까지의 행동 전체를 env 에서 다시 실행해 성공을 판정하고, 한 번 성공한 에피소드는 성공으로 남는다(이후 행동은 0). `logs.json` 의 `mpc/success_rate`(step = 반복 횟수)가 예산별 성공률이다.
- **오픈루프 CEM = MPC 1회차**: `n_taken_actions = horizon` 이라 MPC 의 첫 반복은 같은 CEM 설정으로 계획한 행동열을 피드백 없이 끝까지 실행하는 것과 같은 계산이다(같은 seed·목표·난수 순서). Table 8 "CEM" 과의 비교에는 MPC step 1 값을 쓴다. 논문은 오픈루프 CEM 의 opt steps 를 적지 않았으므로 planner 그룹 기본값(30) 변형 `cem30` 을 PointMaze·Wall 에 따로 돌린다 (PushT 는 MPC 설정이 이미 30).
- **오픈루프 GD** (`conf/planner/gd.yaml`): 행동을 표준정규 초기화, SGD lr 1, 1000 스텝, 매 스텝 잡음 0.003. 10 스텝마다 env 에서 평가하고 전부 성공하면 멈춘다.
- CEM·GD 모두 env 의 실제 성공을 조기 종료에만 쓴다(upstream 그대로, dino_wm issue #26).

### 5.4 예측 품질 (Table 4, 9)

upstream `train.py` 의 검증과 같은 계산(`model.eval()`, `model(obs, act)`, `metrics/image_metrics.eval_images` — VGG LPIPS, 11×11 가우시안 SSIM, 정규화된 [-1, 1] 이미지)을 검증 분할 전체에 대해 한다 (`eval/eval_pred_quality.py`). 슬라이스가 5000 개를 넘으면 5000 개만 쓴다: `train.py` 처럼 `seed(training.seed)` 뒤에 데이터셋을 만들어 슬라이스 순서(무작위 순열)를 학습 때와 같게 하고, 그 순서에서 균등 간격으로 고른다. upstream 검증은 첫 배치(32개)에서만 이미지 지표를 계산한다. 논문은 예측 길이와 표본 수를 적지 않았다.

## 6. upstream 코드·논문과 다른 점

main 의 코드 변경은 upstream `0a9492f` 위의 커밋이다 (7개 파일, 수정 블록마다 `[repro]` 주석, `git diff 0a9492f -- . ':!repro' ':!*.md' ':!.gitignore'`). `max_iter`·`eval_every` 는 실행 인자(`jobs/benchmark.txt`)다.
GPU 환경에 맞춘 메모리·속도 수정(예: 24 GB GPU 용 attention·청크·평가 디코딩)은 main 에 두지 않고 환경별 브랜치에 둔다 — 예: [`hanbin5/local`](https://github.com/jongmin-s-multicore/thread0/tree/hanbin5/local) 의 `repro/LOCAL.md`.

| 변경 | 이유 | 결과에 주는 영향 (확인 방법) |
|---|---|---|
| MPC `max_iter` 상한 (위 표) | 원본 null 은 50개 중 하나라도 실패하면 끝나지 않는다 (issue #29). 논문은 반복 예산을 적지 않았다 | 성공은 sticky 라 반복별 성공률이 모두 남는다. 예산별 곡선으로 보고한다 |
| `train.py` 가 epoch 지표를 `epoch_logs.jsonl` 에도 쓴다 | wandb 를 끄고 돌린다 (`WANDB_MODE=disabled`) | 없음 |
| `flex_env.py`: `pyflex.init()` 을 프로세스당 한 번만 | `plan.py` 가 한 프로세스에 FlexEnv 10개를 만들면 4–7번째에서 segfault | env 마다 `set_scene` 으로 장면을 다시 만들고 롤아웃은 env 단위로 순차 실행된다. 같은 초기 상태·행동에서 10개 env 결과가 같았다 |
| Rope·Granular 내부 CEM 의 시뮬 평가를 opt step 0 에서만 (`planner.sub_planner.eval_every=1000`) | FleX 롤아웃이 push 한 번에 수 초 걸린다 | deformable 은 success 가 항상 False 라 조기 종료가 일어나지 않는다. 계획 결과는 같고 중간 로그만 줄어든다 |
| PushT 평가 그림·영상의 왼쪽 칸(실제 env, 월드모델 예측)에 목표 블록 자세를 빨간 윤곽으로 (`plan.py`, `planning/evaluator.py`, `env/pusht/goal_outline.py`) | env 가 항상 그리는 연두색 T 는 고정 목표 (256, 256, π/4) 라 planning 목표(state_g)와 관계없어 영상을 잘못 읽게 된다 (§5.1) | 그림·영상만 바뀐다. 성공 판정·지표는 그리기 전에 계산하고 관측은 그대로다 (윤곽을 켜고 끈 같은 설정의 실행에서 `logs.json`·`plan_targets.pkl` 내용 동일) |
| PointMaze 평가 그림·영상의 왼쪽 칸에 목표 위치를 빨간 원(성공 반경 0.5)과 안쪽 눈금으로 (`plan.py`, `planning/evaluator.py`, `env/pointmaze/goal_marker.py`) | env 가 목표를 그리지 않아(목표 마커는 화면 밖) 에이전트가 어디로 가야 하는지 오른쪽 칸과 견줘야만 보인다 | 그림·영상만 바뀐다. 성공 판정·지표는 그리기 전에 계산하고 관측은 그대로다 (같은 설정으로 표시를 그리는 실행과 그리기 함수를 항등 함수로 바꾼 실행에서 `logs.json`·`plan_targets.pkl` 바이트 단위로 동일, PointMaze MPC 에피소드 10개) |
| 평가 영상(모든 환경·planner, `planning/evaluator.py`): 칸마다 한 단어 라벨 — 왼쪽 위 `Real`(실제 env 에서 행동을 실행한 화면), 왼쪽 아래 `Model`(같은 행동을 월드모델로 굴려 decoder 로 그린 예측), 오른쪽 `Goal`. 샘플마다 실행한 마지막 프레임(`action_len·frameskip+1`)에서 끊고 그 프레임을 3초(36 프레임) 더 보여 준다 | upstream 은 배치에서 가장 긴 롤아웃 길이로 저장해, MPC 에서 일찍 성공한 에피소드는 성공 뒤가 가려진 회색 프레임으로 채워진다. 위아래 줄이 무엇인지 화면에 없다 | 영상만 바뀐다 (프레임 수 = 실행 길이 + 36). png 그리드·성공 판정·지표는 그대로 (같은 설정의 실행에서 `logs.json` 동일) |
| `models/dino.py`: DINOv2 hub 코드를 `facebookresearch/dinov2:85a2460` 으로 고정 | 현재 main 은 Python 3.10 문법 (issue #25) | 가중치(`dinov2_vits14_pretrain.pth`)는 같다. 고정 커밋과 이전에 쓴 hub 캐시(같은 커밋)의 patch 특징 차이 0 |

시뮬레이터 대조 (`repro/setup/check_env.sh`): PointMaze 는 데이터셋 프레임·상태와 완전히 같다. PushT 는 상태가 같고 픽셀 MAE 0.3/255. Wall 은 프레임이 같고, 데이터셋의 상태 배열은 env 상태와 1스텝부터 다르다(기록 방식 차이로 보인다. 성공 판정은 env 상태끼리 비교하므로 영향 없다). Granular 는 3스텝까지 완전히 같다. Rope 는 2스텝까지 입자 위치 오차 3e-4, 3–4스텝에서 평균 0.013–0.019 (이동량 약 1.0 대비, 재실행해도 같은 값).
