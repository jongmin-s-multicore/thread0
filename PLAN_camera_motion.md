# 실험 계획: 카메라가 움직일 때 Rope·Granular 예측

- 브랜치: `hanbin5/occlusion`
- 작성: 2026-10-13
- 결과는 이슈에 올린다.

## 1. 목적

Rope, Granular 에서 에피소드 도중 카메라 pose 가 변할 때 DINO-WM 의 다음 프레임 예측 오차를 측정한다.

확인할 항목:

| 번호 | 항목 | 모델 (4절) | 평가 조건·지표 (5절) |
|---|---|---|---|
| 1 | 고정 카메라로 학습한 모델의 오차가 다른 시점, 움직이는 시점에서 얼마나 증가하는지 | `fixed` | E-static, E-move |
| 2 | 카메라 이동량을 action 에 넣었을 때 움직이는 시점에서의 예측 오차 | `move-told-h1` | E-move |
| 3 | 카메라 이동량을 넣지 않았을 때 context 프레임 수(1, 2)에 따른 차이 | `move-blind-h1`, `move-blind-h2` | E-move |
| 4 | 에피소드마다 시점이 다르고 에피소드 안에서는 정지한 데이터로 학습한 경우와의 비교 | `static` | E-static, E-move |
| 5 | 예측에 카메라 이동과 물체 이동이 각각 반영되는지 | 전체 | `err_camonly`, `err_objonly` |
| 6 | 학습 범위보다 빠른 카메라 이동에서의 오차 | `move-blind-h1`, `move-blind-h2`, `move-told-h1` | E-move-fast |

## 2. 데이터

원본은 DINO-WM 의 deformable 데이터다. Rope, Granular 각 1000 에피소드, 에피소드당 20 프레임.

재렌더 방법:

- 원본은 프레임별 입자 위치(`states.pth`)를 저장한다. 프레임을 저장할 때 로봇 팔은 home 자세다.
- [scripts/render_camera_data.py](scripts/render_camera_data.py) 가 저장된 입자 위치를 PyFleX 에 설정하고 지정한 카메라로 렌더링한다. 시뮬레이션은 다시 실행하지 않는다.
- 상태, push action, train/val 분할(seed 42)은 원본과 같다.

검증:

| 항목 | 값 |
|---|---|
| 데이터셋 카메라로 재렌더한 프레임과 원본 `obses.pth` 의 평균 절대 차 (0–255 기준) | Rope 0.13 (10 에피소드), Granular 0.10 (1 에피소드) |
| 원본의 연속 프레임 간 평균 절대 차 | Rope 0.41, Granular 1.33 |
| 카메라 pose 식과 데이터셋 카메라 4대의 위치·각도 차 | 1e-15 이하 |

### 카메라 pose

코드는 [env/deformable_env/camera_motion.py](env/deformable_env/camera_motion.py).

- pose = (azimuth, height, radius)
- 위치: `(radius·sin(azimuth), height, radius·cos(azimuth))`
- 방향: yaw = azimuth, pitch = −atan2(height, radius + 1.515)
- 데이터셋 카메라: (135°, 10, 8.49)

| 변수 | 범위 | 비고 |
|---|---|---|
| azimuth | −55° ~ 235° | 240°–300° 구간은 로봇 팔이 테이블을 가려서 제외했다. 데이터셋 카메라 4대(45°, 135°, 225°, 315°)는 범위 안이다. |
| height | 7 ~ 13 | 데이터셋 값 10 |
| radius | 6.5 ~ 10.5 | 데이터셋 값 8.49 |

### 카메라 변형

| 이름 | 에피소드 안의 움직임 | azimuth 속도 | 용도 |
|---|---|---|---|
| `fixed` | 데이터셋 카메라로 고정 | 0 | 평가 |
| `static_rand` | 에피소드마다 무작위 pose, 정지 | 0 | 학습, 평가 |
| `move` | 등속 이동. azimuth 는 스텝당 일정 각도(부호 무작위), height 와 radius 는 무작위 두 값 사이를 선형으로 이동 | 스텝당 2°–8° (에피소드 전체 38°–152°) | 학습, 평가 |
| `move_fast` | `move` 와 같은 방식 | 스텝당 10°–15° | 평가 |

- 한 스텝은 push 1회다.
- 궤적은 (변형, 에피소드 번호, seed) 로 결정된다.

## 3. 사전 측정

기존 모델을 재학습 없이 측정했다.

- 모델: `train_rope`, `train_granular` (100 epoch)
- 데이터: val 40 에피소드
- 카메라: 데이터셋 pose 에서 azimuth 만 변경했다. "정지"는 고정 각도만큼 회전한 위치, "이동"은 스텝당 일정 각도 회전이다.
- 값: 예측 latent 와 다음 프레임 latent 의 MSE. "복사"는 현재 프레임의 latent 를 예측으로 사용한 경우다.

| 조건 | Rope 1-step 예측 | 복사 | 복원 LPIPS / SSIM | Granular 1-step 예측 | 복사 | 복원 LPIPS / SSIM |
|---|---|---|---|---|---|---|
| 데이터셋 카메라 | 0.077 | 0.390 | 0.044 / 0.963 | 0.145 | 0.307 | 0.102 / 0.900 |
| 정지, 5° | 0.939 | 0.387 | 0.246 / 0.495 | 0.632 | 0.297 | 0.328 / 0.434 |
| 정지, 15° | 1.388 | 0.346 | 0.363 / 0.469 | 0.973 | 0.287 | 0.440 / 0.405 |
| 정지, 45° | 2.106 | 0.303 | 0.456 / 0.487 | 1.322 | 0.275 | 0.491 / 0.411 |
| 정지, 90° | 1.919 | 0.333 | 0.444 / 0.285 | 1.240 | 0.285 | 0.516 / 0.245 |
| 이동, 스텝당 0.5° | 0.835 | 0.580 | 0.225 / 0.547 | 0.655 | 0.494 | 0.299 / 0.490 |
| 이동, 스텝당 1° | 1.134 | 0.669 | 0.297 / 0.499 | 0.886 | 0.602 | 0.371 / 0.442 |
| 이동, 스텝당 2° | 1.578 | 0.837 | 0.372 / 0.478 | 1.152 | 0.770 | 0.424 / 0.425 |
| 이동, 스텝당 4.5° | 2.063 | 1.289 | 0.419 / 0.451 | 1.592 | 1.208 | 0.463 / 0.398 |
| 이동, 스텝당 9° | 2.754 | 2.353 | 0.468 / 0.390 | 2.374 | 2.251 | 0.502 / 0.354 |

- 데이터셋 카메라의 Rope LPIPS 0.044, SSIM 0.963 은 val 전체에서 측정한 기존 값(0.043, 0.963)과 같은 수준이다.
- 정지 5° 에서 예측 오차는 Rope 0.939, Granular 0.632 로 복사 오차(0.387, 0.297)보다 크다.
- 이동 조건에서는 측정한 모든 속도에서 예측 오차가 복사 오차보다 크다.
- 이동 조건의 예측은 카메라가 움직이지 않았을 때의 다음 프레임에 더 가깝다. 스텝당 4.5° 에서 그 프레임과의 오차는 Rope 1.661, Granular 1.099 이고 실제 다음 프레임과의 오차는 2.063, 1.592 다.

## 4. 모델

객체당 5개다. `fixed` 는 기존 모델이고 나머지 4개를 학습한다. 학습할 모델은 총 8개다.

| 이름 | 학습 데이터 | 카메라 이동량 입력 | `num_hist` | decoder | 비고 |
|---|---|---|---|---|---|
| `fixed` | 원본 (고정 카메라) | - | 1 | 있음 | 기존 `train_rope`, `train_granular`. 10 epoch 마다 체크포인트가 있다. |
| `static` | `static_rand` | 없음 | 1 | 없음 | |
| `move-blind-h1` | `move` | 없음 | 1 | 없음 | |
| `move-blind-h2` | `move` | 없음 | 2 | 없음 | |
| `move-told-h1` | `move` | 있음 | 1 | 있음 | action 은 push 4차원에 (Δazimuth, Δheight, Δradius) 를 붙인 7차원 |

공통 설정:

- `frameskip=1`, batch 32, 50 epoch, 10 epoch 마다 체크포인트. 그 외는 `conf/train.yaml` 기본값이다.
- 설정 파일 [conf/env/deformable_cam.yaml](conf/env/deformable_cam.yaml), 로더 [datasets/deformable_cam_dset.py](datasets/deformable_cam_dset.py)
- seed 0

epoch 수:

- 기존 모델의 val latent 오차는 Rope 30 epoch 0.076, 100 epoch 0.071 이고 Granular 20 epoch 0.134, 100 epoch 0.136 이다 (각 모델의 `epoch_logs.jsonl`). 이 값을 보고 50 epoch 로 정했다.
- 모델 간 비교는 같은 epoch 의 체크포인트로 한다. `fixed` 는 `model_50.pth` 를 쓴다.

decoder:

- `move-told-h1` 만 decoder 를 학습한다. 나머지는 `has_decoder=False` 다.
- decoder loss 는 `z_pred.detach()` 로 계산하므로 예측기 학습에 영향을 주지 않는다.
- 평가에서는 모든 모델의 예측 latent 를 `move-told-h1` 의 decoder 로 복원한다. 이 decoder 는 다른 모델의 예측 latent 로 학습하지 않았다.

## 5. 평가

val 100 에피소드를 조건별로 렌더링하고 같은 프레임으로 모든 모델을 평가한다.

| 조건 | 카메라 |
|---|---|
| E-fixed | 데이터셋 카메라 |
| E-static | `static_rand` |
| E-move | `move` |
| E-move-fast | `move_fast` |

지표는 latent MSE 다. 1-step (t → t+1) 과 open-loop horizon 1–5 를 계산한다.

| 지표 | 예측과 비교하는 프레임 |
|---|---|
| `err_true` | 상태 t+1, 카메라 t+1 (실제 다음 프레임) |
| `err_copy` | 예측 대신 현재 프레임을 사용했을 때의 오차 |
| `err_camonly` | 상태 t, 카메라 t+1 (카메라만 이동한 프레임) |
| `err_objonly` | 상태 t+1, 카메라 t (물체만 이동한 프레임) |

해석:

- `err_true` 가 셋 중 최소이면 예측에 카메라 이동과 물체 이동이 모두 반영된 것으로 본다.
- `err_objonly` 가 최소이면 카메라 이동이 반영되지 않은 것으로 본다.
- `err_camonly` 가 최소이면 물체 이동이 반영되지 않은 것으로 본다.
- 카메라 이동에 의한 프레임 변화는 물체 이동에 의한 변화보다 크다. 스텝당 4.5° 회전에서 연속 프레임 간 평균 절대 차는 13.8 이고 물체만 이동할 때는 0.41 이다. 그래서 `err_true` 와 함께 `err_camonly`, `err_objonly` 를 본다.

추가로 측정하는 것:

- 복원 영상의 LPIPS, SSIM
- 실제 프레임과 예측 프레임을 비교하는 영상

planning (MPC, Chamfer distance) 은 이번 범위에 넣지 않는다.

평가 스크립트는 [scripts/eval_camera_pred.py](scripts/eval_camera_pred.py) 다. 기존 `fixed` 모델로 실행을 확인했다 (Rope, val 30 에피소드). 3절의 사전 측정은 별도 스크립트로 했다.

## 6. 실행

| 단계 | 장소 | 내용 |
|---|---|---|
| 렌더링 | 로컬 (PyFleX) | `static_rand`, `move` 전체와 평가용 val 세트. Rope 는 프레임당 약 28 ms, 2만 프레임에 약 9분이다. |
| 학습 | Colab GPU | 세션 하나의 GPU 에서 작업 2–3개를 동시에 실행한다. 동시에 열 수 있는 세션이 3개여서 8개 작업을 2, 3, 3개로 나눴다. 체크포인트는 VM 에서 비공개 Hugging Face 저장소로 올린다. |
| 평가 | 로컬 | 체크포인트를 받아 5절의 평가를 실행한다. |

```bash
# 렌더링 (source repro/env.sh 실행 후)
python scripts/render_camera_data.py --object rope --variant move
python scripts/render_camera_data.py --object rope --variant static_rand
# 학습 (move-told-h1 의 예)
python train.py env=deformable_cam env.dataset.object_name=rope env.kwargs.object_name=rope \
    env.dataset.variant=move env.dataset.cam_action=true frameskip=1 num_hist=1 \
    training.epochs=50 training.save_every_x_epoch=10
```

## 7. 한계

- 카메라는 위치 3자유도만 변한다. 방향은 테이블을 향하도록 정해진다. 화각은 고정이다. 흔들림은 넣지 않았다.
- 모든 프레임에서 로봇 팔은 home 자세다. 원본 데이터도 같다.
- `fixed` 모델은 torch 2.3 에서 decoder 와 함께 학습했다. 새 모델은 다른 torch 버전에서 decoder 없이 학습한다. 이 차이의 영향은 확인하지 않았다.
- seed 1개, 객체당 val 100 에피소드.

## 8. 브랜치의 다른 파일

이 브랜치에는 thread1 `occlusion0` 의 Distracting Reacher 코드가 merge 되어 있다 ([README_experiment.md](README_experiment.md), `CLAUDE.md`, `scripts/run_all.sh` 등). Distracting Reacher 는 thread1 의 실험이고 이 브랜치의 실험에 포함되지 않는다.
