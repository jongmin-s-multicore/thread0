# 실험 계획: Rope·Granular 에서 카메라가 움직일 때의 예측

브랜치 `hanbin5/occlusion`. Distracting Reacher 실험([README_experiment.md](README_experiment.md))의 확장이다.
그 실험이 "작업과 무관한 배경 변화"를 다뤘다면, 이 실험은 **카메라의 3D pose 가 에피소드 안에서 움직일 때**
DINO-WM 이 다음 관측을 예측할 수 있는지를 본다. 작성 2026-10-13. 결과는 이 문서가 아니라 이슈에 올린다.

## 1. 질문

| | 질문 | 답을 주는 비교 (§4 모델 × §5 조건) |
|---|---|---|
| Q1 | 고정 카메라로 학습한 모델은 시점이 바뀌거나 움직이면 얼마나 무너지는가 | `fixed` 모델을 E-static·E-move 에서 |
| Q2 | 카메라 이동을 입력으로 알려 주면 움직이는 시점의 다음 관측을 예측하는가 | `move-told-h1` 을 E-move 에서 |
| Q3 | 알려 주지 않으면? context 가 1장일 때(이동 속도를 알 수 없음)와 2장일 때(추정 가능) | `move-blind-h1` 대 `move-blind-h2` 대 `move-told-h1` |
| Q4 | 어려운 것이 "시점이 다양한 것"인가 "시점이 움직이는 것"인가 | `static` 을 E-static 에서, `move-*` 를 E-move 에서 |
| Q5 | 예측이 카메라 이동과 물체 이동을 둘 다 담는가, 한쪽만 담는가 | counterfactual 지표 (§5) |
| Q6 | 학습한 범위보다 빠른 카메라 이동으로 외삽되는가 | `move-*` 를 E-move-fast 에서 |

## 2. 데이터: 같은 궤적, 다른 카메라

DINO-WM 의 deformable 데이터(Rope·Granular 각 1000 에피소드 × 20 프레임)는 프레임마다 입자 위치(`states.pth`)를
저장하고, 프레임을 저장하는 시점에는 로봇 팔이 항상 home 자세다. 그래서 저장된 상태와 카메라 pose 만 있으면 그
프레임을 다시 그릴 수 있다. [scripts/render_camera_data.py](scripts/render_camera_data.py) 는 시뮬레이션을 다시
돌리지 않고 저장된 상태를 PyFleX 에 넣어 원하는 카메라로 렌더링한다.

- 모든 변형은 **상태·push action 이 원본과 동일**하고 픽셀만 다르다 (Distracting Reacher 의 paired rendering 과 같은 원칙).
  train/val 분할도 원본과 같다 (같은 에피소드 순서, seed 42).
- 재렌더 검증: 데이터셋 카메라로 다시 그린 프레임과 원본 `obses.pth` 의 평균 절대 차는 Rope 0.13 / 255
  (에피소드 10개, `--variant fixed` 의 `meta.json`), Granular 0.10 / 255 (에피소드 1개). 원본의 연속 두 프레임 사이 차이는
  Rope 0.41, Granular 1.33 이다.
- 카메라 pose 식은 데이터셋의 카메라 4대 위치·각도와 오차 1e-15 로 일치한다.

### 카메라 pose 와 궤적 ([env/deformable_env/camera_motion.py](env/deformable_env/camera_motion.py))

pose = (azimuth, height, radius). 카메라는 `(radius·sin(az), height, radius·cos(az))` 에 있고 테이블 쪽을 본다
(yaw = azimuth, pitch = −atan2(height, radius + 1.515)). 데이터셋 카메라는 (135°, 10, 8.49) 다.

| 범위 | 값 | 이유 |
|---|---|---|
| azimuth | −55° … 235° (290° 호) | 로봇 팔 뒤쪽(약 240°–300°)에서는 팔이 테이블 대부분을 가리고, 낮고 가까운 카메라는 팔 안으로 들어간다. 데이터셋 카메라 4대(45°·135°·225°·315°)는 호 안에 있다 |
| height | 7 … 13 | 데이터셋 10 |
| radius | 6.5 … 10.5 | 데이터셋 8.49 |

| 변형 | 에피소드 안에서 | azimuth 속도 | 용도 |
|---|---|---|---|
| `fixed` | 데이터셋 카메라 고정 | 0 | 기준 조건 |
| `static_rand` | 에피소드마다 무작위 pose, 움직이지 않음 | 0 | 시점 다양성만 |
| `move` | 등속 이동: azimuth 는 스텝당 일정 각도(부호 무작위), height·radius 는 무작위 두 값 사이를 선형으로 | 스텝당 2°–8° (에피소드 전체 38°–152°) | 학습·평가 |
| `move_fast` | `move` 와 같고 더 빠름 | 스텝당 10°–15° | 평가 전용 (학습 범위 밖) |

궤적은 (변형, 에피소드 번호, seed) 의 순수 함수라 다시 만들면 같은 값이 나온다. 한 스텝은 push 한 번이다.

## 3. 사전 측정: 기존 고정 카메라 모델 (재학습 없이)

기존 `train_rope` (100 epoch) 를 val 에피소드 40개에서 쟀다. 카메라는 데이터셋 pose 에서 azimuth 만 바꿨다
(정지: 일정 각도만큼 옮김, 이동: 스텝당 일정 각도 회전). 값은 예측 latent 와 실제 다음 프레임 latent 의 MSE 다
(학습 loss·planning 비용과 같은 양). "복사"는 현재 프레임을 그대로 다음 프레임으로 쓰는 기준선이다.

| 조건 | 1-step 예측 | 복사 기준선 | 복원 LPIPS / SSIM |
|---|---|---|---|
| 데이터셋 카메라 | 0.077 | 0.390 | 0.044 / 0.963 |
| 정지, 5° 옮김 | 0.939 | 0.387 | 0.246 / 0.495 |
| 정지, 15° | 1.388 | 0.346 | 0.363 / 0.469 |
| 정지, 45° | 2.106 | 0.303 | 0.456 / 0.487 |
| 정지, 90° | 1.919 | 0.333 | 0.444 / 0.285 |
| 이동, 스텝당 0.5° | 0.835 | 0.580 | 0.225 / 0.547 |
| 이동, 스텝당 1° | 1.134 | 0.669 | 0.297 / 0.499 |
| 이동, 스텝당 2° | 1.578 | 0.837 | 0.372 / 0.478 |
| 이동, 스텝당 4.5° | 2.063 | 1.289 | 0.419 / 0.451 |

- 데이터셋 카메라에서의 LPIPS 0.044 / SSIM 0.963 은 예전에 val 전체로 잰 값(0.043 / 0.963)과 맞는다.
- 시점을 5° 만 옮겨도 예측 오차가 12배가 되고 "아무것도 안 변한다"고 답하는 것보다 나빠진다.
  즉 고정 카메라 모델은 시점 변화에 대한 일반화가 없다 (Q1 의 첫 답).
- 스텝당 9° 와 Granular 는 이 문서를 쓰는 시점에 측정 중이다.

## 4. 학습할 모델

객체(Rope, Granular)마다 아래 5개. `fixed` 는 이미 있는 모델을 쓰고 나머지 4개를 새로 학습한다 (총 8개).

| 이름 | 학습 데이터 | 카메라 이동 입력 | `num_hist` | decoder | 비고 |
|---|---|---|---|---|---|
| `fixed` | 원본 (고정 카메라) | – | 1 | 있음 | 기존 `train_rope`·`train_granular`. 10 epoch 마다 체크포인트가 있다 |
| `static` | `static_rand` | 없음 | 1 | 없음 | |
| `move-blind-h1` | `move` | 없음 | 1 | 없음 | 프레임 한 장으로는 이동 방향·속도를 알 수 없다 |
| `move-told-h1` | `move` | 있음 | 1 | 있음 | action = push(4) + (Δazimuth, Δheight, Δradius) = 7차원 |
| `move-blind-h2` | `move` | 없음 | 2 | 없음 | 두 장에서 이동을 추정할 수 있다 |

- 공통: `frameskip=1`, batch 32, 50 epoch, 10 epoch 마다 체크포인트, 나머지는 `conf/train.yaml` 기본값 (기존 `fixed` 모델과 같은 설정).
  설정 파일은 [conf/env/deformable_cam.yaml](conf/env/deformable_cam.yaml), 로더는 [datasets/deformable_cam_dset.py](datasets/deformable_cam_dset.py).
- **50 epoch 인 이유**: 기존 모델의 val latent 오차는 20–30 epoch 에서 멈춘다 (Rope 30 epoch 0.076 → 100 epoch 0.071,
  Granular 20 epoch 0.134 → 100 epoch 0.136; 각 모델의 `epoch_logs.jsonl`). 비교는 같은 epoch 의 체크포인트끼리 한다 (`fixed` 는 `model_50.pth`).
- **decoder 를 한 모델에만 두는 이유**: decoder 는 예측기에서 분리된 latent 로 학습하므로 (`z_pred.detach()`) 빼도 예측기는 같고 학습이 빨라진다.
  영상·LPIPS 용 decoder 는 `move-told-h1` 에서만 학습해 모든 모델의 예측 latent 를 복원하는 데 쓴다 (`move` 의 시점 범위가 다른 변형을 포함한다).
  이 decoder 는 다른 모델의 예측 latent 로는 학습하지 않았다는 점을 영상·LPIPS 해석에 감안한다.
- seed 는 하나다 (0).

## 5. 평가

val 에피소드 100개를 아래 조건으로 로컬에서 렌더링하고, 모든 모델을 같은 프레임으로 평가한다.

| 조건 | 카메라 |
|---|---|
| E-fixed | 데이터셋 카메라 |
| E-static | `static_rand` (학습 분포) |
| E-move | `move` (학습 분포) |
| E-move-fast | `move_fast` (학습 범위 밖) |

지표 (모두 latent MSE, 스텝 t → t+1; open-loop 은 horizon 1–5):

| 지표 | 예측과 비교하는 대상 | 읽는 법 |
|---|---|---|
| `err_true` | 상태 t+1 을 카메라 t+1 에서 본 프레임 | 실제 다음 관측에 대한 오차 |
| `err_copy` | (예측 대신 현재 프레임을 넣음) | 이것보다 크면 예측이 "변화 없음"보다 못하다 |
| `err_camonly` | 상태 t 를 카메라 t+1 에서 본 프레임 | 카메라만 움직이고 물체는 그대로인 가상의 다음 프레임 |
| `err_objonly` | 상태 t+1 을 카메라 t 에서 본 프레임 | 물체만 움직이고 카메라는 그대로인 가상의 다음 프레임 |

- `err_true` 가 `err_camonly`·`err_objonly` 보다 작으면 예측이 두 변화를 모두 담은 것이다. `err_objonly` 가 가장 작으면
  카메라 이동을 놓친 것이고, `err_camonly` 가 가장 작으면 시점만 따라가고 물체 dynamics 를 놓친 것이다 (Q5).
  카메라 이동이 만드는 픽셀 변화가 물체 이동보다 훨씬 커서 (스텝당 4.5° 회전 시 프레임 간 차 13.8 / 255, 물체만 움직일 때 0.41),
  `err_true` 만으로는 물체 dynamics 를 맞혔는지 알 수 없다.
- 복원 영상의 LPIPS·SSIM, 실제 프레임과 예측을 나란히 놓은 영상.
- planning (MPC, Chamfer distance) 은 이 단계에 넣지 않는다. 예측 결과를 보고 정한다.

평가 스크립트(`scripts/eval_camera_pred.py`)는 아직 작성하지 않았다. §3 의 사전 측정은 같은 계산을 하는 임시 스크립트로 했다.

## 6. 실행

| 단계 | 어디서 | 내용 |
|---|---|---|
| 렌더링 | 로컬 (PyFleX) | `static_rand`·`move` 전체, 평가용 val 렌더. 프레임당 약 28 ms (Rope), 세트(2만 프레임)당 약 9분 |
| 학습 | Colab GPU | 객체별 세션 하나에 4개 작업을 한 GPU 에서 동시에 돌린다. 체크포인트는 VM 에서 비공개 Hugging Face 저장소로 바로 올린다 |
| 평가 | 로컬 | 체크포인트를 받아 §5 |

```bash
# 렌더링 (source repro/env.sh 뒤)
python scripts/render_camera_data.py --object rope --variant move
python scripts/render_camera_data.py --object rope --variant static_rand
# 학습 한 개 (예: move-told-h1)
python train.py env=deformable_cam env.dataset.object_name=rope env.kwargs.object_name=rope \
    env.dataset.variant=move env.dataset.cam_action=true frameskip=1 num_hist=1 \
    training.epochs=50 training.save_every_x_epoch=10
```

## 7. 한계

- 카메라는 위치 3자유도만 움직이고 방향은 테이블을 보도록 따라간다. 내부 파라미터(화각)는 고정이다. 흔들림(jitter)은 없다.
- 모든 프레임에서 로봇 팔은 home 자세다 (원본 데이터와 같다). 팔이 움직이는 모습은 없다.
- `fixed` 모델은 torch 2.3, decoder 포함, 로컬 GPU 에서 학습했고 새 모델은 Colab GPU 에 맞는 torch 버전, decoder 없이 학습한다.
  예측기 학습에는 영향이 없다고 보지만 직접 대조하지는 않았다.
- seed 하나, 객체당 val 에피소드 100개.
