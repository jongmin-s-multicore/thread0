# hanbin5/local — 24 GB GPU 2장 환경

이 브랜치는 main 위에 **24 GB GPU 에 맞춘 구현**과 **이 환경에서 돌린 실행 기록**을 더한다 ([AGENTS.md §1](../AGENTS.md#1-무엇을-어디에-두나)).
벤치마크 프로토콜과 공용 도구는 main 의 [README](README.md)·[SETTINGS.md](SETTINGS.md) 를 따른다. 결과와 해석은 [이슈](https://github.com/jongmin-s-multicore/thread0/issues)로 올리고, 이 브랜치의 커밋 해시를 적는다.

## 1. 환경

| 항목 | 값 |
|---|---|
| GPU | RTX 3090 Ti 24 GB + RTX 3090 24 GB, NVIDIA driver 580.178.04 |
| 호스트 | Ubuntu 24.04, 32 코어 CPU, RAM 48 GB + 스왑 64 GB |
| 소프트웨어 | main 과 같다 (`repro/requirements/`) |

머신 하나의 값(작업 루트 경로, 작업별 GPU 고정·메모리 조건)은 이 브랜치에도 넣지 않고 그 머신의 `.claude/` 에 둔다.

## 2. main 에 더한 코드 (`git diff main -- . ':!repro' ':!*.md'`)

수정 블록마다 `[repro]` 주석이 있다. 아래 환경변수는 이 브랜치의 `repro/queue/run_plan.sh` 가 기본으로 켠다 (`DINO_WM_SDPA=1 DINO_WM_SKIP_SOLVED=1 DINO_WM_ROLLOUT_CHUNK=150 DINO_WM_GD_CHUNK=8`). 끄려면 0 으로 export 한다.

| 변경 | 이유 | 결과에 주는 영향 (확인 방법) |
|---|---|---|
| MPC 에서 이미 성공한 에피소드의 CEM 생략 (`DINO_WM_SKIP_SOLVED=1`) | 그 에피소드의 계획 행동은 어차피 0 으로 바뀐다. 2회차부터 계산량이 성공률만큼 준다 | 1회차는 동일. 이후는 같은 알고리즘이지만 남은 에피소드의 난수 순서가 달라진다. 내부 조기 종료 조건은 남은 에피소드에 대해서만 본다 |
| predictor attention 을 fp32 `scaled_dot_product_attention` 으로 (`DINO_WM_SDPA=1`, eval 모드에서만) | 원본은 마스크·softmax 를 따로 계산해 300 샘플 × 588 토큰에서 행렬 하나가 6.2 GB 다 | 300 샘플 5스텝 롤아웃 latent 상대오차 3e-7 – 1.3e-6 (PointMaze·PushT·Wall 체크포인트로 측정), 1.2–1.6배 빠르다. TF32 는 상대오차 1e-3 이라 쓰지 않았다 |
| CEM 롤아웃을 N 샘플씩 (`DINO_WM_ROLLOUT_CHUNK`, 24 GB 에서 150) | 메모리 | 샘플끼리 독립이라 같은 계산 |
| GD 순전파·역전파를 에피소드 N 개씩 (`DINO_WM_GD_CHUNK`, 24 GB 에서 8 안팎) | 메모리 (h=3 모델 역전파가 에피소드 10개에 12 GB, 25개는 24 GB 에서 OOM) | 에피소드별 손실의 합이라 그래디언트가 같다 (상대오차 ≤ 1.4e-6, 측정) |
| 평가 시 그림에 쓰는 `n_plot_samples` 개만, 한 개씩 디코딩하고 평가 뒤 `empty_cache` | 원본은 50개 × 지금까지의 프레임 전부를 한 번에 디코딩해 MPC 반복마다 메모리가 늘어난다 (issue #23) | 지표는 디코딩 전에 계산되므로 같다. 그림·영상도 같다 |
| `train.py` 검증을 `torch.no_grad()` 로 | 원본은 검증 forward 가 그래프를 만들어 24 GB 에서 OOM | 검증은 역전파하지 않으므로 학습은 같다 |
| `VWorldModel.forward` 에서 `decoder_loss_pred` 용 디코딩(`z_pred.detach()`)을 `torch.no_grad()` 로 | 그 값은 `loss` 에 들어가지 않고 로그·그림에만 쓰는데, 원본은 디코딩 그래프(예측 프레임 num_hist 장)를 역전파가 끝날 때까지 들고 있다. num_hist 3 (Pour Water) 에서 배치 32 가 24 GB 에서 OOM 이었다 | 손실은 비트 단위로 같고, 그래디언트는 GPU 비결정성 수준만 다르다 (PointMaze 체크포인트, 배치 4: 원본과의 차이가 같은 코드를 두 번 돌린 차이와 같은 크기 — 텐서별 상대오차 최대 6.8e-6 대 8.9e-6). 최대 메모리 4.90 → 4.14 GiB (배치 4) |

## 3. 24 GB GPU 에서의 메모리 (이 환경에서 잰 값)

- h=3 모델(PointMaze·PushT)의 GD 역전파는 에피소드 10개에 약 12 GB(원본 attention 기준), 25개는 24 GB 에서 OOM 이다. 청크 0(50개 한 번에)으로는 돌지 않는다.
- CEM 롤아웃 300 샘플은 SDPA 에서 한 번에 약 5.8 GB, 150 개씩이면 약 3.6 GB 다 (원본 attention 은 한 번에 OOM — attention 행렬 하나가 6.2 GB).
- planning 작업 하나는 GPU 를 약 6–10 GB, 학습 작업(Rope·Granular, batch 32)은 약 11.5 GB 쓴다.
- PointMaze 작업은 env 워커 50개의 mujoco-py EGL 컨텍스트로 약 5 GB 를 더 쓴다. 학습과 같은 GPU 에 두면 MPC 는 시작 직후, GD 는 첫 스텝·첫 평가에서 OOM 이었다. GPU 하나에 작업 여러 개를 돌릴 때는 overlay 로 PointMaze 작업에 `need=` 를 크게 준다.
- 원본 코드로는 MPC 반복이 쌓이면 평가 디코딩(50개 × 프레임 전부)이, 학습에서는 검증 forward 가 OOM 이었다 (2절의 수정으로 해결).
- Pour Water 학습 (num_hist 3, frameskip 5, 224): 배치 32 는 24 GB 에서 OOM 이다 (디코더 conv 에서, 23.7 GB 점유 뒤). 2절의 `decoder_loss_pred` 수정을 넣고도 같았다. 배치 24 는 최대 19.8 GB (nvidia-smi). num_hist 1 은 배치 24 에서 7.1 GB.
- Cloth Move·Push Sand·Stack Cube 학습 (num_hist 1, frameskip 5, 224, 배치 24): 7.1 GB. 평가(`eval_acwm_phys.py`)는 학습 작업과 같은 GPU 에서 같이 돈다.

## 4. 계산 시간 (이 환경에서 측정)

| 항목 | 값 |
|---|---|
| world model 롤아웃 300 샘플 × 5 스텝, h=3 (PointMaze), RTX 3090 Ti 단독 | 원본 attention 3.8 s, SDPA 3.0 s |
| 같은 것, h=1 (Wall) | 원본 1.3 s (단독). SDPA 는 GPU 를 다른 작업과 나눠 쓰는 중에만 쟀다 (원본 2.2 s → 1.8 s, 단독 값과 비교할 수 없다) |
| CEM opt step 1번 (에피소드 50개, MPC 1회차) | PointMaze 약 371 s, PushT 약 373 s — 24 GB GPU 하나를 두 planning 작업이 나눠 쓸 때 (단독이 아니다), `plan_0_output_<k>.png` 저장 시각 간격 |
| Rope 학습 1 epoch (535 iteration + 검증) | RTX 3090 Ti 에서 약 5분 (GPU 를 거의 혼자 쓸 때), planning 작업과 GPU 를 나눠 쓸 때 8.5–10.3분 — `rollout_plots/e<k>_rollout` 생성 시각 간격 |
| FleX 롤아웃 (env 1개, prepare 포함 push 3번) | Rope 18.7 s, Granular 45 s |
| Pour Water 학습, 배치 24, 6 epoch (epoch = 슬라이스 전부 한 번) | num_hist 3: 0.88 s/it (RTX 3090), epoch 10,327 it + 검증 약 2.7 h, 총 16.1 h. num_hist 1: 0.37 s/it (RTX 3090 Ti), epoch 10,695 it 약 1.2 h, 총 7.0 h. 샘플당 약 37 ms 로 배치와 무관 |
| Pour Water 평가 (`eval_pour_water.py`, 에피소드 198개) | 약 23–26 분 (num_hist 1 23 분, num_hist 3 26 분) |
| Cloth Move·Push Sand·Stack Cube 학습, num_hist 1, 배치 24 | GPU 를 혼자 쓸 때 2.75 it/s (0.36 s/it), 같은 GPU 에 학습 둘이면 각 1.3 it/s. Cloth Move 15 epoch 6.8 h (RTX 3090 Ti), Push Sand 18 epoch 7.4 h (RTX 3090), Stack Cube 25 epoch 7.1 h (RTX 3090 Ti). Push Sand 와 Stack Cube 는 일부 구간에서 평가 작업과 GPU 를 나눠 썼다. 슬라이스 하나를 읽는 데 13–26 ms, DataLoader 12 worker 로 배치 24 에 0.09–0.19 s |
| 위 세 환경의 평가 | `eval_acwm_phys.py`: Cloth Move 400 에피소드 48 분, Push Sand 379 에피소드 57 분 (둘 다 학습과 GPU 공유), Stack Cube 476 에피소드 35 분 (단독). `eval_acwm_phys_quant.py` 약 1 분 (CPU), `render_acwm_phys.py` 20–90 초 |

## 5. 실행 기록

`repro/runs/{train,eval}/<run-id>/run.yaml` 에 실행별 설정을 둔다 (결과 수치 없음). `source repro/env.sh && python3 repro/make_run_yaml.py` 로 `repro/jobs/` 의 작업 목록(`benchmark.txt`, 추가 실행 `pointmaze_extra.txt`·`rope_plr.txt`·`pour_water.txt`·`acwm_phys.txt`)과 `$DINO_RUNS/<job>/run_info.txt` 에서 다시 만든다. 큐 밖에서 돌린 팔 영상 재실행도 같이 만든다.

| run-id | job | 내용 |
|---|---|---|
| `dinowm_rope-dinov2s14-100ep` | `train_rope` | Rope world model, DINOv2 ViT-S/14 동결, 100 epoch |
| `dinowm_granular-dinov2s14-100ep` | `train_granular` | Granular world model |
| `dinowm_{pointmaze,pusht,wall}-released-at{10,2,65}-mpccem` | `{pointmaze,pusht,wall}_mpc` | 공개 체크포인트 MPC-CEM (step 1 = 오픈루프 CEM) |
| `dinowm_{pointmaze,pusht,wall}-released-at{10,2,65}-gd` | `{pointmaze,pusht,wall}_gd` | 오픈루프 GD |
| `dinowm_{pointmaze,wall}-released-at{10,65}-cem30` | `{pointmaze,wall}_cem30` | 오픈루프 CEM, 30 opt steps |
| `dinowm_{rope,granular}-dinov2s14-100ep-at100-mpccem` | `{rope,granular}_mpc` | 학습한 모델 MPC-CEM, Chamfer distance |
| `dinowm_{pointmaze,pusht,wall}-released-at{10,2,65}-predq`, `dinowm_{rope,granular}-dinov2s14-100ep-at100-predq` | `pq_{pointmaze,pusht,wall,rope,granular}` | 예측 품질 |
| `dinowm_pointmaze-released-at10-{mpccem,cem30,gd}-{s1,s101}` | `pointmaze_{mpc,cem30,gd}_{s1,s101}` | PointMaze seed 1·101, 나머지는 seed 99 실행과 같다 (`jobs/pointmaze_extra.txt`) |
| `dinowm_pointmaze-released-at10-{mpccem,cem30,gd}-orig` | `pointmaze_{mpc,cem30,gd}_orig` | seed 99, 원본 코드 경로 (`DINO_WM_SDPA=0 DINO_WM_SKIP_SOLVED=0`) |
| `dinowm_rope-dinov2s14-100ep-plr5e-5`, `…-plr5e-5-at100-{mpccem,predq}` | `train_rope_plr5e-5`, `rope_mpc_plr5e-5`, `pq_rope_plr5e-5` | Rope 재학습, predictor lr 5e-5 (논문 Table 12 값, `jobs/rope_plr.txt`) |
| `dinowm_pourwater-dinov2s14-6ep-{h3f5,h1f5}` | `train_pw_{h3f5,h1f5}` | ACWM-Phys Pour Water world model, num_hist 3 / 1, frameskip 5, predictor lr 5e-5, 6 epoch, 배치 24 (`jobs/pour_water.txt`, SETTINGS.md §7) |
| `dinowm_pourwater-dinov2s14-6ep-{h3f5,h1f5}-at6-predq` | `pq_pw_{h3f5,h1f5}` | Pour Water 예측 품질·롤아웃·물 양 평가 (`repro/eval/eval_pour_water.py`) |
| `dinowm_{clothmove,pushsand,stackcube}-dinov2s14-{15,18,25}ep-h1f5` | `train_{cloth,sand,stack}_h1f5` | ACWM-Phys Cloth Move·Push Sand·Stack Cube world model, num_hist 1, frameskip 5, predictor lr 5e-5, 배치 24 (`jobs/acwm_phys.txt`, SETTINGS.md §8) |
| `dinowm_{clothmove,pushsand,stackcube}-dinov2s14-{15,18,25}ep-h1f5-at{15,18,25}-predq` | `pq_{cloth,sand,stack}_h1f5`, 이어서 `quant_*`, `render_*` | 위 세 모델의 예측 품질·롤아웃 (`eval_acwm_phys.py`), 물리량 (`eval_acwm_phys_quant.py`), 영상 (`render_acwm_phys.py`). 물리량·영상 명령은 run.yaml 의 `then` 에 있다 |
| `dinowm_{rope,granular}-dinov2s14-100ep-at100-mpccem-armvideo` | `{rope,granular}_mpc_arm` | `{rope,granular}_mpc` 와 같은 설정의 재실행 + 팔이 움직이는 평가 영상 (`repro/eval/deform_arm_video.py`, 큐 밖에서 실행). 같은 시작·목표 입자 상태지만 비트 단위 재현은 아니다 |

## 6. 코드 버전과 결과의 대응

결과를 낸 코드는 모두 upstream `0a9492f` + main 공용 수정 + 2절 수정이고, 작업마다 그 코드가 놓였던 곳만 다르다. 아래 작업이 모두 끝난 뒤 main 의 평가 영상 수정(칸 라벨·실행 끝에서 끊고 3초 멈춤, PushT·PointMaze 목표 표시: `planning/evaluator.py` `plan.py` `env/pusht/goal_outline.py` `env/pointmaze/goal_marker.py`)을 merge 했다 (merge `2daa991`). 그래서 `git diff archive/2026-10-01-pre-split hanbin5/local -- . ':!repro' ':!*.md' ':!.gitignore'` 에는 이 그림·영상 수정만 나온다. 이 수정은 지표를 계산한 뒤 그림·영상만 바꾼다 (표시를 그리는 실행과 그리지 않는 실행에서 `logs.json` 이 같다, SETTINGS.md §6). 지표를 내는 코드는 아래 모든 경우에 같다.

| 코드가 있던 곳 | 작업 |
|---|---|
| fork 이전: upstream clone `0a9492f` 에 같은 diff 를 커밋 없이 패치, DINOv2 는 hub 캐시 `85a2460` (그 clone 의 diff 는 이 브랜치와 `[repro]` 주석·hub 고정 줄만 다르다) | `train_rope` `train_granular` `{pointmaze,pusht,wall}_mpc` `pusht_gd` |
| fork 의 나누기 전 main `c34ddd3` (`3a4341b` 의 조상) | `wall_gd` |
| 나누기 전 main `3a4341b`, 태그 [`archive/2026-10-01-pre-split`](https://github.com/jongmin-s-multicore/thread0/tree/archive/2026-10-01-pre-split) | `wall_cem30` |
| 이 브랜치 `798234f` (`run_info.txt` 의 `commit=`) | `pointmaze_gd` `pointmaze_cem30` `{rope,granular}_mpc` `pq_*` |
| 이 브랜치 `a482f6e` (`run_info.txt` 의 `commit=`, `repro/` 밖 코드는 `798234f` 와 같다) | `pointmaze_{mpc,cem30,gd}_{s1,s101,orig}` `train_rope_plr5e-5` `rope_mpc_plr5e-5` `pq_rope_plr5e-5`; `{rope,granular}_mpc_arm` 은 이 코드의 `plan.py` 를 `repro/eval/deform_arm_video.py`(그때는 main 작업 트리의 커밋 전 파일, 같은 동작으로 main `5180303` 에 커밋)로 돌렸다 |

| 이 브랜치 `41606dc` (`run_info.txt` 의 `commit=`; 학습에 쓰는 코드는 `e2bbc0e` 의 `VWorldModel.forward` 수정과 main `d0c66eb` 의 로더) | `train_pw_{h3f5,h1f5}` (파일럿 `train_pw_pilot` 은 `e2bbc0e`, 결과에 쓰지 않는다) |
| 이 브랜치 `0a75239` (평가 스크립트는 main `1c22b7c` + `dbb03d8`) | `pq_pw_{h3f5,h1f5}` |
| 브랜치 `hanbin5/acwm-phys` 의 커밋 전 작업 트리 (`run_info.txt` 의 `commit=d0e27aa` 는 그 바탕 커밋). 그 파일들을 내용 그대로 커밋했다: 로더·설정 `ebe1aea`, 작업 목록 `060b292`, 평가 도구 `10e8192` (학습에 쓴 로더·설정 4개는 sha256 을 대조했다) | `train_{cloth,sand,stack}_h1f5` `pq_{cloth,sand,stack}_h1f5` `quant_*` `render_*` (파일럿 `train_{cloth,sand,stack}_pilot` 은 결과에 쓰지 않는다) |

- 앞의 세 줄에 해당하는 작업은 `run_info.txt` 가 이전 형식(`chunk= sdpa= skip_solved= gdchunk=`, 브랜치·커밋 없음)이다. 작업별 코드 위치는 run.yaml 의 `code` 에 적었다. 결과 이슈에는 `hanbin5/local` 의 커밋과 이 대응을 함께 적는다.
- 태그 설명의 "나누기 전에 시작한 작업은 이 커밋에서 돌았다" 는 `wall_cem30` 에만 맞다. 이 표가 정확하다.
- `train_rope` 는 planning 쪽 수정(`cem.py`·`mpc.py`·`evaluator.py`)이 들어가기 전에 시작했지만 학습은 그 파일을 쓰지 않는다.
- OOM 이나 코드 수정으로 중간에 멈춘 시도는 결과에 쓰지 않는다.
- Pour Water 실험(`jobs/pour_water*.txt`, 로더, 평가, 위 두 행)은 브랜치 `hanbin5/water` 에 있다. 실행 당시 이 커밋들의 브랜치 이름은 `hanbin5/local` 이었다 (`run_info.txt` 의 `branch=`). 이 문서의 Pour Water 항목은 그 브랜치의 것이다.
- Pour Water 의 물 양 분석·그림·정성 비교(`eval_pour_water_volume.py`, `plot_pour_water.py`, `render_pour_water.py`)는 큐 밖에서 main `36834f5` 와 같은 내용의 파일(그때는 커밋 전 main 작업 트리)로 돌렸다. `eval_pour_water.py` 의 "action 0" 변형은 `36834f5` 에서 같은 값(정규화한 0)을 명시적으로 쓰도록 바꿨고 동작은 같다.
- `repro/eval/eval_pred_quality.py` 는 공개 전 검토에서 부분 표본이 실행마다 달라지는 것을 찾아 `seed(training.seed)` 를 넣었다 (예측 품질 작업은 그 뒤에 시작한다).
- Cloth Move·Push Sand·Stack Cube 실험(`jobs/acwm_phys*.txt`, 로더, 평가, 위 행)은 브랜치 `hanbin5/acwm-phys` (`hanbin5/water` 위)에 있다. 이 문서의 해당 항목은 그 브랜치의 것이다.
- 그 실험의 에피소드별 영상(`render_acwm_phys.py --per_episode 10`, Stack Cube 는 `--groups success,fail`)과 면적 그림(`plot_acwm_phys.py`)은 평가가 끝난 뒤 큐 밖에서 만들었다. 저장된 프레임과 JSON 만 읽는다.
