# PRIOR_WORK: ACWM-Phys 실험의 선행연구 조사

조사일은 2026-10-14 다. 이 브랜치의 ACWM-Phys 실험([루트 README](../README.md#summary))과 겹치는 선행연구를 정리한다. 이 실험의 결과 수치는 여기 두지 않는다. 루트 README 와 [SETTINGS.md](SETTINGS.md) 를 가리킨다.

## 1. 질문과 결론

질문: DINO-WM 으로, ACWM-Phys 처럼 InD 와 OoD 를 나눠, 모델이 물리 법칙에 맞게 예측하는지 잰 연구가 이미 있는가.

- DINO-WM 을 ACWM-Phys 에서 평가한 연구는 찾지 못했다. DINO-WM 을 변형체·유체·입자 장면에서 InD/OoD 로 나눠 물리량으로 잰 연구도 찾지 못했다.
- DINO-WM 을 물리 OoD 에서 물리 상태로 잰 연구는 두 편 있다 (§3.1). 두 편 모두 강체·관절 장면이고 학습 범위 밖으로 바꾼 요인이 중력과 관절 damping 이다.
- ACWM-Phys 저자들은 ICLR 2027 제출판에서 ACWM-DiT 의 보존 오차를 InD/OoD 로 쟀다 (§3.3). 분할한 양으로 물리량을 재는 지표는 이 실험의 새 기여가 아니다.
- 이 실험이 더한 것은 네 가지다. 모델 계열(DINO-WM), 예측된 양이 학습 때 본 값 쪽으로 간다는 방향, decoder 대조, 실패한 에피소드가 쌓인 것으로 그려지는 비율이다.

## 2. 조사 방법

- 검색한 곳: arXiv(API 와 HTML 전문), Semantic Scholar 인용 그래프(DINO-WM 을 인용한 472편), OpenReview 검색 API, GitHub 코드 검색(`"ACWM-Phys"`), Hugging Face(`t1an/ACWM-Phys` 를 쓰는 모델·space). 검색 기록은 레포에 두지 않았다.
- 진행: LLM 에이전트 6개가 영역을 나눠 검색했다. 그 뒤 서로의 후보를 원문으로 다시 확인하고 등급을 정했다.
- OpenReview 에만 있는 ICLR 2027 제출작 4편은 2026-10-14 에 받은 PDF 를 읽었다. 심사 중인 익명 제출작이라 내용이 바뀔 수 있다.

판정에 쓴 축은 다음과 같다.

| 축 | 본 것 |
|---|---|
| 모델 | DINO-WM 을 실제로 학습·평가했는가 (인용만 한 것은 제외) |
| 물리량 | 물리량·물리 상태를 직접 쟀는가, planning 성공률·영상 지표뿐인가 |
| InD/OoD | 학습 분포를 통제하고 그 밖의 조건을 따로 평가했는가 |
| 예측 | frame 과 action 에서 미래를 예측하는 rollout 을 평가했는가 |

아래 표의 "확인" 열은 확인 수준이다.

- **대조**: arXiv HTML 전문 또는 PDF 에서 인용문·표를 한 번 더 대조했다.
- **1회**: 검색 단계에서 원문을 읽고 적은 내용이다. 다시 대조하지 않았다.

## 3. 선행연구

### 3.1 DINO-WM 을 물리 OoD 에서 물리 상태로 잰 연구

| 논문 | 한 것 | 이 실험과 다른 점 | 확인 |
|---|---|---|---|
| Semigroup-JEPA [1] | DINO-WM 을 baseline 으로 학습한다 (frozen DINOv2 ViT-S/14, 112×112 입력). MuJoCo 강체 장면에서 학습 중력을 N(4, 0.5²) 또는 N(9.8, 2.0²) 로 좁히고 더 넓은 격자에서 평가한다. open-loop rollout(최대 44 step)의 latent 에 probe 를 붙여 위치·속도·누적 회전 오차를 읽는다. Table 8 의 정사각형 위치 L2(m, 5/16/32/44 step)는 DINO-WM 0.454 / 1.117 / 1.570 / 2.006, SG-JEPA(GRU) 0.374 / 0.658 / 0.862 / 1.034 다 | 강체만 다룬다. 중력 값을 모델에 입력으로 준다. decoder 는 시각화에만 쓴다. DINO-WM 의 InD 와 OoD 를 나눈 표가 없고 중력별 값은 그림에만 있다. 틀리는 방향을 분석하지 않았다. seed 는 하나다. ACWM-Phys 를 관련 연구로 인용한다 | 대조: 인용문, Table 8, seed. 1회: 중력 입력, InD/OoD 표 없음 |
| PhysiCalWorld [2] | DINO-WM 과 LeWM 을 기반 모델로 학습한다. 환경은 Push-T(질량 중심), Reacher(관절 damping), BeerPong(접촉 damping), FetchSlide(질량·마찰·drag)다. 물리 설정 135개로 학습하고 못 본 15개로 평가한다. Reacher 는 damping 을 학습 범위 [0.005, 0.020] 밖으로도 평가한다 (Fig. 8). 지표는 첫 프레임부터의 open-loop rollout 의 PSNR·SSIM·LPIPS·물체 마스크 IoU 와 planning 성공률이다 | 바꾼 요인이 프레임에 보이지 않는다. 강체·관절 장면이다. 범위 밖 수치는 그림에만 있다. DINO-WM 이 틀리는 방향을 분석하지 않았다. 초록의 "regress toward an empirical prior" 는 동기 문장이고 측정값이 아니다 | 대조 (PDF) |

### 3.2 DINO-WM 의 OoD 평가 중 지표가 planning 결과나 latent 오차인 연구

| 논문 | 바꾼 요인 | 지표와 수치 | 확인 |
|---|---|---|---|
| DINO-WM §4.5 GranularRandom [3] | 입자 양. 본문은 "fewer than half the particles" 라고 적는다 | planning 뒤의 Chamfer distance 0.63 (Table 3). rollout 의 입자 양은 재지 않았다 | 대조 |
| Better Slots, Better Worlds [4] | 물체 크기 40 → 20, 60. 모양. 색 | PushT planning 성공률(%). DINO-WM 은 InD 72, agent 크기 68, block 크기 46, block 모양 26 이다. Fig. 3 의 막대 값이고 본문은 크기 결과를 서술하지 않는다 | 대조: baseline 구성. 1회: 수치 |
| stable-worldmodel-v1 [5] | Push-T 의 요인을 하나씩 바꾼다 (색, 크기, 각도, 위치, 모양, 속도) | planning 성공률(%). 변화 없음 94.0, 크기 Anchor 14.0 / Agent 4.0 / Block 16.0, 모양 Agent 18.0 / Block 8.0 (Table 2). 확장판 [6] 은 같은 표의 기준 행이 20.0 이다. 질량·마찰·중력 실험은 없다 | 대조: Table 2. 1회: 확장판 |
| Sandwich-Residuals [7] | 조명 0.7 → 0.3, 카메라 10° 회전, 팔 관절 position gain 절반 | OGBench-Cube planning 성공률(%, 3 seed × 50 에피소드). 적응하지 않은 DINO-WM 은 62.7±3.1 / 58.7±4.2 / 54.0±5.3 / 59.3±5.8 이다 (Table IV. 순서는 변화 없음, 조명, 카메라, 팔 gain) | 대조 |
| Fast Residual Memory [8] | PushT damping 과 수평 중력, Wall 바람, PointMaze action gain. 보정 모듈을 학습한 값의 범위 안과 밖 | 공개 DINO-WM 체크포인트를 고정하고 1-step latent 예측 오차를 잰다. 0.217 / 0.153 / 0.229 / 0.027 (Table 1, Base 행) | 대조 (PDF) |

### 3.3 모델 계열은 다르지만 인용할 연구

| 논문 | 내용 | 확인 |
|---|---|---|
| ACWM-Phys, arXiv v2 [9] | 데이터 출처다. ACWM-DiT 만 평가했다. 지표는 MSE, M-MSE, SSIM, PSNR 이다. OoD 에서의 관찰은 정성 서술이다. Push Sand 는 "the sand pile appears to shrink", Cloth Move 는 "hallucinates incorrect cloth extents", Pour Water 는 "underestimates the water amount" 라고 적는다. 전문에 DINO 와 JEPA 가 나오지 않는다 | 대조 |
| ACWM-Phys, ICLR 2027 제출판 [10] | ACWM-DiT-S 의 물리 지표를 InD 50, OoD 50 에피소드로 낸다 (Table 2). 보존 오차는 rollout 첫 프레임 대비 평균 상대 변화 \|q_t − q_0\| / q_0 다. q 는 물·모래의 분할한 질량, 로프·천의 윤곽 길이, 큐브의 개수다. InD → OoD 값은 Stack Cube 0.03 → 0.05, Cloth Move 0.85 → 3.00, Push Sand 0.08 → 0.31, Pour Water 0.06 → 0.15 다. 절댓값이라 늘었는지 줄었는지 알 수 없다. 실제 영상과 비교하지 않는다. 쌓기 성공·실패는 분석하지 않는다. 두 번째 모델은 Cosmos-Predict2.5-2B 이고 Push Cube 와 Push Rope 의 PSNR 만 있다. 전문에 DINO 와 JEPA 가 나오지 않는다 | 대조 (PDF) |
| Schott et al. [11] | 이미지에서 요인 값을 회귀하는 표현학습 모델을 요인 범위 밖에서 평가했다. 5.3절: "Instead of extrapolating, all models regress the OOD factor towards the mean in the training set." world model 과 rollout 은 다루지 않는다 | 대조 |
| Kang et al. (Phyworld) [12] | 비디오 diffusion 을 2D 공 장면에서 속도·크기 범위를 통제해 InD, OoD, 조합으로 평가했다. 지표는 생성 영상의 속도 오차다. OoD 에서 "mimicking the closest training example" 이라고 결론짓는다. "the size was consistently transformed" 는 빨갛고 작은 공과 파랗고 큰 공으로 학습한 색–크기 충돌 실험의 결과다. 한 가지 크기만 학습한 경우의 근거로 쓰지 않는다. VAE 로 실제 영상을 복원한 오차가 OoD 오차보다 한 자릿수 낮다는 대조가 부록에 있다 | 대조 |
| ReOI [13] | DINO-WM 을 로봇 조작 궤적 500개로 학습했다. 학습에 없던 distractor 가 있을 때 "DINO-WM to hallucinate and predict that a failed action plan would succeed" 라고 적는다 (Fig. 4 설명). 사례 제시이고 비율은 재지 않았다 | 대조: 인용문 |

### 3.4 검토했으나 근접 연구로 보지 않은 것

| 논문 | 이유 | 확인 |
|---|---|---|
| H-Lens [14] | DINO-WM 을 Pendulum, MountainCar, LunarLander 에서 평가한다. 그러나 동역학을 바꾼 환경에서 모델을 따로 학습한다 ("We train a second set of world models on the same environments but with dynamics-shifted parameters"). OoD 평가가 아니다 | 대조 (PDF) |
| Where World Models Break [15] | DINO-WM 의 decode 한 rollout 에 사건 기준을 적용한다. InD/OoD 분할이 없다 | 1회 |
| LeWorldModel [16] | DINO-WM 행이 physical probing 표에 있다. 실제 프레임 feature 의 probing 이고 OoD 가 없다 | 1회 |
| Tan et al. [17] | JEPA 식 모델의 latent 에서 질량·drag·강성을 probe 한다. 물리 파라미터의 범위 밖 평가가 없다. DINO-WM 은 인용만 한다 | 1회 |
| Neural Voxel Dynamics [18] | V-JEPA latent 위의 모델에서 부피 비를 잰다 (학습 도메인 1.08, zero-shot 1.63). DINO-WM 이 아니다 | 1회 |
| Do-JEPA [19], DWM [20], AdaJEPA [21] | JEPA 계열의 동역학 shift 평가다. DINO-WM 의 InD/OoD 대조가 없다 | 1회 |
| IMPLY [22] | V-JEPA 2-AC 가 물체와 무관한 값을 예측하는 실패를 잰다. 물체 단서를 뺀 조건의 결과다 | 1회 |
| MiraBench [23] | 비디오 생성 world model 이 실패해야 하는 action 에서 성공을 그리는 비율을 Optimism Bias 로 정의한다. 실패는 합성 perturbation 으로 만든다. DINO-WM 이 아니다 | 1회 |

## 4. 이 실험에 대해 쓸 수 있는 문장

| 문장 | 판정 | 근거와 붙일 것 |
|---|---|---|
| DINO-WM 을 ACWM-Phys 에서 평가한 연구는 2026-10-14 까지 찾지 못했다 | 쓸 수 있다 | ACWM-Phys 의 두 판 [9, 10] 에 DINO 와 JEPA 가 없다. ACWM-Phys 를 인용한 논문 10편, GitHub 코드 검색, Hugging Face 에서 DINO-WM 을 돌린 것을 찾지 못했다 (1회) |
| DINO-WM 의 물리 OoD 일반화를 물리량으로 잰 연구는 없다 | 쓰지 않는다 | [1] 과 [2] 가 있다. 대신 "첫 프레임에 보이는 물질의 양·크기가 학습 범위 밖일 때 DINO-WM rollout 의 양을 잰 연구는 찾지 못했다"로 쓴다 |
| OoD 로 준 양·크기가 rollout 중에 학습 때 본 값 쪽으로 간다는 것을 수치로 보인 연구는 없다 | 조건부 | DINO-WM·JEPA 계열로 한정하고 "찾지 못했다"로 쓴다. [11] (표현학습에서의 평균 회귀), [9, 10] (같은 데이터의 정성 서술과 보존 오차. 크기만 재고 방향은 없다), [12] (속도 외삽)를 함께 인용한다 |
| DINO-WM 에서 실패한 에피소드가 성공으로 그려지는 비율을 잰 연구는 없다 | 조건부 | "찾지 못했다"로 쓰고 [13] 의 사례 보고를 인용한다 |
| OoD 실패가 decoder 가 아니라 predictor 에 있음을 처음 분리했다 | 쓰지 않는다 | 측정이 decoder 에 막히지 않는다는 대조로 쓴다. 같은 논리의 선례가 있다. [1] 은 실제 latent 와 예측 latent 에 같은 probe 를 써서 probe 오차를 뺀다. [12] 는 VAE 복원 오차를 잰다 |
| Push Sand 결과가 DINO-WM §4.5 의 가설을 반박한다 | 쓰지 않는다 | §5 의 셋째 항목 |
| 색 마스크로 잰 물리량 지표가 새롭다 | 쓰지 않는다 | [10] 이 같은 데이터에서 분할한 양의 보존 오차를 잰다 |

## 5. 서술할 때 주의할 점

- Pour Water 의 OoD 는 첫 프레임 물 픽셀 수(W0)로 보면 학습 범위와 일부 겹친다. 학습 2,351–6,202, OoD 낮음 1,675–2,795, OoD 높음 5,360–8,366 이다 ([SETTINGS.md §7.1](SETTINGS.md#71-데이터)). "학습 범위 밖"의 근거는 Cloth Move 와 Push Sand 로 든다 ([SETTINGS.md §8.1](SETTINGS.md#81-데이터)).
- Stack Cube 결과를 "낙관 편향"(optimism bias)이라고 부르지 않는다. 루트 README 의 물리량 표에 있는 반올림된 비율로 전체 쌓임 비율을 계산하면, InD 는 실제 (50×100 % + 17×12 %) / 67 ≈ 78 %, 모델 (50×96 % + 17×24 %) / 67 ≈ 78 % 다. OoD 는 실제 (200×100 % + 71×8 %) / 271 ≈ 76 %, 모델 (200×90 % + 71×23 %) / 271 ≈ 72 % 다. 모델은 실패를 쌓인 것으로 그리는 만큼 성공을 안 쌓인 것으로도 그린다.
- DINO-WM §4.5 를 이 실험의 반대 결과로 쓰지 않는다. 그 절의 가설은 observation model 에 관한 것이다 ("DINO-WM's observation model encoding the scene as patch features, making the variance in particle number still within the distribution for each image patch"). 이 실험의 decoder 대조(실제 프레임의 feature 를 decode 하면 OoD 양이 대체로 그려진다)는 그 가설과 어긋나지 않는다. 조건도 다르다. 원 논문은 입자가 줄어든 장면의 planning 결과이고, 이 실험은 입자가 약 2배인 장면의 rollout 이다. 원 논문 Table 1 의 Granular CD 0.26 (1회)과 Table 3 의 0.63 은 과제와 입자 수가 달라 InD 대 OoD 비교로 읽을 수 없다. 입자 수 설정은 `conf/env/granular.yaml` 의 `area: 5.0 # change particle numbers` 와 주석 처리된 `# area: 1.5` 에 남아 있다. 1.5 를 실제로 썼는지는 확인하지 않았다.
- decoder 대조를 인용할 때 Push Sand OoD 에서는 decoder 도 실제보다 17 % 적게 그린다는 것을 함께 적는다 (루트 README 의 "Limits of these numbers").

## 6. 한계

- Semantic Scholar 인용 그래프는 최신 인용을 다 담지 않는다. [1] 은 인용 그래프가 아니라 arXiv 검색에서 나왔다. Google Scholar 는 검색하지 않았다.
- arXiv API 는 초록만 검색한다. 본문 표에만 DINO-WM 이 나오는 논문은 놓쳤을 수 있다.
- OpenReview 에만 있는 제출작은 "DINO-WM" 검색 결과의 초록을 훑었다. PDF 를 읽은 것은 [2], [8], [10], [14] 네 편이다.
- 확인 수준이 "1회"인 내용은 다시 대조하지 않았다. 인용하기 전에 원문을 확인한다.
- 조사일 뒤에 나온 문헌은 없다. ACWM-Phys 제출판 [10] 은 모든 모델로의 확장을 남은 과제로 적는다.

## 7. 문헌

1. Liu, Sun, Baker, Balestriero, Sous. Semigroup-JEPA: Latent Dynamics Consistency for Zero-Shot Physics Generalization. 2026-09-09. [arXiv 2609.10464](https://arxiv.org/abs/2609.10464)
2. 저자 익명. PhysiCalWorld: In-Context Physics Calibration for World Models. ICLR 2027 제출작. [OpenReview rtlzkXdJe9](https://openreview.net/forum?id=rtlzkXdJe9)
3. Zhou, Pan, LeCun, Pinto. DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning. ICML 2025. [arXiv 2411.04983](https://arxiv.org/abs/2411.04983)
4. Nazirjonov, Prasanna, Manasyan, Martius. Better Slots, Better Worlds: Representation Quality & Robustness in Object-Centric World Models. 2026-08-12. [arXiv 2608.12078](https://arxiv.org/abs/2608.12078)
5. Maes et al. stable-worldmodel-v1: Reproducible World Modeling Research and Evaluation. 2026-02-09. [arXiv 2602.08968](https://arxiv.org/abs/2602.08968)
6. Maes et al. stable-worldmodel: A Platform for Reproducible World Modeling Research and Evaluation. 2026-05-20. [arXiv 2605.21800](https://arxiv.org/abs/2605.21800)
7. Soni, Sehgal, Dave, Rueckert. Sandwich-Residuals: Parameter-Efficient Test-time Adaptation of World Models. 2026-09-18. [arXiv 2609.21740](https://arxiv.org/abs/2609.21740)
8. 저자 익명. Fast Residual Memory for Test-Time Training of Latent World Models. ICLR 2027 제출작. [OpenReview JvQYIoQxQ0](https://openreview.net/forum?id=JvQYIoQxQ0)
9. Xue, Chen, Ma, Zhao, Moukheiber, Zhu, Chen. ACWM-Phys: Investigating Generalized Physical Interaction in Action-Conditioned Video World Models. v2 2026-05-16. [arXiv 2605.08567](https://arxiv.org/abs/2605.08567)
10. 같은 제목의 ICLR 2027 제출작. [OpenReview l4rkpIZvDq](https://openreview.net/forum?id=l4rkpIZvDq)
11. Schott et al. Visual Representation Learning Does Not Generalize Strongly Within the Same Domain. ICLR 2022. [arXiv 2107.08221](https://arxiv.org/abs/2107.08221)
12. Kang et al. How Far is Video Generation from World Model: A Physical Law Perspective. ICML 2025. [arXiv 2411.02385](https://arxiv.org/abs/2411.02385)
13. Chen et al. Reimagination with Test-time Observation Interventions: Distractor-Robust World Model Predictions for Visual Model Predictive Control. 2025-06-19. [arXiv 2506.16565](https://arxiv.org/abs/2506.16565)
14. 저자 익명. H-Lens: Extracting Phase-Space from World Model Representations. ICLR 2027 제출작. [OpenReview P1J97EDjIK](https://openreview.net/forum?id=P1J97EDjIK)
15. Shi et al. Where World Models Break: Natural-Input Failure Discovery. 2026-08-23. [arXiv 2608.22421](https://arxiv.org/abs/2608.22421)
16. Maes, Le Lidec, Scieur, LeCun, Balestriero. LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels. 2026-03-13. [arXiv 2603.19312](https://arxiv.org/abs/2603.19312)
17. Tan et al. What Can Latent World Models Know? Physical Information in Multimodal Predictive Representations. 2026-07-29. [arXiv 2607.27017](https://arxiv.org/abs/2607.27017)
18. Wang, Mitra. Neural Voxel Dynamics: Learning Volumetric Feature Advection for 3D Physics in V-JEPA Latent Space. 2026-06-24. [arXiv 2606.26410](https://arxiv.org/abs/2606.26410)
19. Resani, Shi. Do-JEPA: From Masking to Intervention in Latent World Models. 2026-09-29. [arXiv 2609.37378](https://arxiv.org/abs/2609.37378)
20. Zhang, Du, Zhang, Wang. DWM: Separating World Effects from Actions in Latent World Models. 2026-07-21. [arXiv 2607.18715](https://arxiv.org/abs/2607.18715)
21. Wang, Bounou, LeCun, Ren. AdaJEPA: An Adaptive Latent World Model. 2026-06-30. [arXiv 2606.32026](https://arxiv.org/abs/2606.32026)
22. Mehta, Baviskar. IMPLY: Physically Anchored Consistency for World-Model Rollouts. 2026-09-11. [arXiv 2609.12441](https://arxiv.org/abs/2609.12441)
23. Yang et al. MiraBench: Evaluating Action-Conditioned Reliability in Robotic World Models. 2026-05-28. [arXiv 2605.29360](https://arxiv.org/abs/2605.29360)
