# ACWM-Phys Cloth Move · Push Sand · Stack Cube 데이터 검사

`repro/SETTINGS.md` §8.1–§8.3 에 적은 데이터 사실(프레임과 action 의 정렬, OoD 의 크기, frameskip·num_hist 근거, 쌓임 판정)을 낸 스크립트다. 학습·평가에는 필요 없다. 데이터를 받은 뒤 (`python repro/setup/download_acwm_phys.py`) 레포 루트에서 돌린다.

```bash
source repro/env.sh
OUT=$DINO_WORK/results/acwm_phys_check && mkdir -p $OUT
python repro/setup/checks/acwm_phys/scan.py $DATASET_DIR/acwm_phys $OUT          # 전 프레임 스캔 (20 프로세스, 약 2분)
python repro/setup/checks/acwm_phys/report.py $OUT                              # 분포·이상치, 프레임-action 정렬
python repro/setup/checks/acwm_phys/details.py $OUT $DATASET_DIR/acwm_phys      # 천 크기, 쿼터니언 부호, Stack Cube action 필드
python repro/setup/checks/acwm_phys/history.py $OUT                             # 이전 스텝이 예측에 보태는 정도 (선형)
python repro/setup/checks/acwm_phys/stack_rule.py $OUT                          # 쌓임 화면 규칙 대 success 라벨
python repro/setup/checks/acwm_phys/copy_baseline.py $DATASET_DIR/acwm_phys $OUT/copy_baseline.json   # k 프레임 복사 오차 (GPU)
```

| 스크립트 | SETTINGS.md 에서 쓰인 곳 |
|---|---|
| `scan.py` | 나머지의 입력 (`<env>_scan.npz`) |
| `report.py` | §8.1: 제외한 에피소드 없음, action t 가 프레임 t→t+1 을 만든다 (Cloth Move·Stack Cube), Push Sand 의 action 은 프레임 t 의 자세 |
| `details.py` | §8.1: Cloth Move 는 학습 천 크기가 하나고 OoD 는 두 크기, Push Sand 쿼터니언 부호가 뒤집힌다, Stack Cube 는 `actions` 가 `actions_abs` 차분보다 낫다 |
| `copy_baseline.py` | §8.2: frameskip 근거 (k=1 / k=5 복사 LPIPS) |
| `history.py` | §8.2: num_hist 근거 |
| `stack_rule.py` | §8.3: 쌓임 판정이 `success` 와 98.8 % 일치 |

색 마스크의 HSV 범위는 원본 프레임에서 정했다. 사람이 영상을 보고 확인한 것(빠른 움직임 이상치가 천이 미끄러지는 장면이라는 것 등)은 스크립트에 없다.
