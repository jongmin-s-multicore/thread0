# ACWM-Phys 추가 실험 스크립트

InD 로만 학습한 DINO-WM 이 OoD 에서 실패하는 원인을 확인한 세 실험과 표를 만든 스크립트다.
작업 루트(`$DINO_WORK`)의 결과 파일을 읽고 쓴다. 레포 루트에서 `source repro/env.sh` 한 뒤 실행한다.

| 실험 | 내용 | 스크립트 | 결과 |
|---|---|---|---|
| 디코더 확인 | 실제 프레임을 인코더에 넣고 디코더로 바로 복원해 물체 양을 잰다 | `talk_render.py`, `lpips_table.py` | `<out>/<env>_summary.json`, `<env>_lpips.json` |
| OoD 를 섞어 학습 | OoD 테스트 에피소드 일부를 학습에 넣은 모델과 원래 모델을 학습에 쓰지 않은 OoD 에피소드에서 비교한다 | `mix_eval.py` (물체 양), `acwm_metrics_mix.py` (MSE, M-MSE, PSNR, SSIM) | `results/acwm_phys/ood_mix/<env>_mix_eval.json`, `<env>_mix_acwm.json` |
| 더 학습 | InD 로 2배, 4배, 8배 step 까지 이어 학습한 모델 | `water_long_eval.py` (Pour Water). Cloth Move, Push Sand, Stack Cube 는 `repro/eval/trend_acwm_phys.py` | `results/acwm_phys/long/` |
| Push Sand 디코더 학습량 | 디코더만 1.3배, 2배, 4배 step 까지 학습한 체크포인트 | `dec_eval.py` | `results/acwm_phys/ood_mix/sand_dec_eval.json` |

표 (LaTeX booktabs, tectonic 으로 PDF, pdftoppm 으로 PNG):

| 스크립트 | 표 |
|---|---|
| `make_table_pixel.py <out stem>` | InD / OoD 화소 지표 (MSE, M-MSE, PSNR, SSIM) |
| `make_table.py <render dir> <out stem>` | 디코더 확인: 마지막 프레임 물체 양과 LPIPS |
| `make_table_oodmix.py <out stem>` | OoD 를 섞어 학습: 마지막 프레임 물체 양 |
| `make_table_oodmix_pixel.py {rollout,window} <out stem>` | OoD 를 섞어 학습: 화소 지표 |
| `make_table_more_training.py <out stem>` | 더 학습: 물체 양과 검증 손실 |

학습은 `repro/colab/predictor_long.py` 로 했다. 저장된 DINOv2 토큰으로 predictor 와 action encoder 만 학습하고, 디코더는 원래 실행의 것을 쓴다.

```bash
python repro/colab/ckpt_tools.py slim $DINO_TRAIN train_sand_h1f5 sand_slim.pt
python repro/colab/predictor_long.py sand_slim.pt $DATASET_DIR/acwm_phys out --extra_split ood_test --extra_train all:even --workers 16 --tf32 --amp --epochs 6,12,18 --from_scratch --tag oodmix
python repro/colab/predictor_long.py sand_slim.pt $DATASET_DIR/acwm_phys out --epochs 36,72,144 --workers 12 --tf32 --amp
```

`ckpt_tools.py merge` 가 결과 가중치를 원래 체크포인트(디코더 포함)에 넣어 평가 스크립트가 읽는 형식으로 저장한다.

| 환경 | OoD 를 섞을 때 `--extra_train` | epoch (OoD 섞기) | epoch (더 학습) |
|---|---|---|---|
| Cloth Move | `smaller:even,larger:even` | 5, 10, 15 | 30, 60, 120 |
| Push Sand | `all:even` | 6, 12, 18 | 36, 72, 144 |
| Stack Cube | `all:even` | 10, 20, 25 | 50, 100, 200 |
| Pour Water | `all:mod4=0.1` | 2, 4, 6 | 12, 24, 48 |

학습한 체크포인트는 Hugging Face [`hanbin5/dino-wm-acwm-phys`](https://huggingface.co/hanbin5/dino-wm-acwm-phys) 의 `outputs/train_<env>_h1f5_oodmix`, `outputs/train_<env>_h1f5_long` 이다. `split.json` 에 학습에 넣은 에피소드와 남긴 에피소드가 있다.
