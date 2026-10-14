#!/bin/bash
# ACWM-Phys baseline (이슈 #4) 학습 준비: 시작점 체크포인트와 학습 설정 파일을 $ACWM_BASE 에 만든다. 이미 있는 파일은 건너뛴다.
#   source repro/env.sh && bash repro/colab/acwm_baseline_prep.sh
# 1) HF hanbin5/dino-wm-acwm-phys 에서 baseline_x15/ (이슈 #4 §8 의 Colab 실행이 멈춘 state·best 가중치·기록)와
#    기존 4개 환경 실행의 hydra.yaml, Pour Water 6 epoch 체크포인트를 $HF_ACWM 에 받는다.
# 2) $ACWM_BASE/cfg/<run>.yaml: 기존 환경은 hydra.yaml 그대로, 새 환경은 train_cloth_h1f5 의 hydra.yaml 에 conf/env/acwm_<env>.yaml 을 넣는다.
# 3) Pour Water 시작 state ($ACWM_BASE/init_train_pw_h1f5.pt).
set -e
cd "$THREAD0"
mkdir -p "$HF_ACWM" "$ACWM_BASE/cfg"
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("hanbin5/dino-wm-acwm-phys", local_dir="$HF_ACWM", max_workers=8,
                  allow_patterns=["README.md", "baseline_x15/*", "outputs/*/hydra.yaml", "outputs/*/epoch_logs.jsonl",
                                  "outputs/train_pw_h1f5/checkpoints/model_latest.pth"])
EOF
for r in cloth sand stack pw; do
  [ -f "$ACWM_BASE/cfg/train_${r}_h1f5.yaml" ] || cp "$HF_ACWM/outputs/train_${r}_h1f5/hydra.yaml" "$ACWM_BASE/cfg/train_${r}_h1f5.yaml"
done
for e in push_rope push_cube robot_arm reacher; do
  [ -f "$ACWM_BASE/cfg/train_${e}_h1f5.yaml" ] || python repro/colab/ckpt_tools.py cfg "$HF_ACWM/outputs/train_cloth_h1f5/hydra.yaml" \
    "conf/env/acwm_${e}.yaml" "train_${e}_h1f5" "$ACWM_BASE/cfg/train_${e}_h1f5.yaml"
done
[ -f "$ACWM_BASE/init_train_pw_h1f5.pt" ] || python repro/colab/ckpt_tools.py init "$ACWM_BASE/cfg/train_pw_h1f5.yaml" \
  "$HF_ACWM/outputs/train_pw_h1f5/checkpoints/model_latest.pth" "$ACWM_BASE/init_train_pw_h1f5.pt"
echo "prepared $ACWM_BASE"
