#!/bin/bash
# 새 일이 생길 때까지 기다린다 (최대 $1 초): 작업 DONE·실패, OOM 재시도, MPC/final 로그 줄 추가.
#   wait_event.sh 3300 major   # major = MPC 반복 로그는 무시
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"
snap() { echo "$(ls "$DINO_RUNS"/*/DONE 2>/dev/null | wc -l) $(grep -ls 'rc=[1-9]' "$DINO_RUNS"/*/run_info.txt 2>/dev/null | wc -l) $(cat "$DINO_RUNS"/scheduler*.log 2>/dev/null | grep -c 'OOM') $(cat "$DINO_RUNS"/*/logs.json 2>/dev/null | grep -c 'mpc/\|final_eval')"; }
MODE=${2:-all}
key() { if [ "$MODE" == "major" ]; then snap | cut -d' ' -f1-3; else snap; fi; }
S0=$(snap); K0=$(key); T=${1:-3300}; t=0
while [ "$(key)" == "$K0" ] && [ $t -lt $T ]; do sleep 60; t=$((t+60)); done
date -u +%T; echo "before: $S0  after: $(snap)  (done failed oom loglines)"
tail -4 "$DINO_RUNS/scheduler.log" 2>/dev/null
python3 "$HERE/../eval/summarize.py" > /dev/null 2>&1; sed -n '/^## Planning/,/^## Prediction/p' "$DINO_WORK/results/summary.md"
for f in "$DINO_TRAIN"/outputs/*/epoch_logs.jsonl; do
  [ -f "$f" ] && echo "$f: $(wc -l < "$f") epochs, last: $(tail -n1 "$f" | python3 -c 'import sys,json; d=json.load(sys.stdin); print({k:round(v,4) for k,v in d.items() if k in ("train_loss","val_loss","val_img_lpips_pred","val_img_ssim_pred")})')"
done
nvidia-smi --query-compute-apps=pid,gpu_bus_id,used_memory --format=csv,noheader
