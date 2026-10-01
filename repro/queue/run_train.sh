#!/bin/bash
# usage: run_train.sh <job> <gpu> <train.py hydra args...>
# 모델: $DINO_TRAIN/outputs/<job>/ (hydra.yaml, checkpoints/, epoch_logs.jsonl), 로그: $DINO_RUNS/<job>.log
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"
NAME=$1; GPU=$2; shift 2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=$DINO_RUNS/$NAME; MODEL=$DINO_TRAIN/outputs/$NAME
if [ -f "$OUT/DONE" ]; then echo "$NAME already done"; exit 0; fi
mkdir -p "$OUT" "$MODEL"
cd "$DINO_WM"
echo "START $(date -Is) pid=$$ gpu=$GPU args=$*" > "$OUT/run_info.txt"
CUDA_VISIBLE_DEVICES=$GPU python train.py --config-name train.yaml "$@" ckpt_base_path=$DINO_TRAIN hydra.run.dir=$MODEL > "$DINO_RUNS/$NAME.log" 2>&1
RC=$?
echo "END $(date -Is) rc=$RC" >> "$OUT/run_info.txt"
[ $RC -eq 0 ] && touch "$OUT/DONE"
exit $RC
