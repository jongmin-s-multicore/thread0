#!/bin/bash
# 작업 큐를 백그라운드로 띄운다. 이미 떠 있으면 먼저 끈다 (실행 중인 작업은 건드리지 않는다 — 상태는 파일에 있다).
#   bash repro/queue/start.sh [jobs 파일]   (기본 repro/jobs/benchmark.txt)
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"
JOBS=${1:-$HERE/../jobs/benchmark.txt}
mkdir -p "$DINO_RUNS"
for p in $(pgrep -f "python3 $HERE/scheduler.py"); do kill "$p"; done
nohup python3 "$HERE/scheduler.py" "$JOBS" >> "$DINO_RUNS/scheduler.log" 2>&1 &
echo "scheduler pid $! — 로그 $DINO_RUNS/scheduler.log"
