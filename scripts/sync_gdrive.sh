#!/usr/bin/env bash
# Push metrics and checkpoints to Google Drive with rclone.
#
#   scripts/sync_gdrive.sh --once  --remote gdrive:dinowm-reacher
#   scripts/sync_gdrive.sh --watch --remote gdrive:dinowm-reacher &
#
# Design notes
#   * One-way `rclone copy`, never `sync`: nothing on Drive is ever deleted, so a
#     bug or a dead instance cannot destroy already-uploaded results.
#   * --min-age 5m: skips files written in the last 5 minutes, so a checkpoint
#     still being written by torch.save is never uploaded half-finished. This is
#     why train.py needs no modification. Pass --no-min-age for the final sync.
#   * Metrics (KBs) and checkpoints (hundreds of MBs) are on separate intervals;
#     uploading checkpoints as often as metrics would saturate the uplink.
#   * Raw frames are NOT uploaded: they are reproducible from the seed. Only
#     meta.json and the sample grid go up.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REMOTE="${RCLONE_REMOTE:-}"
EXP_ROOT="${EXP_ROOT:-/workspace/exp}"
DATASET_DIR="${DATASET_DIR:-/workspace/data}"
MODE="once"
METRICS_INTERVAL=180
CKPT_INTERVAL=1800
KEEP_EVERY=10
BWLIMIT="40M"
MIN_AGE="5m"
NO_IMAGES=0
PRUNE=0

usage() { sed -n '2,6p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) REMOTE="$2"; shift 2;;
    --once) MODE="once"; shift;;
    --watch) MODE="watch"; shift;;
    --exp_root) EXP_ROOT="$2"; shift 2;;
    --dataset_dir) DATASET_DIR="$2"; shift 2;;
    --metrics-interval) METRICS_INTERVAL="$2"; shift 2;;
    --ckpt-interval) CKPT_INTERVAL="$2"; shift 2;;
    --keep-every) KEEP_EVERY="$2"; shift 2;;
    --bwlimit) BWLIMIT="$2"; shift 2;;
    --min-age) MIN_AGE="$2"; shift 2;;
    --no-min-age) MIN_AGE=""; shift;;
    --no-images) NO_IMAGES=1; shift;;
    --prune) PRUNE=1; shift;;
    -h|--help) usage 0;;
    *) echo "unknown arg: $1" >&2; usage 2;;
  esac
done

[[ -n "$REMOTE" ]] || { echo "sync_gdrive: --remote is required (e.g. gdrive:dinowm-reacher)" >&2; exit 2; }
command -v rclone >/dev/null 2>&1 || { echo "sync_gdrive: rclone not installed" >&2; exit 2; }

LOG_DIR="$EXP_ROOT/results"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sync.log"
HEARTBEAT="$LOG_DIR/sync_heartbeat.json"
LOCK="$EXP_ROOT/.state/sync.lock"
mkdir -p "$(dirname "$LOCK")"

RC_COMMON=(
  copy
  --checksum --no-traverse --transfers 4 --drive-chunk-size 64M
  --retries 5 --low-level-retries 20
  --bwlimit "$BWLIMIT"
  --log-file "$LOG_FILE" --log-level INFO --stats-one-line
)
[[ -n "$MIN_AGE" ]] && RC_COMMON+=( --min-age "$MIN_AGE" )

say() { printf '[sync %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

heartbeat() {  # heartbeat <status> <what> <seconds>
  cat > "$HEARTBEAT" <<EOF
{
  "last_run": "$(date -Is)",
  "status": "$1",
  "scope": "$2",
  "duration_s": $3,
  "remote": "$REMOTE",
  "keep_every": $KEEP_EVERY,
  "min_age": "$MIN_AGE"
}
EOF
}

# Only the latest checkpoint plus every KEEP_EVERY-th epoch: on 50GB of Drive,
# uploading all 100 epochs x 2 models would not fit.
ckpt_filters() {
  local -n out=$1
  out=( --include "outputs/*/checkpoints/model_latest.pth" )
  for ((e = KEEP_EVERY; e <= 2000; e += KEEP_EVERY)); do
    out+=( --include "outputs/*/checkpoints/model_${e}.pth" )
  done
}

sync_metrics() {
  local t0=$SECONDS
  local filters=(
    --include "results/**/*.jsonl"
    --include "results/**/*.csv"
    --include "results/**/*.md"
    --include "results/**/*.json"
    --include "results/**/*.png"
    --include "outputs/*/hydra.yaml"
    --include "outputs/*/*.log"
    --include "outputs/*/train_stdout.log"
    --include ".state/*.done"
  )
  rclone "${RC_COMMON[@]}" "${filters[@]}" --exclude "**" \
    "$EXP_ROOT" "$REMOTE/exp" || return 1
  # dataset provenance only (never the frames)
  if [[ -d "$DATASET_DIR/distracting_reacher" ]]; then
    rclone "${RC_COMMON[@]}" \
      --include "**/meta.json" --include "samples/*.png" --exclude "**" \
      "$DATASET_DIR/distracting_reacher" "$REMOTE/data/distracting_reacher" || return 1
  fi
  heartbeat ok metrics $((SECONDS - t0))
  say "metrics synced in $((SECONDS - t0))s"
}

sync_checkpoints() {
  local t0=$SECONDS
  local filters=()
  ckpt_filters filters
  if (( ! NO_IMAGES )); then
    filters+=(
      --include "outputs/*/train/*.png"
      --include "outputs/*/valid/*.png"
      --include "outputs/*/rollout_plots/*.png"
    )
  fi
  filters+=( --include "results/videos/**/*failure*.mp4" )
  rclone "${RC_COMMON[@]}" "${filters[@]}" --exclude "**" \
    "$EXP_ROOT" "$REMOTE/exp" || return 1
  heartbeat ok checkpoints $((SECONDS - t0))
  say "checkpoints synced in $((SECONDS - t0))s"
}

run_once() {
  # never let two syncs overlap (a slow checkpoint upload + a metrics tick)
  exec 9>"$LOCK"
  if ! flock -n 9; then
    say "another sync is running; skipping this tick"
    return 0
  fi
  sync_metrics || { heartbeat error metrics 0; say "metrics sync FAILED (see $LOG_FILE)"; }
  sync_checkpoints || { heartbeat error checkpoints 0; say "checkpoint sync FAILED (see $LOG_FILE)"; }
  flock -u 9
}

if [[ "$MODE" == "once" ]]; then
  say "one-shot sync to $REMOTE"
  run_once
  say "done. log: $LOG_FILE"
  exit 0
fi

say "watching $EXP_ROOT -> $REMOTE (metrics ${METRICS_INTERVAL}s, checkpoints ${CKPT_INTERVAL}s)"
trap 'say "stopping"; exit 0' INT TERM

last_ckpt=0
while true; do
  exec 9>"$LOCK"
  if flock -n 9; then
    sync_metrics || { heartbeat error metrics 0; say "metrics sync FAILED"; }
    now=$SECONDS
    if (( now - last_ckpt >= CKPT_INTERVAL )); then
      sync_checkpoints || { heartbeat error checkpoints 0; say "checkpoint sync FAILED"; }
      last_ckpt=$now
    fi
    flock -u 9
  fi
  sleep "$METRICS_INTERVAL"
done
