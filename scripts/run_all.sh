#!/usr/bin/env bash
# Full experiment: data -> (features) -> train -> eval -> analyze.
#
#   bash scripts/run_all.sh --smoke              # tiny end-to-end check
#   bash scripts/run_all.sh --num_gpus 2         # the real run, 2 GPUs
#   bash scripts/run_all.sh --num_gpus 1 --gdrive gdrive:dinowm-reacher
#
# Each finished stage drops a marker in $EXP_ROOT/.state/<stage>.done, so
# re-running skips what already completed. Use --redo <stage> to force one.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---- defaults (real run) ---------------------------------------------------
SMOKE=0
NUM_GPUS=1
NUM_EPISODES=500
EPISODE_LEN=100
EPOCHS=100
BATCH_SIZE=32
N_EVAL_EPISODES=50
EVAL_BATCH=10
CEM_SAMPLES=300
CEM_ITERS=30
MPC_ITERS=3
CONDITIONS="T1_clean,T2_train_bg,T3_val_bg,T4_mismatch"
PHASES="plan,rollout,latent"
SEED=0
EVAL_SEED=99
BG_SOURCE="davis"
CACHE_FEATURES=0
GDRIVE_REMOTE="${RCLONE_REMOTE:-}"
REDO=""
COLLECT_WORKERS=""

DATASET_DIR="${DATASET_DIR:-/workspace/data}"
EXP_ROOT="${EXP_ROOT:-/workspace/exp}"

usage() { sed -n '2,10p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift;;
    --num_gpus) NUM_GPUS="$2"; shift 2;;
    --num_episodes) NUM_EPISODES="$2"; shift 2;;
    --episode_len) EPISODE_LEN="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --n_eval_episodes) N_EVAL_EPISODES="$2"; shift 2;;
    --conditions) CONDITIONS="$2"; shift 2;;
    --phases) PHASES="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --bg_source) BG_SOURCE="$2"; shift 2;;
    --cache_features) CACHE_FEATURES=1; shift;;
    --gdrive) GDRIVE_REMOTE="$2"; shift 2;;
    --dataset_dir) DATASET_DIR="$2"; shift 2;;
    --exp_root) EXP_ROOT="$2"; shift 2;;
    --collect_workers) COLLECT_WORKERS="$2"; shift 2;;
    --redo) REDO="$2"; shift 2;;
    -h|--help) usage 0;;
    *) echo "unknown arg: $1" >&2; usage 2;;
  esac
done

if (( SMOKE )); then
  NUM_EPISODES=8; EPISODE_LEN=30; EPOCHS=2; BATCH_SIZE=8
  N_EVAL_EPISODES=2; EVAL_BATCH=2
  CEM_SAMPLES=24; CEM_ITERS=2; MPC_ITERS=1
  echo "[smoke] tiny settings: ${NUM_EPISODES}ep x ${EPISODE_LEN} frames, ${EPOCHS} epochs, ${N_EVAL_EPISODES} eval episodes/condition"
fi

export DATASET_DIR EXP_ROOT
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export DAVIS_PATH="${DAVIS_PATH:-/workspace/DAVIS}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

DATA_ROOT="$DATASET_DIR/distracting_reacher"
RESULTS_DIR="$EXP_ROOT/results"
STATE_DIR="$EXP_ROOT/.state"
mkdir -p "$STATE_DIR" "$RESULTS_DIR" "$EXP_ROOT/outputs"

GPU_LIST="0"
(( NUM_GPUS >= 2 )) && GPU_LIST="0,1"

if [[ -z "$COLLECT_WORKERS" ]]; then
  COLLECT_WORKERS=$(python -c 'import os;print(max(1,min(16,(os.cpu_count() or 4))))')
  (( SMOKE )) && COLLECT_WORKERS=1
fi

log()  { printf '\n\033[1m[run_all %s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
marker() { echo "$STATE_DIR/$1.done"; }
is_done() { [[ -f "$(marker "$1")" && "$REDO" != "$1" && "$REDO" != "all" ]]; }
mark_done() {
  { echo "finished=$(date -Is)"; echo "duration_s=${2:-0}"; } > "$(marker "$1")"
}
timed() {  # timed <stage> <command...>
  local stage="$1"; shift
  local t0=$SECONDS
  "$@"
  mark_done "$stage" "$((SECONDS - t0))"
  log "stage '$stage' took $((SECONDS - t0))s"
}

# ---- background sync daemon ------------------------------------------------
SYNC_PID=""
start_sync() {
  [[ -z "$GDRIVE_REMOTE" ]] && return 0
  if ! command -v rclone >/dev/null 2>&1; then
    echo "[run_all] rclone not installed; skipping Google Drive sync"
    return 0
  fi
  log "starting Google Drive sync daemon -> $GDRIVE_REMOTE"
  bash scripts/sync_gdrive.sh --watch --remote "$GDRIVE_REMOTE" \
      --exp_root "$EXP_ROOT" --dataset_dir "$DATASET_DIR" &
  SYNC_PID=$!
}
stop_sync() {
  [[ -z "$SYNC_PID" ]] && return 0
  kill "$SYNC_PID" 2>/dev/null || true
  wait "$SYNC_PID" 2>/dev/null || true
  SYNC_PID=""
  log "final Google Drive sync"
  bash scripts/sync_gdrive.sh --once --remote "$GDRIVE_REMOTE" \
      --exp_root "$EXP_ROOT" --dataset_dir "$DATASET_DIR" --no-min-age || true
}
trap stop_sync EXIT
start_sync

# --------------------------------------------------------------------------- #
# 1. data
# --------------------------------------------------------------------------- #
if is_done collect; then
  log "stage 'collect' already done -> skipping ($DATA_ROOT)"
else
  log "collecting paired clean/distract data ($NUM_EPISODES episodes x $EPISODE_LEN frames)"
  timed collect python scripts/collect_data.py \
    --out_dir "$DATA_ROOT" \
    --num_episodes "$NUM_EPISODES" \
    --episode_len "$EPISODE_LEN" \
    --seed "$SEED" \
    --num_workers "$COLLECT_WORKERS" \
    --gpu_ids "$GPU_LIST" \
    --mujoco_gl "$MUJOCO_GL" \
    --bg_source "$BG_SOURCE" \
    --obs_format mp4
  echo "[run_all] check the sample grid: $DATA_ROOT/samples/grid_clean_vs_distract.png"
fi

# --------------------------------------------------------------------------- #
# 2. optional feature cache
# --------------------------------------------------------------------------- #
if (( CACHE_FEATURES )); then
  if is_done cache; then
    log "stage 'cache' already done -> skipping"
  else
    log "caching DINOv2 features (has_decoder must be false to use these)"
    timed cache bash -c '
      set -e
      for arm in clean distract; do
        python scripts/cache_features.py --data_path "'"$DATA_ROOT"'/$arm" --gpu 0
      done'
  fi
fi

# --------------------------------------------------------------------------- #
# 3. train A and B
# --------------------------------------------------------------------------- #
if is_done train; then
  log "stage 'train' already done -> skipping"
else
  log "training models A (clean) and B (distract) on GPUs [$GPU_LIST]"
  timed train bash scripts/train.sh \
    --gpus "$GPU_LIST" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
    --exp_root "$EXP_ROOT"
fi

# --------------------------------------------------------------------------- #
# 4. eval
# --------------------------------------------------------------------------- #
eval_common=(
  --ckpt_base_path "$EXP_ROOT"
  --results_dir "$RESULTS_DIR"
  --n_episodes "$N_EVAL_EPISODES"
  --batch "$EVAL_BATCH"
  --seed "$EVAL_SEED"
  --cem_samples "$CEM_SAMPLES"
  --cem_iters "$CEM_ITERS"
  --mpc_iters "$MPC_ITERS"
  --phases "$PHASES"
  --save_videos
)

if is_done eval; then
  log "stage 'eval' already done -> skipping"
else
  log "evaluating models x conditions [$CONDITIONS]"
  if (( NUM_GPUS >= 2 )); then
    # split the conditions across the two GPUs
    IFS=',' read -r -a CONDS <<< "$CONDITIONS"
    half=$(( (${#CONDS[@]} + 1) / 2 ))
    first=$(IFS=,; echo "${CONDS[*]:0:half}")
    second=$(IFS=,; echo "${CONDS[*]:half}")
    log "GPU0: $first   GPU1: ${second:-<none>}"
    pids=()
    python scripts/eval.py "${eval_common[@]}" --gpu 0 --conditions "$first" &
    pids+=($!)
    if [[ -n "$second" ]]; then
      python scripts/eval.py "${eval_common[@]}" --gpu 1 --conditions "$second" &
      pids+=($!)
    fi
    status=0
    for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
    (( status == 0 )) || { echo "[run_all] eval failed" >&2; exit "$status"; }
    mark_done eval
  else
    timed eval python scripts/eval.py "${eval_common[@]}" --gpu 0 \
      --conditions "$CONDITIONS"
  fi
fi

# --------------------------------------------------------------------------- #
# 5. analyze (always re-run: it is cheap and reflects whatever exists)
# --------------------------------------------------------------------------- #
log "analyzing"
python scripts/analyze.py --results_dir "$RESULTS_DIR"
mark_done analyze

log "ALL DONE"
cat <<EOF
  data     : $DATA_ROOT  (sample grid in samples/)
  models   : $EXP_ROOT/outputs/{reacher_clean,reacher_distract}
  results  : $RESULTS_DIR
  tables   : $RESULTS_DIR/analysis/summary.md
  stages   : $STATE_DIR  (delete a .done file, or use --redo <stage>, to re-run)
EOF
