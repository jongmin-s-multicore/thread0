#!/usr/bin/env bash
# Train model A (clean data) and model B (distract data).
#
# Both models differ ONLY in which render of the paired dataset they read; every
# other setting comes from conf/train_reacher.yaml.
#
#   scripts/train.sh --gpus 0            # sequential: A then B on GPU 0
#   scripts/train.sh --gpus 0,1          # parallel:   A on GPU 0, B on GPU 1
#   scripts/train.sh --gpus 0 --epochs 5 --only A
#
# Checkpoints land in $EXP_ROOT/outputs/<exp_name>/checkpoints/. hydra.run.dir is
# pinned to that fixed path (not a timestamp) so re-running this script RESUMES:
# train.py reloads model_latest.pth and continues from its epoch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GPUS="0"
EPOCHS=""
BATCH_SIZE=""
ONLY="both"
EXP_ROOT="${EXP_ROOT:-$PWD}"
EXTRA=()
NAME_A="reacher_clean"
NAME_B="reacher_distract"

usage() { sed -n '2,13p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;          # A | B | both
    --exp_root) EXP_ROOT="$2"; shift 2;;
    --name_a) NAME_A="$2"; shift 2;;
    --name_b) NAME_B="$2"; shift 2;;
    -h|--help) usage 0;;
    *) EXTRA+=("$1"); shift;;             # passed through to hydra
  esac
done

: "${DATASET_DIR:?set DATASET_DIR to the directory holding distracting_reacher/}"
mkdir -p "$EXP_ROOT/outputs"

# wandb: offline by default so a missing API key can never block a long run.
# Sync later with: wandb sync $EXP_ROOT/outputs/*/wandb/offline-run-*
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$EXP_ROOT/outputs}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM=false

overrides=( "ckpt_base_path=$EXP_ROOT" )
[[ -n "$EPOCHS" ]] && overrides+=( "training.epochs=$EPOCHS" )
[[ -n "$BATCH_SIZE" ]] && overrides+=( "training.batch_size=$BATCH_SIZE" )
# guarded: an unquoted empty array would pass hydra an empty override and fail
if (( ${#EXTRA[@]} > 0 )); then overrides+=( "${EXTRA[@]}" ); fi

run_one() {
  local label="$1" env_cfg="$2" exp_name="$3" gpu="$4"
  local log_dir="$EXP_ROOT/outputs/$exp_name"
  mkdir -p "$log_dir"
  echo "[train] model $label  env=$env_cfg  exp_name=$exp_name  GPU=$gpu"
  echo "[train] log: $log_dir/train_stdout.log"
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
    python train.py --config-name train_reacher \
      "env=$env_cfg" "exp_name=$exp_name" "${overrides[@]}" \
      >> "$log_dir/train_stdout.log" 2>&1
}

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
N_GPUS=${#GPU_ARR[@]}

declare -a JOBS=()
[[ "$ONLY" == "both" || "$ONLY" == "A" ]] && JOBS+=("A|distracting_reacher_clean|$NAME_A")
[[ "$ONLY" == "both" || "$ONLY" == "B" ]] && JOBS+=("B|distracting_reacher_distract|$NAME_B")

if (( N_GPUS >= 2 && ${#JOBS[@]} > 1 )); then
  echo "[train] 2+ GPUs: training the two models in parallel"
  pids=()
  for i in "${!JOBS[@]}"; do
    IFS='|' read -r label env_cfg exp_name <<< "${JOBS[$i]}"
    run_one "$label" "$env_cfg" "$exp_name" "${GPU_ARR[$((i % N_GPUS))]}" &
    pids+=($!)
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
  if (( status != 0 )); then
    echo "[train] FAILED (see train_stdout.log in the output dirs)" >&2
    exit "$status"
  fi
else
  for job in "${JOBS[@]}"; do
    IFS='|' read -r label env_cfg exp_name <<< "$job"
    run_one "$label" "$env_cfg" "$exp_name" "${GPU_ARR[0]}"
  done
fi

echo "[train] done. checkpoints:"
for job in "${JOBS[@]}"; do
  IFS='|' read -r _ _ exp_name <<< "$job"
  ls -la "$EXP_ROOT/outputs/$exp_name/checkpoints/" 2>/dev/null | tail -3 || true
done
