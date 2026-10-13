#!/usr/bin/env bash
# Pull checkpoints/results back from Google Drive onto a fresh instance.
#
#   bash scripts/restore_gdrive.sh --remote gdrive:dinowm-reacher
#
# After this, `bash scripts/run_all.sh` resumes rather than restarting:
#   * train.py reloads outputs/<exp_name>/checkpoints/model_latest.pth
#   * scripts/eval.py skips episodes already present in results/*/*.jsonl
#   * run_all.sh skips stages whose .state/<stage>.done marker came back
#
# The raw frames are deliberately not on Drive; they are regenerated from the
# seed by the collect stage (delete .state/collect.done to force it).
set -euo pipefail

REMOTE="${RCLONE_REMOTE:-}"
EXP_ROOT="${EXP_ROOT:-/workspace/exp}"
DATASET_DIR="${DATASET_DIR:-/workspace/data}"
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) REMOTE="$2"; shift 2;;
    --exp_root) EXP_ROOT="$2"; shift 2;;
    --dataset_dir) DATASET_DIR="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    -h|--help) sed -n '2,6p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[[ -n "$REMOTE" ]] || { echo "restore_gdrive: --remote is required" >&2; exit 2; }
command -v rclone >/dev/null 2>&1 || { echo "restore_gdrive: rclone not installed" >&2; exit 2; }

args=( copy --progress --transfers 4 --retries 5 --low-level-retries 20 )
(( DRY )) && args+=( --dry-run )

mkdir -p "$EXP_ROOT" "$DATASET_DIR/distracting_reacher"

echo "[restore] $REMOTE/exp -> $EXP_ROOT"
rclone "${args[@]}" "$REMOTE/exp" "$EXP_ROOT"

echo "[restore] $REMOTE/data/distracting_reacher -> $DATASET_DIR/distracting_reacher (meta only)"
rclone "${args[@]}" "$REMOTE/data/distracting_reacher" \
  "$DATASET_DIR/distracting_reacher" || true

echo
echo "[restore] what came back:"
for d in "$EXP_ROOT"/outputs/*/; do
  [[ -d "$d" ]] || continue
  latest=$(ls -1 "$d/checkpoints" 2>/dev/null | grep -c '^model_' || true)
  echo "  $(basename "$d"): $latest checkpoint file(s)"
done
echo "  results records: $(find "$EXP_ROOT/results" -name '*.jsonl' 2>/dev/null | wc -l) jsonl file(s)"
echo "  stage markers  : $(ls -1 "$EXP_ROOT/.state" 2>/dev/null | tr '\n' ' ')"
cat <<'EOF'

NOTE: the raw dataset frames are not restored (not uploaded by design). If
.state/collect.done came back but $DATASET_DIR has no frames, delete that marker
so run_all.sh regenerates the data from the seed:
    rm -f $EXP_ROOT/.state/collect.done
EOF
