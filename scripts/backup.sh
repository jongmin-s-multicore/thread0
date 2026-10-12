#!/usr/bin/env bash
# Mirror checkpoints / eval results / analysis to any rsync target.
#
#   scripts/backup.sh /mnt/backup/dinowm                 # local or mounted path
#   scripts/backup.sh user@host:/data/dinowm             # over ssh
#   scripts/backup.sh /mnt/backup/dinowm --with-data     # include the raw frames
#   scripts/backup.sh /mnt/backup/dinowm --dry-run
#
# For Google Drive use scripts/sync_gdrive.sh instead (rsync cannot talk to Drive).
set -euo pipefail

EXP_ROOT="${EXP_ROOT:-/workspace/exp}"
DATASET_DIR="${DATASET_DIR:-/workspace/data}"
WITH_DATA=0
DRY=0
DEST=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-data) WITH_DATA=1; shift;;
    --dry-run) DRY=1; shift;;
    --exp_root) EXP_ROOT="$2"; shift 2;;
    --dataset_dir) DATASET_DIR="$2"; shift 2;;
    -h|--help) sed -n '2,9p' "$0"; exit 0;;
    *) DEST="$1"; shift;;
  esac
done

[[ -n "$DEST" ]] || { echo "usage: scripts/backup.sh <rsync-destination> [--with-data] [--dry-run]" >&2; exit 2; }
command -v rsync >/dev/null 2>&1 || { echo "backup: rsync not installed" >&2; exit 2; }

args=( -a --info=progress2 --human-readable --partial )
(( DRY )) && args+=( --dry-run )

echo "[backup] $EXP_ROOT -> $DEST/exp"
rsync "${args[@]}" \
  --exclude 'wandb/run-*/files/media' \
  --exclude '*.tmp' \
  "$EXP_ROOT/" "$DEST/exp/"

if (( WITH_DATA )); then
  echo "[backup] $DATASET_DIR/distracting_reacher -> $DEST/data (frames included)"
  rsync "${args[@]}" "$DATASET_DIR/distracting_reacher/" "$DEST/data/distracting_reacher/"
else
  echo "[backup] dataset: metadata only (pass --with-data for the frames)"
  rsync "${args[@]}" \
    --include '*/' --include 'meta.json' --include 'samples/***' --exclude '*' \
    "$DATASET_DIR/distracting_reacher/" "$DEST/data/distracting_reacher/"
fi

echo "[backup] done -> $DEST"
