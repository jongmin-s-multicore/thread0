#!/usr/bin/env bash
# One-shot setup for a fresh Vast.ai A100 instance.
#
#   bash scripts/setup_instance.sh                  # everything
#   bash scripts/setup_instance.sh --skip-davis     # packages + render test only
#   bash scripts/setup_instance.sh --davis-dir /data/DAVIS
#
# Steps: system packages (EGL) -> python packages -> DAVIS 2017 -> EGL render test
# -> optional rclone/Google Drive check. Every step prints what to do if it fails.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DAVIS_DIR="${DAVIS_PATH:-/workspace/DAVIS}"
SKIP_DAVIS=0
SKIP_APT=0
SKIP_PIP=0
RCLONE_REMOTE="${RCLONE_REMOTE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --davis-dir) DAVIS_DIR="$2"; shift 2;;
    --skip-davis) SKIP_DAVIS=1; shift;;
    --skip-apt) SKIP_APT=1; shift;;
    --skip-pip) SKIP_PIP=1; shift;;
    --rclone-remote) RCLONE_REMOTE="$2"; shift 2;;
    -h|--help) sed -n '2,9p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

ok()   { printf '\033[32m[ ok ]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[FAIL]\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

SUDO=""
if [[ "$(id -u)" != "0" ]]; then SUDO="sudo"; fi

# --------------------------------------------------------------------------- #
step "1/5 system packages (EGL / GL / ffmpeg)"
# --------------------------------------------------------------------------- #
if (( SKIP_APT )); then
  warn "skipped (--skip-apt)"
else
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -qq || die "apt-get update failed. Check network/mirrors."
  # libegl1 + libgles2 + nvidia's EGL vendor config are what MUJOCO_GL=egl needs;
  # libosmesa6-dev is the CPU fallback; libglew is needed by some mujoco builds.
  $SUDO apt-get install -y -qq \
    libegl1 libegl1-mesa-dev libgles2-mesa-dev libgl1 libglew-dev libglfw3 \
    libosmesa6-dev patchelf ffmpeg unzip wget curl git rsync \
    || die "apt-get install failed. On a minimal image try: apt-get install -y software-properties-common"
  ok "system packages installed"
  if [[ -d /usr/share/glvnd/egl_vendor.d ]]; then
    ok "EGL vendor configs: $(ls /usr/share/glvnd/egl_vendor.d | tr '\n' ' ')"
  else
    warn "no /usr/share/glvnd/egl_vendor.d; EGL may fail. On NVIDIA images this is
     normally provided by libnvidia-gl-*. If the render test below fails, run it
     with MUJOCO_GL=osmesa to confirm the rest of the pipeline works."
  fi
fi

# --------------------------------------------------------------------------- #
step "2/5 python packages"
# --------------------------------------------------------------------------- #
if (( SKIP_PIP )); then
  warn "skipped (--skip-pip)"
else
  python -c 'import sys; print("python", sys.version)' || die "no python on PATH"
  # dm_control 1.0.27 pairs with mujoco 3.x; this repo's environment.yaml pins
  # mujoco==3.2.7. Do NOT install google-research/distracting_control: it writes
  # to mjModel.tex_rgb, removed in mujoco 3.x. env/distracting_reacher.py
  # reimplements the background mechanism against the modern API.
  python - <<'PY' || exit 1
import importlib, subprocess, sys

REQUIRED = {
    "mujoco": "mujoco==3.2.7",
    "dm_control": "dm-control==1.0.27",
    "imageio": "imageio==2.34.1",
    "imageio_ffmpeg": "imageio-ffmpeg",
    "decord": "decord==0.6.0",
    "einops": "einops",
    "hydra": "hydra-core==1.2.0",
    "omegaconf": "omegaconf",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "PIL": "pillow",
    "psutil": "psutil",
    "tqdm": "tqdm",
}
missing = []
for mod, pkg in REQUIRED.items():
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(pkg)
if missing:
    print("installing:", " ".join(missing))
    rc = subprocess.call([sys.executable, "-m", "pip", "install", "-q", *missing])
    if rc:
        sys.exit(rc)
else:
    print("all python packages already present")

import mujoco, dm_control
print("mujoco", mujoco.__version__)
# the attribute our background painter needs
m = mujoco.MjModel.from_xml_string("<mujoco/>")
assert hasattr(m, "tex_data") or hasattr(m, "tex_rgb"), "no tex_data/tex_rgb on mjModel"
print("texture buffer attribute:", "tex_data" if hasattr(m, "tex_data") else "tex_rgb")
PY
  ok "python packages verified"
  python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count(), "gpu(s)")' \
    || warn "torch import failed; training will not run until it is installed"
fi

# --------------------------------------------------------------------------- #
step "3/5 DAVIS 2017 backgrounds"
# --------------------------------------------------------------------------- #
DAVIS_ZIP_URL="https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"
if (( SKIP_DAVIS )); then
  warn "skipped (--skip-davis). Use --bg_source synthetic for smoke tests."
elif [[ -f "$DAVIS_DIR/ImageSets/2017/train.txt" && -f "$DAVIS_DIR/ImageSets/2017/val.txt" ]]; then
  ok "DAVIS already present at $DAVIS_DIR"
else
  mkdir -p "$(dirname "$DAVIS_DIR")"
  tmp_zip="$(dirname "$DAVIS_DIR")/DAVIS-2017-trainval-480p.zip"
  echo "downloading DAVIS 2017 trainval 480p (~4GB) -> $tmp_zip"
  wget -c -q --show-progress -O "$tmp_zip" "$DAVIS_ZIP_URL" \
    || die "DAVIS download failed. Mirror: https://davischallenge.org/davis2017/code.html
     Then unzip so that $DAVIS_DIR/ImageSets/2017/train.txt exists."
  # extract into its own directory: this is downloaded third-party data
  unzip -q -o "$tmp_zip" -d "$(dirname "$DAVIS_DIR")" || die "unzip failed"
  rm -f "$tmp_zip"
  [[ -f "$DAVIS_DIR/ImageSets/2017/train.txt" ]] \
    || die "unexpected DAVIS layout; expected $DAVIS_DIR/ImageSets/2017/train.txt"
  ok "DAVIS extracted to $DAVIS_DIR"
fi

if [[ -f "$DAVIS_DIR/ImageSets/2017/train.txt" ]]; then
  n_train=$(wc -l < "$DAVIS_DIR/ImageSets/2017/train.txt")
  n_val=$(wc -l < "$DAVIS_DIR/ImageSets/2017/val.txt")
  ok "DAVIS splits: train=$n_train videos, val=$n_val videos (val is NEVER used in training)"
  echo "    export DAVIS_PATH=$DAVIS_DIR"
fi

# --------------------------------------------------------------------------- #
step "4/5 render test"
# --------------------------------------------------------------------------- #
OUT_DIR="${EXP_ROOT:-$PWD}/setup_check"
mkdir -p "$OUT_DIR"
export DAVIS_PATH="$DAVIS_DIR"
for gl in egl osmesa; do
  echo "--- MUJOCO_GL=$gl"
  if MUJOCO_GL="$gl" DAVIS_PATH="$DAVIS_DIR" python - "$OUT_DIR" "$gl" <<'PY'
import sys, numpy as np
out_dir, gl = sys.argv[1], sys.argv[2]
sys.path.insert(0, ".")
from PIL import Image
from env.distracting_reacher import DistractingReacherWrapper, build_background_sources

bg_source = "davis"
try:
    build_background_sources("davis", "train")
except Exception as exc:
    print("  no DAVIS available (%s); using synthetic backgrounds" % type(exc).__name__)
    bg_source = "synthetic"

frames = {}
for arm, enabled in (("clean", False), ("distract", True)):
    env = DistractingReacherWrapper(
        bg_enabled=enabled, bg_source=bg_source if enabled else "none", img_size=224
    )
    init, goal = env.sample_random_init_goal_states(0)
    obs, _ = env.prepare(0, init)
    frames[arm] = obs["visual"]
    env.close()

gap = abs(float(frames["clean"].mean()) - float(frames["distract"].mean()))
grid = np.concatenate([frames["clean"], frames["distract"]], axis=1)
path = "%s/render_test_%s.png" % (out_dir, gl)
Image.fromarray(grid).save(path)
print("  wrote", path, "(left: clean, right: distract)")
print("  mean-pixel gap between the two renders: %.2f" % gap)
if gap < 1.0:
    print("  WARNING: the two renders look almost identical -> background "
          "replacement is probably not taking effect")
    sys.exit(3)
PY
  then
    ok "render works with MUJOCO_GL=$gl"
    [[ "$gl" == "egl" ]] && break
  else
    rc=$?
    if [[ "$gl" == "egl" ]]; then
      warn "EGL render failed (exit $rc). Common causes:
     - missing libegl1 / NVIDIA EGL vendor json (step 1)
     - container without GPU access: check nvidia-smi
     - wrong device: try MUJOCO_EGL_DEVICE_ID=0/1
     Trying osmesa next; it is CPU-only and much slower, fine for smoke tests."
    else
      die "both EGL and OSMesa rendering failed. Fix this before collecting data.
     Debug with: MUJOCO_GL=osmesa python -c 'from dm_control import suite; \
     e=suite.load(\"reacher\",\"easy\"); print(e.physics.render(64,64).shape)'"
    fi
  fi
done

# --------------------------------------------------------------------------- #
step "5/5 Google Drive sync (rclone)"
# --------------------------------------------------------------------------- #
if ! command -v rclone >/dev/null 2>&1; then
  echo "installing rclone"
  curl -fsSL https://rclone.org/install.sh | $SUDO bash >/dev/null 2>&1 \
    || warn "rclone install failed; checkpoints will stay on the instance only"
fi
if command -v rclone >/dev/null 2>&1; then
  ok "rclone $(rclone version | head -1 | awk '{print $2}')"
  if [[ -n "$RCLONE_REMOTE" ]] && rclone lsd "$RCLONE_REMOTE" >/dev/null 2>&1; then
    ok "remote '$RCLONE_REMOTE' is reachable"
  else
    cat <<'EOF'
[warn] no working rclone remote yet. Headless setup (personal Google Drive):

  1. On your own machine (which has a browser):
         rclone authorize "drive" --drive-scope=drive.file
     The drive.file scope limits access to files rclone itself creates, so the
     token cannot read the rest of your Drive.
  2. Copy the JSON blob it prints, then on THIS instance:
         mkdir -p ~/.config/rclone
         cat > ~/.config/rclone/rclone.conf <<'CONF'
         [gdrive]
         type = drive
         scope = drive.file
         token = {"access_token":"...","refresh_token":"...","expiry":"..."}
         CONF
         chmod 600 ~/.config/rclone/rclone.conf
  3. Verify:
         rclone lsd gdrive:
  4. Re-run with --rclone-remote gdrive:dinowm-reacher to confirm.
EOF
  fi
fi

step "summary"
cat <<EOF
Next steps:
  export DATASET_DIR=/workspace/data
  export EXP_ROOT=/workspace/exp
  export DAVIS_PATH=$DAVIS_DIR
  bash scripts/run_all.sh --smoke          # ~20 min end-to-end sanity check
  bash scripts/run_all.sh --num_gpus 2     # the real run

Inspect $OUT_DIR/render_test_*.png before collecting data: the right half must
clearly show a video background behind the arm.
EOF
