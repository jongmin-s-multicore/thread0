# Distracting Reacher: testing DINO-WM's latent-L2 planning cost

Does DINO-WM still plan when the background is a moving video? Two world models
are trained on the *same* trajectories rendered two ways (clean vs. video
background) and planned under four background conditions. See
[CLAUDE.md](CLAUDE.md) for the design rationale; this file is the operating manual.

---

## 1. Instance requirements (Vast.ai)

| | recommended | why |
|---|---|---|
| GPU | 1-2x A100 (40/80GB) | 2 GPUs train both models in parallel and halve eval wall time |
| Disk | **>= 200GB** | mp4 dataset ~4GB, DAVIS ~4GB, checkpoints ~0.3GB x epochs x 2 models, videos |
| CPU | **>= 16 cores** | rendering is CPU-bound per worker; `--num_workers 16` |
| RAM | **>= 64GB** | 16 render workers each hold a MuJoCo context plus DAVIS frames |
| Host | **on-demand**, not interruptible | a preemption mid-training is recoverable but costs a restart |
| CUDA | 12.x image with NVIDIA EGL libs | `MUJOCO_GL=egl` needs `/usr/share/glvnd/egl_vendor.d` |

`nvidia-smi` must work inside the container, otherwise EGL rendering falls back
to CPU OSMesa and data collection becomes the bottleneck.

---

## 2. Run order

```bash
# 0. environment
export DATASET_DIR=/workspace/data     # datasets
export EXP_ROOT=/workspace/exp         # checkpoints + results
export DAVIS_PATH=/workspace/DAVIS     # background videos

# 1. one-shot setup: packages, DAVIS, EGL render test, rclone check
bash scripts/setup_instance.sh --rclone-remote gdrive:dinowm-reacher
#    -> inspect $EXP_ROOT/setup_check/render_test_egl.png
#       left half = clean, right half = video background. If the right half has
#       no visible video, STOP and fix that before collecting data.

# 2. smoke test (~20-40 min): 8 episodes, 2 epochs, 2 eval episodes/condition
bash scripts/run_all.sh --smoke --num_gpus 1
#    -> check $DATASET_DIR/distracting_reacher/samples/grid_clean_vs_distract.png
#       and $EXP_ROOT/results/analysis/summary.md

# 3. the real run
bash scripts/run_all.sh --num_gpus 2 --gdrive gdrive:dinowm-reacher
```

Without DAVIS (e.g. a laptop), swap in procedurally generated backgrounds:

```bash
MUJOCO_GL=osmesa bash scripts/run_all.sh --smoke --bg_source synthetic
```

### Running the smoke test on a local workstation

The smoke test is small enough for a single consumer GPU (it was sized for a
16GB card). Three things differ from the instance:

```bash
# 1) dataloader workers: conf/env/* ships num_workers=16, sized for the A100 box
# 2) --obs_format pth avoids the decord dependency for tiny datasets
# 3) MUJOCO_GL=osmesa if the local box has no working EGL
bash scripts/run_all.sh --smoke --num_gpus 1 \
     --train_extra "env.num_workers=2" \
     --obs_format pth \
     --collect_workers 2
```

Local smoke runs are a code check only. Delete the artifacts before the real run
so that training does not resume from a 2-epoch checkpoint, and do not upload
locally generated frames: the instance regenerates them from the seed.

```bash
rm -rf "$EXP_ROOT"/{.state,outputs,results} "$DATASET_DIR"/distracting_reacher
```

`run_all.sh` writes a marker per finished stage into `$EXP_ROOT/.state/`, so
re-running it skips completed stages. Force one with `--redo collect|train|eval|all`.

---

## 3. What each file does

**Environment and data**
| file | role |
|---|---|
| [env/distracting_reacher.py](env/distracting_reacher.py) | DMC Reacher + skybox-video background, DINO-WM env interface, numeric-IK goal sampling, self-registers as gym id `distracting_reacher` |
| [datasets/distracting_reacher_dset.py](datasets/distracting_reacher_dset.py) | `TrajDataset` over the generated arm (mp4 or pth frames, optional feature cache) |
| [scripts/collect_data.py](scripts/collect_data.py) | generates trajectories once, renders them twice (clean/distract), verifies the two arms are identical, writes the sample grid |

**Configs**
| file | role |
|---|---|
| [conf/env/distracting_reacher_clean.yaml](conf/env/distracting_reacher_clean.yaml) / [..._distract.yaml](conf/env/distracting_reacher_distract.yaml) | the only difference between models A and B |
| [conf/train_reacher.yaml](conf/train_reacher.yaml) | training hyperparameters, `hydra.run.dir` pinned to `outputs/${exp_name}` for resume |
| [conf/plan_distracting_reacher.yaml](conf/plan_distracting_reacher.yaml) | MPC-CEM planner, `alpha=0` (visual latent cost only), `max_iter=3` |

**Pipeline**
| file | role |
|---|---|
| [scripts/train.sh](scripts/train.sh) | trains A and B; `--gpus 0` sequential, `--gpus 0,1` parallel; resumes automatically |
| [scripts/cache_features.py](scripts/cache_features.py) | optional DINOv2 feature precompute (`has_decoder=false` only) |
| [scripts/eval.py](scripts/eval.py) | models x T1-T4 x phases(plan/rollout/latent), per-episode jsonl, resumable, `--gpu` splittable |
| [scripts/analyze.py](scripts/analyze.py) | csv + markdown tables, success bars, cost-vs-truth scatter, latent-error curves |
| [scripts/run_all.sh](scripts/run_all.sh) | the whole chain with `--smoke` and stage markers |
| [scripts/sync_gdrive.sh](scripts/sync_gdrive.sh) | rclone push to Google Drive, metrics every 3 min, checkpoints every 30 min |
| [scripts/restore_gdrive.sh](scripts/restore_gdrive.sh) | pull back onto a fresh instance and resume |
| [scripts/backup.sh](scripts/backup.sh) | rsync mirror to any path/host |
| [scripts/setup_instance.sh](scripts/setup_instance.sh) | packages, DAVIS, EGL render test, rclone guidance |

---

## 4. Results layout

```
$EXP_ROOT/
├── outputs/reacher_clean/            # model A (and reacher_distract = model B)
│   ├── hydra.yaml                    # resolved training config (eval reads this)
│   ├── checkpoints/model_latest.pth  # resume point, overwritten each epoch
│   ├── checkpoints/model_<N>.pth     # per-epoch snapshots
│   ├── train_stdout.log, train.log
│   └── train/ valid/ rollout_plots/  # reconstruction grids from train.py
├── results/
│   ├── eval/<model>__<cond>.jsonl    # 1 line per episode: success, distances, latent cost, bg
│   ├── rollout/<model>__<cond>.jsonl # 1 line per (episode, horizon): latent error
│   ├── latent/<model>__<cond>.jsonl  # 1 line per (episode, sample): cost vs true distance
│   ├── planner/<model>__<cond>.json  # per-CEM-iteration planner logs
│   ├── videos/<model>__<cond>/       # rollout videos (failures kept by default)
│   ├── analysis/summary.{csv,md}, rollout.csv, correlation.csv, fig_*.png
│   ├── sync.log, sync_heartbeat.json
└── .state/*.done                     # stage markers for run_all.sh
```

Every record carries `git_sha`, `eval_seed`, planner budget, `alpha` and the
background in use, so any number in the summary table traces back to an exact
configuration.

Reading `summary.md`: a large T1 -> T3/T4 drop in `success` **with** a collapse in
`r (cost,dist)` but flat `z_err h*` means the *cost function* failed, not the
predictor. That is the claim this experiment exists to test.

---

## 5. Timing: measure, don't guess

Every stage prints its own duration and `run_all.sh` records it in the stage
marker:

```bash
cat $EXP_ROOT/.state/collect.done     # finished=... duration_s=...
```

Live throughput while collecting (frames/s and ETA) is printed every ~2% of
episodes. For training, one epoch's wall time appears in `train_stdout.log`; to
project the full run:

```bash
grep -c 'Epoch' $EXP_ROOT/outputs/reacher_clean/train_stdout.log
```

To size the real run from the smoke run, scale by episodes x epochs; rendering is
linear in frames and CEM cost is linear in `num_samples x opt_steps x episodes`.

---

## 6. Google Drive sync

One-time headless auth (personal Drive, ~50GB free assumed):

```bash
# on your own machine (has a browser)
rclone authorize "drive" --drive-scope=drive.file     # drive.file = rclone sees only its own files

# on the instance
mkdir -p ~/.config/rclone && chmod 700 ~/.config/rclone
cat > ~/.config/rclone/rclone.conf <<'CONF'
[gdrive]
type = drive
scope = drive.file
token = {"access_token":"...","refresh_token":"...","expiry":"..."}
CONF
chmod 600 ~/.config/rclone/rclone.conf
rclone lsd gdrive:
```

Then either let `run_all.sh --gdrive gdrive:dinowm-reacher` manage the daemon, or
run it yourself:

```bash
bash scripts/sync_gdrive.sh --watch --remote gdrive:dinowm-reacher \
    --keep-every 10 --bwlimit 40M &
```

What goes up: jsonl metrics, logs, `hydra.yaml`, analysis tables and figures,
`model_latest.pth`, every 10th epoch checkpoint, failure videos, dataset
`meta.json` and the sample grid. What does not: the raw frames (regenerable from
the seed). With ~0.3GB checkpoints and `--keep-every 10`, 100 epochs x 2 models is
roughly 6-8GB, comfortably inside 50GB. If Drive fills up, raise `--keep-every`.

Safety properties: `rclone copy` only ever adds, so nothing on Drive is deleted;
`--min-age 5m` prevents uploading a checkpoint that `torch.save` is still
writing. `results/sync_heartbeat.json` shows the last successful sync, so you can
tell from Drive alone whether the daemon is alive.

Instance died? On a new one:

```bash
bash scripts/setup_instance.sh
bash scripts/restore_gdrive.sh --remote gdrive:dinowm-reacher
rm -f $EXP_ROOT/.state/collect.done     # frames are not backed up; regenerate them
bash scripts/run_all.sh --num_gpus 2
```

---

## 7. Knobs you are most likely to touch

```bash
# smaller/faster eval
python scripts/eval.py --ckpt_base_path $EXP_ROOT --n_episodes 10 \
    --cem_samples 64 --cem_iters 5 --mpc_iters 1 --phases plan

# just the diagnostics, no planning (cheap, no CEM)
python scripts/eval.py --ckpt_base_path $EXP_ROOT --phases rollout,latent

# one condition on one GPU
python scripts/eval.py --ckpt_base_path $EXP_ROOT --conditions T4_mismatch --gpu 1

# stronger T4 mismatch
python scripts/eval.py --ckpt_base_path $EXP_ROOT --conditions T4_mismatch --mismatch_shift 80

# re-analyze at any time (safe while eval is running)
python scripts/analyze.py --results_dir $EXP_ROOT/results
```

---

## 8. Troubleshooting

| symptom | cause / fix |
|---|---|
| `render_test_egl.png` right half shows no video | ground-plane alpha or skybox upload failing. Check `ground_plane_alpha=0.0`; run with `MUJOCO_GL=osmesa` to isolate GL issues |
| collect aborts: "clean and distract states differ" | rendering is perturbing physics; do not add reward/termination logic to `step()` |
| collect warns "renders look nearly identical" | background replacement is a no-op; verify the skybox texture id and `mat_rgba['grid',3]` |
| `mjModel has neither tex_data nor tex_rgb` | unsupported mujoco; pin `mujoco==3.2.7` |
| eval: `hydra.yaml not found` | `--ckpt_base_path` must be the dir *containing* `outputs/`, not `outputs/` itself |
| eval hangs forever | upstream `max_iter: null` in a planner config; set `--mpc_iters` |
| training OOM | lower `--batch_size`; `has_decoder=false` also cuts memory |
| `wandb` login prompt | `WANDB_MODE=offline` is the default in `train.sh`; sync later with `wandb sync` |
| Drive uploads but checkpoints never appear | `--min-age 5m` means a file must be 5 minutes old; wait one checkpoint interval |
