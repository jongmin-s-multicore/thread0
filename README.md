# thread0 — DINO-WM reproduction

> **This is the [`hanbin5/local`](repro/LOCAL.md) branch**: `main` plus memory/speed changes for two 24 GB GPUs and the run records of this environment (`repro/runs/`). The shared code and docs are on [main](https://github.com/jongmin-s-multicore/thread0/tree/main).

This is a fork of [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm), the code of DINO-WM ([arXiv:2411.04983](https://arxiv.org/abs/2411.04983), "DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning"). We re-ran the paper's planning benchmarks with the released code. For PointMaze, PushT and Wall we used the authors' released world models. For Rope and Granular we trained the world models ourselves, because no checkpoints were released.

- **Evaluation videos:** [Google Drive folder `dino-wm-videos`](https://drive.google.com/drive/folders/17M0MZGSDPzu7CCT3ez3ABMyZGjavTfx3), one sub-folder per experiment (listed below).
- **Detailed results and discussion** (Korean): issues [#1](https://github.com/jongmin-s-multicore/thread0/issues/1) (PointMaze, PushT, Wall) and [#2](https://github.com/jongmin-s-multicore/thread0/issues/2) (Rope, Granular). Full numbers for the main results table are in the JSON comments of those issues. The additional runs reported below (PointMaze seeds 1 and 101, the original-code-path re-run, the Rope predictor-lr 5e-5 retrain, the Rope/Granular arm-video re-runs) are not in the issues yet; their run records are on the `hanbin5/local` branch (`repro/runs/`).
- **Protocol** (models, data, hyper-parameters, evaluation, differences from upstream; Korean): [repro/SETTINGS.md](repro/SETTINGS.md). **How to run** (Korean): [repro/README.md](repro/README.md).

## Results

Each row is one sub-folder of the [video folder](https://drive.google.com/drive/folders/17M0MZGSDPzu7CCT3ez3ABMyZGjavTfx3). The planning seed is 99 throughout. Paper numbers are from arXiv v2: Table 1 for MPC, Table 8 for open-loop CEM and GD.

| Experiment (video folder) | Environment | World model | Planner | Metric | Ours | Paper |
|---|---|---|---|---|---|---|
| [`pointmaze_mpc`](https://drive.google.com/drive/folders/158v3AxloqlmTjQle8--d_khoe-WufC1x) | PointMaze | released | MPC-CEM | success rate, 50 episodes ↑ | **1.00** | 0.98 |
| [`pointmaze_cem30`](https://drive.google.com/drive/folders/1syrPyrz8DvJUqtj0KMOJCGxLqb5_vUJ9) | PointMaze | released | open-loop CEM, 30 opt steps | success rate, 50 episodes ↑ | **0.90** | 0.80 ¹ |
| [`pointmaze_gd`](https://drive.google.com/drive/folders/18LXWuFazAfT0JonZzf79qJVFoJS9TC3o) | PointMaze | released | open-loop GD | success rate, 50 episodes ↑ | **0.20** | 0.22 |
| [`pusht_mpc`](https://drive.google.com/drive/folders/10991WBPOl-Y1ujpCPWMH6cm01HYiuEEF) | PushT | released | MPC-CEM | success rate, 50 episodes ↑ | **0.92** | 0.90 |
| [`pusht_gd`](https://drive.google.com/drive/folders/1q4bBv2dlsNlqEcVhYRjAE8il6mcGtX28) | PushT | released | open-loop GD | success rate, 50 episodes ↑ | **0.56** | 0.28 |
| [`wall_mpc`](https://drive.google.com/drive/folders/1xQaHS_Q6p0j__xbPLCeQz4wrfGnIdl4G) | Wall | released | MPC-CEM | success rate, 50 episodes ↑ | **0.94** | 0.96 |
| [`wall_cem30`](https://drive.google.com/drive/folders/1nezRBUN0DhYKTAR48FHCLnMnBY9Eyp7A) | Wall | released | open-loop CEM, 30 opt steps | success rate, 50 episodes ↑ | **0.28** | 0.74 ¹ |
| [`wall_gd`](https://drive.google.com/drive/folders/1MaVPC-GqGOIfp_qdinaCBVTbWNc2jgM3) | Wall | released | open-loop GD | success rate, 50 episodes ↑ | **0.04** | – (not reported) |
| [`rope_mpc`](https://drive.google.com/drive/folders/1KgGV_Ic2AcpQVooBJCnv4EPqivl_qxbk) | Rope | ours (100 epochs) | MPC-CEM | Chamfer distance, 10 instances ↓ | **0.933** ² | 0.41 |
| [`granular_mpc`](https://drive.google.com/drive/folders/178kKGzYE0xebNJbyfsw1ji9PyrAZUA6C) | Granular | ours (100 epochs) | MPC-CEM | Chamfer distance, 10 instances ↓ | **0.216** ² | 0.26 |

¹ The paper does not state the number of optimization steps behind its open-loop CEM results (Table 8); the only CEM setting it gives is for the planning-time measurement (Appendix A.8: 100 samples, 10 steps). Our open-loop CEM with the MPC configuration's 10 steps (identical to MPC iteration 1, see below) gives PointMaze **0.86** (paper 0.80) and Wall **0.54** (paper 0.74). With 30 steps (these folders), PointMaze rises to 0.90 and Wall drops to 0.28. Wall open-loop CEM peaks at 0.58 at 9 optimization steps, and the gap between the world model's predicted latent and the latent of the real outcome grows with more optimization steps (issue #1).
² CD of the original evaluation run (the numbers in issue #2). The videos in these two folders come from a separate re-run that also renders the robot arm (see [How to read the videos](#how-to-read-the-videos)); that re-run's final CD is 1.045 for Rope and 0.224 for Granular.

**Summary.** MPC-CEM matches the paper on PointMaze, PushT and Wall: every gap is 0.02, within the binomial standard error of 50 episodes (0.02–0.04 at success rates of 0.90–0.98). Granular reaches the paper's level (CD 0.216 vs 0.26, lower is better). Rope does not reproduce (0.933 vs 0.41), the same outcome reported in upstream issue [gaoyuezhou/dino_wm#24](https://github.com/gaoyuezhou/dino_wm/issues/24). Open-loop CEM on Wall stays below the paper at every number of optimization steps we measured (at most 0.58 vs 0.74). Open-loop GD on PushT is higher than in the paper (0.56 vs 0.28) with the upstream GD config. We did not find the cause.

## What each experiment is

### Environments and tasks

| Environment | Task | Goal of an episode | Success |
|---|---|---|---|
| **PointMaze** (D4RL U-maze, MuJoCo) | Move a point mass (green dot) through a U-shaped maze | A start and a goal position drawn independently from the free space (`goal_source=random_state`) | Position within 0.5 of the goal position |
| **PushT** | Push a T-shaped block with a circular agent | Start: a state on a held-out demonstration. Goal: the state 25 env steps later on the same demonstration (`goal_source=dset`), so it is reachable in 25 steps | Agent and block positions within 20 (4-D L2, 512 px canvas) and block angle within π/9 |
| **Wall** | Move a dot from one room to the other through a door in a wall | Start and goal in opposite rooms; wall and door positions taken from validation trajectories | Position within 4.5 (env units) of the goal |
| **Rope** (NVIDIA FleX) | Push a rope with an xArm6 pusher | Start: a particle state from a random validation trajectory. Goal: the reset rope shape, translated and rotated | No success test (upstream's test is always false); the metric is the Chamfer distance |
| **Granular** (NVIDIA FleX) | Push granular particles with an xArm6 pusher | Start as for Rope. Goal: the reset particle pile, translated and scaled | as Rope |

### World models

- **Released** (PointMaze, PushT, Wall): checkpoints from the authors' [OSF project](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28). They are frozen DINOv2 ViT-S/14 patch features + ViT predictor + VQ-VAE decoder (decoder used only for the videos and the prediction-quality metric).
- **Ours** (Rope, Granular): trained with upstream `train.py` on the OSF datasets for 100 epochs (history 1, frameskip 1, batch 32, upstream default learning rates; [SETTINGS.md §4.2](repro/SETTINGS.md)).

### Planners (upstream `conf/plan*.yaml` and `conf/planner/*.yaml` unless noted)

- **MPC-CEM**: CEM plans 5 actions ahead (300 samples, top 30; 10 opt steps for PointMaze and Wall, 30 for PushT, Rope and Granular). All 5 actions are executed in the simulator, then the agent replans from the observation it reached. Upstream has no limit on the number of replanning iterations, so we capped them: PointMaze and Wall at 20, PushT at 10, Rope and Granular at 5. An episode counts as solved as soon as it succeeds after any iteration (upstream behaviour). Iterations used: PointMaze 4 (all episodes solved), PushT 10, Wall 20.
- **Open-loop CEM**: CEM plans the whole horizon (5 actions) once and the plan is executed without feedback. With the MPC settings (10 opt steps for PointMaze and Wall, 30 for PushT) this is exactly MPC iteration 1. That is what we compare with the paper's "CEM" column: PointMaze 0.86, PushT 0.90, Wall 0.54 vs. 0.80 / 0.86 / 0.74. The `pointmaze_cem30` and `wall_cem30` folders, the open-loop CEM rows of the results table, are an extra run with the default of upstream `conf/planner/cem.yaml` (30 opt steps); PushT needs no such run because its MPC setting already uses 30.
- **Open-loop GD**: gradient descent on the action sequence through the world model (upstream `conf/planner/gd.yaml`: SGD lr 1, 1000 steps, noise 0.003).
- The planning objective is the MSE between the world model's predicted final latent and the goal observation's latent (plus a proprioception term for PushT and Wall). The simulator executes and scores the plan. As in upstream, the planners also read the simulator's success test while planning: CEM and GD stop optimizing once every episode has succeeded, and MPC stops replanning an episode once it has succeeded ([gaoyuezhou/dino_wm#26](https://github.com/gaoyuezhou/dino_wm/issues/26)).

### Metrics

- **Success rate**: fraction of the 50 episodes that pass the success test above. Open-loop planners are scored on the state after the whole plan (5 planned actions = 25 env steps). MPC is scored on the state at the end of the first replanning iteration after which the episode passed, or after the last iteration if it never passed (upstream behaviour, so MPC success is sticky).
- **Chamfer distance (CD)**: between the final particle positions and the goal particle positions (xyz, sum of the two directional mean nearest-neighbour distances), averaged over the 10 instances. Lower is better.

## How to read the videos

Every folder holds `output_final_<i>_<tag>.mp4` for episodes `i` = 0–9: the first 10 of the 50 evaluated episodes, or all 10 instances for Rope and Granular. `<tag>` is `success` or `failure` from the final evaluation. Rope and Granular videos are always tagged `failure`, because upstream's success test for them is always false; judge them by CD.

Each frame is a 2 × 2 grid with a label in every panel:

| | left | right |
|---|---|---|
| top | **Real**: the simulator executing the planned actions | **Goal**: the goal observation |
| bottom | **Model**: the world model's prediction for the same actions, decoded to pixels | **Goal** |

- **PointMaze, PushT, Wall**: 12 fps, one frame per simulator step. The Model panel changes every 5 frames, once per planned action (frameskip 5). MPC videos show the whole executed trajectory up to the iteration at which the episode succeeded. Every video ends by holding its last frame for 3 s; there are no gray padding frames.
- **PushT**: the light-green T is drawn by the environment at a fixed pose in every image (also in training data). It is **not** the goal. The red outline in Real and Model is the goal pose of the block (in these converted videos it was fitted to the Goal panel, within 3 px (512 px canvas) and 2° of the true goal pose; new runs draw the goal state directly).
- **PointMaze**: the red ring marks the goal position. Its radius is the success radius 0.5, with four ticks pointing at the goal. Frames show the position one 0.01 s physics sub-step behind the state used for scoring. So an episode that ends right at the boundary can look just outside the ring (episode 4 of `pointmaze_mpc`).
- **Wall**: no overlay. The goal is the dot in the Goal panel.
- **Rope, Granular**: 24 fps. These come from a re-run that keeps the intermediate simulator frames, so the xArm is seen pushing ([`repro/eval/deform_arm_video.py`](repro/eval/deform_arm_video.py)). The data and the standard evaluator keep only one frame per push, taken after the arm has returned home. The Model panel advances when each push finishes. The re-run used the same model, settings, seed and start/goal particle states. FleX simulation and rendering are not bit-reproducible across processes, though (the Rope re-run also ran on the other of our two GPUs, and its rendered start/goal images already differ slightly; the Granular re-run matches the evaluated run for two MPC iterations and then diverges). So per-instance results differ from the evaluated run: final CD is 1.045 vs 0.933 for Rope and 0.224 vs 0.216 for Granular.

The plotted frames, labels and overlays are drawn after the metrics are computed. With the overlay on and off, `logs.json` is byte-identical and the start/goal file (`plan_targets.pkl`) holds the same values (also byte-identical for PointMaze; for PushT only the pickle bytes differ). Old videos were converted with `repro/eval/{pusht_goal_overlay,pointmaze_goal_overlay,reformat_eval_videos}.py`.

## Additional results

These are not in the video folder.

**MPC success rate vs. number of replanning iterations** (iteration 1 = open-loop CEM):

| Iterations | 1 | 2 | 3 | 4 | 5–10 | 11–20 |
|---|---|---|---|---|---|---|
| PointMaze | 0.86 | 0.96 | 0.96 | 1.00 | | |
| PushT | 0.90 | 0.92 | 0.92 | 0.92 | 0.92 | (cap 10) |
| Wall | 0.54 | 0.86 | 0.94 | 0.94 | 0.94 | 0.94 |

**PointMaze over three seeds** (99, 1, 101; 50 episodes each). Mean success: MPC 0.987 (paper 0.98), open-loop CEM with 10 steps 0.82 (0.80), CEM with 30 steps 0.87, GD 0.25 (0.22). The gaps to the paper are within seed-to-seed variation. Re-running seed 99 with the original code path (original attention, no skipping of solved episodes) gives the same success rates for open-loop CEM (at every one of the 30 opt steps) and for MPC iteration 1 (0.86); the other logged metrics agree to within 3e-4 relative, consistent with the ~1e-6 latent difference of fp32 SDPA attention. GD is chaotic and changes (0.20 → 0.14).

**Rope and Granular, MPC iterations** (CD after each iteration; the actions so far replayed from the start):

| Iteration | 1 | 2 | 3 | 4 | 5 | final evaluation | paper |
|---|---|---|---|---|---|---|---|
| Rope | 1.199 | 1.320 | 0.918 | 0.686 | 0.909 | 0.933 | 0.41 |
| Granular | 0.409 | 0.304 | 0.223 | 0.223 | 0.222 | 0.216 | 0.26 |

The final evaluation replays the same actions as iteration 5 and gives a different CD, because FleX results depend on execution history (issue #2). We also retrained Rope with the predictor learning rate listed in the paper's Table 12 (5e-5 instead of upstream's 5e-4). Decoded prediction quality improved (LPIPS 0.043 → 0.018, SSIM 0.963 → 0.985), but the gain comes from the decoder (reconstruction LPIPS 0.039 → 0.012), whose settings were unchanged and which planning does not use; the predictor's latent prediction loss got slightly worse (0.071 → 0.079). Planning did not improve (CD 1.157; two runs of the 5e-4 model gave 0.933 and 1.045), so the learning rate does not explain the Rope gap.

**Prediction quality** (decoded one-step prediction on the validation split — for PointMaze and Wall 5000 evenly spaced slices of 16200 / 7872 — VGG LPIPS ↓ / SSIM ↑; paper Table 4 and 9. The paper does not state its prediction horizon or sample count, so the comparison is approximate):

| | PointMaze | PushT | Wall | Rope | Granular |
|---|---|---|---|---|---|
| Ours | 0.0006 / 0.999 | 0.0067 / 0.987 | 0.0022 / 0.997 | 0.043 / 0.963 | 0.102 / 0.901 |
| Paper | – | 0.007 / 0.985 | 0.0016 / 0.997 | 0.009 / 0.985 | 0.035 / 0.94 |

For Rope and Granular, plain encode–decode reconstruction of the target frame is already almost as poor as the prediction. So the gap comes mostly from the decoder, not the predictor.

**Not reproduced:** Reacher (data and environment were not released), the baselines IRIS / DreamerV3 / TD-MPC2 / AVDC (not in the codebase), the generalization environments (the WallRandom and PushObj datasets were not released; GranularRandom needs changes to the environment code), and the encoder and ablation studies (they need many retrained models, and the mask and decoder-loss ablations need code changes). See [SETTINGS.md §1](repro/SETTINGS.md).

## Running it

Hardware used: two 24 GB GPUs (RTX 3090 Ti + RTX 3090) on Ubuntu 24.04. Setup scripts and the job queue are in `repro/` ([repro/README.md](repro/README.md), Korean).

```bash
git clone https://github.com/jongmin-s-multicore/thread0.git ~/thread0 && cd ~/thread0
export DINO_WORK=~/dinowm                    # work root outside the repo (env, data, checkpoints, runs)
bash repro/setup/install_env.sh              # micromamba env (Python 3.9) + MuJoCo 2.1.0 + mujoco-py
bash repro/setup/install_pyflex.sh           # only for Rope/Granular (docker build)
bash repro/setup/download_data.sh core checkpoints   # PointMaze, PushT, Wall + released checkpoints (OSF)
bash repro/setup/download_data.sh deformable         # Rope, Granular
source repro/env.sh && bash repro/setup/check_env.sh

# one quick planning run with the released PointMaze model (2 episodes)
python plan.py --config-name plan_point_maze.yaml model_name=point_maze ckpt_base_path=$DINO_CKPT \
  n_evals=2 planner.sub_planner.opt_steps=2 planner.max_iter=1 hydra.run.dir=$DINO_RUNS/smoke/point_maze

bash repro/queue/start.sh                    # all benchmark jobs (repro/jobs/benchmark.txt)
```

The setup scripts are the commands we ran by hand on the machine above. Every command was run (pip packages were installed with uv; the plain-pip and CPU-renderer fallbacks were not run), but the scripts have not been run end to end on a fresh machine.

## Branches and changes to upstream

- **`main`**: upstream `0a9492f` plus fixes and features that apply to any GPU environment, and the reproduction tooling in `repro/`. Every edited upstream block carries a `[repro]` comment: `git diff 0a9492f -- . ':!repro' ':!*.md' ':!.gitignore'`.
- **[`hanbin5/local`](https://github.com/jongmin-s-multicore/thread0/tree/hanbin5/local)**: `main` plus memory/speed changes for 24 GB GPUs and the run records of the results above (`repro/LOCAL.md`, `repro/runs/`). All results in this README were produced with this branch's code (LOCAL.md §6 maps each job to the commit it ran from). The changes are fp32 SDPA attention, chunked CEM rollouts and GD, decoding only the plotted rollouts, skipping CEM for episodes MPC already solved, and validation under `no_grad`. They keep the computation identical or within 1.4e-6 relative error, except that skipping solved episodes changes MPC from iteration 2 on (iteration 1 is identical): the remaining episodes draw random numbers in a different order, and the inner CEM's early stop checks only the still-unsolved episodes instead of all 50.

Changes on `main`:

| File | Change | Effect on results |
|---|---|---|
| `models/dino.py` | Pin the DINOv2 torch.hub code to `85a2460` (Python 3.9; upstream issue #25) | none (identical features) |
| `env/deformable_env/.../flex_env.py` | Call `pyflex.init()` once per process (several envs segfaulted) | none |
| `train.py` | Also write epoch metrics to `epoch_logs.jsonl` (for runs without wandb) | none |
| `planning/evaluator.py` | Evaluation videos: panel labels, end at the last executed frame, 3 s hold | videos only |
| `plan.py`, `env/pusht/goal_outline.py` | PushT: outline the goal block pose on plots/videos | plots/videos only (`logs.json` identical) |
| `plan.py`, `env/pointmaze/goal_marker.py` | PointMaze: ring of the success radius at the goal position on plots/videos | plots/videos only (`logs.json` identical) |

Protocol choices that differ from upstream defaults, such as the MPC iteration caps, are listed in [SETTINGS.md §6](repro/SETTINGS.md).

## License

Code is MIT like upstream ([LICENSE](LICENSE), © gaoyuezhou); `repro/` and our changes are MIT as well. Datasets and released checkpoints come from the authors' [OSF project](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28) and are not redistributed here, nor are our trained weights. The upstream README is kept as [README_upstream.md](README_upstream.md); repository conventions are in [AGENTS.md](AGENTS.md) (Korean).
