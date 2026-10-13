# thread0 — DINO-WM reproduction

> **This is the `hanbin5/acwm-phys` branch**: [`hanbin5/local`](repro/LOCAL.md) (`main` plus memory/speed changes for two 24 GB GPUs and that environment's run records) plus the **ACWM-Phys experiments**: DINO-WM trained on Pour Water, Cloth Move, Push Sand and Stack Cube. Their results, videos and the steps to reproduce them are in [the next section](#acwm-phys-experiments-this-branch). The shared code and docs are on [main](https://github.com/jongmin-s-multicore/thread0/tree/main).

This is a fork of [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm), the code of DINO-WM ([arXiv:2411.04983](https://arxiv.org/abs/2411.04983), "DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning"). We re-ran the paper's planning benchmarks with the released code. For PointMaze, PushT and Wall we used the authors' released world models. For Rope and Granular we trained the world models ourselves, because no checkpoints were released.

- **Evaluation videos:** [Google Drive folder `dino-wm-videos`](https://drive.google.com/drive/folders/17M0MZGSDPzu7CCT3ez3ABMyZGjavTfx3), one sub-folder per experiment (listed below).
- **Detailed results and discussion** (Korean): issues [#1](https://github.com/jongmin-s-multicore/thread0/issues/1) (PointMaze, PushT, Wall) and [#2](https://github.com/jongmin-s-multicore/thread0/issues/2) (Rope, Granular). Full numbers for the main results table are in the JSON comments of those issues. The additional runs reported below (PointMaze seeds 1 and 101, the original-code-path re-run, the Rope predictor-lr 5e-5 retrain, the Rope/Granular arm-video re-runs) are not in the issues yet; their run records are on the `hanbin5/local` branch (`repro/runs/`).
- **Protocol** (models, data, hyper-parameters, evaluation, differences from upstream; Korean): [repro/SETTINGS.md](repro/SETTINGS.md). **How to run** (Korean): [repro/README.md](repro/README.md).

## ACWM-Phys experiments (this branch)

[ACWM-Phys](https://arxiv.org/abs/2605.08567) (Hugging Face dataset [`t1an/ACWM-Phys`](https://huggingface.co/datasets/t1an/ACWM-Phys)) is a benchmark of simulated physics videos with per-frame actions. Every environment has a training set, an in-distribution (InD) test set and an out-of-distribution (OoD) test set. It is not part of the DINO-WM paper. We trained DINO-WM on four of its environments (frozen DINOv2 ViT-S/14 features, ViT predictor, VQ-VAE decoder, all trained from scratch on each environment) and measured how well the world model predicts future frames from the first frame and the actions. The dataset comes without a simulator, goals or rewards, so there is no planning.

**Summary.**
- **In distribution the model predicts well and follows the actions.** Its one-step error in DINOv2 feature space is 0.17–0.46 of the error of copying the previous frame, and feeding it another episode's actions makes the rollout 1.7–3.1× worse. Averaged over an episode, the gripper and pusher positions in the rollout are 3–6 px from the real ones (224 px frames) while they move 30–38 px.
- **Out of distribution, the amount of material is pulled to what training showed.** A cloth 2.5× larger than the training cloth shrinks to the training size within one episode: in the last frame the real cloth covers 4,475 pixels and the model's 1,836 (1,888 on InD episodes). A smaller cloth grows (real 404, model 1,129), a sand pile with twice the particles shrinks to the training size (real 3,886, model 1,843), and the poured water ends near the training mean whatever the real amount.
- **Rare outcomes are pulled to the common one.** In Stack Cube, 27 % of the training episodes are failed stacks. On failed test episodes the model's last frame shows a stacked cube in 23–24 % of the episodes; the real last frame does in 8–12 %.
- **The decoder is not the cause.** The decoder draws the OoD sizes and the failed stacks when it is given the features of the real frames (last column of the second table). The predicted features are what changes.

### Environments, videos and result files

| Environment (video folder) | What happens | OoD test set | Episodes: train / val · InD test · OoD test | Training |
|---|---|---|---|---|
| [`[acwm] pour_water`](https://drive.google.com/drive/folders/1ejhY8cU0dFgsPEEExnReXy0_kTZ1yRkg) | A cup is tilted and pours water into a box (300 frames) | less or more water: 26 low, 24 high | 882 / 98 · 50 · 50 | num_hist 1 (B) and 3 (A), 6 epochs |
| [`[acwm] cloth_move`](https://drive.google.com/drive/folders/1zackVOkDErFJBOC7ylLhR44QV_rfAZkC) | Two grippers hold two corners of a cloth and drape it over a sphere (65 frames) | cloth area 0.26× (50 episodes) or 2.5× (50); training has one cloth size | 1800 / 200 · 100 · 100 | num_hist 1, 15 epochs |
| [`[acwm] push_sand`](https://drive.google.com/drive/folders/1jPap3D5vxFnut8BHrOugjFDgN9xxZkIy) | A board pushes a pile of particles three times (64 frames) | about twice the particles | 1605 / 179 · 100 · 100 | num_hist 1, 18 epochs |
| [`[acwm] stack_cube`](https://drive.google.com/drive/folders/1r2SIYP-alTeVXvKPwAqPim6WAvd6t6IL) | A Franka arm picks the yellow cube and stacks it on the green one (53–58 frames); 27 % of the training episodes fail | yellow cube starts outside the training region | 1235 / 138 · 67 · 271 | num_hist 1, 25 epochs |

All models: 224 px input, frameskip 5 (the model predicts every 5th frame), predictor learning rate 5e-5, batch 24. The epochs give every run about 62–65k optimizer steps. Data conventions (which action drives which frame, the Push Sand action encoding), settings and the evaluation protocol are in [SETTINGS.md §7](repro/SETTINGS.md) (Pour Water) and [§8](repro/SETTINGS.md) (the other three), in Korean.

Each video folder also holds the numbers behind the tables below: `<env>_h1f5.json` (prediction metrics with bootstrap intervals), `<env>_h1f5_quant.json` (physical quantities), `epoch_logs.jsonl` (training curves) and figures, in the `results/` sub-folder for [Cloth Move](https://drive.google.com/drive/folders/1Kr9HsFg_bmx7FZVMZQNSJRP6MPPDpAB0), [Push Sand](https://drive.google.com/drive/folders/16OXSzpN5lt7vD6LExBdFSZ1D3B48mP_E) and [Stack Cube](https://drive.google.com/drive/folders/1KSYUBFMk1HCRz-X8LrMkkN6ssosXx8bx) and next to the videos for Pour Water. Model weights are not uploaded.

### Results

Prediction quality of the num_hist 1 models. *Copy* predicts that nothing changes. *Decoder ceiling* is the real target frame passed through the encoder and decoder, the best the decoder can do. *Skill* is the share of the gap between copying (through the decoder) and the decoder ceiling that the model closes: 0 is as good as copying, 1 is the decoder ceiling. Rollouts are open loop over 37-frame windows (7 predicted frames at frameskip 5), 10 windows per episode. LPIPS: lower is better.

| Environment | Test set | Episodes | One-step latent error, model ÷ copy | Rollout LPIPS: model / copy-last / decoder ceiling | Skill [95 % CI] | LPIPS with another episode's actions ÷ true actions |
|---|---|---|---|---|---|---|
| Pour Water ¹ | InD test | 50 | 0.46 | 0.090 / 0.167 / 0.072 | 0.84 [0.83, 0.84] | 1.7× |
| Pour Water ¹ | OoD low volume | 26 | 0.47 | 0.085 / 0.160 / 0.065 | 0.81 [0.80, 0.82] | 1.7× |
| Pour Water ¹ | OoD high volume | 24 | 0.59 | 0.182 / 0.215 / 0.153 | 0.74 [0.73, 0.75] | 1.3× |
| Cloth Move | InD test | 100 | 0.31 | 0.038 / 0.113 / 0.018 | 0.79 [0.78, 0.81] | 3.1× |
| Cloth Move | OoD smaller cloth | 50 | 0.74 | 0.050 / 0.059 / 0.012 | 0.22 [0.17, 0.27] | 1.5× |
| Cloth Move | OoD larger cloth | 50 | 0.77 | 0.125 / 0.167 / 0.040 | 0.35 [0.31, 0.38] | 1.5× |
| Push Sand | InD test | 100 | 0.39 | 0.025 / 0.054 / 0.018 | 0.83 [0.82, 0.84] | 2.7× |
| Push Sand | OoD 2× particles | 100 | 0.74 | 0.069 / 0.065 / 0.042 | 0.31 [0.28, 0.33] | 1.4× |
| Stack Cube | InD test | 67 | 0.17 | 0.069 / 0.225 / 0.024 | 0.79 [0.76, 0.81] | 2.1× |
| Stack Cube | OoD start position | 271 | 0.18 | 0.078 / 0.228 / 0.027 | 0.76 [0.74, 0.77] | 1.9× |

¹ The Pour Water rollout columns use window frames 15–35 only, so that its two models can be compared on the same frames; the other environments use all predicted frames (5–35). Its episodes begin with the same tilt in every episode: for windows that start at frame 200 or later the last column is 1.9× / 1.9× / 1.5×.

What the rollout does to the scene, measured with the same colour mask on real and decoded frames (224 px). The rollout starts from the first frame and uses all actions of the episode.

| Environment | Test set | Quantity | Real | Model | Decoder on real frames |
|---|---|---|---|---|---|
| Pour Water | InD test | water pixels in the box, frames ≥ 250 | 1,970 | 2,045 | 1,962 |
| Pour Water | OoD low volume | water pixels in the box, frames ≥ 250 | 1,384 | 2,049 | 1,370 |
| Pour Water | OoD high volume | water pixels in the box, frames ≥ 250 | 2,407 | 2,077 | 2,326 |
| Cloth Move | InD test | cloth pixels, last frame | 1,819 | 1,888 | 1,790 |
| Cloth Move | OoD smaller cloth | cloth pixels, last frame | 404 | 1,129 | 388 |
| Cloth Move | OoD larger cloth | cloth pixels, last frame | 4,475 | 1,836 | 4,352 |
| Push Sand | InD test | sand pixels, last frame | 1,792 | 1,665 | 1,729 |
| Push Sand | OoD 2× particles | sand pixels, last frame | 3,886 | 1,843 | 3,215 |
| Stack Cube | InD test, successful stacks (50) | last frame shows the cube stacked | 100 % | 96 % | 100 % |
| Stack Cube | InD test, failed stacks (17) | last frame shows the cube stacked | 12 % | 24 % | 18 % |
| Stack Cube | OoD, successful stacks (200) | last frame shows the cube stacked | 100 % | 90 % | 100 % |
| Stack Cube | OoD, failed stacks (71) | last frame shows the cube stacked | 8 % | 23 % | 6 % |

Limits of these numbers:
- One seed and one setting per environment (Pour Water has two models; num_hist 3 and 1 did not differ at equal horizon). The validation loss was still falling slowly at the last epoch (0.4–1.8 % per epoch).
- The tables do not compare with the ACWM-DiT numbers of the ACWM-Phys paper: we score every 5th frame at 224 px, and the released test sets differ in size from the ones in the paper.
- The colour masks are checked against the decoder on real frames: on InD episodes the masks overlap with IoU 0.92 (cloth) and 0.95 (sand) and the areas agree within 2–3 %. For Push Sand OoD the decoder itself draws 17 % less sand than the real frame, so part of that row's gap is the decoder.
- The "stacked" test is an image rule (yellow cube centre right above the green one). On real frames it agrees with the dataset's success flag in 98.8 % of the training episodes; that is why failed stacks show 8–12 % in the Real column.

### How to read these videos

- Left panel **Real**: the dataset video. Right panel **Model**: the world model's rollout from the first frame and all actions of the episode, decoded to pixels. Both show every 5th frame at 3 fps and hold the last frame for 2 s. Push Sand is trained on frames squeezed to 224 × 224; the videos stretch them back.
- `<set>_<group>_<rank>_<episode>.mp4`: one episode. Ten episodes per group, picked at evenly spaced quantiles of the whole-episode LPIPS, so rank `01` is the best-predicted one and `10` the worst. Groups: `ind_test_all`, `ood_test_smaller` and `ood_test_larger` for Cloth Move; `ind_test_all` and `ood_test_all` for Push Sand; `{ind,ood}_test_{success,fail}` for Stack Cube.
- `qual_<set>_<group>.mp4`: three episodes of a group in one video (10 %, 50 % and 90 % quantiles), top to bottom.
- Pour Water: `qual_{ind_test,ood_low,ood_high}.mp4` show ground truth, model A (num_hist 3) and model B (num_hist 1) side by side for one episode each.

### Reproduce

**Requirements.** The same as in [Running it](#running-it), without docker and without the OSF data: an NVIDIA driver, `git curl bzip2 unzip zip gcc`, and `libglew-dev libgl1-mesa-dev`. These experiments never call MuJoCo, but the code imports mujoco-py, so `install_env.sh` has to build it. Disk space in the work root `$DINO_WORK`: Python env 9 GB plus about 4 GB of pip downloads, dataset 2.6 GB, and for the outputs about 5 GB of checkpoints and 2 GB of result files for Cloth Move, Push Sand and Stack Cube, and 5 GB of checkpoints for Pour Water. On an RTX 3090 one training run takes about 7 h (Pour Water model A 16 h) and its evaluation 25–60 min.

**GPU memory by training setting.** Peak memory of the training process (`nvidia-smi`) at 224 px input. It depends on `num_hist` (frames per sample) and `training.batch_size`; `frameskip` does not change the number of frames per sample. To fit a smaller GPU, lower `training.batch_size` in the job list.

| `num_hist` | `training.batch_size` | Peak GPU memory | Used for |
|---|---|---|---|
| 1 | 24 | 7.1 GB | Cloth Move, Push Sand, Stack Cube, Pour Water model B |
| 3 | 24 | 19.8 GB | Pour Water model A |
| 3 | 32 (the paper's batch size) | more than 24 GB (out of memory on a 24 GB GPU) | not used |

Everything needed is in this repo and in the public dataset; nothing is read from outside the clone and `$DINO_WORK`. As in [Running it](#running-it), the code blocks contain no `#` comments and work in bash and zsh, from the repo root.

**Install and data.** `install_env.sh` creates the Python env under `$DINO_WORK` (any path outside the repo; `~/dinowm` is the default). `source repro/env.sh` sets the paths and puts the env's `python` first on `PATH`; run it in every new shell. The download script fetches the four environments from Hugging Face at a pinned dataset commit into `$DINO_WORK/data/acwm_phys` and checks the number of episodes of every split. If the Hub rate-limits the download it waits and retries; files already on disk are skipped.

```bash
git clone -b hanbin5/acwm-phys https://github.com/jongmin-s-multicore/thread0.git
cd thread0
export DINO_WORK=~/dinowm
bash repro/setup/install_env.sh
source repro/env.sh
python repro/setup/download_acwm_phys.py
```

**Pilot (optional).** 60 episodes and 3 epochs per environment, a few minutes each. It shows that training, validation and checkpointing work before the long runs.

```bash
bash repro/queue/start.sh repro/jobs/acwm_phys_pilot.txt
```

**Full runs.** The job queue runs in the background and starts one job per GPU; its log is `$DINO_RUNS/scheduler.log`, which ends with `QUEUE EMPTY` when a job list is done. Run one job list at a time: `start.sh` replaces the running scheduler, and jobs still waiting in the old list would not start. `acwm_phys.txt` trains Cloth Move, Push Sand and Stack Cube and then evaluates each model, measures the physical quantities and renders the summary videos (about a day on a single GPU). `pour_water.txt` trains and evaluates Pour Water models A and B.

```bash
bash repro/queue/start.sh repro/jobs/acwm_phys.txt
```

```bash
bash repro/queue/start.sh repro/jobs/pour_water.txt
```

**Videos, figure and tables.** These read the saved frames and JSON files, no GPU: the per-episode videos of the Drive folders, the cloth and sand area figure, and the two tables of this README.

```bash
R=$DINO_WORK/results/acwm_phys
python repro/eval/render_acwm_phys.py $R/cloth_h1f5 $R/figs/cloth_h1f5 --per_episode 10
python repro/eval/render_acwm_phys.py $R/sand_h1f5 $R/figs/sand_h1f5 --per_episode 10
python repro/eval/render_acwm_phys.py $R/stack_h1f5 $R/figs/stack_h1f5 --per_episode 10 --groups success,fail
python repro/eval/plot_acwm_phys.py $R $R/figs/fig_area_cloth_sand.png
python repro/eval/tables_acwm_phys.py $DINO_WORK/results
```

**Where things land and what to compare.** Models: `$DINO_WORK/train_runs/outputs/<job>/`. Logs: `$DINO_WORK/runs/<job>.log`. Results: `$DINO_WORK/results/acwm_phys/<env>_h1f5.json` and `<env>_h1f5_quant.json` (videos and grids under `figs/`), and `$DINO_WORK/results/pour_water/`. `tables_acwm_phys.py` prints the two tables above from your JSON files; the JSON files of our runs are in the Drive folders. A re-run will not be bit-identical (GPU non-determinism), and with one seed per environment we do not know how much the numbers move between runs.

- Pour Water's water-amount analysis and figures are separate scripts (`repro/eval/eval_pour_water_volume.py`, `plot_pour_water.py`, `render_pour_water.py`; usage in each file).
- The data checks behind the conventions in SETTINGS.md §8 (which action drives which frame, the Push Sand action encoding, the frameskip) are in [`repro/setup/checks/acwm_phys/`](repro/setup/checks/acwm_phys/README.md). They are not needed to reproduce the results.
- Run records (what each run was): `repro/runs/{train,eval}/dinowm_{pourwater,clothmove,pushsand,stackcube}-*/run.yaml`.

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

Hardware used: two 24 GB GPUs (RTX 3090 Ti + RTX 3090) on Ubuntu 24.04. Setup scripts and the job queue are in `repro/` ([repro/README.md](repro/README.md), Korean). Run the commands below in one bash or zsh shell, from the repo root. The code blocks contain no `#` comments on purpose: zsh without `setopt interactivecomments` (the default unless e.g. oh-my-zsh turns it on) passes an inline `# ...` to the command as arguments.

**Requirements**

- An NVIDIA driver, `git curl bzip2 unzip zip gcc`, and `libglew-dev libgl1-mesa-dev` for building mujoco-py. mujoco-py builds its GPU (EGL) renderer if `/usr/lib/nvidia` exists (Ubuntu's driver packages create it) and a CPU renderer otherwise, which also needs `libosmesa6-dev`.
- For Rope and Granular planning: docker with the NVIDIA container runtime, usable without sudo (`docker` group).
- Disk space in the work root `$DINO_WORK` (default `~/dinowm`): about 110 GB for PointMaze, PushT and Wall (Python env 9 GB, data 91 GB, checkpoints 1 GB, zip files 6 GB), and about 160 GB more for Rope and Granular (data 144 GB, zip files 15 GB, plus a 15 GB temporary file while unzipping). The zip files in `$DINO_WORK/downloads` can be deleted after unzipping. If your home directory is small, choose a work root on a larger disk.

**Install**

Nothing assumes your home directory. The commands below clone the repo into the current directory; it can live anywhere. `DINO_WORK` is the work root for the env, data, checkpoints and run outputs, and can be any path outside the repo. `~/dinowm` in the third line is only the default: if your home directory is small, replace it with a path on a larger disk (see the disk space above).

```bash
git clone https://github.com/jongmin-s-multicore/thread0.git
cd thread0
export DINO_WORK=~/dinowm
bash repro/setup/install_env.sh
```

Every script reads `DINO_WORK` from the environment and falls back to `~/dinowm`. If you choose another path, set it again in every new shell (for example in `~/.bashrc`), or write `export DINO_WORK=${DINO_WORK:-/your/path}` into `.claude/env.local.sh` in the repo (gitignored; `repro/env.sh` reads it first).

`install_env.sh` creates a micromamba env (Python 3.9 with the pinned versions of `environment.yaml`), installs MuJoCo 2.1.0 into `~/.mujoco/mujoco210` (where mujoco-py looks by default; set `MUJOCO_DIR` to use another path) and builds mujoco-py. It installs the pip packages (about 4 GB of downloads) with uv if `uv` is on `PATH`, otherwise with pip. `bash repro/setup/install_env.sh --dry-run` prints the commands without running them.

For Rope and Granular only, build PyFleX (in docker):

```bash
bash repro/setup/install_pyflex.sh
```

**Data and released checkpoints** from the authors' OSF project. `core` is PointMaze, PushT and Wall; `deformable` is Rope and Granular. If a download is interrupted, run the same command again and it resumes.

```bash
bash repro/setup/download_data.sh core checkpoints
bash repro/setup/download_data.sh deformable
```

**Check and one quick planning run.** `source repro/env.sh` sets the paths and puts the env's `python` first on `PATH`; run it in every new shell. `check_env.sh` tests torch/CUDA, mujoco-py and DINOv2, then replays dataset actions in PointMaze, PushT and Wall and compares the frames with the dataset (`check_env.sh --pyflex` adds Rope and Granular). The `plan.py` command plans 2 episodes with the released PointMaze model (under a minute) and writes to `$DINO_RUNS/smoke/point_maze`.

```bash
source repro/env.sh
bash repro/setup/check_env.sh
python plan.py --config-name plan_point_maze.yaml model_name=point_maze ckpt_base_path=$DINO_CKPT \
  n_evals=2 planner.sub_planner.opt_steps=2 planner.max_iter=1 hydra.run.dir=$DINO_RUNS/smoke/point_maze
```

**All benchmark jobs** (`repro/jobs/benchmark.txt`) run in a background job queue; its log is `$DINO_RUNS/scheduler.log`. GPUs, jobs per GPU and free-memory thresholds are set with environment variables ([repro/README.md](repro/README.md), "머신별 설정").

```bash
bash repro/queue/start.sh
```

## Branches and changes to upstream

- **`main`**: upstream `0a9492f` plus fixes and features that apply to any GPU environment, and the reproduction tooling in `repro/`. Every edited upstream block carries a `[repro]` comment: `git diff 0a9492f -- . ':!repro' ':!*.md' ':!.gitignore'`.
- **[`hanbin5/local`](https://github.com/jongmin-s-multicore/thread0/tree/hanbin5/local)**: `main` plus memory/speed changes for 24 GB GPUs and the run records of the results above (`repro/LOCAL.md`, `repro/runs/`). All results in this README were produced with this branch's code (LOCAL.md §6 maps each job to the commit it ran from). The changes are fp32 SDPA attention, chunked CEM rollouts and GD, decoding only the plotted rollouts, skipping CEM for episodes MPC already solved, and validation under `no_grad`. They keep the computation identical or within 1.4e-6 relative error, except that skipping solved episodes changes MPC from iteration 2 on (iteration 1 is identical): the remaining episodes draw random numbers in a different order, and the inner CEM's early stop checks only the still-unsolved episodes instead of all 50.
- **[`hanbin5/acwm-phys`](https://github.com/jongmin-s-multicore/thread0/tree/hanbin5/acwm-phys)** (this branch): `hanbin5/local` plus the ACWM-Phys loaders (`datasets/acwm_*_dset.py`, `conf/env/acwm_*.yaml`), job lists, evaluation tools and run records. Outside the new files it changes only `models/visual_world_model.py`: the prediction that is decoded for logging is decoded without autograd (LOCAL.md §2).

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
