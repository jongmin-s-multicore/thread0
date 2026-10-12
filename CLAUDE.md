# Distracting-Reacher experiment: core design

This repo is DINO-WM (upstream: gaoyuezhou/dino_wm) plus an experiment that tests
one claim: **DINO-WM's latent-L2 planning cost is fragile to visual distractors.**
Keep the design below intact; it is what makes the numbers interpretable.

## The claim under test

Planning cost is `MSE(z_visual) + alpha * MSE(z_proprio)`
([planning/objectives.py](planning/objectives.py)). With `alpha=0` the cost is
purely the DINOv2 latent L2 between the imagined and the goal observation. A
video background injects large latent variation that is unrelated to the task, so
the cost can be minimized by matching the background rather than the arm.

## Non-negotiable design decisions

1. **Paired rendering.** For every episode, states/actions are generated once from
   `RandomState([seed, episode])` and then rendered twice: `clean` (default
   skybox) and `distract` (DAVIS-train video background). MuJoCo physics is
   independent of the skybox, so both arms share bit-identical
   `states.pth`/`actions.pth`. [scripts/collect_data.py](scripts/collect_data.py)
   verifies this and aborts on any deviation. Models A and B therefore differ in
   *pixels only*.
2. **DAVIS split separation.** Training data uses **train**-split videos only.
   The **val** split is reserved for the unseen-background test condition and must
   never appear in training data or in T2/T4.
3. **proprio excludes the target.** `proprio = [qpos(2), qvel(2)]`;
   `state = [qpos(2), qvel(2), finger_xy(2), target_xy(2)]`
   ([env/distracting_reacher.py](env/distracting_reacher.py)). The target position
   exists only in the image. If proprio ever carries the target, or `alpha > 0`
   starts to dominate, the planner can bypass the visual latent and the
   experiment measures nothing.
4. **The goal image solves the task.** `sample_random_init_goal_states` places the
   goal pose *on the target* via numeric IK, so "match the goal image" and
   "succeed" coincide. A random goal pose would decouple the planner's objective
   from the success metric.
5. **Background replacement mechanism.** Skybox texture is overwritten per frame
   and the ground-plane material alpha is set to 0 so the video is visible from
   the overhead camera. Implemented locally rather than via
   google-research/distracting_control, which writes `mjModel.tex_rgb` (removed in
   MuJoCo 3.x; this repo pins mujoco 3.2.7). Color and camera distractions stay
   off: background only.

## Test conditions

| | background | goal vs observation background, at the time the cost compares them |
|---|---|---|
| T1_clean | default skybox | same |
| T2_train_bg | DAVIS **train** video | same frame |
| T3_val_bg | DAVIS **val** video (unseen) | same frame |
| T4_mismatch | DAVIS **train** video | **half the video's length apart** (`--mismatch_frac 0.5`) |

T4 uses train-split videos on purpose: it isolates "goal and observation
backgrounds disagree" from "the background is novel", which is T3's job.

The video advances one frame per env step, so "same background" is defined at
the env step `t_eval` the cost refers to: the goal image is rendered with frame
`obs_offset + t_eval + delta` (`delta` = 0 except T4). `scripts/eval.py`
re-renders the goal every MPC iteration (`t_eval` = end of that iteration's
plan) and in the latent phase (`t_eval` = end of the random rollout). Records
carry `bg_protocol: goal_sync_v1`.

## Metrics and what each one rules out

* **success rate** - fingertip within the target radius (0.05, reacher *easy*).
* **final fingertip-target distance** - graded version of the above, from true state.
* **open-loop latent error per horizon** (`results/rollout/`) - is the *predictor*
  degrading?
* **latent-cost vs true-distance correlation** (`results/latent/`) - is the *cost
  function* degrading? Computed on real observations, with no model prediction
  involved, so it cannot be confounded by predictor error.

Reading the result: if `r` collapses in T3/T4 while per-horizon latent error stays
flat, the failure is the cost function, not prediction. If both degrade, the
predictor is implicated too.

## Recording

Append-only jsonl, one record per episode, flushed and fsynced immediately;
episode seeds are a pure function of `(seed, episode_id)` so resuming never
changes an episode. Every record carries `git_sha`, planner settings, `alpha`,
and the exact background (split / video idx / frame offset) in use.
`scripts/analyze.py` reads only those jsonl files and is safe to re-run anytime.
See [README_experiment.md](README_experiment.md) for the directory layout.

## Upstream code policy

Extend, do not edit. The only modification to upstream DINO-WM is a 3-line
`register(id="distracting_reacher", ...)` block in
[env/\_\_init\_\_.py](env/__init__.py), needed because `plan.py` calls
`gym.make(cfg.env.name)`. Background config is driven through
`SubprocVectorEnv.set_env_attr` (plain attributes) specifically to avoid
patching [env/venv.py](env/venv.py).

## Gotchas worth remembering

* Never render the goal once at frame `offset + 0`. Before `goal_sync_v1`, it was
  rendered that way: T2/T3 were then mismatched by `goal_H * frameskip` = 25
  frames at cost time, and T4's fixed 25-frame shift made it the *matched*
  condition. `eval.py` refuses to resume into jsonl from another protocol.
* Background frames wrap modulo the video length, so a mismatch must be checked
  against the video's *visual* period, not just its length. The synthetic
  source used to repeat every 15 of its 60 frames, which made a half-length T4
  a no-op.
* `conf/plan_distracting_reacher.yaml` sets `planner.max_iter: 3`. Upstream's
  `null` means "loop until every episode succeeds", which never terminates under
  heavy distractors.
* `scripts/train.sh` pins `hydra.run.dir` to `outputs/<exp_name>` (not a
  timestamp) because that is what makes `train.py` resume from
  `model_latest.pth`.
* Feature caching ([scripts/cache_features.py](scripts/cache_features.py)) only
  works with `has_decoder=false` and is off by default.
* The sync daemon uses `rclone copy --min-age 5m`, which is what keeps a
  half-written `torch.save` from being uploaded. Do not remove it.
