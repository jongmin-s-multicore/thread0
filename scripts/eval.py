#!/usr/bin/env python
"""Evaluate trained DINO-WM models under the four background conditions.

Conditions
----------
    T1_clean      default skybox, as in the clean training data
    T2_train_bg   a DAVIS *train* video background (the ones model B saw)
    T3_val_bg     a DAVIS *val* video background (never seen in training)
    T4_mismatch   DAVIS *train* video, but the goal image shows a different frame
                  of that video (--mismatch_frac of its length away) than the
                  observation at the time the cost is evaluated.
                  Train-split videos are used on purpose, so T4 isolates
                  "background differs between goal and observation" from
                  "background is novel" (which is T3).

Goal background timing
----------------------
The video background advances one frame per env step, so "goal and observation
share a background" only means something at a given time. The goal image is
therefore rendered with the background frame that the observation will show at
the moment the cost compares them:

    goal frame = obs offset + t_eval + delta      (delta = 0 except in T4)

t_eval is the env step the compared observation belongs to: the end of the plan
for each MPC iteration (the goal is re-rendered every iteration), the end of
the random rollout in the latent phase. Rendering the goal at offset+0 instead
makes T2/T3 mismatched by t_eval frames and, with t_eval == delta, makes T4 the
matched one.

Phases (any subset, --phases)
-----------------------------
    plan     run the planner, record success / final fingertip-target distance
    rollout  open-loop latent error per prediction horizon (no planning)
    latent   latent-cost vs true-state-distance pairs, for the correlation that
             separates "the predictor is wrong" from "the cost function is wrong"

Everything is written as one JSON object per line, flushed immediately, keyed by
episode id, so an interrupted run resumes by skipping ids already present.
Episode seeds depend only on (--seed, episode id), never on batch size or order.

Examples
--------
python scripts/eval.py --ckpt_base_path $EXP_ROOT --results_dir $EXP_ROOT/results \
    --conditions T1_clean,T2_train_bg,T3_val_bg,T4_mismatch --n_episodes 50 --gpu 0

# two GPUs: split the conditions
python scripts/eval.py ... --conditions T1_clean,T2_train_bg --gpu 0 &
python scripts/eval.py ... --conditions T3_val_bg,T4_mismatch --gpu 1 &
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CONDITIONS = {
    "T1_clean": {"bg_enabled": False, "bg_split": "train", "mismatch": False},
    "T2_train_bg": {"bg_enabled": True, "bg_split": "train", "mismatch": False},
    "T3_val_bg": {"bg_enabled": True, "bg_split": "val", "mismatch": False},
    "T4_mismatch": {"bg_enabled": True, "bg_split": "train", "mismatch": True},
}

PHASES = ("plan", "rollout", "latent")

# Bumped whenever the meaning of a condition changes. Results written under a
# different protocol are refused on resume rather than silently mixed in.
BG_PROTOCOL = "goal_sync_v1"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def git_sha():
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def episode_seed(base_seed, ep):
    """Seed of one episode: a pure function of (base seed, episode id) so that
    resuming, re-batching or re-ordering cannot change an episode."""
    return int(base_seed) * 1000 + int(ep)


def done_episodes(path):
    """Episode ids already recorded in a jsonl file (tolerates a truncated tail)."""
    done = set()
    if not Path(path).exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["episode"]))
            except Exception:
                continue  # half-written last line from a hard kill
    return done


# phases whose numbers depend on the goal image (rollout never renders one)
GOAL_PHASES = ("plan", "latent")


def check_protocol(path, mismatch_frac):
    """Refuse to resume into a jsonl written under another goal-background
    protocol or T4 shift: its episodes would be skipped as done and the file
    would mix two different definitions of the same condition."""
    if not Path(path).exists():
        return
    with open(path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            got = (rec.get("bg_protocol"), rec.get("mismatch_frac"))
            if got != (BG_PROTOCOL, mismatch_frac):
                raise SystemExit(
                    f"{path} was written with bg_protocol/mismatch_frac={got}, "
                    f"this run uses {(BG_PROTOCOL, mismatch_frac)}. Move it aside "
                    "(do not delete: it is a past result) and re-run."
                )
            return


class JsonlWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "a", buffering=1)

    def write(self, record):
        self.f.write(json.dumps(record, default=_json_default) + "\n")
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.f.close()


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def background_plan(cond, seed, video_lens, mismatch_frac=0.5):
    """Per-episode background choice, derived from the episode seed only.

    The goal background is not fixed here: it depends on when the cost is
    evaluated, see goal_bg_cfg().
    """
    spec = CONDITIONS[cond]
    if not spec["bg_enabled"]:
        return {
            "bg_enabled": False,
            "bg_split": spec["bg_split"],
            "obs": {"bg_video_idx": 0, "bg_frame_offset": 0},
            "goal_delta": 0,
            "video_len": None,
        }
    rs = np.random.RandomState(seed + 777)
    video_idx = int(rs.randint(0, max(len(video_lens), 1)))
    offset = int(rs.randint(0, 10_000))
    n_frames = int(video_lens[video_idx])
    delta = 0
    if spec["mismatch"]:
        # same video, a different point in time. A fraction of the video length
        # (not a fixed shift) because frames wrap modulo the length: a fixed
        # shift equal to a short video's length would be no mismatch at all.
        delta = int(round(mismatch_frac * n_frames)) % n_frames
        if delta == 0:
            raise SystemExit(
                f"T4 mismatch is a no-op: mismatch_frac={mismatch_frac} on a "
                f"{n_frames}-frame video (video idx {video_idx})."
            )
    return {
        "bg_enabled": True,
        "bg_split": spec["bg_split"],
        "obs": {"bg_video_idx": video_idx, "bg_frame_offset": offset},
        "goal_delta": delta,
        "video_len": n_frames,
    }


def goal_bg_cfg(bg_plan, t_eval):
    """Goal background for a cost evaluated at env step t_eval: the frame the
    observation shows at that step, shifted by the T4 delta."""
    obs = bg_plan["obs"]
    return {
        "bg_video_idx": obs["bg_video_idx"],
        "bg_frame_offset": obs["bg_frame_offset"] + int(t_eval) + bg_plan["goal_delta"],
    }


def set_bg(env, cond_spec, per_env_cfgs):
    """Push background config into each sub-env via the existing venv setattr
    channel (no change to env/venv.py needed)."""
    for i, cfg in enumerate(per_env_cfgs):
        env.set_env_attr("bg_enabled", bool(cond_spec["bg_enabled"]), id=i)
        env.set_env_attr("bg_split", cond_spec["bg_split"], id=i)
        env.set_env_attr("bg_video_idx", int(cfg["bg_video_idx"]), id=i)
        env.set_env_attr("bg_frame_offset", int(cfg["bg_frame_offset"]), id=i)


# --------------------------------------------------------------------------- #
# model / env construction
# --------------------------------------------------------------------------- #


def load_everything(model_dir, device, n_envs, env_kwargs_override, dset_n_rollout):
    """Load the trained WM, its dataset statistics and a vector env."""
    import gym
    import hydra
    import torch
    from omegaconf import OmegaConf

    import env  # registers the gym ids, including distracting_reacher
    import env.distracting_reacher  # noqa: F401  (self-registers as well)
    from env.venv import SubprocVectorEnv
    from plan import load_model
    from preprocessor import Preprocessor

    model_dir = Path(model_dir)
    cfg_path = model_dir / "hydra.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found. Point --ckpt_base_path at the directory that "
            "holds outputs/<model_name>/ as written by train.py."
        )
    model_cfg = OmegaConf.load(cfg_path)

    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
        n_rollout=dset_n_rollout,
    )
    dset = dset["valid"]

    ckpt = model_dir / "checkpoints" / "model_latest.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}")
    wm = load_model(ckpt, model_cfg, model_cfg.num_action_repeat, device=device)
    wm.eval()

    preprocessor = Preprocessor(
        action_mean=dset.action_mean,
        action_std=dset.action_std,
        state_mean=dset.state_mean,
        state_std=dset.state_std,
        proprio_mean=dset.proprio_mean,
        proprio_std=dset.proprio_std,
        transform=dset.transform,
    )

    env_kwargs = dict(OmegaConf.to_container(model_cfg.env.kwargs, resolve=True))
    env_kwargs.update(env_kwargs_override)
    venv = SubprocVectorEnv(
        [
            (lambda kw=env_kwargs: gym.make(model_cfg.env.name, **kw))
            for _ in range(n_envs)
        ]
    )
    return model_cfg, wm, dset, preprocessor, venv


def goal_sync_mpc_class():
    """MPCPlanner whose goal image is re-rendered every MPC iteration, so the
    goal background matches the frame the plan's last step will see.

    plan() is upstream planning/mpc.py MPCPlanner.plan with one addition (the
    goal_fn call at the top of each iteration); upstream is subclassed, not
    edited. Built lazily so importing this script does not import torch.
    """
    import torch

    from planning.mpc import MPCPlanner
    from utils import slice_trajdict_with_t

    class GoalSyncMPCPlanner(MPCPlanner):
        def __init__(self, goal_fn, **kwargs):
            super().__init__(**kwargs)
            self.goal_fn = goal_fn  # iteration -> (obs_g, goal frame offsets)
            self.goal_offsets = []  # per iteration, per env

        def plan(self, obs_0, obs_g=None, actions=None):
            n_evals = obs_0["visual"].shape[0]
            self.is_success = np.zeros(n_evals, dtype=bool)
            self.action_len = np.full(n_evals, np.inf)
            init_obs_0, init_state_0 = self.evaluator.get_init_cond()

            cur_obs_0 = obs_0
            memo_actions = None
            while not np.all(self.is_success) and self.iter < self.max_iter:
                obs_g, offsets = self.goal_fn(self.iter)
                self.goal_offsets.append(offsets)
                self.evaluator.assign_goal_cond(obs_g, self.evaluator.state_g)

                self.sub_planner.logging_prefix = f"plan_{self.iter}"
                actions, _ = self.sub_planner.plan(
                    obs_0=cur_obs_0, obs_g=obs_g, actions=memo_actions
                )
                taken_actions = actions.detach()[:, : self.n_taken_actions]
                self._apply_success_mask(taken_actions)
                memo_actions = actions.detach()[:, self.n_taken_actions :]
                self.planned_actions.append(taken_actions)

                print(f"MPC iter {self.iter} Eval ------- ")
                action_so_far = torch.cat(self.planned_actions, dim=1)
                self.evaluator.assign_init_cond(obs_0=init_obs_0, state_0=init_state_0)
                logs, successes, e_obses, e_states = self.evaluator.eval_actions(
                    action_so_far,
                    self.action_len,
                    filename=f"plan{self.iter}",
                    save_video=True,
                )
                new_successes = successes & ~self.is_success
                self.is_success = self.is_success | successes
                self.action_len[new_successes] = (self.iter + 1) * self.n_taken_actions

                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": self.iter + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)

                cur_obs_0 = slice_trajdict_with_t(e_obses, start_idx=-1)
                self.evaluator.assign_init_cond(
                    obs_0=cur_obs_0, state_0=e_states[:, -1]
                )
                self.iter += 1

            planned_actions = torch.cat(self.planned_actions, dim=1)
            self.evaluator.assign_init_cond(obs_0=init_obs_0, state_0=init_state_0)
            return planned_actions, self.action_len

    return GoalSyncMPCPlanner


def make_planner(plan_cfg, wm, venv, dset, frameskip, objective_fn, preprocessor,
                 evaluator, log_filename, goal_fn):
    from omegaconf import OmegaConf

    from plan import DummyWandbRun

    planner_cfg = OmegaConf.to_container(plan_cfg["planner"], resolve=True)
    target = planner_cfg.pop("_target_")
    if target != "planning.mpc.MPCPlanner":
        raise SystemExit(
            f"eval.py re-renders the goal per MPC iteration and needs "
            f"planning.mpc.MPCPlanner; the plan config asks for {target}."
        )
    return goal_sync_mpc_class()(
        goal_fn=goal_fn,
        wm=wm,
        env=venv,
        action_dim=dset.action_dim * frameskip,
        objective_fn=objective_fn,
        preprocessor=preprocessor,
        evaluator=evaluator,
        wandb_run=DummyWandbRun(),
        log_filename=log_filename,
        **planner_cfg,
    )


# --------------------------------------------------------------------------- #
# target preparation (shared by all phases)
# --------------------------------------------------------------------------- #


def render_goal(venv, cond, batch, t_eval):
    """Goal image for a cost evaluated at env step t_eval (scalar or one per
    env). Leaves the observation background set, so whatever rolls out next
    sees the observation background. Returns (obs_g, goal frame offsets)."""
    spec = CONDITIONS[cond]
    n = len(batch["seeds"])
    t_eval = np.broadcast_to(np.asarray(t_eval, dtype=int), (n,))
    cfgs = [goal_bg_cfg(p, t) for p, t in zip(batch["bg_plans"], t_eval)]
    set_bg(venv, spec, cfgs)
    obs_g, _ = venv.prepare(batch["seeds"], batch["goal_states"])
    set_bg(venv, spec, [p["obs"] for p in batch["bg_plans"]])
    obs_g = {k: np.expand_dims(v, axis=1) for k, v in obs_g.items()}
    return obs_g, [c["bg_frame_offset"] for c in cfgs]


def prepare_batch(venv, cond, ep_ids, base_seed, video_lens, mismatch_frac, goal_t):
    """Init/goal states and observations for one batch of episodes, with the
    goal rendered for a cost evaluated at env step goal_t."""
    spec = CONDITIONS[cond]
    seeds = [episode_seed(base_seed, ep) for ep in ep_ids]
    bg_plans = [
        background_plan(cond, s, video_lens, mismatch_frac=mismatch_frac)
        for s in seeds
    ]

    init_states, goal_states = venv.sample_random_init_goal_states(seeds)

    # prepare() only to read the goal state back exactly as the env sets it
    set_bg(venv, spec, [p["obs"] for p in bg_plans])
    _, state_g = venv.prepare(seeds, goal_states)
    obs_0, state_0 = venv.prepare(seeds, init_states)
    obs_0 = {k: np.expand_dims(v, axis=1) for k, v in obs_0.items()}

    batch = {
        "seeds": seeds,
        "bg_plans": bg_plans,
        "goal_states": goal_states,
        "obs_0": obs_0,
        "state_0": state_0,
        "state_g": state_g,
    }
    batch["obs_g"], batch["goal_offsets"] = render_goal(venv, cond, batch, goal_t)
    batch["goal_t"] = int(goal_t)
    return batch


def bg_record(bg_plan, goal_offset):
    """Exact background provenance for one episode record."""
    return {
        "protocol": BG_PROTOCOL,
        "enabled": bg_plan["bg_enabled"],
        "split": bg_plan["bg_split"],
        "video_idx": bg_plan["obs"]["bg_video_idx"],
        "video_len": bg_plan["video_len"],
        "obs_offset": bg_plan["obs"]["bg_frame_offset"],
        "goal_delta": bg_plan["goal_delta"],
        "goal_offset": goal_offset,
    }


def encode_obs_np(wm, preprocessor, obs_np, device):
    """numpy obs dict (b, t, H, W, C) -> latent dict, under no_grad."""
    import torch

    from utils import move_to_device

    with torch.no_grad():
        trans = move_to_device(preprocessor.transform_obs(obs_np), device)
        return wm.encode_obs(trans)


# --------------------------------------------------------------------------- #
# phase: plan
# --------------------------------------------------------------------------- #


def run_plan_phase(ctx, cond, ep_ids):
    import torch

    from planning.evaluator import PlanEvaluator
    from utils import slice_trajdict_with_t

    venv, wm, dset, pre = ctx["venv"], ctx["wm"], ctx["dset"], ctx["preprocessor"]
    args, device = ctx["args"], ctx["device"]
    frameskip = ctx["frameskip"]

    planner_cfg = ctx["plan_cfg"]["planner"]
    horizon = int(planner_cfg["sub_planner"]["horizon"])
    n_taken = int(planner_cfg["n_taken_actions"])

    def plan_end_t(mpc_iter):
        # env step of the plan's last imagined frame in this MPC iteration
        return (mpc_iter * n_taken + horizon) * frameskip

    batch = prepare_batch(
        venv, cond, ep_ids, args.seed, ctx["video_lens"], args.mismatch_frac,
        goal_t=plan_end_t(0),
    )

    evaluator = PlanEvaluator(
        obs_0=batch["obs_0"],
        obs_g=batch["obs_g"],
        state_0=batch["state_0"],
        state_g=batch["state_g"],
        env=venv,
        wm=wm,
        frameskip=frameskip,
        seed=batch["seeds"],
        preprocessor=pre,
        n_plot_samples=args.n_plot_samples,
    )

    objective_fn = ctx["objective_fn"]
    planner = make_planner(
        ctx["plan_cfg"],
        wm,
        venv,
        dset,
        frameskip,
        objective_fn,
        pre,
        evaluator,
        log_filename=str(ctx["planner_log"]),
        goal_fn=lambda it: render_goal(venv, cond, batch, plan_end_t(it)),
    )

    t0 = time.time()
    actions, action_len = planner.plan(obs_0=batch["obs_0"], obs_g=batch["obs_g"])
    plan_seconds = time.time() - t0

    logs, successes, e_obses, e_states = evaluator.eval_actions(
        actions.detach(),
        action_len,
        save_video=args.save_videos,
        filename=f"{cond}_eps{ep_ids[0]}",
    )

    # per-episode metrics (evaluator's `logs` are already batch means)
    T = e_states.shape[1]
    last_idx = np.minimum(
        np.where(np.isfinite(action_len), action_len * frameskip, T - 1), T - 1
    ).astype(int)
    e_final_state = e_states[np.arange(e_states.shape[0]), last_idx]
    per_ep = venv.eval_state(batch["state_g"], e_final_state)

    # Each cost below is scored against the goal rendered for the time of the
    # observation it compares, exactly as during planning.
    obs_g_imag, _ = render_goal(venv, cond, batch, actions.shape[1] * frameskip)
    obs_g_reached, goal_offsets_reached = render_goal(venv, cond, batch, last_idx)
    with torch.no_grad():
        from utils import move_to_device

        # what the planner believed: latent cost of its own imagined rollout
        z_g = encode_obs_np(wm, pre, obs_g_imag, device)
        trans_obs_0 = move_to_device(pre.transform_obs(batch["obs_0"]), device)
        i_z_obses, _ = wm.rollout(obs_0=trans_obs_0, act=actions.detach())
        z_last = slice_trajdict_with_t(i_z_obses, start_idx=-1, end_idx=None)
        latent_cost = objective_fn(z_last, z_g).detach().cpu().numpy()
        # cost the *actually reached* observation would score (prediction-free)
        z_g_reached = encode_obs_np(wm, pre, obs_g_reached, device)
        z_reached = encode_obs_np(
            wm,
            pre,
            {k: np.expand_dims(v[np.arange(v.shape[0]), last_idx], 1)
             for k, v in e_obses.items()},
            device,
        )
        latent_cost_reached = objective_fn(z_reached, z_g_reached).detach().cpu().numpy()

    init_dists = np.linalg.norm(
        np.asarray(batch["state_0"])[:, 4:6] - np.asarray(batch["state_0"])[:, 6:8],
        axis=1,
    )

    records = []
    for i, ep in enumerate(ep_ids):
        bgp = batch["bg_plans"][i]
        records.append(
            {
                "episode": int(ep),
                "cond": cond,
                "model": ctx["label"],
                "model_name": ctx["model_name"],
                "seed": args.seed,
                "eval_seed": batch["seeds"][i],
                "success": bool(per_ep["success"][i]),
                "final_fingertip_target_dist": float(per_ep["finger_target_dist"][i]),
                "final_fingertip_goal_dist": float(per_ep["finger_goal_dist"][i]),
                "state_dist": float(per_ep["state_dist"][i]),
                "init_fingertip_target_dist": float(init_dists[i]),
                "latent_cost_imagined": float(latent_cost[i]),
                "latent_cost_reached": float(latent_cost_reached[i]),
                "action_len": (
                    None if not np.isfinite(action_len[i]) else int(action_len[i])
                ),
                "plan_seconds_batch": round(plan_seconds, 2),
                "bg": {
                    **bg_record(bgp, goal_offsets_reached[i]),
                    "goal_offset_per_mpc_iter": [
                        offs[i] for offs in planner.goal_offsets
                    ],
                },
                **ctx["provenance"],
            }
        )
    return records


# --------------------------------------------------------------------------- #
# phase: rollout (open-loop latent error per horizon)
# --------------------------------------------------------------------------- #


def run_rollout_phase(ctx, cond, ep_ids):
    import torch
    from einops import rearrange

    venv, wm, pre = ctx["venv"], ctx["wm"], ctx["preprocessor"]
    args, device = ctx["args"], ctx["device"]
    frameskip, num_hist = ctx["frameskip"], ctx["num_hist"]
    H = args.rollout_horizon

    # no goal is involved in this phase; goal_t only fixes what gets rendered
    batch = prepare_batch(
        venv, cond, ep_ids, args.seed, ctx["video_lens"], args.mismatch_frac,
        goal_t=0,
    )
    n = len(ep_ids)

    # random action sequence in raw env units, long enough for num_hist + H steps
    rs = np.random.RandomState(args.seed + 31337)
    n_wm_steps = num_hist + H
    raw_actions = rs.uniform(
        -1.0, 1.0, size=(n, n_wm_steps * frameskip, ctx["action_dim"])
    ).astype(np.float32)

    e_obses, e_states = venv.rollout(
        batch["seeds"], batch["state_0"], raw_actions
    )

    norm_actions = pre.normalize_actions(torch.from_numpy(raw_actions))
    wm_actions = rearrange(norm_actions, "b (t f) d -> b t (f d)", f=frameskip).to(
        device
    )

    strided = np.arange(0, n_wm_steps + 1) * frameskip
    strided = strided[strided < e_obses["visual"].shape[1]]

    obs_0 = {k: v[:, : num_hist * frameskip : frameskip] for k, v in e_obses.items()}
    with torch.no_grad():
        from utils import move_to_device

        trans_obs_0 = move_to_device(pre.transform_obs(obs_0), device)
        z_pred, _ = wm.rollout(obs_0=trans_obs_0, act=wm_actions)
        z_true = encode_obs_np(
            wm, pre, {k: v[:, strided] for k, v in e_obses.items()}, device
        )

    records = []
    for i, ep in enumerate(ep_ids):
        bgp = batch["bg_plans"][i]
        for h in range(1, H + 1):
            k = num_hist + h - 1  # z index of the h-th predicted step
            if k >= z_pred["visual"].shape[1] or k >= z_true["visual"].shape[1]:
                break
            zv_err = float(
                torch.mean(
                    (z_pred["visual"][i, k] - z_true["visual"][i, k]) ** 2
                ).item()
            )
            zp_err = float(
                torch.mean(
                    (z_pred["proprio"][i, k] - z_true["proprio"][i, k]) ** 2
                ).item()
            )
            frame = min(k * frameskip, e_states.shape[1] - 1)
            records.append(
                {
                    "episode": int(ep),
                    "cond": cond,
                    "model": ctx["label"],
                    "horizon": h,
                    "z_visual_err": zv_err,
                    "z_proprio_err": zp_err,
                    "fingertip_target_dist": float(
                        np.linalg.norm(
                            e_states[i, frame, 4:6] - e_states[i, frame, 6:8]
                        )
                    ),
                    "bg_split": bgp["bg_split"],
                    "bg_enabled": bgp["bg_enabled"],
                    **ctx["provenance"],
                }
            )
    return records


# --------------------------------------------------------------------------- #
# phase: latent (cost function vs ground truth)
# --------------------------------------------------------------------------- #


def run_latent_phase(ctx, cond, ep_ids):
    """For K random action sequences per episode: what the planning cost says vs
    how close the arm actually got. Uses *real* observations, not model
    predictions, so a low correlation indicts the cost function itself."""
    import torch

    venv, wm, pre = ctx["venv"], ctx["wm"], ctx["preprocessor"]
    args, device = ctx["args"], ctx["device"]
    frameskip = ctx["frameskip"]

    # the cost compares the observation after n_steps env steps, so the goal is
    # rendered with the background of that step
    n_steps = args.goal_H * frameskip
    batch = prepare_batch(
        venv, cond, ep_ids, args.seed, ctx["video_lens"], args.mismatch_frac,
        goal_t=n_steps,
    )
    n = len(ep_ids)
    z_g = encode_obs_np(wm, pre, batch["obs_g"], device)
    objective_fn = ctx["objective_fn"]

    rs = np.random.RandomState(args.seed + 991)
    records = []

    for k in range(args.latent_samples):
        raw_actions = rs.uniform(
            -1.0, 1.0, size=(n, n_steps, ctx["action_dim"])
        ).astype(np.float32)
        e_obses, e_states = venv.rollout(
            batch["seeds"], batch["state_0"], raw_actions
        )
        final_obs = {k2: np.expand_dims(v[:, -1], 1) for k2, v in e_obses.items()}
        with torch.no_grad():
            z_final = encode_obs_np(wm, pre, final_obs, device)
            cost = objective_fn(z_final, z_g).detach().cpu().numpy()
            # the visual term alone, i.e. the pure DINOv2 latent L2
            visual_cost = (
                torch.mean(
                    (z_final["visual"] - z_g["visual"]) ** 2, dim=tuple(
                        range(1, z_final["visual"].ndim)
                    )
                )
                .cpu()
                .numpy()
            )
        for i, ep in enumerate(ep_ids):
            st = e_states[i, -1]
            records.append(
                {
                    "episode": int(ep),
                    "sample": int(k),
                    "cond": cond,
                    "model": ctx["label"],
                    "latent_cost": float(cost[i]),
                    "latent_visual_cost": float(visual_cost[i]),
                    "fingertip_target_dist": float(
                        np.linalg.norm(st[4:6] - st[6:8])
                    ),
                    "fingertip_goal_dist": float(
                        np.linalg.norm(st[4:6] - batch["state_g"][i][4:6])
                    ),
                    "state_dist": float(
                        np.linalg.norm(st[:4] - batch["state_g"][i][:4])
                    ),
                    "bg": bg_record(batch["bg_plans"][i], batch["goal_offsets"][i]),
                    **ctx["provenance"],
                }
            )
    return records


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


PHASE_FN = {
    "plan": run_plan_phase,
    "rollout": run_rollout_phase,
    "latent": run_latent_phase,
}
PHASE_SUBDIR = {"plan": "eval", "rollout": "rollout", "latent": "latent"}


def parse_models(spec):
    """'A_clean:reacher_clean,B_distract:reacher_distract' -> [(label, name)]"""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        label, _, name = item.partition(":")
        out.append((label, name or label))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt_base_path", default=os.environ.get("EXP_ROOT", "."),
                   help="contains outputs/<model_name>/")
    p.add_argument("--models",
                   default="A_clean:reacher_clean,B_distract:reacher_distract",
                   help="comma separated label:model_name")
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--phases", default="plan,rollout,latent")
    p.add_argument("--results_dir", default=None,
                   help="default: <ckpt_base_path>/results")
    p.add_argument("--plan_cfg", default=str(REPO_ROOT / "conf" / "plan_distracting_reacher.yaml"))

    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--batch", type=int, default=10, help="episodes planned together (= number of envs)")
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--goal_H", type=int, default=5)
    p.add_argument("--gpu", default="0")

    # planner budget (shrink these for a smoke run)
    p.add_argument("--cem_samples", type=int, default=None)
    p.add_argument("--cem_iters", type=int, default=None, help="CEM opt_steps")
    p.add_argument("--mpc_iters", type=int, default=None, help="MPC max_iter")

    p.add_argument("--rollout_horizon", type=int, default=10)
    p.add_argument("--latent_samples", type=int, default=16)
    p.add_argument("--mismatch_frac", type=float, default=0.5,
                   help="T4: goal background is this fraction of the video length "
                        "away from the observation's frame at cost time "
                        "(0.5 = the farthest frame, since frames wrap)")

    p.add_argument("--save_videos", action="store_true", help="save rollout videos")
    p.add_argument("--keep_videos", default="failure", choices=["failure", "all", "none"])
    p.add_argument("--n_plot_samples", type=int, default=4)
    p.add_argument("--dset_n_rollout", type=int, default=None,
                   help="limit episodes loaded for dataset statistics (speeds up startup)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if os.environ["MUJOCO_GL"] == "egl":
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(args.gpu).split(",")[0])

    # Env workers must be spawned, not forked. Where d4rl is installed,
    # `import env` pulls in d4rl.kitchen -> dm_control, whose import initializes
    # an EGL display in this process; a forked worker inherits it and
    # eglCreateContext fails with EGL_BAD_ALLOC. env/venv.py starts its workers
    # through multiprocessing's default context, so set that here.
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)

    import hydra
    import torch
    from omegaconf import OmegaConf

    from env.distracting_reacher import build_background_sources

    # absolute: the plan phase chdir()s into the video dir while writing these
    results_dir = Path(
        args.results_dir or Path(args.ckpt_base_path) / "results"
    ).resolve()
    # the vector env has a fixed width; never build more envs than episodes
    n_envs = max(1, min(args.batch, args.n_episodes))
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    phases = [ph.strip() for ph in args.phases.split(",") if ph.strip()]
    for c in conditions:
        if c not in CONDITIONS:
            raise SystemExit(f"Unknown condition {c}; known: {list(CONDITIONS)}")
    for ph in phases:
        if ph not in PHASES:
            raise SystemExit(f"Unknown phase {ph}; known: {list(PHASES)}")

    plan_cfg = OmegaConf.load(args.plan_cfg)
    if args.cem_samples is not None:
        OmegaConf.update(plan_cfg, "planner.sub_planner.num_samples", args.cem_samples)
    if args.cem_iters is not None:
        OmegaConf.update(plan_cfg, "planner.sub_planner.opt_steps", args.cem_iters)
    if args.mpc_iters is not None:
        OmegaConf.update(plan_cfg, "planner.max_iter", args.mpc_iters)
    # planning horizon == goal horizon, as plan.py assumes
    OmegaConf.update(plan_cfg, "planner.sub_planner.horizon", args.goal_H)
    OmegaConf.update(plan_cfg, "planner.n_taken_actions", args.goal_H)
    objective_fn = hydra.utils.call(plan_cfg["objective"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    provenance = {
        "git_sha": git_sha(),
        "planner": OmegaConf.to_container(plan_cfg["planner"], resolve=True)["name"],
        "cem_samples": OmegaConf.select(plan_cfg, "planner.sub_planner.num_samples"),
        "cem_opt_steps": OmegaConf.select(plan_cfg, "planner.sub_planner.opt_steps"),
        "mpc_max_iter": OmegaConf.select(plan_cfg, "planner.max_iter"),
        "alpha": OmegaConf.select(plan_cfg, "objective.alpha"),
        "goal_H": args.goal_H,
        "bg_protocol": BG_PROTOCOL,
        "mismatch_frac": args.mismatch_frac,
    }

    for label, model_name in parse_models(args.models):
        model_dir = Path(args.ckpt_base_path) / "outputs" / model_name
        print(f"\n===== model {label} ({model_dir}) =====", flush=True)

        for cond in conditions:
            spec = CONDITIONS[cond]
            model_cfg, wm, dset, pre, venv = load_everything(
                model_dir,
                device,
                n_envs=n_envs,
                env_kwargs_override={
                    "bg_enabled": spec["bg_enabled"],
                    "bg_split": spec["bg_split"],
                },
                dset_n_rollout=args.dset_n_rollout,
            )
            # the split's videos and their lengths, so episode seeds can index
            # them and T4 can size its shift to each video
            if spec["bg_enabled"]:
                video_lens = [
                    len(src)
                    for src in build_background_sources(
                        model_cfg.env.kwargs.get("bg_source", "davis"),
                        spec["bg_split"],
                        model_cfg.env.kwargs.get("davis_path", None),
                    )
                ]
                if not video_lens:
                    raise SystemExit(
                        f"No background videos for split '{spec['bg_split']}'."
                    )
            else:
                video_lens = [1]  # T1 shows no video at all

            ctx = {
                "args": args,
                "device": device,
                "label": label,
                "model_name": model_name,
                "wm": wm,
                "dset": dset,
                "preprocessor": pre,
                "venv": venv,
                "frameskip": model_cfg.frameskip,
                "num_hist": model_cfg.num_hist,
                "action_dim": dset.action_dim,
                "plan_cfg": plan_cfg,
                "objective_fn": objective_fn,
                "video_lens": video_lens,
                "provenance": {**provenance, "model_epoch": "latest"},
                "planner_log": results_dir / "planner" / f"{label}__{cond}.json",
            }
            ctx["planner_log"].parent.mkdir(parents=True, exist_ok=True)

            for phase in phases:
                out_path = (
                    results_dir / PHASE_SUBDIR[phase] / f"{label}__{cond}.jsonl"
                )
                if phase in GOAL_PHASES:
                    check_protocol(out_path, args.mismatch_frac)
                done = done_episodes(out_path)
                todo = [ep for ep in range(args.n_episodes) if ep not in done]
                if not todo:
                    print(f"[{label}/{cond}/{phase}] already complete "
                          f"({len(done)} episodes), skipping")
                    continue
                print(f"[{label}/{cond}/{phase}] {len(todo)} episodes to run "
                      f"({len(done)} already done) -> {out_path}", flush=True)
                writer = JsonlWriter(out_path)
                work_dir = results_dir / "videos" / f"{label}__{cond}"
                work_dir.mkdir(parents=True, exist_ok=True)
                prev_cwd = os.getcwd()
                try:
                    if phase == "plan":
                        os.chdir(work_dir)  # planner/evaluator write plots relative
                    for start in range(0, len(todo), n_envs):
                        chunk = todo[start : start + n_envs]
                        keep = len(chunk)
                        # The vector env width is fixed, so a short final chunk is
                        # padded by repeating earlier ids; their duplicate records
                        # are dropped below.
                        pad = list(chunk)
                        while len(pad) < n_envs:
                            pad.append(todo[(len(pad) - keep) % len(todo)])
                        t0 = time.time()
                        records = PHASE_FN[phase](ctx, cond, pad)
                        kept_ids = set(pad[:keep])
                        emitted = set()
                        for rec in records:
                            key = (rec["episode"], rec.get("horizon"), rec.get("sample"))
                            if rec["episode"] in kept_ids and key not in emitted:
                                emitted.add(key)
                                writer.write(rec)
                        print(
                            f"  [{label}/{cond}/{phase}] episodes {pad[:keep]} "
                            f"in {time.time()-t0:.1f}s",
                            flush=True,
                        )
                finally:
                    os.chdir(prev_cwd)
                    writer.close()

            prune_videos(results_dir / "videos" / f"{label}__{cond}", args.keep_videos)
            venv.close()

    print("\n[done] results under", results_dir)


def prune_videos(video_dir, keep):
    """The evaluator saves a video per episode per MPC iteration; keep only what
    was asked for so a long eval does not fill the disk."""
    video_dir = Path(video_dir)
    if not video_dir.exists():
        return
    for p in video_dir.rglob("*.mp4"):
        if keep == "all":
            continue
        if keep == "none" or "failure" not in p.name:
            try:
                p.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
