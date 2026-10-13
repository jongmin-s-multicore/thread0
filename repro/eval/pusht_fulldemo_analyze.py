"""Per-episode results of PushT MPC runs made through repro/eval/plan_dump_actions.py (e.g. the full-demo runs).

usage (after `source repro/env.sh`): python repro/eval/pusht_fulldemo_analyze.py <out.json> <run_dir> [<run_dir> ...]
Replays each run's planned actions (diag_actions.pkl) in the env from the run's start states (plan_targets.pkl) and
reports per episode:
  success      plan.py criterion: agent+block position within 20 px (4-D norm) and block angle within 20 deg of the
               goal state, checked after every MPC iteration (25 env steps); must agree with the run's own flags
  block_ok     the same thresholds on the block alone (20 px, 20 deg), at the end and at any MPC iteration
  coverage     the env's own overlap of the block with the green T (Diffusion Policy metric; meaningful only when the
               goal is the green T, as in the full-demo goals)
Episodes of several runs are concatenated in the order given. demo_idx / demo_len come from the goal file if present.
"""
import os, sys, json, pickle
import numpy as np

THREAD0 = os.environ["THREAD0"]
sys.path.insert(0, THREAD0)
os.chdir(THREAD0)

import gym
import torch
from einops import rearrange
from omegaconf import OmegaConf
import env  # registers the gym envs
from datasets.pusht_dset import ACTION_MEAN, ACTION_STD

FRAMESKIP = 5
POS_TOL, ANG_TOL = 20.0, np.pi / 9


def ang_diff(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def analyze_run(e, run_dir):
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra/config.yaml"))
    tg = pickle.load(open(os.path.join(run_dir, "plan_targets.pkl"), "rb"))
    src = pickle.load(open(cfg.goal_file_path, "rb")) if cfg.goal_source == "file" else {}
    d = pickle.load(open(os.path.join(run_dir, "diag_actions.pkl"), "rb"))
    n_taken = d["planned_actions"][0].shape[1]
    step = n_taken * FRAMESKIP  # env steps per MPC iteration
    acts = rearrange(d["actions"], "b t (f d) -> b (t f) d", f=FRAMESKIP)
    acts = (acts * ACTION_STD + ACTION_MEAN).numpy()
    rows = []
    for n in range(acts.shape[0]):
        s0, g = tg["state_0"][n], tg["state_g"][n]
        _, state = e.prepare(cfg.seed * n + 1, s0)
        _, _, _, infos = e.step_multiple(acts[n])
        states = np.vstack([state[None], infos["state"]])
        cov = np.concatenate([[np.nan], np.asarray(infos["final_coverage"])])
        solved = bool(d["is_success"][n])
        end = int(d["action_len"][n]) * FRAMESKIP if solved else len(states) - 1
        ks = np.arange(step, end + 1, step)  # states checked by MPC
        pos4 = np.linalg.norm(states[:, :4] - g[:4], axis=1)
        blk = np.linalg.norm(states[:, 2:4] - g[2:4], axis=1)
        ang = ang_diff(states[:, 4], g[4])
        ok = (pos4 < POS_TOL) & (ang < ANG_TOL)
        blk_ok = (blk < POS_TOL) & (ang < ANG_TOL)
        assert ok[ks].any() == solved and (not solved or ok[end]), (run_dir, n, "replay disagrees with the run")
        e.prepare(0, g)
        _, _, _, ginfo = e.step(np.zeros(2))
        rows.append({
            "run": os.path.basename(run_dir), "ep": n,
            "demo_idx": src.get("demo_idx", [None] * (n + 1))[n],
            "demo_len": src.get("demo_len", [None] * (n + 1))[n],
            "success": solved, "steps": end,
            "block_ok_end": bool(blk_ok[end]),
            "block_ok_any_iter": bool(blk_ok[ks].any()),
            "first_block_ok_step": int(ks[blk_ok[ks]][0]) if blk_ok[ks].any() else None,
            "block_dist_start": float(blk[0]), "block_dist_end": float(blk[end]),
            "block_dist_min": float(blk[: end + 1].min()), "block_dist_min_step": int(blk[: end + 1].argmin()),
            "angle_deg_start": float(np.degrees(ang[0])), "angle_deg_end": float(np.degrees(ang[end])),
            "agent_dist_end": float(np.linalg.norm(states[end, :2] - g[:2])),
            "block_moved": float(np.linalg.norm(states[end, 2:4] - states[0, 2:4])),
            "coverage_end": float(cov[end]), "coverage_max": float(np.nanmax(cov[: end + 1])),
            "coverage_goal": float(ginfo["final_coverage"]),
            "success_by_iter": [bool(ok[ks[: i + 1]].any()) for i in range(len(d["planned_actions"]))] if not solved
            else [i + 1 >= end // step for i in range(len(d["planned_actions"]))],
            "block_ok_by_iter": [bool(blk_ok[np.arange(step, min(end, (i + 1) * step) + 1, step)].any())
                                 for i in range(len(d["planned_actions"]))],
        })
    return rows


def main(out_json, run_dirs):
    e = gym.make("pusht", with_velocity=True, with_target=True).unwrapped  # step() directly, no gym wrappers
    rows = [r for rd in run_dirs for r in analyze_run(e, os.path.abspath(rd))]
    n = len(rows)
    n_iter = max(len(r["success_by_iter"]) for r in rows)
    by_iter = lambda key: [float(np.mean([r[key][min(i, len(r[key]) - 1)] for r in rows])) for i in range(n_iter)]
    summary = {
        "n": n,
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "block_ok_end_rate": float(np.mean([r["block_ok_end"] for r in rows])),
        "block_ok_any_iter_rate": float(np.mean([r["block_ok_any_iter"] for r in rows])),
        "coverage_max_mean": float(np.mean([r["coverage_max"] for r in rows])),
        "coverage_end_mean": float(np.mean([r["coverage_end"] for r in rows])),
        "coverage_max_ge_0.95": int(np.sum([r["coverage_max"] >= 0.95 for r in rows])),
        "coverage_goal_mean": float(np.mean([r["coverage_goal"] for r in rows])),
        "success_rate_by_iter": by_iter("success_by_iter"),
        "block_ok_rate_by_iter": by_iter("block_ok_by_iter"),
    }
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "episodes": rows}, f, indent=1, default=lambda o: o.item())  # numpy scalars
    print("demo len | ok steps | blk_ok(end/any) | block dist start->end (min@step) | angle start->end | agent end | cov end/max/goal | moved")
    for r in rows:
        print("%4s %4s | %-5s %4d | %-5s %-5s | %6.1f -> %6.1f (%5.1f @%3d) | %6.1f -> %6.1f | %6.1f | %.2f %.2f %.2f | %6.1f" % (
            r["demo_idx"], r["demo_len"], r["success"], r["steps"], r["block_ok_end"], r["block_ok_any_iter"],
            r["block_dist_start"], r["block_dist_end"], r["block_dist_min"], r["block_dist_min_step"],
            r["angle_deg_start"], r["angle_deg_end"], r["agent_dist_end"],
            r["coverage_end"], r["coverage_max"], r["coverage_goal"], r["block_moved"]))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
