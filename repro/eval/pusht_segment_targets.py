"""PushT goal files with start-goal pairs d env steps apart on validation demos (for plan.py goal_source=file).

As plan.py goal_source=dset: draw a validation demo and an offset, start from the dataset state there, replay the
demo's next d actions in the env; the goal is the state reached. d = 25 is the DINO-WM evaluation (goal_H 5);
larger d gives the same kind of goal further away, and a file lets several planner settings share the same pairs.

usage (after `source repro/env.sh`): python repro/eval/pusht_segment_targets.py <out.pkl> <d> <n> [seed]
Extra keys (ignored by plan.py): demo_idx, demo_offset, demo_len (= d).
"""
import os, sys, pickle
import numpy as np

sys.path.insert(0, os.getcwd())
import gym
import env  # registers the gym envs
from datasets.pusht_dset import PushTDataset, ACTION_MEAN, ACTION_STD

out, d, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 99
val = PushTDataset(data_path=os.path.join(os.environ["DATASET_DIR"], "pusht_noise", "val"))
e = gym.make("pusht", with_velocity=True, with_target=True).unwrapped
rng = np.random.default_rng(seed)
long_enough = [i for i in range(len(val)) if val.get_seq_length(i) >= d + 1]
keys = ("obs_0", "obs_g", "state_0", "state_g", "demo_idx", "demo_offset")
rec = {k: [] for k in keys}
for j in range(n):
    i = int(rng.choice(long_enough))
    off = int(rng.integers(0, val.get_seq_length(i) - d))
    s0 = val.states[i, off].numpy()
    act = (val.actions[i, off : off + d] * ACTION_STD + ACTION_MEAN).numpy()
    obses, states = e.rollout(seed * j + 1, s0, act)
    rec["obs_0"].append({k: v[:1] for k, v in obses.items()})
    rec["obs_g"].append({k: v[-1:] for k, v in obses.items()})
    rec["state_0"].append(s0)
    rec["state_g"].append(states[-1])
    rec["demo_idx"].append(i)
    rec["demo_offset"].append(off)
tg = {
    "obs_0": {k: np.stack([o[k] for o in rec["obs_0"]]) for k in rec["obs_0"][0]},
    "obs_g": {k: np.stack([o[k] for o in rec["obs_g"]]) for k in rec["obs_g"][0]},
    "state_0": np.stack(rec["state_0"]).astype(np.float32),
    "state_g": np.stack(rec["state_g"]).astype(np.float32),
    "gt_actions": None, "goal_H": 5,
    "demo_idx": rec["demo_idx"], "demo_offset": rec["demo_offset"], "demo_len": [d] * n,
}
pickle.dump(tg, open(out, "wb"))
blk = np.linalg.norm(tg["state_g"][:, 2:4] - tg["state_0"][:, 2:4], axis=1)
ang = np.abs((tg["state_g"][:, 4] - tg["state_0"][:, 4] + np.pi) % (2 * np.pi) - np.pi)
print(f"{out}: {n} pairs, d={d}; block moves {blk.mean():.1f} px (median {np.median(blk):.1f}), "
      f"turns {np.degrees(ang).mean():.1f} deg (median {np.degrees(np.median(ang)):.1f}); "
      f"already within 20 px / 20 deg of the goal at the start: {int(((blk < 20) & (ang < np.pi / 9)).sum())}")
