"""PushT goal files where one episode = one whole validation demo (for plan.py goal_source=file).

start = the demo's first state, goal = the state reached by replaying the demo's actions to its end
(same recipe as plan.py goal_source=dset, which replays a 25-step segment; here the segment is the whole demo).

usage (after `source repro/env.sh`): python repro/eval/pusht_fulldemo_targets.py <out_dir> [n_parts]
Writes <out_dir>/targets_all.pkl and, if n_parts > 1, targets_p<k>.pkl (contiguous demo ranges, one per GPU).
Extra keys (ignored by plan.py): demo_idx, demo_len (env steps), ds_state_g (dataset's last state),
replay_err (max |replayed state - dataset state| over the demo, per state dim).
"""
import os, sys, pickle
import numpy as np

THREAD0 = os.environ["THREAD0"]
sys.path.insert(0, THREAD0)
os.chdir(THREAD0)

import gym
import hydra
import torch
from omegaconf import OmegaConf
import env  # registers the gym envs
from preprocessor import Preprocessor

SEED = 99  # plan.py default; env seed of episode n is SEED * n + 1


def main(out_dir, n_parts):
    os.makedirs(out_dir, exist_ok=True)
    model_cfg = OmegaConf.load(os.path.join(os.environ["DINO_CKPT"], "outputs/pusht/hydra.yaml"))
    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = dset["valid"]
    pre = Preprocessor(
        action_mean=dset.action_mean, action_std=dset.action_std,
        state_mean=dset.state_mean, state_std=dset.state_std,
        proprio_mean=dset.proprio_mean, proprio_std=dset.proprio_std,
        transform=dset.transform,
    )
    e = gym.make(model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs)

    out = {k: [] for k in ("obs_0", "obs_g", "state_0", "state_g", "demo_idx", "demo_len", "ds_state_g", "replay_err")}
    for i in range(len(dset)):
        T = dset.get_seq_length(i)
        act, state = dset.actions[i, : T - 1], dset.states[i, :T].numpy()
        e.update_env({"shape": dset.shapes[i]})
        obses, states = e.rollout(SEED * i + 1, state[0], pre.denormalize_actions(act).numpy())
        out["obs_0"].append({k: v[:1] for k, v in obses.items()})
        out["obs_g"].append({k: v[-1:] for k, v in obses.items()})
        out["state_0"].append(state[0])
        out["state_g"].append(states[-1])
        out["demo_idx"].append(i)
        out["demo_len"].append(T - 1)
        out["ds_state_g"].append(state[-1])
        d = np.abs(states - state)
        d[:, 4] = np.minimum(d[:, 4] % (2 * np.pi), 2 * np.pi - d[:, 4] % (2 * np.pi))
        out["replay_err"].append(d.max(axis=0))

    def pack(idx):
        return {
            "obs_0": {k: np.stack([out["obs_0"][j][k] for j in idx]) for k in out["obs_0"][0]},
            "obs_g": {k: np.stack([out["obs_g"][j][k] for j in idx]) for k in out["obs_g"][0]},
            "state_0": np.stack([out["state_0"][j] for j in idx]).astype(np.float32),
            "state_g": np.stack([out["state_g"][j] for j in idx]).astype(np.float32),
            "gt_actions": None,
            "goal_H": 5,
            "demo_idx": [out["demo_idx"][j] for j in idx],
            "demo_len": [out["demo_len"][j] for j in idx],
            "ds_state_g": np.stack([out["ds_state_g"][j] for j in idx]),
            "replay_err": np.stack([out["replay_err"][j] for j in idx]),
        }

    n = len(dset)
    parts = {"all": list(range(n))}
    if n_parts > 1:
        for k, idx in enumerate(np.array_split(np.arange(n), n_parts)):
            parts[f"p{k}"] = idx.tolist()
    for name, idx in parts.items():
        with open(os.path.join(out_dir, f"targets_{name}.pkl"), "wb") as f:
            pickle.dump(pack(idx), f)
        print(f"targets_{name}.pkl: demos {idx[0]}..{idx[-1]} ({len(idx)})")

    a = pack(parts["all"])
    g, s0 = a["state_g"], a["state_0"]
    ang = lambda x: np.abs((x + np.pi) % (2 * np.pi) - np.pi)
    print("demo  len  start block->green (px, deg)   goal block->green (px, deg)   replay-vs-dataset max err (block px, deg)")
    for j in range(n):
        print("%4d %4d   %6.1f %6.1f   %6.1f %6.1f   %6.2f %6.2f" % (
            j, a["demo_len"][j],
            np.linalg.norm(s0[j, 2:4] - 256), np.degrees(ang(s0[j, 4] - np.pi / 4)),
            np.linalg.norm(g[j, 2:4] - 256), np.degrees(ang(g[j, 4] - np.pi / 4)),
            np.linalg.norm(a["replay_err"][j, 2:4]), np.degrees(a["replay_err"][j, 4])))


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 1)
