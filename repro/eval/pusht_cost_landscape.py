"""Planning cost to the goal along the demo vs along the MPC rollout (full-demo PushT runs).

The planner minimizes objective(z(final predicted frame), z(goal)) = visual latent MSE + alpha * proprio latent MSE
(plan_pusht.yaml: alpha 1). Here the same cost is evaluated on REAL frames (env renders, no world-model prediction):
  demo  the validation demo replayed in the env (the known solution), every 5 env steps
  mpc   the run's executed actions (diag_actions.pkl of repro/eval/plan_dump_actions.py) replayed, every 5 env steps
so it shows what the objective itself prefers, independent of model error and of the CEM search.

usage (after `source repro/env.sh`): CUDA_VISIBLE_DEVICES=<gpu> python repro/eval/pusht_cost_landscape.py <out_dir> <run_dir> [<run_dir> ...]
Writes <out_dir>/cost_landscape.json and cost_landscape.png.
"""
import os, sys, json, pickle
import numpy as np

THREAD0 = os.environ["THREAD0"]
sys.path.insert(0, THREAD0)
os.chdir(THREAD0)

import gym
import torch
import hydra
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf
import env  # registers the gym envs
from plan import load_model
from preprocessor import Preprocessor
from utils import move_to_device

out_dir = os.path.abspath(sys.argv[1])
run_dirs = [os.path.abspath(p) for p in sys.argv[2:]]
model_path = Path(os.environ["DINO_CKPT"]) / "outputs" / "pusht"
cfg = OmegaConf.load(model_path / "hydra.yaml")
_, dset = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred, frameskip=cfg.frameskip)
dset = dset["valid"]
pre = Preprocessor(
    action_mean=dset.action_mean, action_std=dset.action_std, state_mean=dset.state_mean, state_std=dset.state_std,
    proprio_mean=dset.proprio_mean, proprio_std=dset.proprio_std, transform=dset.transform,
)
device = "cuda"
model = load_model(model_path / "checkpoints" / "model_latest.pth", cfg, cfg.num_action_repeat, device)
model.eval()
e = gym.make("pusht", with_velocity=True, with_target=True).unwrapped
FS = cfg.frameskip


def encode(obs):
    """obs: {"visual": (n, H, W, C) uint8, "proprio": (n, 4)} -> latents with a time dim of 1"""
    out = {"visual": [], "proprio": []}
    with torch.no_grad():
        for i in range(0, len(obs["visual"]), 32):
            o = {k: v[i : i + 32, None] for k, v in obs.items()}
            z = model.encode_obs(move_to_device(pre.transform_obs(o), device))
            for k in out:
                out[k].append(z[k])
    return {k: torch.cat(v) for k, v in out.items()}


def costs(z, z_g):
    """(visual MSE, proprio MSE) of every frame in z against the single goal latent z_g"""
    vis = ((z["visual"] - z_g["visual"]) ** 2).flatten(1).mean(1).cpu().numpy()
    prop = ((z["proprio"] - z_g["proprio"]) ** 2).flatten(1).mean(1).cpu().numpy()
    return vis, prop


def ang_diff(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


rows = []
for rd in run_dirs:
    rcfg = OmegaConf.load(os.path.join(rd, ".hydra/config.yaml"))
    alpha = rcfg.objective.alpha
    tg = pickle.load(open(os.path.join(rd, "plan_targets.pkl"), "rb"))
    src = pickle.load(open(rcfg.goal_file_path, "rb"))
    d = pickle.load(open(os.path.join(rd, "diag_actions.pkl"), "rb"))
    acts = rearrange(d["actions"], "b t (f d) -> b (t f) d", f=FS)
    acts = pre.denormalize_actions(acts).numpy()
    for n in range(acts.shape[0]):
        i, g = src["demo_idx"][n], tg["state_g"][n]
        z_g = encode({k: v[n] for k, v in tg["obs_g"].items()})
        T = dset.get_seq_length(i)
        traj = {
            "demo": e.rollout(rcfg.seed * n + 1, tg["state_0"][n],
                              pre.denormalize_actions(dset.actions[i, : T - 1]).numpy()),
            "mpc": e.rollout(rcfg.seed * n + 1, tg["state_0"][n], acts[n]),
        }
        row = {"run": os.path.basename(rd), "ep": n, "demo_idx": int(i), "alpha": float(alpha)}
        for name, (obses, states) in traj.items():
            idx = np.unique(np.concatenate([np.arange(0, len(states), FS), [len(states) - 1]]))
            vis, prop = costs(encode({k: v[idx] for k, v in obses.items()}), z_g)
            row[name] = {
                "env_step": idx.tolist(), "visual": vis.tolist(), "proprio": prop.tolist(),
                "cost": (vis + alpha * prop).tolist(),
                "block_dist": np.linalg.norm(states[idx, 2:4] - g[2:4], axis=1).tolist(),
                "block_angle_deg": np.degrees(ang_diff(states[idx, 4], g[4])).tolist(),
                "agent_dist": np.linalg.norm(states[idx, :2] - g[:2], axis=1).tolist(),
            }
        rows.append(row)
        print(f"{row['run']} ep {n} (demo {i}): start {row['demo']['cost'][0]:.3f}, "
              f"mpc end {row['mpc']['cost'][-1]:.3f}", flush=True)

# demo cost at fixed fractions of the demo, and how the MPC end state ranks against the demo's own path
fracs = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
at = lambda tr, key, f: float(np.interp(f * tr["env_step"][-1], tr["env_step"], tr[key]))
summary = {"n": len(rows), "fractions": fracs}
for key in ("cost", "visual", "proprio", "block_dist", "block_angle_deg", "agent_dist"):
    summary[f"demo_{key}"] = [float(np.mean([at(r["demo"], key, f) for r in rows])) for f in fracs]
    summary[f"mpc_end_{key}"] = float(np.mean([r["mpc"][key][-1] for r in rows]))
    summary[f"mpc_min_{key}"] = float(np.mean([np.min(r["mpc"][key]) for r in rows]))
# last point of the demo (as a fraction of its length) where its cost is still above the MPC end cost
last_above = []
for r in rows:
    c, s = np.array(r["demo"]["cost"]), np.array(r["demo"]["env_step"])
    above = np.where(c > r["mpc"]["cost"][-1])[0]
    last_above.append(float(s[above[-1]] / s[-1]) if len(above) else 0.0)
summary["demo_frac_until_cost_below_mpc_end"] = {"mean": float(np.mean(last_above)), "min": float(np.min(last_above)),
                                                 "max": float(np.max(last_above)), "per_demo": last_above}
# does following the demo for 25 env steps lower the cost? (what a 25-step planner can see)
win = []
for r in rows:
    c, s = np.array(r["demo"]["cost"]), np.array(r["demo"]["env_step"])
    for a in range(len(s)):
        b = np.where(s == s[a] + 25)[0]
        if len(b):
            win.append((s[a] / s[-1], c[b[0]] - c[a]))
win = np.array(win)
summary["demo_25step_cost_change"] = {
    "n_windows": int(len(win)), "frac_decreasing": float(np.mean(win[:, 1] < 0)),
    "frac_decreasing_first_half": float(np.mean(win[win[:, 0] < 0.5, 1] < 0)),
    "frac_decreasing_second_half": float(np.mean(win[win[:, 0] >= 0.5, 1] < 0)),
}
os.makedirs(out_dir, exist_ok=True)
json.dump({"summary": summary, "episodes": rows}, open(os.path.join(out_dir, "cost_landscape.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 7, figsize=(17.5, 7.2), sharey=True)
for ax, r in zip(axes.flat, rows):
    ax.plot(r["demo"]["env_step"], r["demo"]["cost"], color="tab:blue", label="demo (known solution)")
    ax.plot(r["mpc"]["env_step"], r["mpc"]["cost"], color="tab:red", label="MPC rollout")
    ax.set_title(f"demo {r['demo_idx']}", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel("env step", fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("cost to goal (real frames)", fontsize=8)
axes[0, 0].legend(fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "cost_landscape.png"), dpi=130)
