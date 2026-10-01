"""usage (cwd = $DINO_WM, env sourced): python render_check.py point_maze pusht wall_single

Check that simulator renders + dynamics match the released datasets.

For each env: load the validation trajectory dataset exactly as plan.py does (but without
image transform), replay the first K raw actions from the dataset's init state in the env,
and compare rendered frames with dataset frames (mean abs pixel diff in [0,255]).
"""
import sys, os
import numpy as np
import torch
import gym
import hydra
import imageio
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
import env  # noqa: registers envs

CKPT = os.environ["DINO_CKPT"]
OUT = os.path.join(os.environ["DINO_WORK"], "setup_check")
os.makedirs(OUT, exist_ok=True)
K = 25

for name in sys.argv[1:]:
    cfg = OmegaConf.load(f"{CKPT}/outputs/{name}/hydra.yaml")
    dcfg = cfg.env.dataset
    dcfg.transform = None
    _, dset = hydra.utils.call(dcfg, num_hist=cfg.num_hist, num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    dset = dset["valid"]
    # raw (un-normalized) actions
    obs, act, state, info = dset[0]
    act_raw = act * dset.action_std + dset.action_mean if cfg.normalize_action else act
    e = gym.make(cfg.env.name, *cfg.env.args, **cfg.env.kwargs)
    e.update_env(info)
    state = state.numpy()
    o0, s0 = e.prepare(0, state[0])
    obs_r, st_r = e.rollout(0, state[0], act_raw[:K].numpy())
    vis_e = obs_r["visual"]  # T H W C uint8-like
    vis_d = (obs["visual"][: K + 1].permute(0, 2, 3, 1).numpy() * 255.0)
    T = min(len(vis_e), len(vis_d))
    diffs = [float(np.abs(vis_e[t].astype(np.float32) - vis_d[t]).mean()) for t in range(T)]
    sdiff = [float(np.abs(st_r[t] - state[t]).max()) for t in range(min(T, len(st_r), len(state)))]
    print(f"[{name}] env={cfg.env.name} env_frame shape={vis_e.shape} dtype={vis_e.dtype} dset_frame shape={vis_d.shape}")
    print(f"[{name}] pixel MAE per t (first 6, last): {[round(d,2) for d in diffs[:6]]} ... {round(diffs[-1],2)}")
    print(f"[{name}] state max-abs-diff per t (first 6, last): {[round(d,4) for d in sdiff[:6]]} ... {round(sdiff[-1],4)}")
    side = np.concatenate([np.concatenate([vis_e[t].astype(np.uint8), vis_d[t].astype(np.uint8)], 1) for t in [0, T // 2, T - 1]], 0)
    imageio.imwrite(f"{OUT}/{name}_env_vs_dset.png", side)
    print(f"[{name}] wrote {OUT}/{name}_env_vs_dset.png (left=env, right=dataset; rows t=0,{T//2},{T-1})")
