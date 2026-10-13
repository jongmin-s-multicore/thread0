"""Re-render the Rope/Granular dataset from another camera, without re-simulating.

The DINO-WM deformable dataset stores the particle positions of every frame (states.pth) and
the arm is parked at its home pose whenever a frame is saved, so a stored state plus a camera
pose determines the image. This script sets each stored state in PyFleX and renders it from
the pose given by env/deformable_env/camera_motion.py. States and actions are therefore
exactly the dataset's; the variants differ in pixels only.

    python scripts/render_camera_data.py --object rope --variant move
    python scripts/render_camera_data.py --object rope --variant move_fast --episodes val --counterfactual

Output: $DATASET_DIR/deformable_cam/<object>/<variant>/
    frames.npy    (N, T, 224, 224, 3) uint8, RGB
    cam_pose.npy  (N, T, 3) float32: azimuth [rad, not wrapped], height, radius
    actions.npy   (N, T, 4) float32, copied from the dataset (unnormalized)
    episodes.npy  (N,) dataset episode index of each row
    meta.json
  with --counterfactual also (for telling camera motion from object motion at evaluation):
    frames_camonly.npy  (N, T-1, ...) state t   seen from the camera of step t+1
    frames_objonly.npy  (N, T-1, ...) state t+1 seen from the camera of step t

Needs PyFleX (source repro/env.sh); runs at roughly 20-35 ms per frame.
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

from env.deformable_env import camera_motion as cm  # noqa: E402


def val_episodes(n, split_ratio=0.9, seed=42):
    """The validation trajectories of datasets.traj_dset.split_traj_datasets."""
    import torch

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return sorted(perm[int(split_ratio * n):])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--object", required=True, choices=["rope", "granular"])
    p.add_argument("--variant", required=True, choices=list(cm.VARIANTS))
    p.add_argument("--episodes", default="all", help="all | val | a:b")
    p.add_argument("--seed", type=int, default=0, help="camera seed")
    p.add_argument("--counterfactual", action="store_true")
    p.add_argument("--src", default=None, help="default: $DATASET_DIR/deformable")
    p.add_argument("--out", default=None, help="default: $DATASET_DIR/deformable_cam")
    p.add_argument("--name", default=None, help="output folder name (default: the variant)")
    return p.parse_args()


def main():
    args = parse_args()
    import torch

    src = Path(args.src or Path(os.environ["DATASET_DIR"]) / "deformable") / args.object
    out = Path(args.out or Path(os.environ["DATASET_DIR"]) / "deformable_cam") / args.object / (args.name or args.variant)
    out.mkdir(parents=True, exist_ok=True)

    states = torch.load(src / "states.pth")  # (N, T, P, 4)
    actions = torch.load(src / "actions.pth").float().numpy()
    n_all, T = states.shape[:2]
    if args.episodes == "all":
        episodes = list(range(n_all))
    elif args.episodes == "val":
        episodes = val_episodes(n_all)
    else:
        a, b = args.episodes.split(":")
        episodes = list(range(int(a), int(b)))
    N = len(episodes)

    from env.deformable_env.FlexEnvWrapper import FlexEnvWrapper
    import pyflex

    env = FlexEnvWrapper(args.object)
    env.seed(0)
    env.reset(save_data=True)
    H, W = env.screenHeight, env.screenWidth
    zero_vel = np.zeros(states.shape[2] * 3, dtype=np.float32)

    def render(state, pose):
        pyflex.set_positions(np.ascontiguousarray(state, dtype=np.float32).reshape(-1))
        pyflex.set_velocities(zero_vel)
        pos, angle = cm.pose_to_pyflex(pose)
        pyflex.set_camPos(pos)
        pyflex.set_camAngle(angle)
        img = env.render()[..., :3][..., ::-1]  # BGR -> RGB, float 0..255
        return np.clip(np.rint(img), 0, 255).astype(np.uint8)

    def memmap(name, n_t):
        return np.lib.format.open_memmap(out / name, mode="w+", dtype=np.uint8, shape=(N, n_t, H, W, 3))

    frames = memmap("frames.npy", T)
    cam_only = memmap("frames_camonly.npy", T - 1) if args.counterfactual else None
    obj_only = memmap("frames_objonly.npy", T - 1) if args.counterfactual else None
    poses = np.zeros((N, T, cm.POSE_DIM), dtype=np.float32)

    t0 = time.time()
    for i, ep in enumerate(episodes):
        st = states[ep].numpy()
        traj = cm.sample_trajectory(args.variant, ep, T, seed=args.seed)
        poses[i] = traj
        for t in range(T):
            frames[i, t] = render(st[t], traj[t])
        if args.counterfactual:
            for t in range(T - 1):
                cam_only[i, t] = render(st[t], traj[t + 1])
                obj_only[i, t] = render(st[t + 1], traj[t])
        if (i + 1) % max(1, N // 20) == 0 or i + 1 == N:
            el = time.time() - t0
            print(f"[render] {i + 1}/{N} episodes  {el:6.0f}s  eta {el / (i + 1) * (N - i - 1):5.0f}s", flush=True)
    frames.flush()

    # how close is a re-render from the dataset camera to the frames the dataset ships?
    fixed_check = None
    if args.variant == "fixed":
        diffs = []
        for i in np.linspace(0, N - 1, min(N, 10)).round().astype(int):
            ref = torch.load(src / f"{episodes[i]:06d}" / "obses.pth").numpy()
            diffs.append(float(np.abs(frames[i].astype(np.float32) - ref).mean()))
        fixed_check = {"episodes_checked": len(diffs), "mean_abs_diff_255": float(np.mean(diffs)), "max": float(np.max(diffs))}
        print("[check] re-render vs dataset obses.pth, mean abs diff /255:", fixed_check, flush=True)

    np.save(out / "cam_pose.npy", poses)
    np.save(out / "actions.npy", actions[episodes])
    np.save(out / "episodes.npy", np.asarray(episodes, dtype=np.int64))
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        sha = None
    meta = {
        "object": args.object,
        "variant": args.variant,
        "variant_spec": cm.VARIANTS[args.variant],
        "camera_seed": args.seed,
        "episodes": args.episodes,
        "n_episodes": N,
        "n_steps": int(T),
        "n_particles": int(states.shape[2]),
        "counterfactual": bool(args.counterfactual),
        "height_range": cm.HEIGHT_RANGE,
        "radius_range": cm.RADIUS_RANGE,
        "azimuth_arc_deg": [float(np.rad2deg(a)) for a in cm.AZIMUTH_ARC],
        "source": f"deformable/{args.object} (states.pth, actions.pth)",
        "fixed_check": fixed_check,
        "git_sha": sha,
        "seconds": round(time.time() - t0, 1),
    }
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(f"[done] {out}  {N} episodes x {T} frames in {meta['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
