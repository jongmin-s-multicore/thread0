#!/usr/bin/env python
"""Generate the paired clean/distract DMC Reacher dataset.

The experiment hinges on both arms sharing *identical* trajectories, so the
pipeline is strictly two-phase:

  1. plan_trajectories(): for every episode, derive the initial state, the target
     and the whole action sequence from np.random.RandomState(seed, episode).
     MuJoCo physics does not depend on what the skybox looks like, so the states
     that follow are identical for both renders.
  2. render_episode(): replay that action sequence twice, once with the default
     background ("clean") and once with a DAVIS-train video background
     ("distract"), writing both into DINO-WM dataset layout.

After generation both arms' states.pth are hashed and compared; the hashes land in
meta.json and a mismatch is a hard error.

Examples
--------
# on the A100 instance (EGL, 16 render workers)
python scripts/collect_data.py --out_dir $DATASET_DIR/distracting_reacher \
    --num_episodes 500 --episode_len 100 --num_workers 16 --seed 0

# locally, no GPU and no DAVIS download
python scripts/collect_data.py --out_dir /tmp/dr --num_episodes 3 --episode_len 20 \
    --mujoco_gl osmesa --bg_source synthetic --num_workers 1
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ARMS = ("clean", "distract")


# --------------------------------------------------------------------------- #
# trajectory planning (pure numpy, no MuJoCo: cheap and fully reproducible)
# --------------------------------------------------------------------------- #


def episode_rng(seed, episode_idx):
    """One independent stream per episode, so episodes can be rendered in any
    order / by any worker and still come out identical."""
    return np.random.RandomState(np.array([seed, episode_idx], dtype=np.uint32))


def plan_trajectories(args, n_bg_videos):
    """Per-episode init state, target and action sequence. No rendering here."""
    from env.distracting_reacher import STATE_DIM, TARGET_RADIUS_RANGE, WRIST_LIMIT

    plans = []
    for ep in range(args.num_episodes):
        rng = episode_rng(args.seed, ep)

        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(*TARGET_RADIUS_RANGE)
        init = np.zeros(STATE_DIM, dtype=np.float32)
        init[0] = rng.uniform(-np.pi, np.pi)
        init[1] = rng.uniform(-WRIST_LIMIT, WRIST_LIMIT)
        init[2:4] = 0.0
        init[6] = radius * np.sin(angle)
        init[7] = radius * np.cos(angle)

        # Smoothed random policy: an OU-like walk explores far more of the state
        # space than i.i.d. uniform torques, which mostly jitter in place.
        actions = np.zeros((args.episode_len, 2), dtype=np.float32)
        a = rng.uniform(-1, 1, size=2)
        rho = args.action_smoothing
        for t in range(args.episode_len):
            noise = rng.uniform(-1, 1, size=2)
            a = rho * a + np.sqrt(max(1.0 - rho * rho, 0.0)) * noise
            actions[t] = np.clip(a, -1.0, 1.0)

        plans.append(
            {
                "episode": ep,
                "init_state": init,
                "actions": actions,
                # background assignment, drawn from the SAME stream so it is
                # reproducible and recorded below
                "bg_video_idx": int(rng.randint(0, max(n_bg_videos, 1))),
                "bg_frame_offset": int(rng.randint(0, 10_000)),
                "env_seed": int(rng.randint(0, 2**31 - 1)),
            }
        )
    return plans


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

_ENV_CACHE = {}


def get_env(arm, args):
    """One env per (process, arm). Building a dm_control env is expensive."""
    if arm in _ENV_CACHE:
        return _ENV_CACHE[arm]
    from env.distracting_reacher import DistractingReacherWrapper

    env = DistractingReacherWrapper(
        difficulty=args.difficulty,
        img_size=args.img_size,
        action_repeat=args.action_repeat,
        bg_enabled=(arm == "distract"),
        bg_source=args.bg_source if arm == "distract" else "none",
        bg_split="train",  # training data NEVER sees the val backgrounds
        bg_dynamic=not args.bg_static,
        ground_plane_alpha=args.ground_plane_alpha,
        davis_path=args.davis_path,
        n_synthetic=args.n_synthetic,
    )
    _ENV_CACHE[arm] = env
    return env


def write_video(path, frames, fps, crf):
    import imageio

    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        ffmpeg_params=["-crf", str(crf), "-pix_fmt", "yuv420p"],
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def render_episode(task):
    """Render one episode for both arms. Runs inside a worker process."""
    plan, args, out_dirs = task
    import torch

    ep = plan["episode"]
    result = {"episode": ep, "states": {}, "bg": {}}

    for arm in ARMS:
        env = get_env(arm, args)
        if arm == "distract":
            env.bg_video_idx = plan["bg_video_idx"]
            env.bg_frame_offset = plan["bg_frame_offset"]
        obses, states = env.rollout(
            plan["env_seed"], plan["init_state"], plan["actions"]
        )
        # keep T frames so obs/action/state lengths match (DINO-WM convention)
        visual = obses["visual"][: args.episode_len]
        states = states[: args.episode_len]

        obs_dir = Path(out_dirs[arm]) / "obses"
        if args.obs_format == "mp4":
            write_video(obs_dir / f"episode_{ep:04d}.mp4", visual, args.fps, args.crf)
        else:
            torch.save(
                torch.from_numpy(np.ascontiguousarray(visual)),
                obs_dir / f"episode_{ep:04d}.pth",
            )

        result["states"][arm] = states.astype(np.float32)
        result["bg"][arm] = env.current_background_info()
        result[f"mean_pixel_{arm}"] = float(visual.mean())

    # the two renders must describe the same physics
    sc, sd = result["states"]["clean"], result["states"]["distract"]
    max_dev = float(np.abs(sc - sd).max()) if sc.shape == sd.shape else float("inf")
    result["state_max_dev"] = max_dev
    # how different do the pixels actually look? guards against a silently
    # no-op background replacement
    result["pixel_gap"] = abs(
        result["mean_pixel_clean"] - result["mean_pixel_distract"]
    )
    return result


def _init_worker(args, worker_id):
    os.environ["MUJOCO_GL"] = args.mujoco_gl
    if args.mujoco_gl == "egl":
        gpus = [g.strip() for g in str(args.gpu_ids).split(",") if g.strip()]
        os.environ["MUJOCO_EGL_DEVICE_ID"] = gpus[worker_id % len(gpus)]
    # keep per-worker BLAS threads from oversubscribing the box
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")


_WORKER_COUNTER = {"n": 0}


def _pool_init(args):
    import multiprocessing

    name = multiprocessing.current_process().name
    try:
        worker_id = int(name.rsplit("-", 1)[-1]) - 1
    except ValueError:
        worker_id = 0
    _init_worker(args, worker_id)


# --------------------------------------------------------------------------- #
# disk accounting
# --------------------------------------------------------------------------- #


def estimate_bytes(args):
    """Rough expected size for both arms together."""
    n_frames = args.num_episodes * args.episode_len
    raw = n_frames * args.img_size * args.img_size * 3
    if args.obs_format == "pth":
        per_arm = raw
    else:
        # h264 at crf ~10 on 224x224: ~8-20 kB/frame. Use the pessimistic end.
        per_arm = n_frames * 20_000
    tabular = args.num_episodes * args.episode_len * (8 + 2) * 4 * 2
    return 2 * per_arm + tabular


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def check_disk(out_dir, needed, force):
    usage = shutil.disk_usage(out_dir)
    print(f"[disk] estimated dataset size : {human(needed)}")
    print(f"[disk] free on {out_dir}      : {human(usage.free)}")
    if usage.free < needed * 1.2:
        msg = (
            f"Not enough free disk: need ~{human(needed * 1.2)} "
            f"(estimate + 20% margin), have {human(usage.free)}."
        )
        if not force:
            raise SystemExit(f"[disk] ABORT. {msg}\nRe-run with --force to override.")
        print(f"[disk] WARNING: {msg} Continuing because --force was given.")


# --------------------------------------------------------------------------- #
# verification grid
# --------------------------------------------------------------------------- #


def save_sample_grid(out_root, args, episodes, n_rows=4, n_cols=4):
    """One PNG with clean/distract pairs, so the render can be eyeballed."""
    from PIL import Image
    import torch

    rng = np.random.RandomState(args.seed)
    picks = rng.choice(len(episodes), size=min(n_rows, len(episodes)), replace=False)
    rows = []
    for ep in sorted(int(e) for e in picks):
        frame_ids = np.linspace(0, args.episode_len - 1, n_cols).astype(int)
        for arm in ARMS:
            frames = []
            if args.obs_format == "mp4":
                import imageio

                reader = imageio.get_reader(
                    str(out_root / arm / "obses" / f"episode_{ep:04d}.mp4")
                )
                all_frames = [f for f in reader]
                reader.close()
                frames = [all_frames[i] for i in frame_ids]
            else:
                vid = torch.load(
                    out_root / arm / "obses" / f"episode_{ep:04d}.pth"
                ).numpy()
                frames = [vid[i] for i in frame_ids]
            rows.append(np.concatenate(frames, axis=1))
    grid = np.concatenate(rows, axis=0)
    samples_dir = out_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    path = samples_dir / "grid_clean_vs_distract.png"
    Image.fromarray(grid).save(path)
    print(f"[grid] wrote {path}  (row pairs = clean / distract of the same episode)")
    return path


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", required=True, help="dataset root; clean/ and distract/ are created inside")
    p.add_argument("--num_episodes", type=int, default=500)
    p.add_argument("--episode_len", type=int, default=100, help="stored frames (= actions) per episode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--action_repeat", type=int, default=2, help="physics steps per stored frame")
    p.add_argument("--action_smoothing", type=float, default=0.8, help="random-policy AR(1) coefficient")
    p.add_argument("--difficulty", default="easy", choices=["easy", "hard"])

    p.add_argument("--mujoco_gl", default="egl", choices=["egl", "osmesa", "glfw"])
    p.add_argument("--gpu_ids", default="0", help="comma separated; workers round-robin over these for MUJOCO_EGL_DEVICE_ID")
    p.add_argument("--num_workers", type=int, default=1)

    p.add_argument("--bg_source", default="davis", choices=["davis", "synthetic"])
    p.add_argument("--davis_path", default=os.environ.get("DAVIS_PATH"))
    p.add_argument("--n_synthetic", type=int, default=30)
    p.add_argument("--bg_static", action="store_true", help="freeze the background frame within an episode")
    p.add_argument("--ground_plane_alpha", type=float, default=0.0, help="0 = floor fully transparent, so the video shows")

    p.add_argument("--obs_format", default="mp4", choices=["mp4", "pth"])
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--crf", type=int, default=10, help="h264 quality, lower = better")

    p.add_argument("--force", action="store_true", help="ignore the disk-space check")
    p.add_argument("--overwrite", action="store_true", help="delete an existing dataset first")
    p.add_argument("--skip_grid", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ["MUJOCO_GL"] = args.mujoco_gl
    if args.mujoco_gl == "egl":
        os.environ.setdefault(
            "MUJOCO_EGL_DEVICE_ID", str(args.gpu_ids).split(",")[0].strip()
        )

    out_root = Path(args.out_dir)
    if out_root.exists() and args.overwrite:
        print(f"[init] removing existing {out_root}")
        shutil.rmtree(out_root)
    out_dirs = {}
    for arm in ARMS:
        d = out_root / arm
        (d / "obses").mkdir(parents=True, exist_ok=True)
        out_dirs[arm] = str(d)

    check_disk(out_root, estimate_bytes(args), args.force)

    # how many background videos does the train split offer?
    from env.distracting_reacher import build_background_sources, STATE_KEYS

    bg_sources = build_background_sources(
        args.bg_source, "train", args.davis_path, args.n_synthetic
    )
    n_bg = len(bg_sources)
    print(f"[bg] {n_bg} background videos in the DAVIS/synthetic TRAIN split")
    if n_bg == 0:
        raise SystemExit("[bg] ABORT: no background videos found.")

    plans = plan_trajectories(args, n_bg)
    print(
        f"[plan] {len(plans)} episodes x {args.episode_len} frames, "
        f"seed={args.seed}, action_repeat={args.action_repeat}"
    )

    tasks = [(plan, args, out_dirs) for plan in plans]
    results = []
    t0 = time.time()
    frames_per_ep = args.episode_len * len(ARMS)

    def report(i):
        done = i + 1
        elapsed = time.time() - t0
        fps = done * frames_per_ep / max(elapsed, 1e-9)
        eta = (len(tasks) - done) * frames_per_ep / max(fps, 1e-9)
        print(
            f"[render] {done}/{len(tasks)} episodes  "
            f"{fps:7.1f} frames/s  elapsed {elapsed/60:5.1f}m  ETA {eta/60:5.1f}m",
            flush=True,
        )

    if args.num_workers <= 1:
        _init_worker(args, 0)
        for i, task in enumerate(tasks):
            results.append(render_episode(task))
            if (i + 1) % max(1, len(tasks) // 50) == 0 or i == len(tasks) - 1:
                report(i)
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")  # EGL contexts do not survive fork
        with ctx.Pool(
            processes=args.num_workers, initializer=_pool_init, initargs=(args,)
        ) as pool:
            for i, res in enumerate(pool.imap_unordered(render_episode, tasks)):
                results.append(res)
                if (i + 1) % max(1, len(tasks) // 50) == 0 or i == len(tasks) - 1:
                    report(i)

    results.sort(key=lambda r: r["episode"])

    # ---- consistency checks -------------------------------------------------
    worst_dev = max(r["state_max_dev"] for r in results)
    if worst_dev > 0.0:
        raise SystemExit(
            f"[verify] ABORT: clean and distract states differ (max {worst_dev:.3g}). "
            "The two arms must share identical physics."
        )
    print("[verify] clean/distract state trajectories are bit-identical")

    min_gap = min(r["pixel_gap"] for r in results)
    print(f"[verify] smallest clean/distract mean-pixel gap: {min_gap:.3f}")
    if min_gap < 1.0:
        print(
            "[verify] WARNING: the two renders look nearly identical. The background "
            "replacement may be a no-op (check the skybox/ground-plane handling)."
        )

    # ---- tabular data -------------------------------------------------------
    import torch

    states = torch.from_numpy(np.stack([r["states"]["clean"] for r in results]))
    actions = torch.from_numpy(np.stack([p["actions"] for p in plans]))
    seq_lengths = torch.full((len(results),), args.episode_len, dtype=torch.long)

    hashes = {}
    for arm in ARMS:
        d = Path(out_dirs[arm])
        torch.save(states, d / "states.pth")
        torch.save(actions, d / "actions.pth")
        torch.save(seq_lengths, d / "seq_lengths.pth")
        hashes[arm] = hashlib.sha256(
            (d / "states.pth").read_bytes()
        ).hexdigest()[:16]
    if hashes["clean"] != hashes["distract"]:
        raise SystemExit(f"[verify] ABORT: states.pth hash mismatch {hashes}")

    # ---- metadata -----------------------------------------------------------
    common = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": git_sha(),
        "seed": args.seed,
        "num_episodes": args.num_episodes,
        "episode_len": args.episode_len,
        "img_size": args.img_size,
        "action_repeat": args.action_repeat,
        "action_smoothing": args.action_smoothing,
        "difficulty": args.difficulty,
        "obs_format": args.obs_format,
        "fps": args.fps,
        "crf": args.crf,
        "bg_source": args.bg_source,
        "bg_split": "train",
        "n_bg_videos": n_bg,
        "state_keys": list(STATE_KEYS),
        "states_sha256_16": hashes["clean"],
        "mujoco_gl": args.mujoco_gl,
    }
    for arm in ARMS:
        meta = dict(common)
        meta["arm"] = arm
        meta["backgrounds"] = {
            str(r["episode"]): r["bg"][arm] for r in results
        }
        (Path(out_dirs[arm]) / "meta.json").write_text(json.dumps(meta, indent=2))

    meta_sha = hashlib.sha256(
        json.dumps(common, sort_keys=True).encode()
    ).hexdigest()[:16]
    (out_root / "meta.json").write_text(
        json.dumps({**common, "data_meta_sha": meta_sha}, indent=2)
    )

    if not args.skip_grid:
        save_sample_grid(out_root, args, results)

    total = time.time() - t0
    print(
        f"[done] {len(results)} episodes x 2 arms in {total/60:.1f} min  "
        f"({len(results)*frames_per_ep/max(total,1e-9):.1f} frames/s)"
    )
    print(f"[done] dataset root: {out_root}   data_meta_sha={meta_sha}")


if __name__ == "__main__":
    main()
