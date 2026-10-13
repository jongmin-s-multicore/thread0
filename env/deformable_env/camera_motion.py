"""Camera poses and trajectories for the moving-camera experiment on Rope and Granular.

This module uses numpy only. It is imported by the renderer and by the evaluation script.

A pose is (azimuth [rad], height, radius). The camera position is

    (radius * sin(azimuth), height, radius * cos(azimuth))

with yaw = azimuth and pitch = -atan2(height, radius + LOOK_PAST).

The dataset cameras are at (+-6, 10, +-6) with pitch -45 deg
(env/deformable_env/src/sim/sim_env/cameras.py). LOOK_PAST = 10 - 6*sqrt(2) reproduces
these poses. The dataset frames (obses.pth) use camera index 1: (6, 10, -6), azimuth 135 deg.
"""

import numpy as np

DATASET_AZIMUTH = np.deg2rad(135.0)
DATASET_HEIGHT = 10.0
DATASET_RADIUS = 6.0 * np.sqrt(2.0)
LOOK_PAST = DATASET_HEIGHT - DATASET_RADIUS

HEIGHT_RANGE = (7.0, 13.0)
RADIUS_RANGE = (6.5, 10.5)
# The xArm is at azimuth ~270 deg. Between about 240 and 300 deg the arm occludes the table,
# so this range is excluded. The four dataset cameras (45, 135, 225, 315 deg) are in range.
AZIMUTH_ARC = (np.deg2rad(-55.0), np.deg2rad(235.0))

# kind: fixed  - dataset camera for every frame
#       static - one random pose per episode, no motion within the episode
#       move   - constant velocity within the episode. The azimuth changes by omega_deg per
#                step (random sign). Height and radius change linearly between two random values.
VARIANTS = {
    "fixed": dict(id=0, kind="fixed"),
    "static_rand": dict(id=1, kind="static"),
    "move": dict(id=2, kind="move", omega_deg=(2.0, 8.0)),
    "move_fast": dict(id=3, kind="move", omega_deg=(10.0, 15.0)),
}

POSE_DIM = 3  # azimuth, height, radius


def sample_trajectory(variant, episode, n_steps, seed=0):
    """Return (n_steps, 3) poses for one episode.

    The result depends on (variant, episode, n_steps, seed) only. The azimuth is not wrapped
    to [0, 2*pi).
    """
    spec = VARIANTS[variant]
    if spec["kind"] == "fixed":
        return np.tile([DATASET_AZIMUTH, DATASET_HEIGHT, DATASET_RADIUS], (n_steps, 1))
    rng = np.random.RandomState([int(seed), spec["id"], int(episode)])
    lo, hi = AZIMUTH_ARC
    if spec["kind"] == "static":
        pose = [rng.uniform(lo, hi), rng.uniform(*HEIGHT_RANGE), rng.uniform(*RADIUS_RANGE)]
        return np.tile(pose, (n_steps, 1))
    frac = np.linspace(0.0, 1.0, n_steps)
    sweep = np.deg2rad(rng.uniform(*spec["omega_deg"])) * (n_steps - 1) * rng.choice([-1.0, 1.0])
    if abs(sweep) > hi - lo:
        raise ValueError(f"{variant}: a sweep of {np.rad2deg(abs(sweep)):.0f} deg does not fit the azimuth arc")
    # sample the start so that the sweep stays within the arc
    az0 = rng.uniform(lo - min(sweep, 0.0), hi - max(sweep, 0.0))
    h0, h1 = rng.uniform(*HEIGHT_RANGE, size=2)
    r0, r1 = rng.uniform(*RADIUS_RANGE, size=2)
    return np.stack([az0 + sweep * frac, h0 + (h1 - h0) * frac, r0 + (r1 - r0) * frac], axis=1)


def pose_to_pyflex(pose):
    """Convert a pose (azimuth, height, radius) to (camPos, camAngle) for pyflex."""
    az, h, r = float(pose[0]), float(pose[1]), float(pose[2])
    cam_pos = np.array([r * np.sin(az), h, r * np.cos(az)])
    cam_angle = np.array([az, -np.arctan2(h, r + LOOK_PAST), 0.0])
    return cam_pos, cam_angle
