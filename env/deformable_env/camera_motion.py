"""Camera trajectories for the moving-camera Rope/Granular experiment.

Pure numpy on purpose: the renderer (PyFleX, local) and the evaluation import this, the
training side does not need it (the rendered dataset stores the poses it was rendered with).

A pose is (azimuth [rad], height, radius). The camera sits at

    (radius * sin(azimuth), height, radius * cos(azimuth))

with yaw = azimuth and pitch = -atan2(height, radius + LOOK_PAST), i.e. it looks at the
point of the table plane LOOK_PAST behind the table center. LOOK_PAST is what makes the
dataset camera come out exactly: env/deformable_env/src/sim/sim_env/cameras.py puts the four
cameras at (+-6, 10, +-6) with pitch -45 deg, so their view axis meets the table at distance
10 from the camera foot, 10 - 6*sqrt(2) past the center. The dataset frames (obses.pth) are
camera index 1: (6, 10, -6), azimuth 135 deg.
"""

import numpy as np

DATASET_AZIMUTH = np.deg2rad(135.0)
DATASET_HEIGHT = 10.0
DATASET_RADIUS = 6.0 * np.sqrt(2.0)
LOOK_PAST = DATASET_HEIGHT - DATASET_RADIUS

HEIGHT_RANGE = (7.0, 13.0)
RADIUS_RANGE = (6.5, 10.5)
# The xArm stands at azimuth ~270 deg. From behind it (about 240-300 deg) the arm hides most
# of the table, and a low, close camera ends up inside it, so that sector is left out. The
# four dataset cameras (45, 135, 225, 315 deg) are inside the allowed arc.
AZIMUTH_ARC = (np.deg2rad(-55.0), np.deg2rad(235.0))

# kind: fixed  - the dataset camera for every frame
#       static - one random pose per episode, no motion inside the episode
#       move   - constant velocity inside the episode: the azimuth turns omega_deg per step
#                (random sign), height and radius go linearly from one random value to another
VARIANTS = {
    "fixed": dict(id=0, kind="fixed"),
    "static_rand": dict(id=1, kind="static"),
    "move": dict(id=2, kind="move", omega_deg=(2.0, 8.0)),
    "move_fast": dict(id=3, kind="move", omega_deg=(10.0, 15.0)),
}

POSE_DIM = 3  # azimuth, height, radius


def sample_trajectory(variant, episode, n_steps, seed=0):
    """(n_steps, 3) poses for one episode; a pure function of (variant, episode, n_steps, seed).

    The azimuth is not wrapped, so consecutive differences are the per-step rotation.
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
    # start anywhere that keeps the whole sweep inside the arc
    az0 = rng.uniform(lo - min(sweep, 0.0), hi - max(sweep, 0.0))
    h0, h1 = rng.uniform(*HEIGHT_RANGE, size=2)
    r0, r1 = rng.uniform(*RADIUS_RANGE, size=2)
    return np.stack([az0 + sweep * frac, h0 + (h1 - h0) * frac, r0 + (r1 - r0) * frac], axis=1)


def pose_to_pyflex(pose):
    """pose (azimuth, height, radius) -> (camPos, camAngle) for pyflex.set_camPos/set_camAngle."""
    az, h, r = float(pose[0]), float(pose[1]), float(pose[2])
    cam_pos = np.array([r * np.sin(az), h, r * np.cos(az)])
    cam_angle = np.array([az, -np.arctan2(h, r + LOOK_PAST), 0.0])
    return cam_pos, cam_angle
