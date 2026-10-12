"""gym registrations for the DINO-WM environments.

The legacy environments are registered best-effort. point_maze pulls in d4rl and
mujoco_py (via gym.envs.mujoco), and deformable_env needs PyFleX; neither is
required by the distracting-Reacher experiment, and on an instance that only has
dm_control an unconditional import of them would make `import env` -- and
therefore training and evaluation -- fail outright. Each registration is
therefore guarded, and anything unavailable is reported once at import time.
"""

import warnings

from gym.envs.registration import register

_unavailable = []

register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

try:
    from .pointmaze import U_MAZE

    register(
        id='point_maze',
        entry_point='env.pointmaze:PointMazeWrapper',
        max_episode_steps=300,
        kwargs={
            'maze_spec':U_MAZE,
            'reward_type':'sparse',
            'reset_target': False,
            'ref_min_score': 23.85,
            'ref_max_score': 161.86,
            'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
        }
    )
except Exception as exc:  # d4rl / mujoco_py missing
    _unavailable.append(f"point_maze ({type(exc).__name__}: {exc})")

register(
    id="wall",
    entry_point="env.wall.wall_env_wrapper:WallEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="deformable_env",
    entry_point="env.deformable_env.FlexEnvWrapper:FlexEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

# DMC Reacher with Distracting-Control-Suite video backgrounds. The module
# self-registers too, so importing it directly is enough; this entry is what
# makes plan.py's gym.make(cfg.env.name) work without any extra import.
register(
    id="distracting_reacher",
    entry_point="env.distracting_reacher:DistractingReacherWrapper",
    max_episode_steps=1000,
    reward_threshold=1.0,
)

if _unavailable:
    warnings.warn(
        "env: some environments are not available and were skipped: "
        + "; ".join(_unavailable),
        stacklevel=2,
    )
