from gym.envs.registration import register
# [repro] point_maze needs mujoco_py (through gym.envs.mujoco). On a machine without MuJoCo, skip it instead of
# failing `import env`, so that the environments that do not need it (pusht, wall) still run.
try:
    from .pointmaze import U_MAZE
except Exception as e:
    U_MAZE = None
    print(f"env: point_maze is not registered ({type(e).__name__}: {str(e).strip().splitlines()[0][:80]})")
register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
if U_MAZE is not None:  # [repro]
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