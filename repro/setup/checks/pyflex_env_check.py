"""Construct DINO-WM deformable env exactly as plan.py does (gym.make('deformable_env', object_name=...)),
run sample_random_init_goal_states / prepare / rollout with a few actions, save RGB frames, test eval_state.
Usage (cwd = $DINO_WM, env sourced): python pyflex_env_check.py rope|granular [n_actions] [n_envs]
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.getcwd())
import gym
import env  # noqa: registers deformable_env
import imageio
from env.serial_vector_env import SerialVectorEnv

OUT = os.path.join(os.environ["DINO_WORK"], "setup_check")
os.makedirs(f"{OUT}/pyflex", exist_ok=True)
obj = sys.argv[1]
n_act = int(sys.argv[2]) if len(sys.argv) > 2 else 3
n_envs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
t0 = time.time()
venv = SerialVectorEnv([gym.make("deformable_env", object_name=obj) for _ in range(n_envs)])
print(f"[{obj}] constructed {n_envs} env(s) in {time.time()-t0:.1f}s", flush=True)
e = venv.envs[0]
print("property params", e.get_property_params(), flush=True)

seeds = [99 * n + 1 for n in range(n_envs)]
init_state, goal_state = venv.sample_random_init_goal_states(seeds)
print("init_state", init_state.shape, "goal_state", goal_state.shape, f"{time.time()-t0:.1f}s", flush=True)

obs_0, state_0 = venv.prepare(seeds, init_state)
obs_g, state_g = venv.prepare(seeds, goal_state)
print("prepare visual", obs_0["visual"].shape, obs_0["visual"].dtype, "state", state_0.shape, flush=True)
if n_envs == 1:
    imageio.imwrite(f"{OUT}/{obj}_prepare_init.png", np.concatenate([obs_0["visual"][0], obs_g["visual"][0]], 1).astype(np.uint8))

# sample actions from the init state (env.sample_action uses current particle positions)
e.set_states(init_state[0])
acts = []
for _ in range(n_act):
    a = e.sample_action()
    if a is None:
        a = np.array([1.0, 1.0, -0.5, -0.5])
    acts.append(a)
acts = np.stack(acts)[None].repeat(n_envs, 0)
print("actions", acts[0], flush=True)
t1 = time.time()
obses, states = venv.rollout(seeds, init_state, acts)
print(f"rollout done {time.time()-t1:.1f}s; visual {obses['visual'].shape} states {states.shape}", flush=True)
vis = obses["visual"]
for t in range(vis.shape[1] if n_envs == 1 else 0):
    imageio.imwrite(f"{OUT}/{obj}_rollout_t{t}.png", vis[0, t].astype(np.uint8))
if n_envs == 1:
    imageio.imwrite(f"{OUT}/{obj}_rollout_strip.png", np.concatenate(list(vis[0]), 1).astype(np.uint8))
else:
    imageio.imwrite(f"{OUT}/pyflex/{obj}_multienv{n_envs}_grid.png",
                    np.concatenate([np.concatenate(list(vis[i]), 1) for i in range(n_envs)], 0).astype(np.uint8))
for i in range(n_envs):
    print(f"env{i} pixel mean per frame", [round(float(v.mean()), 1) for v in vis[i]])
# eval_state (Chamfer distance) as in planning: goal_state vs final env state
res = venv.eval_state(goal_state, states[:, -1])
print("eval_state(goal, final)", {k: v for k, v in res.items()}, flush=True)
res0 = venv.eval_state(states[:, -1], states[:, -1])
print("eval_state(final, final) (should be 0)", res0, flush=True)
print("particle displacement init->final", np.abs(states[0, -1, :, :3] - states[0, 0, :, :3]).mean())
print(f"total {time.time()-t0:.1f}s")
