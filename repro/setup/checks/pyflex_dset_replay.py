"""Replay the first K actions of dataset episode EP from its initial particle state in the DINO-WM deformable env
(constructed as plan.py does) and compare rendered frames + particle states with the dataset.
Usage: python pyflex_dset_replay.py rope|granular [K] [EP] [DATA_ROOT]
DATA_ROOT defaults to $DATASET_DIR/deformable. Run with cwd = $DINO_WM and repro/env.sh sourced.
Dataset layout: <obj>/actions.pth (N,20,4); <obj>/<ep:06d>/obses.pth (20,224,224,3) == cam_1 color of h5 00..19;
<obj>/<ep:06d>/<t:02d>.h5 positions (1,P,4); actions[t] maps frame t -> t+1 (== h5[t+1]['action'])."""
import sys, os, time
import numpy as np
import torch, h5py, imageio
sys.path.insert(0, os.getcwd())
import gym
import env  # noqa
from env.deformable_env.FlexEnvWrapper import chamfer_distance

OUT = os.path.join(os.environ["DINO_WORK"], "setup_check")
os.makedirs(OUT, exist_ok=True)
obj = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
EP = int(sys.argv[3]) if len(sys.argv) > 3 else 0
data_root = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.environ["DATASET_DIR"], "deformable")
print("data_root", data_root, flush=True)
ep_dir = f"{data_root}/{obj}/{EP:06d}"
actions = torch.load(f"{data_root}/{obj}/actions.pth")[EP].numpy()
d_obs = torch.load(f"{ep_dir}/obses.pth").numpy()  # (20,224,224,3) float 0..255
d_pos = np.stack([h5py.File(f"{ep_dir}/{t:02d}.h5")["positions"][0] for t in range(K + 1)])  # (K+1,P,4)
if os.path.exists(f"{data_root}/{obj}/states.pth"):
    st = torch.load(f"{data_root}/{obj}/states.pth")[EP, : K + 1].numpy()
    print("states.pth vs h5 positions max abs diff", np.abs(st - d_pos).max(), flush=True)

e = gym.make("deformable_env", object_name=obj)
seed = 1
init_state = d_pos[0].reshape(-1)  # dataset states are flattened (P*4,), as plan.py passes them
t0 = time.time()
obses, states = e.rollout(seed, init_state, actions[:K])
print(f"rollout {K} actions: {time.time()-t0:.1f}s; sim visual {obses['visual'].shape} {obses['visual'].dtype}, states {states.shape}", flush=True)
s_vis = obses["visual"].astype(np.float32)
for t in range(K + 1):
    pix = np.abs(s_vis[t] - d_obs[t]).mean()
    pos_err = np.linalg.norm(states[t][:, :3] - d_pos[t][:, :3], axis=1)
    cd = chamfer_distance(torch.tensor(d_pos[t][None]), torch.tensor(states[t][None].astype(np.float32))).item()
    w_err = np.abs(states[t][:, 3] - d_pos[t][:, 3]).max()
    moved = np.linalg.norm(d_pos[t][:, :3] - d_pos[0][:, :3], axis=1).mean()
    print(f"t={t}: mean|px diff|={pix:6.2f}/255  pos err mean={pos_err.mean():.4f} max={pos_err.max():.4f}  "
          f"chamfer(dset,sim)={cd:.4f}  invmass maxdiff={w_err:.3g}  (dset displacement from t0: {moved:.4f})", flush=True)
# colors: mean RGB of non-background "object" pixels (saturated) in dset vs sim
def obj_rgb(img):
    sat = img.max(-1) - img.min(-1)
    m = sat > 60
    return img[m].mean(0).round(1), int(m.sum())
print("object-pixel mean RGB t0: dset", obj_rgb(d_obs[0]), "sim", obj_rgb(s_vis[0]))
print("whole-image mean RGB t0: dset", d_obs[0].reshape(-1, 3).mean(0).round(1), "sim", s_vis[0].reshape(-1, 3).mean(0).round(1))
# eval_state as used by planning: goal = dataset state at K, cur = sim state at K
print("eval_state(dset_K, sim_K):", e.eval_state(d_pos[K], states[K].astype(np.float32)))
rows = [np.concatenate(list(d_obs[: K + 1]), 1), np.concatenate(list(s_vis), 1),
        np.concatenate([np.clip(np.abs(s_vis[t] - d_obs[t]) * 3, 0, 255) for t in range(K + 1)], 1)]
out = f"{OUT}/{obj}_env_vs_dset.png"
imageio.imwrite(out, np.concatenate(rows, 0).astype(np.uint8))
print("saved", out, "(rows: dataset / sim / 3x|diff|; cols t=0..K)")
