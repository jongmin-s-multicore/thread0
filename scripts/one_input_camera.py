"""Run one camera-experiment model on one evaluation episode and record GPU memory and time.

Input: one episode (20 frames) of a test-camera folder under deformable_cam/<object>/. The computation is the
one of eval_camera_pred.py and of the 5-step numbers: encode the 20 real frames, roll out 5 steps from every start
t = 1..14 with the push actions (14 rollouts in one batch), latent MSE against the encoded real frames. A single
5-step rollout from t = 1 is timed as well. Each step runs twice: the first run includes CUDA start-up, the second
is the repeat time.

Memory: torch.cuda.max_memory_allocated, max_memory_reserved, and the GPU memory of this process as nvidia-smi
shows it (NVML, sampled every 0.05 s, CUDA context included).

    source repro/env.sh
    python scripts/one_input_camera.py --object rope --model train_rope@50 --cond eval_move --out out.json
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pynvml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_camera_pred import DEVICE, Model, load_condition, mse  # noqa: E402
from datasets.img_transforms import default_transform  # noqa: E402

GB = 2 ** 30
K = 5


class NvmlPeak:
    def __init__(self, index):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
        n = pynvml.nvmlDeviceGetName(self.h)
        self.name = n.decode() if isinstance(n, bytes) else n
        self.peak, self.stop = 0, threading.Event()
        self.th = threading.Thread(target=self.run, daemon=True)
        self.th.start()

    def run(self):
        pid = os.getpid()
        while not self.stop.is_set():
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(self.h):
                if p.pid == pid and p.usedGpuMemory:
                    self.peak = max(self.peak, p.usedGpuMemory)
            time.sleep(0.05)

    def close(self):
        self.stop.set()
        self.th.join()
        return self.peak


def timed(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    out = fn()
    torch.cuda.synchronize()
    return out, time.time() - t, torch.cuda.max_memory_allocated()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--object", required=True, choices=["rope", "granular"])
    p.add_argument("--model", required=True, help="name@epoch")
    p.add_argument("--cond", default="eval_move")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_grad_enabled(False)
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    mon = NvmlPeak(int(vis) if vis.isdigit() else 0)
    t_all = time.time()
    root = Path(os.environ["DATASET_DIR"]) / "deformable_cam"
    res = {"object": a.object, "model": a.model, "cond": a.cond, "episode": a.episode, "horizon": K, "gpu": mon.name,
           "torch": torch.__version__, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    m, res["load_model_s"], _ = timed(lambda: Model("m=" + a.model, a.object, root, os.environ["DINO_TRAIN"]))
    res["num_hist"], res["cam_action"] = m.num_hist, m.cam_action
    t = time.time()
    c = load_condition(root / a.object / a.cond, a.episode + 1)
    i = a.episode
    x = torch.from_numpy(np.ascontiguousarray(c["frames"][i])).permute(0, 3, 1, 2).float() / 255.0
    T = x.shape[0]
    frames = default_transform(224)(x).to(DEVICE)
    act = m.actions(c["push"][i:i + 1], c["pose"][i:i + 1])[0]
    res["read_episode_s"] = round(time.time() - t, 3)
    res["episode_frames"] = int(T)
    ts = list(range(1, T - K))
    res["rollout_starts"] = len(ts)

    def encode():
        return m.wm.encode_obs({"visual": frames[:, None], "proprio": torch.zeros(T, 1, 1, device=DEVICE)})["visual"][:, 0]

    def all_starts():
        zF = encode()
        zr, _ = m.predict(frames, act, ts, K)
        return torch.stack([mse(zr[:, k - 1], zF[[s + k for s in ts]]).mean() for k in range(1, K + 1)])

    for name, fn in (("encode", encode), ("episode", all_starts), ("one_rollout", lambda: m.predict(frames, act, [1], K))):
        for run in (1, 2):
            out, s, mem = timed(fn)
            res[f"{name}_run{run}_s"] = round(s, 4)
            res[f"{name}_run{run}_torch_peak_GiB"] = round(mem / GB, 3)
            if name == "episode":
                res["episode_latent_mse_per_step"] = [round(float(v), 4) for v in out]
    res["torch_max_reserved_GiB"] = round(torch.cuda.max_memory_reserved() / GB, 3)
    res["total_s"] = round(time.time() - t_all, 1)
    res["nvml_peak_GiB"] = round(mon.close() / GB, 3)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
