"""ACWM-Phys DINO-WM 모델 하나를 테스트 에피소드 하나에 실행하고 GPU 메모리와 시간을 잰다.

입력: 지정한 split 의 에피소드 하나 (기본 ind_test 의 첫 에피소드). 평가(eval_acwm_phys.py, eval_pour_water.py)와 같은 계산을 한다.
  windows  37프레임 창 10개 (시작 = 평가와 같은 linspace), 처음 num_hist 프레임과 action 으로 열린 루프 예측 + 디코딩. 배치 10.
  episode  전체 에피소드 롤아웃: 처음 num_hist 프레임과 에피소드의 모든 action 으로 끝까지 예측 + 디코딩. 배치 1.
각 단계를 두 번 실행한다. 첫 번째는 CUDA 초기화가 들어간 시간이고 두 번째가 반복 실행 시간이다.

메모리: torch.cuda.max_memory_allocated (단계별, 전체), max_memory_reserved, 그리고 NVML 로 0.05초마다 잰 이 프로세스의 GPU 메모리
(nvidia-smi 에 보이는 값, CUDA context 포함) 최댓값.

usage (cwd = $DINO_WM, env sourced):
  python repro/eval/one_input_acwm_phys.py <ckpt_base_path> <model_name> <out_json> [--set ind_test] [--episode 0]
"""
import os, sys, json, time, argparse, threading
from pathlib import Path
import numpy as np
import torch
import hydra
import pynvml
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan import load_model
from utils import seed

GB = 2 ** 30


class NvmlPeak:
    """GPU memory of this process as nvidia-smi shows it, sampled every 0.05 s."""

    def __init__(self, index):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
        self.name = pynvml.nvmlDeviceGetName(self.h)
        self.name = self.name.decode() if isinstance(self.name, bytes) else self.name
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
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_base"); ap.add_argument("model_name"); ap.add_argument("out_json")
    ap.add_argument("--set", default="ind_test")
    ap.add_argument("--episode", type=int, default=0)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = "cuda"
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    mon = NvmlPeak(int(vis) if vis.isdigit() else 0)
    t_all = time.time()

    model_path = Path(args.ckpt_base) / "outputs" / args.model_name
    cfg = OmegaConf.load(model_path / "hydra.yaml")
    ck = model_path / "checkpoints" / "model_latest.pth"
    env = cfg.env.name
    H, f = cfg.num_hist, cfg.frameskip
    seed(cfg.training.seed)
    ds_cfg = cfg.env.dataset
    res = {"model": args.model_name, "env": env, "set": args.set, "episode_index": args.episode, "num_hist": H, "frameskip": f,
           "gpu": mon.name, "torch": torch.__version__, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    model, res["load_model_s"], _ = timed(lambda: load_model(ck, cfg, cfg.num_action_repeat, dev))
    model.eval()
    res["ckpt_epoch"] = int(torch.load(ck, map_location="cpu")["epoch"])
    res["params_M"] = {k: round(sum(p.numel() for p in getattr(model, k).parameters()) / 1e6, 2)
                       for k in ("encoder", "predictor", "decoder", "action_encoder") if getattr(model, k, None) is not None}

    t = time.time()
    if env == "acwm_pour_water":
        import eval_pour_water as E
        from datasets.img_transforms import default_transform
        tf = default_transform(cfg.img_size)
        d = E.load_acwm_pour_water_split(tf, ds_cfg.data_path, args.set, n_rollout=ds_cfg.n_rollout, exclude=ds_cfg.exclude,
                                         normalize_action=cfg.normalize_action)
        base, idxs = E.unwrap(d)
        ep = E.Episode(base, idxs[args.episode], tf)
        n_obs = ep.vis.shape[0]
        K = 36 // f + 1
        obs, acts = E.make_windows(ep, f, K)
        cuts = E.CUTS
    else:
        import eval_acwm_phys as E
        from datasets.acwm_phys_dset import load_acwm_phys_split
        tf = hydra.utils.instantiate(ds_cfg.transform)
        d = load_acwm_phys_split(tf, ds_cfg.data_path, args.set, n_rollout=ds_cfg.n_rollout, exclude=list(ds_cfg.exclude),
                                 normalize_action=cfg.normalize_action, action_dims=None if ds_cfg.action_dims is None else list(ds_cfg.action_dims),
                                 action_format=ds_cfg.action_format)
        base, idxs = E.unwrap(d)
        ep = E.Episode(base, idxs[args.episode], tf, E.OBS_HW[env])
        n_obs = ep.n_obs
        K = (E.WINDOW - 1) // f + 1
        cuts = np.linspace(0, n_obs - E.WINDOW, E.N_CUTS, dtype=int)
        obs, acts = E.make_windows(ep, f, K, cuts)
    res["read_episode_s"] = round(time.time() - t, 2)
    res["episode_name"] = str(base.video_names[idxs[args.episode]])
    res["episode_frames"] = int(n_obs)
    res["window"] = {"n_windows": int(len(cuts)), "frames_per_window_model": int(K), "predicted_per_window": int(K - H)}

    kf = min(len(ep.act) // f, (n_obs - 1) // f) + 1  # frames 0, f, 2f, ... of the whole episode
    fo = ep.vis[: f * (kf - 1) + 1 : f][None]
    fa = ep.act[: f * (kf - 1)].reshape(1, kf - 1, -1)
    res["episode_rollout"] = {"model_frames": int(kf), "predicted": int(kf - H)}

    rollout = E.rollout
    for name, fn in (("windows", lambda: rollout(model, obs, acts, H, dev)), ("episode", lambda: rollout(model, fo, fa, H, dev))):
        for run in (1, 2):
            _, s, m = timed(fn)
            res[f"{name}_run{run}_s"] = round(s, 3)
            res[f"{name}_run{run}_torch_peak_GiB"] = round(m / GB, 3)
    res["torch_max_reserved_GiB"] = round(torch.cuda.max_memory_reserved() / GB, 3)
    res["total_s"] = round(time.time() - t_all, 1)
    res["nvml_peak_GiB"] = round(mon.close() / GB, 3)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    json.dump(res, open(args.out_json, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
