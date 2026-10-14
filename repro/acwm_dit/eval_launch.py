"""Run the official eval.py (repo untouched) with a fixed seed and a per-window log.

Usage (cwd = ACWM-Phys code root, see README.md here):  python <this folder>/eval_launch.py <eval.py arguments>
Env: ACWM_EVAL_SEED (default 0), ACWM_EVAL_WINDOWS (path of the per-window jsonl; default <output_root>/<env>_windows.jsonl),
     ACWM_EVAL_SAVE_PRED (directory; if set, the predicted frames of every batch are saved there as uint8 npz)

What it adds on top of eval.py, without changing what eval.py computes:
  - torch / numpy / random seeds, so the sampling noise is repeatable
  - torch.load forced to weights_only=True (files come from Hugging Face)
  - prints missing / unexpected keys of load_state_dict
  - writes the window list of each split and the metrics of every window to a jsonl file
  - optionally saves the predicted frames (for metrics that eval.py does not compute)
"""
import importlib, json, os, random, sys, time
import numpy as np
import torch

REPO = os.getcwd()
sys.path.insert(0, REPO)
seed = int(os.environ.get("ACWM_EVAL_SEED", "0"))
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

_torch_load = torch.load
def _safe_load(*a, **k):
    k["weights_only"] = True
    return _torch_load(*a, **k)
torch.load = _safe_load

ev = importlib.import_module("eval")

def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

out_root = _arg("--output_root", "results")
os.makedirs(out_root, exist_ok=True)
win_path = os.environ.get("ACWM_EVAL_WINDOWS") or os.path.join(out_root, f"{_arg('--env')}_windows.jsonl")
fout = open(win_path, "w")
def emit(rec):
    fout.write(json.dumps(rec) + "\n"); fout.flush()

emit({"type": "header", "argv": sys.argv[1:], "seed": seed, "torch": torch.__version__,
      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
      "data_root": os.environ.get("ACWM_DATA_ROOT"), "vae": os.environ.get("WAN_VAE_PATH"),
      "time": time.strftime("%F %T %Z")})

state = {"split": None, "batch": 0, "window": 0}
pred_dir = os.environ.get("ACWM_EVAL_SAVE_PRED")
if pred_dir:
    os.makedirs(pred_dir, exist_ok=True)

_load_checkpoint = ev.load_checkpoint
def load_checkpoint(model, ckpt_path, device):
    orig = model.load_state_dict
    def lsd(sd, strict=True, **kw):
        r = orig(sd, strict=strict, **kw)
        print(f"load_state_dict: {len(sd)} keys given, missing={list(r.missing_keys)}, unexpected={list(r.unexpected_keys)}")
        emit({"type": "state_dict", "given": len(sd), "missing": list(r.missing_keys), "unexpected": list(r.unexpected_keys)})
        return r
    model.load_state_dict = lsd
    try:
        step = _load_checkpoint(model, ckpt_path, device)
    finally:
        del model.load_state_dict
    emit({"type": "checkpoint", "path": ckpt_path, "step": int(step),
          "params_dit": sum(p.numel() for p in model.model.parameters())})
    return step
ev.load_checkpoint = load_checkpoint

_get_dataset = ev.RoboticsDatasetWrapper.get_dataset
def get_dataset(name, split=None, max_trajs=None, **kw):
    ds = _get_dataset(name, split=split, max_trajs=max_trajs, **kw)
    meta = _safe_load(ds.metadata_path, mmap=True)
    wins = []
    for i, s in ds.indices:
        e = meta[i]
        wins.append([int(i), int(s), str(e["video_path"]), int(e["length"]) if "length" in e else -1])
    emit({"type": "windows", "split": split, "root": ds.effective_root, "n_meta": len(meta),
          "seq_len": ds.config.seq_len, "obs_shape": list(ds.config.obs_shape), "windows": wins})
    return ds
ev.RoboticsDatasetWrapper.get_dataset = staticmethod(get_dataset)

_compute_metrics = ev.compute_metrics
def compute_metrics(pred, gt):
    m = _compute_metrics(pred, gt)
    per = [_compute_metrics(pred[b:b + 1], gt[b:b + 1]) for b in range(pred.shape[0])]
    emit({"type": "batch", "split": state["split"], "batch": state["batch"], "window0": state["window"],
          "t": round(time.time() - state["t0"], 2), "batch_metrics": m, "per_window": per})
    if pred_dir:
        np.savez_compressed(os.path.join(pred_dir, f"{state['split']}_{state['batch']:05d}.npz"),
                            pred=(pred.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy(), window0=state["window"])
    state["batch"] += 1; state["window"] += pred.shape[0]
    return m
ev.compute_metrics = compute_metrics

_eval_split = ev.eval_split
def eval_split(model, split_name, *a, **k):
    state.update(split=split_name, batch=0, window=0, t0=time.time())
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    acc = _eval_split(model, split_name, *a, **k)
    emit({"type": "split_done", "split": split_name, "batches": state["batch"], "windows": state["window"],
          "seconds": round(time.time() - state["t0"], 1),
          "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
          "mean": {k2: float(np.mean(v)) for k2, v in acc.items()}})
    return acc
ev.eval_split = eval_split

ev.main()
emit({"type": "done", "time": time.strftime("%F %T %Z")})
