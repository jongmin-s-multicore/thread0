"""Summarise the logs written by eval_launch.py (ACWM-DiT, released checkpoints).

usage (cwd = ACWM-Phys code root, see README.md here):  python <this folder>/analyze.py <results dir> <env> [<env> ...]
writes <results dir>/<env>_summary.json

Three kinds of numbers per split:
  official    what eval.py prints: mean over batches of the batch metrics (all 37 frames, batch of 2).
              "shipped" = only the first 250 batches (= eval_all.sh with --max_trajs 50, see README notes on the loader).
              psnr_from_mse = 10 log10(1 / mean MSE), the convention of the paper tables.
  per_episode official per-window metrics (37 frames), averaged per episode, mean and 95 % bootstrap interval over episodes.
  dino        the metric our DINO-WM evaluation reports (repro/eval/eval_acwm_phys.py, "acwm" block): only the frames a
              frameskip-5 model predicts (window frames 5, 10, ..., 35; Pour Water 15, ..., 35), motion mask from window
              frames 0, 5, ..., 35, averaged per episode. Needs the saved predictions. "copy_first" repeats window frame 0.
"""
import json, os, sys, glob
from collections import OrderedDict
from multiprocessing import Pool
import numpy as np
import cv2
import torch
from skimage.metrics import structural_similarity

sys.path.insert(0, os.getcwd())
from acwm.dataset.data_config import get_config_by_name

MASK_FRAMES = list(range(0, 36, 5))
SUB = {"pour_water": [15, 20, 25, 30, 35]}
SUB_DEFAULT = [5, 10, 15, 20, 25, 30, 35]
W0_SPLIT = 4000  # Pour Water ood_test: first-frame water pixels at 360x360 (low / high)
# groups used by our DINO-WM evaluation (repro/eval/eval_acwm_phys.py, branch hanbin5/acwm-baseline)
ROPE_PUSHER_OFF = {f"episode_{i}.mp4" for i in (0, 1, 3, 4, 5, 6, 10, 12, 13, 15, 16)}  # pusher leaves the recorded actions
ARM_INSIDE = {f"episode_{i}.mp4" for i in (13, 25, 27, 28, 30, 39, 49, 52, 54, 72)}  # cumulative actions inside the train range
REACHER_OLD = {f"episode_{i}.mp4" for i in (0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18)}  # ood_test: old 480x480 / 50-frame videos
KEYS = ("mse", "masked_mse", "psnr", "ssim")


def boot(vals, n=2000, seed_=0):
    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [None, None, None]
    idx = np.random.RandomState(seed_).randint(0, len(x), (n, len(x)))
    b = x[idx].mean(1)
    return [float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def acwm_metrics(pred, gt, first, gt_win):
    mse = float(((pred - gt) ** 2).mean())
    mask = np.abs(gt_win - first[None]).max(axis=(0, 3))
    w = (0.01 + mask)[None, :, :, None]
    mmse = float((w * (pred - gt) ** 2).sum() / (w.sum() * pred.shape[0] * pred.shape[3] + 1e-8))
    psnr = float(10 * np.log10(1.0 / (mse + 1e-8)))
    ss = float(np.mean([structural_similarity(p, g, data_range=1.0, channel_axis=-1) for p, g in zip(pred, gt)]))
    return {"mse": mse, "masked_mse": mmse, "psnr": psnr, "ssim": ss}


def water_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (hsv[..., 0] >= 98) & (hsv[..., 0] <= 110) & (hsv[..., 1] > 70) & (hsv[..., 2] > 100)


_cache = OrderedDict()
def load_video(path, hw):
    """Same decoding as the official loader: cv2, INTER_AREA resize to obs_shape."""
    key = (path, hw)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if fr.shape[:2] != hw:
            fr = cv2.resize(fr, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    v = np.stack(frames)
    if len(_cache) >= 8:
        _cache.popitem(last=False)
    _cache[key] = v
    return v


def gt_window(path, hw, start, seq_len):
    v = load_video(path, hw)
    idx = np.minimum(np.arange(start, start + seq_len), len(v) - 1)  # the loader pads with the last frame
    return v[idx].astype(np.float32) / 255.0


def batch_job(job):
    pred_file, wins, hw, seq_len, sub = job
    pred = np.load(pred_file)["pred"].astype(np.float32) / 255.0
    out = []
    for b, (root, vp, start) in enumerate(wins):
        gt = gt_window(os.path.join(root, vp), hw, start, seq_len)
        p = pred[b, : gt.shape[0]]
        gw = gt[MASK_FRAMES]
        r = {"model": acwm_metrics(p[sub], gt[sub], gt[0], gw),
             "copy_first": acwm_metrics(np.repeat(gt[:1], len(sub), 0), gt[sub], gt[0], gw),
             "mse37_u8": float(((p - gt) ** 2).mean())}
        out.append(r)
    return out


def groups_of(env, split, root, names):
    g = {"all": np.arange(len(names))}
    if env == "clothmove" and split == "ood_test":
        larger = np.array([os.path.basename(n).startswith("larger") for n in names])
        g["smaller"], g["larger"] = np.where(~larger)[0], np.where(larger)[0]
    if env == "stack_cube":
        meta = {os.path.basename(e["video_path"]): e for e in torch.load(os.path.join(root, "metadata.pt"), weights_only=True)}
        succ = np.array([bool(meta[os.path.basename(n)]["success"]) for n in names])
        g["success"], g["fail"] = np.where(succ)[0], np.where(~succ)[0]
        if split == "ood_test":
            lab = np.array([str(meta[os.path.basename(n)]["ood_label"]) for n in names])
            for l in sorted(set(lab)):
                g[l] = np.where(lab == l)[0]
    base = [os.path.basename(n) for n in names]
    num = np.array([int("".join(c for c in b.split(".")[0] if c.isdigit()) or -1) for b in base])
    if env == "push_cube" and split == "ood_test":
        g["1cube"], g["5cubes"] = np.where(num < 20)[0], np.where(num >= 20)[0]
    if env == "push_cube" and split == "ind_test":
        g["3cubes"], g["4cubes"] = np.where(num < 32)[0], np.where((num >= 32) & (num < 48))[0]
    if env == "push_rope" and split == "ood_test":
        off = np.array([b in ROPE_PUSHER_OFF for b in base])
        g["pusher_ok"], g["pusher_off"] = np.where(~off)[0], np.where(off)[0]
    if env == "robot_arm" and split == "ood_test":
        ins = np.array([b in ARM_INSIDE for b in base])
        g["inside"], g["outside"] = np.where(ins)[0], np.where(~ins)[0]
    if env == "reacher" and split == "ood_test":  # "usable" is what our DINO-WM evaluation calls "all"
        old = np.array([b in REACHER_OLD for b in base])
        g["usable"], g["old_format"] = np.where(~old)[0], np.where(old)[0]
    if env == "pour_water" and split == "ood_test":
        w0 = []
        for n in names:
            cap = cv2.VideoCapture(os.path.join(root, n)); ok, fr = cap.read(); cap.release()
            w0.append(int(water_mask(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).sum()))
        w0 = np.array(w0)
        g["low"], g["high"] = np.where(w0 < W0_SPLIT)[0], np.where(w0 >= W0_SPLIT)[0]
    return g


def summarise(res_dir, env, pool):
    recs = [json.loads(l) for l in open(os.path.join(res_dir, f"{env}_windows.jsonl"))]
    head = next(r for r in recs if r["type"] == "header")
    ck = next((r for r in recs if r["type"] == "checkpoint"), {})
    out = {"env": env, "argv": head["argv"], "seed": head["seed"], "torch": head["torch"], "gpu": head["gpu"],
           "checkpoint": ck.get("path"), "checkpoint_step": ck.get("step"), "params_dit": ck.get("params_dit"),
           "finished": any(r["type"] == "done" for r in recs), "splits": {}}
    pred_dir = os.path.join(res_dir, "pred", env)
    sub = SUB.get(env, SUB_DEFAULT)
    for wrec in [r for r in recs if r["type"] == "windows"]:
        split, root = wrec["split"], wrec["root"]
        hw, seq_len = tuple(wrec["obs_shape"][1:]), wrec["seq_len"]
        batches = sorted([r for r in recs if r["type"] == "batch" and r["split"] == split], key=lambda r: r["batch"])
        done = next((r for r in recs if r["type"] == "split_done" and r["split"] == split), None)
        if not batches:
            continue
        n_win = sum(len(b["per_window"]) for b in batches)
        wins = wrec["windows"][:n_win]
        s = {"root": root, "n_meta": wrec["n_meta"], "windows_total": len(wrec["windows"]), "windows_done": n_win,
             "batches_done": len(batches), "complete": done is not None and n_win == len(wrec["windows"]),
             "seconds": done["seconds"] if done else None, "gpu_peak_gb": done["gpu_peak_gb"] if done else None}
        # official aggregation
        def off(bs):
            m = {k: float(np.mean([b["batch_metrics"][k] for b in bs])) for k in KEYS}
            m["psnr_from_mse"] = float(10 * np.log10(1.0 / m["mse"]))
            m["batches"] = len(bs)
            return m
        s["official"] = {"all": off(batches)}
        if len(batches) > 250:
            s["official"]["shipped_first_250_batches"] = off(batches[:250])
        # episodes
        names = list(OrderedDict.fromkeys(w[2] for w in wins))
        ep_of = np.array([names.index(w[2]) for w in wins])
        groups = groups_of(env, split, root, names)
        s["n_episodes"] = len(names)
        s["group_sizes"] = {g: int(len(v)) for g, v in groups.items()}
        per_win = [pw for b in batches for pw in b["per_window"]]
        def per_ep(vals):
            vals = np.asarray(vals, float)
            return np.array([vals[ep_of == e].mean() for e in range(len(names))])
        ep37 = {k: per_ep([pw[k] for pw in per_win]) for k in KEYS}
        s["per_episode"] = {g: {k: boot(ep37[k][mem]) for k in KEYS} for g, mem in groups.items()}
        for g, mem in groups.items():
            s["per_episode"][g]["psnr_from_mse"] = float(10 * np.log10(1.0 / ep37["mse"][mem].mean())) if len(mem) else None
        # DINO-WM style metric from the saved predictions
        files = [os.path.join(pred_dir, f"{split}_{b['batch']:05d}.npz") for b in batches]
        if files and all(os.path.exists(f) for f in files):
            jobs, w0 = [], 0
            for b, f in zip(batches, files):
                nb = len(b["per_window"])
                jobs.append((f, [(root, w[2], w[1]) for w in wins[w0 : w0 + nb]], hw, seq_len, sub))
                w0 += nb
            rows = [r for chunk in pool.imap(batch_job, jobs, chunksize=4) for r in chunk]
            u8 = np.array([r["mse37_u8"] for r in rows]); fl = np.array([pw["mse"] for pw in per_win])
            s["pred_check"] = {"max_abs_diff_mse37": float(np.abs(u8 - fl).max()), "mean_mse37_float": float(fl.mean()), "mean_mse37_u8": float(u8.mean())}
            s["dino"] = {"frames": sub, "mask_frames": MASK_FRAMES}
            for m in ("model", "copy_first"):
                ep = {k: per_ep([r[m][k] for r in rows]) for k in KEYS}
                s["dino"][m] = {g: {k: boot(ep[k][mem]) for k in KEYS} for g, mem in groups.items()}
        out["splits"][split] = s
    json.dump(out, open(os.path.join(res_dir, f"{env}_summary.json"), "w"), indent=1)
    return out


def show(o):
    print(f"\n== {o['env']}  checkpoint step {o['checkpoint_step']}  finished={o['finished']}")
    for split, s in o["splits"].items():
        print(f" {split}: {s['windows_done']}/{s['windows_total']} windows, {s['n_episodes']} episodes, {s['seconds']} s, groups {s['group_sizes']}")
        for name, m in s["official"].items():
            print(f"   official {name:<26} MSE {1e3*m['mse']:.3f}e-3  M-MSE {1e3*m['masked_mse']:.2f}e-3  PSNR {m['psnr']:.2f} (from MSE {m['psnr_from_mse']:.2f})  SSIM {m['ssim']:.4f}  [{m['batches']} batches]")
        for g, m in s["per_episode"].items():
            print(f"   per-episode {g:<8} MSE {1e3*m['mse'][0]:.3f}e-3 [{1e3*m['mse'][1]:.3f}, {1e3*m['mse'][2]:.3f}]  M-MSE {1e3*m['masked_mse'][0]:.2f}e-3  SSIM {m['ssim'][0]:.4f}")
        if "dino" in s:
            print(f"   saved-prediction check: max |MSE(uint8) - MSE(float)| = {s['pred_check']['max_abs_diff_mse37']:.2e}")
            for g in s["group_sizes"]:
                a, c = s["dino"]["model"][g], s["dino"]["copy_first"][g]
                print(f"   DINO-style {g:<8} MSE {1e3*a['mse'][0]:.3f}e-3  M-MSE {1e3*a['masked_mse'][0]:.2f}e-3  PSNR {a['psnr'][0]:.2f}  SSIM {a['ssim'][0]:.4f}   | copy_first MSE {1e3*c['mse'][0]:.3f}e-3")


if __name__ == "__main__":
    res_dir, envs = sys.argv[1], sys.argv[2:]
    cv2.setNumThreads(1); torch.set_num_threads(1)
    with Pool(int(os.environ.get("ANALYZE_PROCS", "8"))) as pool:
        for env in envs:
            show(summarise(res_dir, env, pool))
