"""Whole-episode rollout of a released ACWM-DiT-S checkpoint by chaining 37-frame windows.

--fresh_first --ckpt <latest.pt>: window 1 is generated here from the real frame 0 (decoded and resized like the
official loader) instead of being read from the evaluation run, for every episode of the split in metadata order;
<main dir> is not used then.

Window 1 (frames 0-36) is taken from the official evaluation run (eval_launch.py saved its prediction for the window that
starts at frame 0). Window k+1 starts at frame 36k: its condition frame is the last predicted frame of window k and its
actions are actions[36k : 36k+37], padded with the last action like the official loader does. This is the
"autoregressive rollout" of the paper's Figure 6; eval.py itself has no such mode.

usage (cwd = ACWM-Phys code root, see README.md here):
  python <this folder>/rollout_chain.py <env> <main results dir> <out dir> [--splits ind_test,ood_test] [--max_eps N]
  python <this folder>/rollout_chain.py <env> - <out dir> --fresh_first --ckpt <latest.pt>
writes <out dir>/<env>/<split>.npz: names, length, frames (E, T, H, W, 3) uint8, all predicted (frame 0 is the model's
reconstruction of the real first frame)
"""
import argparse, importlib, json, os, sys, time
import numpy as np
import cv2
import torch
import yaml

sys.path.insert(0, os.getcwd())
ev = importlib.import_module("eval")
from acwm.dataset.data_config import get_config_by_name

WIN = 37


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env"); ap.add_argument("main_dir"); ap.add_argument("out_dir")
    ap.add_argument("--splits", default="ind_test,ood_test")
    ap.add_argument("--max_eps", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fresh_first", action="store_true")
    ap.add_argument("--ckpt", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg_path, _ = ev.ENV_MAP[args.env]
    config = yaml.safe_load(open(cfg_path))
    if args.fresh_first:
        recs, ckpt_path = [], args.ckpt
    else:
        recs = [json.loads(l) for l in open(os.path.join(args.main_dir, f"{args.env}_windows.jsonl"))]
        ckpt_path = next(r for r in recs if r["type"] == "checkpoint")["path"]
    dev = torch.device("cuda")
    model = ev.load_model(config, dev)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    sd = {k: v for k, v in ckpt["model_state_dict"].items() if k not in ev.SCHEDULER_BUFFERS}
    r = model.load_state_dict(sd, strict=False)
    assert not r.missing_keys and not r.unexpected_keys, r
    step = int(ckpt["step"]); del ckpt, sd
    print(f"{args.env}: {ckpt_path} step {step}", flush=True)

    ds_kw = {k: v for k, v in config["dataset"].items() if k not in ("name", "test_cuts", "train_size", "ind_test_size", "ood_test_size")}
    dcfg = get_config_by_name(config["dataset"]["name"], **ds_kw)
    os.makedirs(os.path.join(args.out_dir, args.env), exist_ok=True)
    pred_dir = os.path.join(args.main_dir, "pred", args.env)

    for split in args.splits.split(","):
        if args.fresh_first:
            root = os.path.join(dcfg.root_dir, split)
            meta = torch.load(os.path.join(root, "metadata.pt"), weights_only=True)
            first = {i: None for i in range(len(meta))}
        else:
            wrec = next(r for r in recs if r["type"] == "windows" and r["split"] == split)
            root = wrec["root"]
            meta = torch.load(os.path.join(root, "metadata.pt"), weights_only=True)
            first = {}  # traj index -> global window index of the window starting at frame 0
            for w, (i, s, vp, ln) in enumerate(wrec["windows"]):
                if s == 0 and i not in first:
                    first[i] = w
        trajs = sorted(first)[: args.max_eps or None]
        names = [os.path.basename(meta[i]["video_path"]) for i in trajs]
        lengths = [int(meta[i]["length"]) for i in trajs]
        T = max(lengths)
        H, W = dcfg.obs_shape[1], dcfg.obs_shape[2]
        frames = np.zeros((len(trajs), T, H, W, 3), np.uint8)
        t0 = time.time()
        for b0 in range(0, len(trajs), args.batch_size):
            idx = list(range(b0, min(b0 + args.batch_size, len(trajs))))
            cur, acts = [], []
            for e in idx:
                acts.append(meta[trajs[e]]["actions"].float())
                if args.fresh_first:  # the real frame 0, as the official loader reads it
                    cap = cv2.VideoCapture(os.path.join(root, os.path.basename(meta[trajs[e]]["video_path"])))
                    ok, fr = cap.read(); cap.release()
                    if fr.shape[:2] != (H, W):
                        fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
                    cur.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                    continue
                w = first[trajs[e]]
                p = np.load(os.path.join(pred_dir, f"{split}_{w // 2:05d}.npz"))
                assert int(p["window0"]) == (w // 2) * 2
                win = p["pred"][w % 2]
                n = min(WIN, lengths[e])
                frames[e, :n] = win[:n]
                cur.append(win[WIN - 1])
            s = 0 if args.fresh_first else WIN - 1
            while s < max(lengths[e] for e in idx) - 1:
                a = []
                for e, act in zip(idx, acts):
                    sl = act[s : s + WIN]
                    if len(sl) == 0:
                        sl = act[-1:]
                    if len(sl) < WIN:
                        sl = torch.cat([sl, sl[-1:].repeat(WIN - len(sl), 1)])
                    a.append(sl)
                o0 = torch.from_numpy(np.stack(cur)).float().div(255.0).to(dev)
                with torch.no_grad():
                    pred = model.generate(o0, torch.stack(a).to(dev), num_inference_steps=50, noise_level=0.0, mode="parallel")
                pred = (pred.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
                for j, e in enumerate(idx):
                    n = min(WIN, lengths[e] - s)
                    if s == 0:
                        frames[e, 0] = pred[j, 0]  # the model's reconstruction of frame 0
                    if n > 1:
                        frames[e, s + 1 : s + n] = pred[j, 1:n]
                cur = [pred[j, WIN - 1] for j in range(len(idx))]
                s += WIN - 1
            print(f"  {split} episodes {idx[0]}-{idx[-1]} done, {time.time() - t0:.0f} s", flush=True)
        np.savez_compressed(os.path.join(args.out_dir, args.env, f"{split}.npz"), names=np.array(names), length=np.array(lengths),
                            traj_idx=np.array(trajs), frames=frames, checkpoint_step=step, seed=args.seed, fresh_first=args.fresh_first)
        print(f"{split}: {len(trajs)} episodes, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
