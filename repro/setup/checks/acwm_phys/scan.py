"""ACWM-Phys Cloth Move / Push Sand / Stack Cube 의 모든 split·모든 프레임을 훑는다: 프레임 수, 프레임 간 변화량,
색 마스크 픽셀 수와 중심(천·공, 모래·밀판, 큐브), action·state 를 에피소드별 배열로 <out>/<env>_scan.npz 에 쓴다.
같은 폴더의 report.py, details.py, history.py, stack_rule.py 가 이 npz 를 읽는다 (SETTINGS.md §8.1 의 근거).
20 프로세스로 약 2분, npz 합쳐 약 60 MB.

usage: python repro/setup/checks/acwm_phys/scan.py $DATASET_DIR/acwm_phys <out dir> [cloth_move push_sand stack_cube]"""
import sys, os, numpy as np, cv2, decord, torch
from multiprocessing import Pool

ROOT, OUT = sys.argv[1], sys.argv[2]
ENVS = {"cloth_move": "deformable/clothmove", "push_sand": "particle/push_sand", "stack_cube": "rigid_dynamics/stack_cube"}
SPLITS = ("ind_train", "ind_test", "ood_test")
TMAX = 66


def hue(hsv, lo, hi, s=100, v=80):
    h = hsv[..., 0]
    m = (h >= lo) & (h <= hi) if lo <= hi else ((h >= lo) | (h <= hi))
    return m & (hsv[..., 1] > s) & (hsv[..., 2] > v)


def cen(m):
    n = int(m.sum())
    if n < 8:
        return n, np.nan, np.nan
    ys, xs = np.nonzero(m)
    return n, float(xs.mean()), float(ys.mean())


MASKS = {
    "cloth_move": {"cloth": lambda h: hue(h, 8, 16, 120), "back": lambda h: hue(h, 158, 170, 60), "blue": lambda h: hue(h, 105, 113), "red": lambda h: hue(h, 175, 4, 90)},
    "push_sand": {"sand": lambda h: hue(h, 90, 102), "stick": lambda h: hue(h, 172, 6, 90), "board": lambda h: (h[..., 2] < 90) & (h[..., 1] < 90)},
    "stack_cube": {"yellow": lambda h: hue(h, 26, 33, 180, 130), "green": lambda h: hue(h, 55, 66, 150, 60)},
}


def scan(a):
    env, sp, name = a
    vr = decord.VideoReader(os.path.join(ROOT, ENVS[env], sp, name), num_threads=1)
    n = len(vr)
    fr = vr.get_batch(list(range(n))).asnumpy()
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)[::4, ::4] for f in fr]).astype(np.int16)
    motion = np.full(TMAX, np.nan); motion[: n - 1] = np.abs(np.diff(gray, axis=0)).mean((1, 2))
    dyn = (np.abs(gray - gray[0]).max(0) > 30)
    ys, xs = np.nonzero(dyn)
    bbox = [int(xs.min()) * 4, int(ys.min()) * 4, int(xs.max()) * 4 + 3, int(ys.max()) * 4 + 3] if len(xs) else [-1, -1, -1, -1]
    feats = {}
    for k in MASKS[env]:
        feats[k] = np.full((TMAX, 3), np.nan)
    for t, f in enumerate(fr):
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        for k, fn in MASKS[env].items():
            feats[k][t] = cen(fn(hsv))
    return dict(n=n, hw=fr.shape[1:3], motion=motion, bbox=bbox, mean0=float(fr[0].mean()), **feats)


if __name__ == "__main__":
    envs = sys.argv[3:] or list(ENVS)
    with Pool(20) as p:
        for env in envs:
            out = {}
            for sp in SPLITS:
                sd = os.path.join(ROOT, ENVS[env], sp)
                meta = torch.load(os.path.join(sd, "metadata.pt"), map_location="cpu", weights_only=True)
                names = [os.path.basename(e["video_path"]) for e in meta]
                have = [os.path.exists(os.path.join(sd, n)) for n in names]
                rs = p.map(scan, [(env, sp, n) for n, h in zip(names, have) if h], chunksize=8)
                out[f"{sp}/names"] = np.array([n for n, h in zip(names, have) if h])
                out[f"{sp}/n_frames"] = np.array([r["n"] for r in rs])
                out[f"{sp}/n_actions"] = np.array([len(e["actions"]) for e, h in zip(meta, have) if h])
                out[f"{sp}/meta_len"] = np.array([int(e["length"]) for e, h in zip(meta, have) if h])
                out[f"{sp}/hw"] = np.array([r["hw"] for r in rs])
                out[f"{sp}/motion"] = np.stack([r["motion"] for r in rs])
                out[f"{sp}/bbox"] = np.array([r["bbox"] for r in rs])
                out[f"{sp}/mean0"] = np.array([r["mean0"] for r in rs])
                for k in MASKS[env]:
                    out[f"{sp}/{k}"] = np.stack([r[k] for r in rs])
                A = meta[0]["actions"].shape[1]
                acts = np.full((len(rs), TMAX, A), np.nan, np.float32)
                i = 0
                for e, h in zip(meta, have):
                    if h:
                        acts[i, : len(e["actions"])] = e["actions"].numpy(); i += 1
                out[f"{sp}/actions"] = acts
                for key in ("cloth_dimx", "cloth_dimy", "success", "episode_seconds"):
                    if key in meta[0]:
                        out[f"{sp}/{key}"] = np.array([float(e[key]) for e, h in zip(meta, have) if h])
                if "ood_label" in meta[0]:
                    out[f"{sp}/ood_label"] = np.array([str(e["ood_label"]) for e, h in zip(meta, have) if h])
                if "states" in meta[0]:
                    st = np.full((len(rs), TMAX, meta[0]["states"].shape[1]), np.nan, np.float32); i = 0
                    for e, h in zip(meta, have):
                        if h:
                            st[i, : len(e["states"])] = e["states"].numpy(); i += 1
                    out[f"{sp}/states"] = st
                print(env, sp, "episodes", len(rs), "missing videos", int(len(names) - sum(have)), flush=True)
            np.savez_compressed(os.path.join(OUT, f"{env}_scan.npz"), **out)
