"""scan.py 결과로 보는 세부 사항 (SETTINGS.md §8.1):
Cloth Move 의 프레임별 변화량과 처음 action 이 0 인 구간, 천 크기와 metadata cloth_dim 의 관계, OoD 두 크기;
Push Sand 의 쿼터니언 부호 뒤집힘과 제자리 회전 스텝, 모래 픽셀 수; Stack Cube 의 길이·성공 여부,
그리고 Stack Cube 의 actions (delta) 와 actions_abs 중 어느 쪽이 관절 움직임을 설명하는지.

usage: python repro/setup/checks/acwm_phys/details.py <dir with *_scan.npz> $DATASET_DIR/acwm_phys"""
import sys, os, numpy as np, torch
D, ROOT = sys.argv[1], sys.argv[2]
np.set_printoptions(linewidth=220, suppress=True, precision=3)
z = np.load(os.path.join(D, "cloth_move_scan.npz"))
print("cloth motion curve (mean over episodes) per step t:")
print(" train ", np.nanmean(z["ind_train/motion"], 0)[:64])
nm = z["ood_test/names"]; lg = np.array([n.startswith("larger") for n in nm])
print(" ood larger ", np.nanmean(z["ood_test/motion"][lg], 0)[:64])
print(" ood smaller", np.nanmean(z["ood_test/motion"][~lg], 0)[:64])
a = z["ind_train/actions"]; zero = (np.abs(a[:, :, [0, 1, 2, 4, 5, 6]]).max(2) == 0)
print(" fraction of train episodes with all-zero action at step t:", zero.mean(0)[:10], "... last", zero.mean(0)[58:64])
c0 = z["ood_test/cloth"][:, 0, 0]; tr = np.median(z["ind_train/cloth"][:, 0, 0])
print(" ood frame-0 cloth px / train median: smaller", np.round(np.sort(c0[~lg] / tr)[[0, 12, 25, 37, 49]], 2), "larger", np.round(np.sort(c0[lg] / tr)[[0, 12, 25, 37, 49]], 2))
tot = lambda sp: np.nan_to_num(z[f"{sp}/cloth"][:, :, 0]) + np.nan_to_num(z[f"{sp}/back"][:, :, 0])
print(" train cloth+back px over time (median):", np.median(tot("ind_train"), 0)[::8])
z = np.load(os.path.join(D, "push_sand_scan.npz"))
a = z["ind_train/actions"]; n = z["ind_train/n_frames"]
q = a[:, :64, [4, 6]]
print("push_sand quaternion (qy,qw): norm min/max", np.linalg.norm(q, axis=2).min(), np.linalg.norm(q, axis=2).max(), "| fraction of steps with qw<0:", (q[..., 1] < 0).mean(), "| max per-step |dq|:", np.abs(np.diff(q, axis=1)).max(), "| episodes with any qw<0:", (q[..., 1] < 0).any(1).mean())
dp = np.linalg.norm(np.diff(a[:, :64, [0, 2]], axis=1), axis=2)
print(" position step size: median", np.median(dp), "p99", np.percentile(dp, 99), "max", dp.max(), "| fraction of steps with zero position change", (dp == 0).mean())
print(" mean |position step| per t:", dp.mean(0)[::4])
print(" motion curve train:", np.nanmean(z["ind_train/motion"], 0)[:63:4])
for sp in ("ind_train", "ind_test", "ood_test"):
    s = z[f"{sp}/sand"][:, :, 0]
    print(f" {sp} sand px frame0 quantiles", np.percentile(s[:, 0], [0, 5, 25, 50, 75, 95, 100]).round(), "| last/first ratio median", np.median(s[:, 63] / s[:, 0]).round(2), "| episode_seconds", np.percentile(z[f"{sp}/episode_seconds"], [0, 50, 100]).round(1))
z = np.load(os.path.join(D, "stack_cube_scan.npz"))
print("stack_cube motion curve train:", np.nanmean(z["ind_train/motion"], 0)[:58:3])
for sp in ("ind_train", "ood_test"):
    n = z[f"{sp}/n_frames"]; suc = z[f"{sp}/success"]
    print(f" {sp} frames by success: success {dict(zip(*np.unique(n[suc > .5], return_counts=True)))} fail {dict(zip(*np.unique(n[suc < .5], return_counts=True)))}")
a = z["ind_train/actions"]; g = a[:, :, 6]
print(" gripper action values:", np.unique(g[np.isfinite(g)]), "| first close step median", np.median([np.argmax(r[np.isfinite(r)] > 0) for r in g]))

z = np.load(os.path.join(D, "cloth_move_scan.npz"))
area = z["ind_train/cloth_dimx"] * z["ind_train/cloth_dimy"]
tot = np.nan_to_num(z["ind_train/cloth"][:, :, 0]) + np.nan_to_num(z["ind_train/back"][:, :, 0])
print("cloth train: corr(dimx*dimy, cloth+back px at frame t):", [(t, round(float(np.corrcoef(area, tot[:, t])[0, 1]), 3)) for t in (0, 1, 2, 3, 4, 8, 16, 32)])
print("   px at frame 1: ", np.percentile(tot[:, 1], [0, 5, 50, 95, 100]).round(), " frame 3:", np.percentile(tot[:, 3], [0, 5, 50, 95, 100]).round())
nm = z["ood_test/names"]; lg = np.array([n.startswith("larger") for n in nm]); t2 = np.nan_to_num(z["ood_test/cloth"][:, :, 0]) + np.nan_to_num(z["ood_test/back"][:, :, 0])
print("   ood px frame 3: smaller", np.percentile(t2[~lg, 3], [0, 50, 100]).round(), "larger", np.percentile(t2[lg, 3], [0, 50, 100]).round(), "| dims smaller", z["ood_test/cloth_dimx"][~lg][:6], "larger", z["ood_test/cloth_dimx"][lg][:6])
# blue/red ball start positions: same in every episode?
for b in ("blue", "red"):
    c = z[f"ind_train/{b}"][:, 0, 1:]; print(f"   {b} ball frame-0 centroid train: std {c.std(0).round(2)} mean {c.mean(0).round(1)} | ood larger mean {z[f'ood_test/{b}'][lg, 0, 1:].mean(0).round(1)} smaller {z[f'ood_test/{b}'][~lg, 0, 1:].mean(0).round(1)}")

# ---- stack cube: how much of the joint motion between kept frames do the action fields determine?
meta = torch.load(os.path.join(ROOT, "rigid_dynamics/stack_cube/ind_train/metadata.pt"), map_location="cpu", weights_only=True)[:500]
def r2(X, Y):
    X = np.concatenate([X, np.ones((len(X), 1))], 1); c, *_ = np.linalg.lstsq(X, Y, rcond=None); res = Y - X @ c
    return 1 - (res ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
for k in (1, 5):
    feats = {"delta a[t..t+k-1]": [], "abs[t+k]-abs[t]": [], "both": [], "abs[t..t+k] pos+rot": []}; Y = []
    for e in meta:
        q = e["states"][:, 1:8].numpy(); a = e["actions"][:, :6].numpy(); ab = e["actions_abs"][:, :6].numpy(); T = len(q)
        for t in range(0, T - k - 1):
            Y.append(q[t + k] - q[t])
            d = a[t:t + k].reshape(-1); da = ab[t + k] - ab[t]
            feats["delta a[t..t+k-1]"].append(d); feats["abs[t+k]-abs[t]"].append(da); feats["both"].append(np.concatenate([d, da])); feats["abs[t..t+k] pos+rot"].append(ab[t:t + k + 1].reshape(-1))
    Y = np.array(Y)
    print(f"stack_cube: joint displacement over {k} kept frame(s), linear R2 per joint (7) and mean:")
    for n_, X in feats.items():
        r = r2(np.array(X), Y); print(f"   {n_:24s} {r.round(3)}  mean {r.mean():.3f}")
e = meta[0]; ab = e["actions_abs"].numpy(); a = e["actions"].numpy()
print("   abs target step |dpos| per frame (m): median", np.median(np.linalg.norm(np.diff(ab[:, :3], axis=0), axis=1)).round(4), "| |delta a[:3]|*0.05 median", np.median(np.linalg.norm(a[:, :3], axis=1) * 0.05).round(4))
print("   gripper: a6 == abs6 everywhere:", bool((e["actions"][:, 6] == e["actions_abs"][:, 6]).all()), "| src_demo_labels unique", torch.unique(e["src_demo_labels"]).tolist(), "src_demo_inds", e["src_demo_inds"].tolist())
