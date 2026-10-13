"""num_hist 를 고르는 참고 지표: 스캔 특징(픽셀 수·중심, Stack Cube 는 관절·큐브 위치)의 한 스텝 변화를
"현재 값 + 이번 action" 으로 맞춘 R2 와, 거기에 "이전 스텝의 변화 + 이전 action" 을 더한 R2 (ridge, 에피소드 5개 중 1개로 검증).
선형 대리 지표다 (SETTINGS.md §8.2).

usage: python repro/setup/checks/acwm_phys/history.py <dir with *_scan.npz>"""
import sys, os, numpy as np
D = sys.argv[1]
def fit(Xtr, Ytr, Xte, Yte, lam=1e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; A = (Xtr - mu) / sd; B = (Xte - mu) / sd; ym = Ytr.mean(0)
    W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (Ytr - ym))
    return 1 - ((Yte - ym - B @ W) ** 2).sum(0) / ((Yte - Yte.mean(0)) ** 2).sum(0)
def run(env, feat_fn, act_fn, f, N=1200):
    z = np.load(os.path.join(D, f"{env}_scan.npz")); nf = z["ind_train/n_frames"][:N]
    F = feat_fn(z)[:N]; A = act_fn(z)[:N]
    rows = {"cur": [], "hist": [], "y": [], "ep": []}
    for i in range(len(nf)):
        T = min(nf[i], np.isfinite(A[i, :, 0]).sum())
        for t in range(f, T - f):
            s, s0, s1 = F[i, t], F[i, t - f], F[i, t + f]
            if not (np.all(np.isfinite(s)) and np.all(np.isfinite(s0)) and np.all(np.isfinite(s1))): continue
            a = A[i, t:t + f].reshape(-1); ap = A[i, t - f:t].reshape(-1)
            rows["cur"].append(np.concatenate([s, a])); rows["hist"].append(np.concatenate([s, a, s - s0, ap])); rows["y"].append(s1 - s); rows["ep"].append(i)
    ep = np.array(rows["ep"]); te = ep % 5 == 0; Y = np.array(rows["y"])
    r1 = fit(np.array(rows["cur"])[~te], Y[~te], np.array(rows["cur"])[te], Y[te]); r2 = fit(np.array(rows["hist"])[~te], Y[~te], np.array(rows["hist"])[te], Y[te])
    print(f"{env} frameskip {f}: n={len(Y)}  R2 of the change over one step, per feature\n    current frame + action     {np.round(r1, 3)}  mean {r1.mean():.3f}\n    + previous step (velocity) {np.round(r2, 3)}  mean {r2.mean():.3f}")
cl = lambda z: np.concatenate([np.nan_to_num(z["ind_train/cloth"][:, :, :1]), np.nan_to_num(z["ind_train/back"][:, :, :1]), z["ind_train/cloth"][:, :, 1:], z["ind_train/blue"][:, :, 1:]], 2)
run("cloth_move", cl, lambda z: z["ind_train/actions"][:, :, [0, 1, 2, 4, 5, 6]], 5)
run("cloth_move", cl, lambda z: z["ind_train/actions"][:, :, [0, 1, 2, 4, 5, 6]], 1)
def sand_act(z):
    a = z["ind_train/actions"]; th = 2 * np.arctan2(a[:, :, 4], a[:, :, 6]); nxt = np.concatenate([a[:, 1:, [0, 2]], np.cos(th)[:, 1:, None], np.sin(th)[:, 1:, None]], 2)
    return np.concatenate([nxt, np.full((len(a), 1, 4), np.nan)], 1)
sd = lambda z: np.concatenate([z["ind_train/sand"][:, :, :]], 2)
run("push_sand", sd, sand_act, 5)
st = lambda z: np.concatenate([z["ind_train/states"][:, :, 1:8], z["ind_train/states"][:, :, 10:13]], 2)
run("stack_cube", st, lambda z: z["ind_train/actions"], 1)
run("stack_cube", st, lambda z: z["ind_train/actions"], 5)
