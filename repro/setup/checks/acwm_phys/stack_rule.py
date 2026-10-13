"""Stack Cube: 마지막 프레임에서 노란 큐브 중심이 초록 큐브 중심 바로 위에 있는지(화면 규칙)가 metadata 의 success 와 얼마나 맞는지,
쌓임이 완성되는 프레임, 실패 에피소드에서 큐브를 들었는지 (SETTINGS.md §8.3 의 쌓임 판정 근거).

usage: python repro/setup/checks/acwm_phys/stack_rule.py <dir with *_scan.npz>"""
import sys, os, numpy as np
z = np.load(os.path.join(sys.argv[1], "stack_cube_scan.npz")); np.set_printoptions(suppress=True, precision=1, linewidth=200)
for sp in ("ind_train", "ood_test"):
    n = z[f"{sp}/n_frames"]; suc = z[f"{sp}/success"] > 0.5; y = z[f"{sp}/yellow"]; g = z[f"{sp}/green"]
    for tsel, name in ((lambda i: n[i] - 1, "last frame"), (lambda i: 50, "frame 50")):
        Y = np.array([y[i, tsel(i)] for i in range(len(n))]); G = np.array([g[i, tsel(i)] for i in range(len(n))])
        dx = Y[:, 1] - G[:, 1]; dy = Y[:, 2] - G[:, 2]
        vis = np.isfinite(dx)
        print(f"[{sp}] {name}: both cubes visible {vis.mean():.3f} (success {vis[suc].mean():.3f}, fail {vis[~suc].mean():.3f})")
        for lab, m in (("success", suc & vis), ("fail", ~suc & vis)):
            print(f"    {lab}: n {m.sum()}  dx pct[1,25,50,75,99] {np.percentile(dx[m], [1, 25, 50, 75, 99])}  dy {np.percentile(dy[m], [1, 25, 50, 75, 99])}  yellow px {np.percentile(Y[m, 0], [1, 50, 99])} green px {np.percentile(G[m, 0], [1, 50, 99])}")
        rule = vis & (np.abs(dx) < 25) & (dy < -15) & (dy > -80)
        tp = (rule & suc).sum(); fp = (rule & ~suc).sum(); fn = (~rule & suc).sum(); tn = (~rule & ~suc).sum()
        print(f"    rule |dx|<25 & -80<dy<-15 (512 px): acc {(tp + tn) / len(n):.3f}  TP {tp} FP {fp} FN {fn} TN {tn}")
    # when does the stack happen: first frame at which the state says stacked (dz in [0.04, 0.075] and dxy < 0.03)
    s = z[f"{sp}/states"]; dz = s[:, :, 12] - s[:, :, 19]; dxy = np.linalg.norm(s[:, :, 10:12] - s[:, :, 17:19], axis=2)
    st = (dz > 0.04) & (dz < 0.075) & (dxy < 0.03)
    first = np.array([np.argmax(st[i, :n[i]]) if st[i, :n[i]].any() else -1 for i in range(len(n))])
    fin = np.array([st[i, n[i] - 1] for i in range(len(n))])
    print(f"    state-stacked at last frame vs success flag: agree {np.mean(fin == suc):.3f} | first stacked frame (success eps) pct[1,50,99] {np.percentile(first[suc & (first >= 0)], [1, 50, 99])} | lifted (cubeA z rise > 0.02 at any time): success {np.mean([(s[i, :n[i], 12] - s[i, 0, 12]).max() > 0.02 for i in np.where(suc)[0]]):.3f} fail {np.mean([(s[i, :n[i], 12] - s[i, 0, 12]).max() > 0.02 for i in np.where(~suc)[0]]):.3f}")
