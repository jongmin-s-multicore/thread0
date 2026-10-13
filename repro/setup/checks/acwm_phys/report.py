"""scan.py 결과에서 (1) split 별 프레임 수·해상도·변화량·색 픽셀 수의 분포와 이상치, (2) 프레임과 action 의 정렬을 낸다.
정렬: 화면에서 잰 조작부(그리퍼 공, 밀판 막대, 큐브·관절)의 한 스텝 변화를 action[t+j] (j = -3..+3) 에 회귀해 어느 j 가 설명하는지 본다.
j = 0 이 크면 action t 가 프레임 t -> t+1 을 만든다. Push Sand 는 action 의 차분 a[t+1] - a[t] 으로 본다 (action 이 자세이므로).

usage: python repro/setup/checks/acwm_phys/report.py <dir with *_scan.npz>"""
import sys, os, numpy as np
D = sys.argv[1]
SPLITS = ("ind_train", "ind_test", "ood_test")
np.set_printoptions(linewidth=200, suppress=True)

def q(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return "min %.4g p1 %.4g med %.4g p99 %.4g max %.4g" % (x.min(), np.percentile(x, 1), np.median(x), np.percentile(x, 99), x.max())

def joint_lag(y_list, x_list, J=(-3, -2, -1, 0, 1, 2, 3)):
    """y_list[i]: (T,) ; x_list[i]: (T, d). Regress y[t] on x[t+j] for all j jointly; return per-lag coef norms (std units) and single-lag R2."""
    rows, ys = [], []
    for y, x in zip(y_list, x_list):
        T = min(len(y), len(x))
        for t in range(3, T - 3):
            v = [x[t + j] for j in J]
            if np.isfinite(y[t]) and np.all(np.isfinite(v)):
                rows.append(np.concatenate(v)); ys.append(y[t])
    X = np.array(rows); y = np.array(ys); y = y - y.mean()
    sd = X.std(0); sd[sd == 0] = 1; X = (X - X.mean(0)) / sd
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ coef) ** 2).sum() / (y ** 2).sum()
    d = X.shape[1] // len(J)
    single = []
    for k in range(len(J)):
        Xj = X[:, k * d:(k + 1) * d]; c, *_ = np.linalg.lstsq(Xj, y, rcond=None)
        single.append(1 - ((y - Xj @ c) ** 2).sum() / (y ** 2).sum())
    return r2, np.abs(coef.reshape(len(J), d)).max(1) / (y.std() + 1e-12), single, len(y)

def show(title, res, J=(-3, -2, -1, 0, 1, 2, 3)):
    r2, cn, single, n = res
    print(f"  {title}: joint R2 {r2:.3f} (n={n})")
    print("     lag j      : " + "  ".join(f"{j:+d}".rjust(6) for j in J))
    print("     max|coef|  : " + "  ".join(f"{c:6.3f}" for c in cn))
    print("     single R2  : " + "  ".join(f"{s:6.3f}" for s in single))

for env in ("cloth_move", "push_sand", "stack_cube"):
    z = np.load(os.path.join(D, f"{env}_scan.npz"))
    print("=" * 20, env)
    for sp in SPLITS:
        nf, na, ml, hw = z[f"{sp}/n_frames"], z[f"{sp}/n_actions"], z[f"{sp}/meta_len"], z[f"{sp}/hw"]
        mo = z[f"{sp}/motion"]
        print(f"[{sp}] n={len(nf)} frames {dict(zip(*np.unique(nf, return_counts=True)))} frames-actions {dict(zip(*np.unique(nf - na, return_counts=True)))} meta_len-actions {dict(zip(*np.unique(ml - na, return_counts=True)))} hw {np.unique(hw, axis=0).tolist()}")
        tot = np.nansum(mo, 1); mx = np.nanmax(mo, 1)
        print(f"   motion per episode: total {q(tot)} | max step {q(mx)} | mean0 {q(z[f'{sp}/mean0'])}")
        bb = z[f"{sp}/bbox"]; ok = bb[:, 0] >= 0
        print(f"   dynamic bbox union x {bb[ok, 0].min()}..{bb[ok, 2].max()} y {bb[ok, 1].min()}..{bb[ok, 3].max()} | p1/p99 x0 {np.percentile(bb[ok,0],1):.0f} x1 {np.percentile(bb[ok,2],99):.0f} y0 {np.percentile(bb[ok,1],1):.0f} y1 {np.percentile(bb[ok,3],99):.0f} | episodes with no change {int((~ok).sum())}")
        lowmo = np.argsort(tot)[:5]; himx = np.argsort(mx)[-5:]
        print("   least motion:", [(str(z[f'{sp}/names'][i]), round(float(tot[i]), 2)) for i in lowmo])
        print("   largest single-step motion:", [(str(z[f'{sp}/names'][i]), round(float(mx[i]), 2), int(np.nanargmax(mo[i]))) for i in himx])
        keys = [k.split("/")[1] for k in z.files if k.startswith(sp + "/") and z[k].ndim == 3 and z[k].shape[2] == 3]
        for k in keys:
            c = z[f"{sp}/{k}"]; last = np.array([c[i, nf[i] - 1, 0] for i in range(len(nf))])
            mn = np.nanmin(c[:, :, 0], 1)
            print(f"   {k:7s} px frame0 {q(c[:, 0, 0])} | last {q(last)} | min over time {q(mn)}")
        act = z[f"{sp}/actions"]
        a = act.reshape(-1, act.shape[2]); a = a[np.isfinite(a[:, 0])]
        print("   action mean", np.round(a.mean(0), 4), "std", np.round(a.std(0), 4))
        # leading / trailing zero-delta steps
    # ---------- alignment (train, first 400 episodes) ----------
    sp = "ind_train"; N = 400
    nf = z[f"{sp}/n_frames"][:N]; act = z[f"{sp}/actions"][:N]
    if env == "cloth_move":
        for ball in ("blue", "red"):
            c = z[f"{sp}/{ball}"][:N]
            for ax, nm in ((1, "x"), (2, "y")):
                ys = [np.diff(c[i, :nf[i], ax]) for i in range(N)]  # d[t] = c[t+1] - c[t]
                for dims, lab in (((0, 1, 2), "a[0:3]"), ((4, 5, 6), "a[4:7]")):
                    xs = [act[i, :nf[i] - 1][:, dims] for i in range(N)]
                    show(f"{ball} ball d{nm}[t]=c[t+1]-c[t] on {lab}[t+j]", joint_lag(ys, xs))
    if env == "push_sand":
        for part in ("stick", "board"):
            c = z[f"{sp}/{part}"][:N]
            for ax, nm in ((1, "x"), (2, "y")):
                ys = [np.diff(c[i, :nf[i], ax]) for i in range(N)]
                xs = [np.diff(act[i, :nf[i]][:, (0, 2)], axis=0) for i in range(N)]  # da[t] = a[t+1] - a[t]
                show(f"{part} d{nm}[t]=c[t+1]-c[t] on da[t+j]=a[t+j+1]-a[t+j] (x,z)", joint_lag(ys, xs))
        # position fit at lags: c[t] ~ affine(a[t+j])
        c = z[f"{sp}/stick"][:N]
        for j in (-2, -1, 0, 1, 2):
            X, Y = [], []
            for i in range(N):
                for t in range(2, nf[i] - 2):
                    if np.isfinite(c[i, t, 1]) and np.isfinite(act[i, t + j, 0]):
                        X.append([act[i, t + j, 0], act[i, t + j, 2], 1.0]); Y.append(c[i, t, 1:3])
            X, Y = np.array(X), np.array(Y); co, *_ = np.linalg.lstsq(X, Y, rcond=None); res = Y - X @ co
            print(f"  stick position c[t] ~ affine(a[t{j:+d}] x,z): rms residual {np.sqrt((res ** 2).mean()):.2f} px  (n={len(Y)})")
    if env == "stack_cube":
        st = z[f"{sp}/states"][:N]
        c = z[f"{sp}/yellow"][:N]
        for ax, nm in ((1, "x"), (2, "y")):
            ys = [np.diff(c[i, :nf[i], ax]) for i in range(N)]
            xs = [np.diff(st[i, :nf[i], 10:13], axis=0) for i in range(N)]  # cube A displacement s[t+1]-s[t]
            show(f"yellow cube image d{nm}[t] on state cubeA d[t+j]", joint_lag(ys, xs))
        ys = [np.diff(st[i, :nf[i], 8]) for i in range(N)]  # finger joint velocity
        xs = [act[i, :nf[i] - 1][:, 6:7] for i in range(N)]
        show("finger dq[t]=q[t+1]-q[t] on gripper action a6[t+j]", joint_lag(ys, xs))
        for jn in (1, 2, 4):
            ys = [np.diff(st[i, :nf[i], jn]) for i in range(N)]
            xs = [act[i, :nf[i] - 1][:, :6] for i in range(N)]
            show(f"arm joint{jn} dq[t] on a[0:6][t+j]", joint_lag(ys, xs))
        print("  state time step:", np.unique(np.round(np.diff(st[:, :50, 0], axis=1), 3)))
        for s2 in SPLITS:
            suc = z[f"{s2}/success"]; lab = z[f"{s2}/ood_label"]; s = z[f"{s2}/states"]; n2 = z[f"{s2}/n_frames"]
            dz = np.array([s[i, n2[i] - 1, 12] - s[i, n2[i] - 1, 19] for i in range(len(n2))])
            dxy = np.array([np.linalg.norm(s[i, n2[i] - 1, 10:12] - s[i, n2[i] - 1, 17:19]) for i in range(len(n2))])
            print(f"  [{s2}] success {suc.mean():.3f} labels {dict(zip(*np.unique(lab, return_counts=True)))} | final cubeA-cubeB dz: success {q(dz[suc > 0.5])} ; fail {q(dz[suc < 0.5]) if (suc < 0.5).any() else 'n/a'} | dxy success {q(dxy[suc > 0.5])}")
            a0 = s[:, 0, 10:12]; b0 = s[:, 0, 17:19]
            print(f"        cubeA xy0 x[{a0[:,0].min():.3f},{a0[:,0].max():.3f}] y[{a0[:,1].min():.3f},{a0[:,1].max():.3f}]  cubeB xy0 x[{b0[:,0].min():.3f},{b0[:,0].max():.3f}] y[{b0[:,1].min():.3f},{b0[:,1].max():.3f}]  mean (B-A) {np.round((b0 - a0).mean(0), 3)}")
            if s2 == "ood_test":
                for L in np.unique(lab):
                    m = lab == L; print(f"        {L}: n {m.sum()} mean (B-A) xy {np.round((b0[m] - a0[m]).mean(0), 3)} success {suc[m].mean():.2f}")
