"""runs/{train,eval}/<run-id>/run.yaml 을 만든다 (설정만, 결과 수치는 넣지 않는다).
jobs/benchmark.txt 의 인자와 아래 표에서 만들고, 시작한 날짜는 $DINO_RUNS/<job>/run_info.txt 에서 읽는다.
usage: source repro/env.sh && python3 repro/make_run_yaml.py
"""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = HERE  # repro/
RUNS = os.environ["DINO_RUNS"]
JOBS = os.path.join(HERE, "jobs", "benchmark.txt")

RELEASED = {  # 공개 체크포인트: hydra.yaml 의 값과 model_latest.pth 의 epoch
    "point_maze": dict(rid="dinowm_pointmaze-released", epoch=10, env="point_maze", hist=3, fs=5),
    "pusht": dict(rid="dinowm_pusht-released", epoch=2, env="pusht", hist=3, fs=5),
    "wall_single": dict(rid="dinowm_wall-released", epoch=65, env="wall", hist=1, fs=5),
}
PLAN_CFG = {  # upstream conf/plan_*.yaml (CEM 설정, goal, alpha)
    "plan_point_maze.yaml": dict(goal_source="random_state", alpha=0, opt_steps=10),
    "plan_pusht.yaml": dict(goal_source="dset", alpha=1, opt_steps=30),
    "plan_wall.yaml": dict(goal_source="random_state", alpha=1, opt_steps=10),
}
# 머신별 실행 이력(배치·결과에 쓰지 않은 시도·OOM)은 레포에 올리지 않는다 (예: .claude/runs.local.md)


def parse_jobs():
    jobs = []
    for line in open(JOBS):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        t = line.split()
        name, kind, rest = t[0], t[1], t[2:]
        opts = {k: v for k, v in (x.split("=", 1) for x in rest if re.match(r"^(gpu|need|after)=", x))}
        env = dict(x[4:].split("=", 1) for x in rest if x.startswith("env:"))
        args = [x for x in rest if not re.match(r"^(gpu|need|after)=", x) and not x.startswith("env:")]
        jobs.append(dict(name=name, kind=kind, opts=opts, env=env, args=args))
    return jobs


def status(job):
    d = os.path.join(RUNS, job)
    if os.path.exists(f"{d}/DONE"):
        return "done"
    if not os.path.exists(f"{d}/run_info.txt"):
        return "queued"
    return "failed" if "END" in open(f"{d}/run_info.txt").read() else "running"


def started(job, path):
    """시작한 날: $DINO_RUNS/<job>/run_info.txt, 없으면 기존 run.yaml 의 date 를 유지한다."""
    p = os.path.join(RUNS, job, "run_info.txt")
    if os.path.exists(p):
        m = re.search(r"START (\d{4}-\d{2}-\d{2})", open(p).read())
        if m:
            return m.group(1)
    if os.path.exists(path):
        m = re.search(r"^date: '?([0-9-]+)'?$", open(path).read(), flags=re.M)
        if m:
            return m.group(1)
    return None


def hydra_args(args):
    return {a.split("=", 1)[0]: a.split("=", 1)[1] for a in args if "=" in a and not a.startswith("--")}


def q(s):
    s = str(s)
    special = re.search(r"[:#{}\[\],&*?|<>=!%@`]", s) or s == "" or s.startswith("-")
    return "'" + s.replace("'", "''") + "'" if special else s


def dump(path, d, header):
    lines = [header]

    def emit(k, v, ind=0):
        pad = "  " * ind
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            for kk, vv in v.items():
                emit(kk, vv, ind + 1)
        elif isinstance(v, str) and len(v) > 90:
            lines.append(f"{pad}{k}: >-")
            words, cur = v.split(" "), ""
            for w in words:
                if len(cur) + len(w) + 1 > 100 and cur:
                    lines.append(f"{pad}  {cur}")
                    cur = w
                else:
                    cur = f"{cur} {w}".strip()
            lines.append(f"{pad}  {cur}")
        else:
            lines.append(f"{pad}{k}: {q(v)}")

    for k, v in d.items():
        emit(k, v)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(lines) + "\n")


HEADER = "# 이 실행이 무엇이었는지의 단일 출처 (repro/make_run_yaml.py 로 생성). 결과 수치는 넣지 않는다 — 이슈로 올린다."
CODE = "hanbin5/local: gaoyuezhou/dino_wm 0a9492f + main 공용 수정 + 이 브랜치의 24 GB 수정 (git diff 0a9492f -- . ':!repro' ':!*.md' ':!.gitignore')"
RUNNER_ENV = "DINO_WM_SDPA=1 DINO_WM_SKIP_SOLVED=1 DINO_WM_ROLLOUT_CHUNK=150 DINO_WM_GD_CHUNK=8 (이 브랜치 run_plan.sh 기본값; 청크는 결과와 무관)"
written = []
for j in parse_jobs():
    name, kind, args = j["name"], j["kind"], j["args"]
    common = dict(stack="dinowm", job=name)
    if kind == "train":
        obj = hydra_args(args)["env.dataset.object_name"]
        rid = f"dinowm_{obj}-dinov2s14-100ep"
        d = dict(id=rid, kind="train", **common, env=obj, dataset=f"$DATASET_DIR/deformable/{obj} (1000 궤적 × 20 프레임, 무작위 0.9/0.1 seed 42)",
                 model=dict(encoder="DINOv2 ViT-S/14 patch tokens, 동결 (hub 85a2460)", predictor="ViT depth 6, heads 16, mlp 2048, dropout 0.1",
                            decoder="VQ-VAE decoder (quantize False)", num_hist=1, num_pred=1, frameskip=1, img_size=224),
                 training=dict(epochs=100, batch_size=32, optimizer="predictor·action encoder AdamW, decoder Adam",
                               lr="decoder 3e-4, predictor 5e-4, action encoder 5e-4 (upstream 기본값)", seed=0, save_every_x_epoch=10),
                 command=f"bash repro/queue/run_train.sh {name} <gpu> " + " ".join(args),
                 output=f"$DINO_TRAIN/outputs/{name}/ (hydra.yaml, checkpoints/model_<epoch>.pth, epoch_logs.jsonl)", code=CODE)
        path = os.path.join(EXP, "runs", "train", rid, "run.yaml")
    elif kind == "plan":
        h = hydra_args(args)
        cfg = [a for a in args if a.startswith("--config-name")]
        cfgname = args[args.index("--config-name") + 1] if "--config-name" in args else None
        mname = h["model_name"]
        if mname in RELEASED:
            r = RELEASED[mname]
            base, env_name, epoch = r["rid"], r["env"], r["epoch"]
            model = dict(source="공개 체크포인트 (OSF checkpoints/outputs.zip)", ckpt=f"$DINO_CKPT/outputs/{mname}/checkpoints/model_latest.pth",
                         epoch=epoch, num_hist=r["hist"], frameskip=r["fs"])
        else:
            obj = mname.replace("train_", "")
            base, env_name, epoch = f"dinowm_{obj}-dinov2s14-100ep", obj, 100
            model = dict(source=f"직접 학습 runs/train/{base}", ckpt=f"$DINO_TRAIN/outputs/{mname}/checkpoints/model_latest.pth", epoch=epoch,
                         num_hist=1, frameskip=1)
        envshort = {"point_maze": "pointmaze", "pusht": "pusht", "wall": "wall"}.get(env_name, env_name)
        if cfgname in PLAN_CFG:
            p = PLAN_CFG[cfgname]
            variant = "mpccem"
            planner = dict(name="MPC-CEM (upstream 설정 파일 내장)", cem=f"horizon = goal_H 5, samples 300, topk 30, var_scale 1, opt_steps {p['opt_steps']}",
                           n_taken_actions="goal_H 5 (plan.py 가 덮어쓴다)", max_iter=f"{h['planner.max_iter']} (원본 null = 무제한)", eval_every=1)
            protocol = dict(n_evals=50, goal_source=p["goal_source"], goal_H=5, objective=f"마지막 프레임 latent MSE, alpha {p['alpha']}", seed=99)
            read = "logs.json 의 mpc/success_rate (step k = MPC k회 예산, sticky). step 1 = 오픈루프 CEM (SETTINGS.md §5.3)"
        else:
            pl = h["planner"]
            n = int(h.get("n_evals", 10))
            protocol = dict(n_evals=n, goal_source=h["goal_source"], goal_H=int(h["goal_H"]),
                            objective=f"마지막 프레임 latent MSE, alpha {h.get('objective.alpha', '1 (plan.yaml 기본값)')}", seed=99)
            if pl == "gd":
                variant = "gd"
                planner = dict(name="오픈루프 GD (conf/planner/gd.yaml)", gd="horizon = goal_H 5, SGD lr 1, 1000 steps, action_noise 0.003, randn 초기화",
                               eval_every=10)
                read = "logs.json 의 final_eval/success_rate"
            elif pl == "cem":
                variant = "cem30"
                planner = dict(name="오픈루프 CEM (conf/planner/cem.yaml)", cem="horizon = goal_H 5, samples 300, topk 30, var_scale 1, opt_steps 30",
                               eval_every=1)
                read = "logs.json 의 final_eval/success_rate"
            else:  # mpc_cem for deformables
                variant = "mpccem"
                planner = dict(name="MPC-CEM (conf/planner/mpc_cem.yaml)", cem="horizon = goal_H 5, samples 300, topk 30, var_scale 1, opt_steps 30",
                               n_taken_actions="goal_H 5", max_iter=f"{h['planner.max_iter']} (원본 null = 무제한; deformable 은 success 가 항상 False)",
                               eval_every=f"{h['planner.sub_planner.eval_every']} (내부 CEM 시뮬 평가는 opt step 0 에서만)")
                read = "logs.json 의 mpc/mean_chamfer_distance (반복별), final_eval/mean_chamfer_distance"
        rid = f"{base}-at{epoch}-{variant}"
        d = dict(id=rid, kind="eval", **common, env=env_name, model=model, planner=planner, protocol=protocol,
                 command=("CKPT_BASE=$DINO_TRAIN " if j["env"] else "") + f"bash repro/queue/run_plan.sh {name} <gpu> " + " ".join(args),
                 runner_env=RUNNER_ENV, code=CODE, output=f"$DINO_RUNS/{name}/", results=read)
        if j["env"]:
            d["job_env"] = " ".join(f"{k}={v}" for k, v in j["env"].items()) + " (jobs/benchmark.txt)"
        path = os.path.join(EXP, "runs", "eval", rid, "run.yaml")
    else:  # cmd: prediction quality
        m = re.search(r"eval_pred_quality\.py (\S+) (\S+) (\S+) (\d+)", " ".join(args))
        ckroot, mname, out, cap = m.groups()
        if mname in RELEASED:
            r = RELEASED[mname]
            base, epoch, env_name = r["rid"], r["epoch"], r["env"]
        else:
            obj = mname.replace("train_", "")
            base, epoch, env_name = f"dinowm_{obj}-dinov2s14-100ep", 100, obj
        rid = f"{base}-at{epoch}-predq"
        d = dict(id=rid, kind="eval", **common, env=env_name,
                 model=dict(ckpt=f"{ckroot}/outputs/{mname}/checkpoints/model_latest.pth", epoch=epoch),
                 protocol=dict(split="검증 분할 (PushT 는 val 폴더 21 궤적, 그 밖은 학습과 같은 seed 42 궤적 분할)", samples=f"슬라이스 전체. {cap} 개를 넘으면 seed(training.seed) 로 만든 슬라이스 순서에서 균등 간격 {cap} 개",
                               metric="model.eval() 의 1-step 예측을 디코딩해 정답 프레임과 LPIPS(VGG)·SSIM(11×11) — upstream train.py 검증과 같은 계산"),
                 command=f"bash repro/queue/run_cmd.sh {name} <gpu> " + " ".join(args), code=CODE,
                 output=out, results="output json 의 pred_lpips, pred_ssim")
        path = os.path.join(EXP, "runs", "eval", rid, "run.yaml")
    d = {**{k: d[k] for k in ("id", "kind", "stack", "job")}, "date": started(name, path) or "-", **{k: v for k, v in d.items() if k not in ("id", "kind", "stack", "job")}}
    dump(path, d, HEADER)
    written.append(os.path.relpath(path, EXP))
print("\n".join(written))
