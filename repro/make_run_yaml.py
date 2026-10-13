"""runs/{train,eval}/<run-id>/run.yaml 을 만든다 (설정만, 결과 수치는 넣지 않는다).
jobs/ 의 작업 목록(JOB_FILES)과 아래 표에서 만들고, 시작한 날짜는 $DINO_RUNS/<job>/run_info.txt 에서 읽는다.
usage: source repro/env.sh && python3 repro/make_run_yaml.py
"""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = HERE  # repro/
RUNS = os.environ["DINO_RUNS"]
# 벤치마크와 그 뒤의 추가 실행 (시드 추가·원본 코드 경로, Rope predictor lr 5e-5 재학습, ACWM-Phys Pour Water, ACWM-Phys Cloth Move·Push Sand·Stack Cube)
JOB_FILES = ["benchmark.txt", "pointmaze_extra.txt", "rope_plr.txt", "pour_water.txt", "acwm_phys.txt"]
# 큐 작업이 아닌 실행: 평가 영상에서 로봇 팔이 움직이게 같은 인자·환경으로 다시 돌린 것 (repro/eval/deform_arm_video.py)
ARM_RERUNS = {"rope_mpc_arm": "rope_mpc", "granular_mpc_arm": "granular_mpc"}
ARM_TOOL = "repro/eval/deform_arm_video.py (실행할 때는 커밋 전 main 작업 트리의 파일, 같은 동작으로 main 5180303 에 커밋)"
LR_DEFAULT = {"decoder": "3e-4", "predictor": "5e-4", "action_encoder": "5e-4"}  # upstream conf/train.yaml

ACWM = {  # ACWM-Phys Cloth Move·Push Sand·Stack Cube (SETTINGS.md §8): job 이름의 환경 -> run-id 의 이름, 데이터, 평가 세트
    "cloth": dict(rid="clothmove", dataset="$DATASET_DIR/acwm_phys/deformable/clothmove (HF t1an/ACWM-Phys 4017e5f). ind_train 2000개를 무작위 0.9/0.1 seed 42 로 나눠 "
                  "학습 1800 / 검증 200 에피소드, 에피소드마다 65프레임·action 64개. action 은 상수인 두 차원을 뺀 6차원 (action_format raw)",
                  sets="val 200, ind_test 100, ood_test 100 (파일 이름으로 smaller 50 / larger 50)",
                  quantities="천(주황 + 분홍 뒷면) 픽셀 수·IoU·중심, 그리퍼 공 두 개의 중심"),
    "sand": dict(rid="pushsand", dataset="$DATASET_DIR/acwm_phys/particle/push_sand (HF t1an/ACWM-Phys 4017e5f). ind_train 1784개를 무작위 0.9/0.1 seed 42 로 나눠 "
                 "학습 1605 / 검증 179 에피소드, 에피소드마다 64프레임. action 은 다음 프레임의 밀판 자세 (x, z, cos yaw, sin yaw) 4차원 (action_format pose_next), "
                 "304×512 프레임을 자르지 않고 224×224 로 줄인다",
                 sets="val 179, ind_test 100, ood_test 100 (입자 약 2배)",
                 quantities="모래(하늘색) 픽셀 수·IoU·중심, 밀판(가장 큰 어두운 덩어리)의 중심"),
    "stack": dict(rid="stackcube", dataset="$DATASET_DIR/acwm_phys/rigid_dynamics/stack_cube (HF t1an/ACWM-Phys 4017e5f). ind_train 1373개(쌓기 실패 373개 포함)를 무작위 0.9/0.1 seed 42 로 나눠 "
                  "학습 1235 / 검증 138 에피소드, 에피소드마다 53–58프레임. action 7차원 그대로 (action_format raw)",
                  sets="val 138, ind_test 67, ood_test 271 (metadata 의 success / fail, ood_test 는 up·down·left·right 도)",
                  quantities="노란·초록 큐브의 픽셀 수·IoU·중심, 마지막 프레임의 쌓임 판정"),
}

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
    for fname in JOB_FILES:
        for line in open(os.path.join(HERE, "jobs", fname)):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            t = line.split()
            name, kind, rest = t[0], t[1], t[2:]
            opts = {k: v for k, v in (x.split("=", 1) for x in rest if re.match(r"^(gpu|need|after)=", x))}
            env = dict(x[4:].split("=", 1) for x in rest if x.startswith("env:"))
            args = [x for x in rest if not re.match(r"^(gpu|need|after)=", x) and not x.startswith("env:")]
            jobs.append(dict(name=name, kind=kind, opts=opts, env=env, args=args, src=fname))
    by_name = {j["name"]: j for j in jobs}
    for name, orig in ARM_RERUNS.items():
        if os.path.exists(os.path.join(RUNS, name, "run_info.txt")):
            jobs.append({**by_name[orig], "name": name, "arm": True})
    return jobs


def trained(mname):
    """직접 학습한 모델의 job 이름 train_<obj>[_<tag>] -> (obj, run-id 앞부분, tag)."""
    obj, _, tag = mname[len("train_"):].partition("_")
    return obj, f"dinowm_{obj}-dinov2s14-100ep" + (f"-{tag}" if tag else ""), tag


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
CODE = "upstream gaoyuezhou/dino_wm 0a9492f + main 공용 수정 + 이 브랜치의 24 GB 수정 (git diff 0a9492f -- . ':!repro' ':!*.md' ':!.gitignore')"
# 나누기 전에 시작한 작업: run_info.txt 에 커밋이 없어 여기에 적는다 (LOCAL.md §6). repro/ 밖 코드는 이 브랜치와 같다.
PRE_SPLIT = {
    **{j: "fork 이전 upstream clone 0a9492f 에 같은 diff 를 커밋 없이 패치 (DINOv2 hub 캐시 85a2460)"
       for j in ("train_rope", "train_granular", "pointmaze_mpc", "pusht_mpc", "wall_mpc", "pusht_gd")},
    "wall_gd": "나누기 전 main c34ddd3 (archive/2026-10-01-pre-split 의 조상)",
    "wall_cem30": "나누기 전 main 3a4341b (태그 archive/2026-10-01-pre-split)",
}
OLD_ENV = {"sdpa": "DINO_WM_SDPA", "skip_solved": "DINO_WM_SKIP_SOLVED", "chunk": "DINO_WM_ROLLOUT_CHUNK", "gdchunk": "DINO_WM_GD_CHUNK"}


def run_start(job):
    p = os.path.join(RUNS, job, "run_info.txt")
    if not os.path.exists(p):
        return ""
    return next((l for l in open(p) if l.startswith("START")), "")


def code_of(job):
    m = re.search(r"\bcommit=(\w+)", run_start(job))
    if m and BY_NAME.get(job, {}).get("src") == "acwm_phys.txt":  # 커밋 전 작업 트리로 돌렸고, 그 파일들을 내용 그대로 커밋했다 (LOCAL.md §6)
        return (f"hanbin5/acwm-phys 의 커밋 전 작업 트리 (run_info.txt 의 commit={m.group(1)} 는 그 바탕 커밋). 같은 내용의 커밋: 로더·설정 ebe1aea, 작업 목록 060b292, "
                f"평가 도구 10e8192. 코드: {CODE}")
    if m and "_pw_" in job:  # Pour Water: 실행 당시 브랜치 이름은 hanbin5/local 이었고 그 커밋들은 hanbin5/water 로 옮겼다
        return f"hanbin5/water {m.group(1)} (run_info.txt 의 branch=hanbin5/local 은 옮기기 전 이름). 코드: {CODE}"
    where = f"hanbin5/local {m.group(1)}" if m else PRE_SPLIT.get(job, "hanbin5/local (실행 전)")
    return f"{where}. 코드: {CODE}"


def runner_env_of(job):
    """run_info.txt 에 기록된 DINO_WM_* 값 (이전 형식은 chunk= sdpa= ... 를 옮긴다)."""
    line = run_start(job)
    m = re.search(r"env=\[([^\]]*)\]", line)
    if m:
        kv = dict(x.split("=", 1) for x in m.group(1).split())
    else:
        kv = {OLD_ENV[k]: v for k, v in re.findall(r"\b(sdpa|skip_solved|chunk|gdchunk)=(\S+)", line)}
    if not kv:
        return "-"
    return " ".join(f"{k}={kv[k]}" for k in sorted(kv)) + " (run_info.txt; 청크는 합산 순서만 바꾼다, LOCAL.md §2)"


written = []
JOBS = parse_jobs()
BY_NAME = {j["name"]: j for j in JOBS}
for j in JOBS:
    name, kind, args = j["name"], j["kind"], j["args"]
    common = dict(stack="dinowm", job=name)
    if kind == "train" and name.startswith("train_pw_"):  # ACWM-Phys Pour Water (SETTINGS.md §7)
        h = hydra_args(args)
        rid = f"dinowm_pourwater-dinov2s14-{h['training.epochs']}ep-{name[len('train_pw_'):]}"
        lr = {k: h.get(f"training.{k}_lr", v) for k, v in LR_DEFAULT.items()}
        changed = [k for k in lr if lr[k] != LR_DEFAULT[k]]
        lr_text = f"decoder {lr['decoder']}, predictor {lr['predictor']}, action encoder {lr['action_encoder']} " + (
            "(upstream 기본값)" if not changed else "(" + ", ".join(f"{k} 를 바꿨다 — upstream {LR_DEFAULT[k]}, 논문 Table 12 는 predictor 5e-5" for k in changed) + ")")
        d = dict(id=rid, kind="train", **common, env=h["env"],
                 dataset="$DATASET_DIR/acwm_phys/particle/pour_water (HF t1an/ACWM-Phys 4017e5f). ind_train 1000개에서 episode_60–79 20개(물이 흩어져 붓는 장면이 없다)를 빼고 "
                         "980개를 무작위 0.9/0.1 seed 42 로 나눠 학습 882 / 검증 98 에피소드, 에피소드마다 300프레임",
                 model=dict(encoder="DINOv2 ViT-S/14 patch tokens, 동결 (hub 85a2460)", predictor="ViT depth 6, heads 16, mlp 2048, dropout 0.1",
                            decoder="VQ-VAE decoder (quantize False), 같은 학습 안에서 따로 최적화: 정답 프레임 잠재(detach)를 재구성하는 MSE 만 학습",
                            num_hist=int(h["num_hist"]), num_pred=1, frameskip=int(h["frameskip"]), img_size=224),
                 training=dict(epochs=int(h["training.epochs"]), batch_size=int(h["training.batch_size"]),
                               optimizer="predictor·action encoder AdamW, decoder Adam", lr=lr_text, seed=0,
                               save_every_x_epoch=int(h["training.save_every_x_epoch"]),
                               batch_note="논문은 32. num_hist 3 은 24 GB GPU 에서 32 가 OOM 이라 24 (LOCAL.md §2·§3)"),
                 command=f"bash repro/queue/run_train.sh {name} <gpu> " + " ".join(args),
                 output=f"$DINO_TRAIN/outputs/{name}/ (hydra.yaml, checkpoints/model_<epoch>.pth, epoch_logs.jsonl)", code=code_of(name))
        path = os.path.join(EXP, "runs", "train", rid, "run.yaml")
    elif kind == "cmd" and "eval_pour_water.py" in " ".join(args):
        m = re.search(r"eval_pour_water\.py (\S+) (\S+) (\S+)(?: --epoch (\d+))?", " ".join(args))
        ckroot, mname, out, ep = m.groups()
        th = hydra_args(BY_NAME[mname]["args"])
        epoch = int(ep or th["training.epochs"])
        rid = f"dinowm_pourwater-dinov2s14-{th['training.epochs']}ep-{mname[len('train_pw_'):]}-at{epoch}-predq"
        d = dict(id=rid, kind="eval", **common, env=th["env"],
                 model=dict(ckpt=f"{ckroot}/outputs/{mname}/checkpoints/model_{epoch}.pth", epoch=epoch, num_hist=int(th["num_hist"]), frameskip=int(th["frameskip"])),
                 protocol=dict(sets="val(학습과 같은 분할의 98 에피소드), ind_test 50, ood_test 50 (첫 프레임 물 색 픽셀 수 4000 기준 낮음 26 / 높음 24)",
                               one_step="슬라이스 5000개 (균등 간격), LPIPS(VGG)·SSIM·MSE — eval_pred_quality.py 와 같은 계산. 기준선 copy_last, recon_copy, recon_ceiling",
                               open_loop="ACWM-Phys eval.py 와 같은 37프레임 창, 시작 linspace(0, 263, 10), 프레임 간격 frameskip. 표는 창 프레임 3 이상, 곡선은 예측 프레임 전부. 잠재 MSE·코사인, ACWM 방식 지표(240×240, [0, 1])와 정적 기준선",
                               whole_episode="처음 num_hist 프레임 + 모든 action 으로 60 스텝. 끝 구간(프레임 250 이상) 상자 바닥 물 색 픽셀 수, 재구성으로 신뢰 관문(상대오차 10 % 이하)",
                               action_sensitivity="다음 에피소드의 action / 정규화한 action 0 으로 교체",
                               intervals="에피소드 단위 bootstrap 95 %"),
                 command=f"bash repro/queue/run_cmd.sh {name} <gpu> " + " ".join(args), runner_env=runner_env_of(name), code=code_of(name),
                 output=out, results="output json 의 sets.<val|ind_test|ood_test>.{m1, m2, m2_curve, m2_latent, acwm, m5, m3_curve, m3_fill}")
        path = os.path.join(EXP, "runs", "eval", rid, "run.yaml")
    elif j["src"] == "acwm_phys.txt" and kind == "train":  # ACWM-Phys Cloth Move·Push Sand·Stack Cube (SETTINGS.md §8)
        h = hydra_args(args)
        short, tag = name.split("_")[1:3]
        a = ACWM[short]
        rid = f"dinowm_{a['rid']}-dinov2s14-{h['training.epochs']}ep-{tag}"
        lr = {k: h.get(f"training.{k}_lr", v) for k, v in LR_DEFAULT.items()}
        d = dict(id=rid, kind="train", **common, env=h["env"], dataset=a["dataset"],
                 model=dict(encoder="DINOv2 ViT-S/14 patch tokens, 동결 (hub 85a2460)", predictor="ViT depth 6, heads 16, mlp 2048, dropout 0.1",
                            decoder="VQ-VAE decoder (quantize False), 같은 학습 안에서 따로 최적화: 정답 프레임 잠재(detach)를 재구성하는 MSE 만 학습",
                            num_hist=int(h["num_hist"]), num_pred=1, frameskip=int(h["frameskip"]), img_size=224),
                 training=dict(epochs=int(h["training.epochs"]), batch_size=int(h["training.batch_size"]),
                               optimizer="predictor·action encoder AdamW, decoder Adam",
                               lr=f"decoder {lr['decoder']}, predictor {lr['predictor']}, action encoder {lr['action_encoder']} (predictor 를 바꿨다 — upstream 5e-4, 논문 Table 12 는 predictor 5e-5)",
                               seed=0, save_every_x_epoch=int(h["training.save_every_x_epoch"]),
                               epochs_note="optimizer step 수가 Pour Water num_hist 1 실행(64,170)과 비슷하도록 정했다 (jobs/acwm_phys.txt 머리말)"),
                 command=f"bash repro/queue/run_train.sh {name} <gpu> " + " ".join(args),
                 output=f"$DINO_TRAIN/outputs/{name}/ (hydra.yaml, checkpoints/model_<epoch>.pth, epoch_logs.jsonl)", code=code_of(name))
        path = os.path.join(EXP, "runs", "train", rid, "run.yaml")
    elif j["src"] == "acwm_phys.txt" and "eval_acwm_phys.py" in " ".join(args):
        ckroot, mname, out = re.search(r"eval_acwm_phys\.py (\S+) (\S+) (\S+)", " ".join(args)).groups()
        th = hydra_args(BY_NAME[mname]["args"])
        short, tag = mname.split("_")[1:3]
        a = ACWM[short]
        epoch = int(th["training.epochs"])
        rid = f"dinowm_{a['rid']}-dinov2s14-{epoch}ep-{tag}-at{epoch}-predq"
        follow = [BY_NAME[n] for n in (f"quant_{short}_{tag}", f"render_{short}_{tag}")]
        d = dict(id=rid, kind="eval", **common, env=th["env"],
                 model=dict(ckpt=f"{ckroot}/outputs/{mname}/checkpoints/model_latest.pth", epoch=epoch, num_hist=int(th["num_hist"]), frameskip=int(th["frameskip"])),
                 protocol=dict(sets=a["sets"],
                               one_step="슬라이스 4000개 (균등 간격), LPIPS(VGG)·SSIM·MSE·잠재 MSE — eval_pred_quality.py 와 같은 계산. 기준선 copy_last, recon_copy, recon_ceiling",
                               open_loop="ACWM-Phys eval.py 와 같은 37프레임 창, 에피소드마다 시작 linspace(0, 프레임 수 - 37, 10), 프레임 간격 frameskip. 예측 프레임 전부를 모은다. 잠재 MSE·코사인, ACWM 방식 지표(환경의 obs_shape, [0, 1])와 정적 기준선",
                               whole_episode="처음 num_hist 프레임 + 모든 action 으로 에피소드 끝까지. 디코딩한 프레임(모델, 정답의 재구성)을 세트별 npz 로 저장",
                               action_sensitivity="다음 에피소드의 action / 정규화한 action 0 으로 교체",
                               quantities=a["quantities"] + " — 저장한 프레임과 정답 프레임을 같은 색 마스크로 잰다 (eval_acwm_phys_quant.py, GPU 불필요)",
                               intervals="에피소드 단위 bootstrap 95 %"),
                 command=f"bash repro/queue/run_cmd.sh {name} <gpu> " + " ".join(args),
                 then=" ; ".join(f"bash repro/queue/run_cmd.sh {f['name']} <gpu> " + " ".join(f["args"]) for f in follow),
                 runner_env=runner_env_of(name), code=code_of(name),
                 output=f"{out}.json, {out}_<set>.npz, {out}_quant.json, {out}_quant_<set>.npz, 영상·격자 그림은 render 작업의 출력 폴더",
                 results="json 의 sets.<val|ind_test|ood_test>.{m1, m2, m2_curve, m2_latent, acwm, m5, m3, m3_curve}, quant json 의 sets.<set>.groups.<group>.<object>")
        path = os.path.join(EXP, "runs", "eval", rid, "run.yaml")
    elif j["src"] == "acwm_phys.txt":  # quant_*, render_*: 위 평가 기록의 then 에 적는다
        continue
    elif kind == "train":
        h = hydra_args(args)
        obj, rid, _ = trained(name)
        assert obj == h["env.dataset.object_name"], name
        lr = {k: h.get(f"training.{k}_lr", v) for k, v in LR_DEFAULT.items()}
        changed = [k for k in lr if lr[k] != LR_DEFAULT[k]]
        lr_text = f"decoder {lr['decoder']}, predictor {lr['predictor']}, action encoder {lr['action_encoder']} " + (
            "(upstream 기본값)" if not changed else "(" + ", ".join(f"{k} 를 바꿨다 — upstream {LR_DEFAULT[k]}" for k in changed) + ")")
        d = dict(id=rid, kind="train", **common, env=obj, dataset=f"$DATASET_DIR/deformable/{obj} (1000 궤적 × 20 프레임, 무작위 0.9/0.1 seed 42)",
                 model=dict(encoder="DINOv2 ViT-S/14 patch tokens, 동결 (hub 85a2460)", predictor="ViT depth 6, heads 16, mlp 2048, dropout 0.1",
                            decoder="VQ-VAE decoder (quantize False)", num_hist=1, num_pred=1, frameskip=1, img_size=224),
                 training=dict(epochs=100, batch_size=32, optimizer="predictor·action encoder AdamW, decoder Adam",
                               lr=lr_text, seed=0, save_every_x_epoch=10),
                 command=f"bash repro/queue/run_train.sh {name} <gpu> " + " ".join(args),
                 output=f"$DINO_TRAIN/outputs/{name}/ (hydra.yaml, checkpoints/model_<epoch>.pth, epoch_logs.jsonl)", code=code_of(name))
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
            obj, base, _ = trained(mname)
            env_name, epoch = obj, 100
            model = dict(source=f"직접 학습 runs/train/{base}", ckpt=f"$DINO_TRAIN/outputs/{mname}/checkpoints/model_latest.pth", epoch=epoch,
                         num_hist=1, frameskip=1)
        envshort = {"point_maze": "pointmaze", "pusht": "pusht", "wall": "wall"}.get(env_name, env_name)
        if cfgname in PLAN_CFG:
            p = PLAN_CFG[cfgname]
            variant = "mpccem"
            planner = dict(name="MPC-CEM (upstream 설정 파일 내장)", cem=f"horizon = goal_H 5, samples 300, topk 30, var_scale 1, opt_steps {p['opt_steps']}",
                           n_taken_actions="goal_H 5 (plan.py 가 덮어쓴다)", max_iter=f"{h['planner.max_iter']} (원본 null = 무제한)", eval_every=1)
            protocol = dict(n_evals=50, goal_source=p["goal_source"], goal_H=5, objective=f"마지막 프레임 latent MSE, alpha {p['alpha']}",
                            seed=int(h.get("seed", 99)))
            read = "logs.json 의 mpc/success_rate (step k = MPC k회 예산, sticky). step 1 = 오픈루프 CEM (SETTINGS.md §5.3)"
        else:
            pl = h["planner"]
            n = int(h.get("n_evals", 10))
            protocol = dict(n_evals=n, goal_source=h["goal_source"], goal_H=int(h["goal_H"]),
                            objective=f"마지막 프레임 latent MSE, alpha {h.get('objective.alpha', '1 (plan.yaml 기본값)')}",
                            seed=int(h.get("seed", 99)))
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
                               eval_every=f"{h['planner.sub_planner.eval_every']} (내부 CEM 시뮬 평가는 첫 opt step 뒤 한 번만)")
                read = "logs.json 의 mpc/mean_chamfer_distance (반복별), final_eval/mean_chamfer_distance"
        # 같은 설정의 변형 실행: job 이름의 꼬리 (pointmaze_mpc_s1 -> s1, _orig, 팔 영상 재실행 -> armvideo)
        parts = name.split("_", 2)
        jtag = parts[2] if len(parts) > 2 and parts[2] != trained(mname)[2] else ""
        jtag = "armvideo" if j.get("arm") else jtag
        rid = f"{base}-at{epoch}-{variant}" + (f"-{jtag}" if jtag else "")
        envs = " ".join(f"{k}={v}" for k, v in j["env"].items() if k != "CKPT_BASE")
        prefix = ("CKPT_BASE=$DINO_TRAIN " if "CKPT_BASE" in j["env"] else "") + (f"{envs} " if envs else "")
        if j.get("arm"):
            # run_plan.sh 가 붙이는 ckpt_base_path 를 직접 준다. DINO_WM_* 는 runner_env 와 같게 export 한다
            command = (f"{envs} " if envs else "") + (f"python repro/eval/deform_arm_video.py $DINO_RUNS/{name} " + " ".join(args)
                       + (" ckpt_base_path=$DINO_TRAIN" if "CKPT_BASE" in j["env"] else ""))
            read = read + ". arm_video/{plan<i>,output_final}_<idx>_<tag>.mp4 = 팔이 움직이는 평가 영상, diag_actions.pkl = 실행한 행동"
            code = code_of(name) + f". 도구: {ARM_TOOL}"
        else:
            command = prefix + f"bash repro/queue/run_plan.sh {name} <gpu> " + " ".join(args)
            code = code_of(name)
        d = dict(id=rid, kind="eval", **common, env=env_name, model=model, planner=planner, protocol=protocol,
                 command=command, runner_env=runner_env_of(name), code=code, output=f"$DINO_RUNS/{name}/", results=read)
        if j["env"]:
            d["job_env"] = " ".join(f"{k}={v}" for k, v in j["env"].items()) + f" (jobs/{j['src']})"
        path = os.path.join(EXP, "runs", "eval", rid, "run.yaml")
    else:  # cmd: prediction quality
        m = re.search(r"eval_pred_quality\.py (\S+) (\S+) (\S+) (\d+)", " ".join(args))
        ckroot, mname, out, cap = m.groups()
        if mname in RELEASED:
            r = RELEASED[mname]
            base, epoch, env_name = r["rid"], r["epoch"], r["env"]
        else:
            obj, base, _ = trained(mname)
            epoch, env_name = 100, obj
        rid = f"{base}-at{epoch}-predq"
        d = dict(id=rid, kind="eval", **common, env=env_name,
                 model=dict(ckpt=f"{ckroot}/outputs/{mname}/checkpoints/model_latest.pth", epoch=epoch),
                 protocol=dict(split="검증 분할 (PushT 는 val 폴더 21 궤적, 그 밖은 학습과 같은 seed 42 궤적 분할)", samples=f"슬라이스 전체. {cap} 개를 넘으면 seed(training.seed) 로 만든 슬라이스 순서에서 균등 간격 {cap} 개",
                               metric="model.eval() 의 1-step 예측을 디코딩해 정답 프레임과 LPIPS(VGG)·SSIM(11×11) — upstream train.py 검증과 같은 계산"),
                 command=f"bash repro/queue/run_cmd.sh {name} <gpu> " + " ".join(args), runner_env=runner_env_of(name), code=code_of(name),
                 output=out, results="output json 의 pred_lpips, pred_ssim")
        path = os.path.join(EXP, "runs", "eval", rid, "run.yaml")
    d = {**{k: d[k] for k in ("id", "kind", "stack", "job")}, "date": started(name, path) or "-", **{k: v for k, v in d.items() if k not in ("id", "kind", "stack", "job")}}
    dump(path, d, HEADER)
    written.append(os.path.relpath(path, EXP))
print("\n".join(written))
