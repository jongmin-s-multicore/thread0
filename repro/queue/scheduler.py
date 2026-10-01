"""파일 상태 기반 GPU 작업 큐 (다시 띄워도 안전하다). start.sh 로 띄운다 (repro/env.sh 가 source 된 환경 필요).

jobs 파일 한 줄:  <job> <kind> [gpu=N] [need=MB] [after=a,b] [env:K=V ...] <인자...>
  kind: plan  -> run_plan.sh   (plan.py, 산출물 $DINO_RUNS/<job>/)
        train -> run_train.sh  (train.py, 모델 $DINO_TRAIN/outputs/<job>/)
        cmd   -> run_cmd.sh    (임의 명령)
  gpu=N 고정 GPU, need=MB 시작에 필요한 GPU 여유 메모리, after= 선행 작업, env: 작업에만 줄 환경변수 ($VAR 펼침).
머신별 조건은 레포 밖에 둔다 (예: .claude/env.local.sh, gitignore):
  DINO_GPUS (기본: nvidia-smi 의 GPU 전부), DINO_SLOTS_PER_GPU (기본 1), DINO_NEED_MB_{PLAN,TRAIN,CMD} (기본 0),
  DINO_QUEUE_COOLDOWN (기본 60 초), DINO_JOBS_LOCAL = 작업별 overlay 파일 (한 줄: <job> [gpu=N] [need=MB] [env:K=V ...]).
상태는 $DINO_RUNS/<job>/run_info.txt (START/END, pid)와 DONE 에 있어서, 앞서 띄운 큐가 시작한 작업도 GPU 슬롯에 센다.
CUDA OOM 으로 죽은 작업은 디렉토리를 $DINO_RUNS/_failed/ 로 옮기고 최대 3번 다시 넣는다.
"""
import os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ["DINO_RUNS"]



def all_gpus():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True).stdout
    return ",".join(x.strip() for x in out.splitlines() if x.strip()) or "0"


SLOTS = {int(g): int(os.environ.get("DINO_SLOTS_PER_GPU", "1")) for g in os.environ.get("DINO_GPUS", "").split(",") if g} \
    or {int(g): int(os.environ.get("DINO_SLOTS_PER_GPU", "1")) for g in all_gpus().split(",")}
NEED_MB = {k: int(os.environ.get(f"DINO_NEED_MB_{k.upper()}", "0")) for k in ("plan", "train", "cmd")}  # 시작에 필요한 여유 메모리
COOLDOWN = int(os.environ.get("DINO_QUEUE_COOLDOWN", "60"))  # 같은 GPU 에 연달아 띄우는 간격 (메모리가 잡힐 때까지)
last_launch = {}


def free_mb(g):
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits", "-i", str(g)],
                         capture_output=True, text=True).stdout.split(",")
    return int(out[1]) - int(out[0])


RUNNER = {"plan": f"{HERE}/run_plan.sh", "train": f"{HERE}/run_train.sh", "cmd": f"{HERE}/run_cmd.sh"}


def parse(path):
    jobs = []
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        toks = line.split()
        name, kind, rest = toks[0], toks[1], toks[2:]
        gpu, after, env, args, need = None, [], {}, [], None
        for t in rest:
            if t.startswith("gpu=") and gpu is None and not args:
                gpu = int(t[4:])
            elif t.startswith("need=") and not args:
                need = int(t[5:])
            elif t.startswith("after=") and not args:
                after = t[6:].split(",")
            elif t.startswith("env:"):
                k, v = t[4:].split("=", 1)
                env[k] = os.path.expandvars(v)
            else:
                args.append(t)
        jobs.append(dict(name=name, kind=kind, gpu=gpu, after=after, env=env, args=args, need=need))
    return jobs


def state(name):
    d = f"{RUNS}/{name}"
    if os.path.exists(f"{d}/DONE"):
        return "done", None
    info = f"{d}/run_info.txt"
    if not os.path.exists(info):
        return "pending", None
    txt = open(info).read()
    m = re.search(r"gpu=(\d+)", txt)
    pid = re.search(r"pid=(\d+)", txt)
    if "END" in txt:
        return "failed", None
    if pid and os.path.exists(f"/proc/{pid.group(1)}"):
        return "running", int(m.group(1)) if m else None
    return "failed", None


def maybe_retry(name, max_retries=3):
    """Requeue a job that died from CUDA OOM (GPU shared by several jobs): archive its dir, mark pending."""
    d, log = f"{RUNS}/{name}", f"{RUNS}/{name}.log"
    if not os.path.exists(log) or "OutOfMemoryError" not in open(log, errors="ignore").read()[-20000:]:
        return False
    os.makedirs(f"{RUNS}/_failed", exist_ok=True)
    n = sum(1 for x in os.listdir(f"{RUNS}/_failed") if x.startswith(name + ".try") and not x.endswith(".log"))
    if n >= max_retries:
        return False
    os.rename(d, f"{RUNS}/_failed/{name}.try{n}")
    os.rename(log, f"{RUNS}/_failed/{name}.try{n}.log")
    if name.startswith("train_"):  # training restarts from scratch
        subprocess.run(["rm", "-rf", os.path.join(os.environ["DINO_TRAIN"], "outputs", name)])
    print(time.strftime("%F %T"), "OOM -> requeued", name, "retry", n + 1, flush=True)
    return True


def apply_overlay(jobs, path):
    """머신별 overlay: <job> [gpu=N] [need=MB] [env:K=V ...] 로 같은 job 의 조건을 덮어쓴다."""
    byname = {j["name"]: j for j in jobs}
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, *toks = line.split()
        if name not in byname:
            print(time.strftime("%F %T"), "overlay: unknown job", name, flush=True)
            continue
        j = byname[name]
        for t in toks:
            if t.startswith("gpu="):
                j["gpu"] = int(t[4:])
            elif t.startswith("need="):
                j["need"] = int(t[5:])
            elif t.startswith("env:"):
                k, v = t[4:].split("=", 1)
                j["env"][k] = os.path.expandvars(v)
            else:
                print(time.strftime("%F %T"), "overlay: unknown option", t, "for", name, flush=True)
    return jobs


jobs = parse(sys.argv[1])
overlay = os.environ.get("DINO_JOBS_LOCAL", "")
if overlay and os.path.exists(overlay):
    jobs = apply_overlay(jobs, overlay)
else:
    if overlay:
        print(time.strftime("%F %T"), "overlay: DINO_JOBS_LOCAL file not found", flush=True)
    overlay = None
for j in jobs:  # GPU 고정이 DINO_GPUS 밖이면 고정을 푼다 (KeyError 방지)
    if j["gpu"] is not None and j["gpu"] not in SLOTS:
        print(time.strftime("%F %T"), "pinned gpu", j["gpu"], "of", j["name"], "not in", sorted(SLOTS), "- unpinned", flush=True)
        j["gpu"] = None
print(time.strftime("%F %T"), "queue: slots", SLOTS, "need", NEED_MB, "cooldown", COOLDOWN, "overlay", overlay is not None, flush=True)
procs = []
blocked = set()
while True:
    st = {j["name"]: state(j["name"]) for j in jobs}
    for nm, (s_, _) in list(st.items()):
        if s_ == "failed" and maybe_retry(nm):
            st[nm] = ("pending", None)
    # 선행 작업이 (재시도 없이) 실패하면 뒤따르는 작업은 막힌 것으로 보고 기다리지 않는다
    for j in jobs:
        if st[j["name"]][0] == "pending" and any(st.get(a, ("done",))[0] == "failed" for a in j["after"]):
            if j["name"] not in blocked:
                print(time.strftime("%F %T"), "blocked", j["name"], "— failed dependency in", j["after"], flush=True)
            blocked.add(j["name"])
    pending = [j for j in jobs if st[j["name"]][0] == "pending" and j["name"] not in blocked]
    if not pending and not any(s == "running" for s, _ in st.values()):
        break
    load = {g: 0 for g in SLOTS}
    for s, g in st.values():
        if s == "running" and g in load:
            load[g] += 1
    for j in pending:
        if any(st.get(a, ("done",))[0] != "done" for a in j["after"]):
            continue
        gpus = [j["gpu"]] if j["gpu"] is not None else sorted(SLOTS, key=lambda g: load[g])
        for g in gpus:
            if load[g] < SLOTS[g] and time.time() - last_launch.get(g, 0) > COOLDOWN and free_mb(g) >= (j["need"] if j["need"] is not None else NEED_MB[j["kind"]]):
                last_launch[g] = time.time()
                env = dict(os.environ, **j["env"])
                p = subprocess.Popen([RUNNER[j["kind"]], j["name"], str(g), *j["args"]], env=env)
                procs.append(p)
                load[g] += 1
                print(time.strftime("%F %T"), "started", j["name"], "gpu", g, flush=True)
                time.sleep(3)  # let run_info.txt appear
                break
    for p in procs:
        p.poll()
    time.sleep(30)
print(time.strftime("%F %T"), "QUEUE EMPTY", {j["name"]: state(j["name"])[0] for j in jobs}, flush=True)
