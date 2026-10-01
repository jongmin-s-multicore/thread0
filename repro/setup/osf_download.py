"""DINO-WM OSF 프로젝트(bmw48)의 데이터셋·체크포인트를 받는다.

OSF 는 연결 하나에 약 0.8 MB/s 로 느려서, HTTP Range 로 64 MB 조각을 여러 연결로 받은 뒤 합치고
OSF API 가 주는 md5 와 대조한다. 끊기면 다시 실행하면 이어 받는다 (완성된 조각은 건너뛴다).

usage: python3 osf_download.py <dest_dir> [--only core|deformable|checkpoints ...] [--threads 12]
"""
import argparse, hashlib, json, os, subprocess, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

VIEW_ONLY = "a56a296ce3b24cceaf408383a175ce28"  # dino_wm README 의 공개 view-only 링크
API = f"https://api.osf.io/v2/nodes/bmw48/files/osfstorage/?view_only={VIEW_ONLY}"
GROUPS = {
    "core": ["point_maze.zip", "pusht_noise.zip", "wall_single.zip"],
    "checkpoints": ["outputs.zip"],
    "deformable": ["deformable.zip", "deformable.z01", "deformable.z02", "deformable.z03"],
}
CHUNK = 64 * 2**20


def list_files(url, out):
    while url:
        d = json.load(urllib.request.urlopen(url))
        for x in d["data"]:
            a = x["attributes"]
            if a["kind"] == "folder":
                list_files(x["relationships"]["files"]["links"]["related"]["href"], out)
            else:
                out.append((a["name"], a["size"], a["extra"]["hashes"].get("md5"), x["links"]["download"]))
        url = d["links"].get("next")


def get_chunk(job):
    path, i, s, e, url = job
    cp = f"{path}.chunks/{i:05d}"
    for k in range(60):
        if os.path.exists(cp) and os.path.getsize(cp) == e - s + 1:
            return
        if k:  # OSF/스토리지가 잠깐 거절하면 기다렸다 다시 (지수 백오프, 최대 60초)
            time.sleep(min(60, 2 ** min(k, 6)))
        subprocess.run(["curl", "-s", "-L", "--fail", "--max-time", "900", "-r", f"{s}-{e}", "-o", cp, url])
    raise RuntimeError(f"failed {cp} (다시 실행하면 받은 조각부터 이어 받는다)")


def finalize(path, size, md5):
    if os.path.isdir(path + ".chunks"):
        with open(path, "wb") as o:
            for c in sorted(os.listdir(path + ".chunks")):
                with open(f"{path}.chunks/{c}", "rb") as f:
                    while b := f.read(2**24):
                        o.write(b)
        subprocess.run(["rm", "-rf", path + ".chunks"])
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(2**24):
            h.update(b)
    ok = os.path.getsize(path) == size and (md5 is None or h.hexdigest() == md5)
    print("FILE", os.path.basename(path), size, h.hexdigest(), "OK" if ok else "MISMATCH", flush=True)
    if not ok:
        raise SystemExit(f"md5/size mismatch: {path} (지우고 다시 받는다)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--only", nargs="*", default=list(GROUPS), choices=list(GROUPS))
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    want = {n for g in a.only for n in GROUPS[g]}
    files = []
    list_files(API, files)
    files = [f for f in files if f[0] in want]
    missing = want - {f[0] for f in files}
    if missing:
        raise SystemExit(f"OSF 목록에 없음: {missing}")
    files.sort(key=lambda f: (f[0].startswith("deformable"), f[1]))  # 작은 것 먼저
    os.makedirs(a.dest, exist_ok=True)
    jobs = []
    for name, size, md5, url in files:
        path = os.path.join(a.dest, name)
        if os.path.exists(path) and os.path.getsize(path) == size and not os.path.isdir(path + ".chunks"):
            continue
        os.makedirs(path + ".chunks", exist_ok=True)
        for i, s in enumerate(range(0, size, CHUNK)):
            jobs.append((path, i, s, min(s + CHUNK, size) - 1, url))
    with ThreadPoolExecutor(a.threads) as ex:
        futs = {}
        for j in jobs:
            futs.setdefault(j[0], []).append(ex.submit(get_chunk, j))
        for name, size, md5, url in files:  # 파일 단위로 끝나는 대로 합치고 검증
            path = os.path.join(a.dest, name)
            for f in futs.get(path, []):
                f.result()
            finalize(path, size, md5)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
