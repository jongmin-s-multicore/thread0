"""frameskip 후보를 고르는 기준선: 224 px 에서 k 프레임 뒤 프레임을 "그대로 복사"한 오차 (LPIPS-VGG, repo 의 SSIM, MSE, 입력 [-1, 1]).
split 마다 에피소드 40개, k = 1, 2, 3, 5 (SETTINGS.md §8.2). GPU 를 쓴다. Push Sand (304x512) 는 자르지 않고 224x224 로 줄인다.

usage (cwd = repo root, env sourced): python repro/setup/checks/acwm_phys/copy_baseline.py $DATASET_DIR/acwm_phys <out.json>"""
import sys, os, json, numpy as np, torch, decord
from torchvision import transforms
ROOT, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.getcwd())
from metrics.image_metrics import ssim
from metrics.lpipsPyTorch.modules.lpips import LPIPS
dev = "cuda"; crit = LPIPS("vgg", "0.1").to(dev).eval()
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.Normalize([0.5] * 3, [0.5] * 3)])
ENVS = {"cloth_move": "deformable/clothmove", "push_sand": "particle/push_sand", "stack_cube": "rigid_dynamics/stack_cube"}
rng = np.random.RandomState(0); res = {}
def lp(a, b):
    fx, fy = crit.net(a), crit.net(b)
    return torch.sum(torch.cat([l((x - y) ** 2).mean((2, 3), True) for x, y, l in zip(fx, fy, crit.lin)], 1), dim=1).flatten()
with torch.no_grad():
    for env, sub in ENVS.items():
        for sp in ("ind_train", "ood_test"):
            sd = os.path.join(ROOT, sub, sp)
            meta = torch.load(os.path.join(sd, "metadata.pt"), map_location="cpu", weights_only=True)
            eps = rng.choice(len(meta), 40, replace=False)
            acc = {k: {"lpips": [], "ssim": [], "mse": []} for k in (1, 2, 3, 5)}
            for i in eps:
                vr = decord.VideoReader(os.path.join(sd, os.path.basename(meta[int(i)]["video_path"])), num_threads=2)
                x = tf(torch.from_numpy(vr.get_batch(list(range(len(vr)))).asnumpy()).permute(0, 3, 1, 2).float() / 255.0).to(dev)
                for k in acc:
                    a, b = x[:-k], x[k:]
                    acc[k]["lpips"].append(lp(a, b).mean().item()); acc[k]["ssim"].append(ssim(a, b).item()); acc[k]["mse"].append(((a - b) ** 2).mean().item())
            for k in acc:
                row = {m: float(np.mean(v)) for m, v in acc[k].items()}; res[f"{env}/{sp}/k{k}"] = row
                print(f"{env:11s} {sp:9s} k={k}  LPIPS {row['lpips']:.4f}  SSIM {row['ssim']:.4f}  MSE {row['mse']:.5f}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
