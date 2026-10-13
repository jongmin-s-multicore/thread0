"""predictor_long.py 의 입출력을 로컬 체크포인트와 잇는다 (로컬에서, 레포 환경으로 실행).

  slim   학습이 끝난 실행의 체크포인트에서 predictor 이어 학습에 필요한 것만 꺼낸다 (가중치 state_dict, predictor 옵티마이저 상태,
         설정, 학습/검증 에피소드 이름, 마지막 검증 손실). 모듈 객체 대신 state_dict 라 다른 PyTorch·Python 버전에서도 읽힌다.
  merge  predictor_long.py 가 낸 가중치를 원래 체크포인트(디코더 포함)에 넣어 평가 스크립트가 읽는 형식으로 저장한다.
         인코더는 동결이고 디코더는 predictor 와 따로 학습되므로(정답 잠재만 본다) 디코더는 원래 것을 그대로 쓴다.

usage (cwd = $DINO_WM, env sourced):
  python repro/colab/ckpt_tools.py slim  <ckpt_base_path> <model_name> <out.pt>
  python repro/colab/ckpt_tools.py merge <ckpt_base_path> <model_name> <state.pt> <new_model_name>
"""
import os, sys, json, shutil
import torch
import hydra
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())


def slim(ckpt_base, name, out):
    d = Path(ckpt_base) / "outputs" / name
    cfg = OmegaConf.load(d / "hydra.yaml")
    ck = torch.load(d / "checkpoints" / "model_latest.pth", map_location="cpu")
    _, trajs = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    base = trajs["train"].dataset
    logs = [json.loads(l) for l in open(d / "epoch_logs.jsonl")]
    opt = ck["predictor_optimizer"]
    torch.save({
        "model_name": name, "cfg": OmegaConf.to_yaml(cfg, resolve=True), "epoch": int(ck["epoch"]),
        "predictor": ck["predictor"].state_dict(), "action_encoder": ck["action_encoder"].state_dict(), "proprio_encoder": ck["proprio_encoder"].state_dict(),
        "predictor_optimizer": getattr(opt, "optimizer", opt).state_dict(),
        "train_names": [base.video_names[i] for i in trajs["train"].indices], "val_names": [base.video_names[i] for i in trajs["valid"].indices],
        "action_mean": base.action_mean, "action_std": base.action_std,
        "val_z_visual_loss": float(logs[-1]["val_z_visual_loss"]), "train_z_visual_loss": float(logs[-1]["train_z_visual_loss"]),
    }, out)
    print("wrote", out, f"{os.path.getsize(out) / 1e6:.0f} MB, epoch {ck['epoch']}, train/val episodes {len(trajs['train'])}/{len(trajs['valid'])}")


def merge(ckpt_base, name, state_path, new_name):
    src, dst = Path(ckpt_base) / "outputs" / name, Path(ckpt_base) / "outputs" / new_name
    ck = torch.load(src / "checkpoints" / "model_latest.pth", map_location="cpu")
    st = torch.load(state_path, map_location="cpu")
    for k in ("predictor", "action_encoder", "proprio_encoder"):
        ck[k].load_state_dict(st[k])
    ck["epoch"] = int(st["epoch"])
    ck.pop("predictor_optimizer", None)  # not the optimizer of these weights; evaluation does not read it
    (dst / "checkpoints").mkdir(parents=True, exist_ok=True)
    if not (dst / "hydra.yaml").exists():
        shutil.copy(src / "hydra.yaml", dst / "hydra.yaml")
    out = dst / "checkpoints" / f"model_{ck['epoch']}.pth"
    torch.save(ck, out)
    print("wrote", out)


if __name__ == "__main__":
    {"slim": slim, "merge": merge}[sys.argv[1]](*sys.argv[2:])
