"""predictor_long.py 의 입출력을 로컬 체크포인트와 잇는다 (로컬에서, 레포 환경으로 실행).

  slim   학습이 끝난 실행의 체크포인트에서 predictor 이어 학습에 필요한 것만 꺼낸다 (가중치 state_dict, predictor 옵티마이저 상태,
         설정, 학습/검증 에피소드 이름, 마지막 검증 손실). 모듈 객체 대신 state_dict 라 다른 PyTorch·Python 버전에서도 읽힌다.
  merge  predictor_long.py 가 낸 가중치를 원래 체크포인트(디코더 포함)에 넣어 평가 스크립트가 읽는 형식으로 저장한다.
         인코더는 동결이고 디코더는 predictor 와 따로 학습되므로(정답 잠재만 본다) 디코더는 원래 것을 그대로 쓴다.

acwm_long.py (이슈 #4 baseline 학습) 용:
  cfg    다른 실행의 hydra.yaml 을 틀로, env 블록을 conf/env/<env>.yaml 로 바꾼 학습 설정을 쓴다 (새 환경).
  init   train.py 체크포인트(model_*.pth)의 가중치·옵티마이저를 acwm_long.py 의 state 형식으로 바꾼다.
         step 은 epoch × (학습 창 수 / 배치 크기, 올림) 로 정한다. action encoder 옵티마이저 상태는 체크포인트에 없어 새로 시작한다.
  build  acwm_long.py 가 낸 *_pred_best.pt 와 *_dec_best.pt 를 평가 스크립트가 읽는 train.py 형식으로 묶는다
         (<ckpt_base_path>/outputs/<new_model_name>/{hydra.yaml, checkpoints/model_latest.pth}).

usage (cwd = $DINO_WM, env sourced):
  python repro/colab/ckpt_tools.py slim  <ckpt_base_path> <model_name> <out.pt>
  python repro/colab/ckpt_tools.py merge <ckpt_base_path> <model_name> <state.pt> <new_model_name>
  python repro/colab/ckpt_tools.py cfg   <template hydra.yaml> <env yaml> <run name> <out hydra.yaml>
  python repro/colab/ckpt_tools.py init  <hydra.yaml> <model_N.pth> <out state.pt>
  python repro/colab/ckpt_tools.py build <hydra.yaml> <pred_best.pt> <dec_best.pt> <ckpt_base_path> <new_model_name>
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


def cfg(template, env_yaml, run, out):
    c = OmegaConf.load(template)
    c.env = OmegaConf.load(env_yaml)
    c.saved_folder = "${ckpt_base_path}/outputs/" + run
    c.wandb_run_id = None
    OmegaConf.save(c, out)  # interpolations (${img_size}, ${normalize_action}, ${oc.env:DATASET_DIR}) stay; they resolve against this file
    print("wrote", out)


def init(hydra_yaml, ckpt, out):
    c = OmegaConf.load(hydra_yaml)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    datasets, _ = hydra.utils.call(c.env.dataset, num_hist=c.num_hist, num_pred=c.num_pred, frameskip=c.frameskip)
    spe = (len(datasets["train"]) + c.training.batch_size - 1) // c.training.batch_size
    step = int(ck["epoch"]) * spe
    opt = lambda o: getattr(o, "optimizer", o).state_dict()
    torch.save({"model_name": c.saved_folder.split("/")[-1], "epoch": float(ck["epoch"]), "steps": step,
                "predictor": ck["predictor"].state_dict(), "action_encoder": ck["action_encoder"].state_dict(),
                "proprio_encoder": ck["proprio_encoder"].state_dict(), "decoder": ck["decoder"].state_dict(),
                "opt_p": opt(ck["predictor_optimizer"]), "opt_d": opt(ck["decoder_optimizer"]), "pred_step": step, "dec_step": step}, out)
    print("wrote", out, f"epoch {ck['epoch']}, {spe} steps per epoch -> step {step}")


def build(hydra_yaml, pred_best, dec_best, ckpt_base, new_name):
    c = OmegaConf.load(hydra_yaml)
    p, d = torch.load(pred_best, map_location="cpu"), torch.load(dec_best, map_location="cpu")
    encoder = hydra.utils.instantiate(c.encoder)
    ds_cfg = c.env.dataset
    _, trajs = hydra.utils.call(ds_cfg, num_hist=c.num_hist, num_pred=c.num_pred, frameskip=c.frameskip)
    A = trajs["train"].dataset.action_dim * c.frameskip
    m = {"proprio_encoder": hydra.utils.instantiate(c.proprio_encoder, in_chans=1, emb_dim=c.proprio_emb_dim),
         "action_encoder": hydra.utils.instantiate(c.action_encoder, in_chans=A, emb_dim=c.action_emb_dim),
         "predictor": hydra.utils.instantiate(c.predictor, num_patches=(c.img_size // 16) ** 2, num_frames=c.num_hist,
                                              dim=encoder.emb_dim + (c.proprio_emb_dim * c.num_proprio_repeat + c.action_emb_dim * c.num_action_repeat) * c.concat_dim),
         "decoder": hydra.utils.instantiate(c.decoder, emb_dim=encoder.emb_dim)}
    for k in ("predictor", "action_encoder", "proprio_encoder"):
        m[k].load_state_dict(p[k])
    m["decoder"].load_state_dict(d["decoder"])
    dst = Path(ckpt_base) / "outputs" / new_name
    (dst / "checkpoints").mkdir(parents=True, exist_ok=True)
    c.saved_folder = "${ckpt_base_path}/outputs/" + new_name
    OmegaConf.save(c, dst / "hydra.yaml")
    ck = {"epoch": int(round(p["epoch"])), "pred_step": int(p["steps"]), "dec_step": int(d["steps"]), **m}
    torch.save(ck, dst / "checkpoints" / "model_latest.pth")
    print("wrote", dst / "checkpoints" / "model_latest.pth", f"predictor step {p['steps']}, decoder step {d['steps']}")


if __name__ == "__main__":
    {"slim": slim, "merge": merge, "cfg": cfg, "init": init, "build": build}[sys.argv[1]](*sys.argv[2:])
