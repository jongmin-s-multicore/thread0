"""predictor_long.py / decoder_long.py 의 입출력을 로컬 체크포인트와 잇는다 (로컬에서, 레포 환경으로 실행).

  slim   학습이 끝난 실행의 체크포인트에서 이어 학습에 필요한 것만 꺼낸다 (predictor·디코더 가중치 state_dict, 두 옵티마이저 상태,
         설정, 학습/검증 에피소드 이름, action 통계, 마지막 검증 손실). 모듈 객체 대신 state_dict 라 다른 PyTorch·Python 버전에서도 읽힌다.
  init   아직 학습하지 않은 환경의 slim: 끝난 실행의 hydra.yaml 을 틀로 쓰고 env 블록만 conf/env/<env>.yaml 로 바꾼다
         (학습 설정은 틀과 같다). 가중치는 없다 (predictor_long.py / decoder_long.py 를 --from_scratch 로 쓴다).
  merge  predictor_long.py 가 낸 가중치(과 decoder_long.py 가 낸 디코더 가중치)를 평가 스크립트가 읽는 체크포인트로 저장한다.
         인코더는 동결이다. 디코더를 주지 않으면 원래 실행의 디코더를 그대로 쓴다.

usage (cwd = $DINO_WM, env sourced):
  python repro/colab/ckpt_tools.py slim  <ckpt_base_path> <model_name> <out.pt>
  python repro/colab/ckpt_tools.py init  <template hydra.yaml> <env name, e.g. acwm_push_cube> <model_name> <out.pt>
  python repro/colab/ckpt_tools.py merge <ckpt_base_path> <model_name> <state.pt> <new_model_name> [<decoder_state.pt>]
  python repro/colab/ckpt_tools.py build <slim.pt> <state.pt> <decoder_state.pt> <ckpt_base_path> <new_model_name>
  python repro/colab/ckpt_tools.py extract <model_N.pth> <out.pt>      (predictor weights of a merged checkpoint, for --init_pred)

ACWM_PHYS_ROOT (optional): read the episodes from <ACWM_PHYS_ROOT>/<env folder> instead of the data_path in the config
(on a Colab VM); the slim keeps the original config.
"""
import os, sys, json, shutil
import torch
import hydra
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, os.getcwd())


def _opt_state(opt):
    return getattr(opt, "optimizer", opt).state_dict()


def _split(cfg):
    ds = cfg.env.dataset
    if os.environ.get("ACWM_PHYS_ROOT"):
        ds = OmegaConf.create(OmegaConf.to_container(ds, resolve=True))
        ds.data_path = os.path.join(os.environ["ACWM_PHYS_ROOT"], str(ds.data_path).split("acwm_phys/", 1)[-1])
    _, trajs = hydra.utils.call(ds, num_hist=cfg.num_hist, num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    base = trajs["train"].dataset
    return ([base.video_names[i] for i in trajs["train"].indices], [base.video_names[i] for i in trajs["valid"].indices],
            base.action_mean, base.action_std)


def slim(ckpt_base, name, out):
    d = Path(ckpt_base) / "outputs" / name
    cfg = OmegaConf.load(d / "hydra.yaml")
    ck = torch.load(d / "checkpoints" / "model_latest.pth", map_location="cpu", weights_only=False)  # our own run (pickled modules)
    train_names, val_names, mean, std = _split(cfg)
    logs = [json.loads(l) for l in open(d / "epoch_logs.jsonl")]
    torch.save({
        "model_name": name, "cfg": OmegaConf.to_yaml(cfg, resolve=True), "epoch": int(ck["epoch"]),
        "predictor": ck["predictor"].state_dict(), "action_encoder": ck["action_encoder"].state_dict(), "proprio_encoder": ck["proprio_encoder"].state_dict(),
        "predictor_optimizer": _opt_state(ck["predictor_optimizer"]),
        "decoder": ck["decoder"].state_dict() if "decoder" in ck else None,
        "decoder_optimizer": _opt_state(ck["decoder_optimizer"]) if "decoder_optimizer" in ck else None,
        "train_names": train_names, "val_names": val_names, "action_mean": mean, "action_std": std,
        "val_z_visual_loss": float(logs[-1]["val_z_visual_loss"]), "train_z_visual_loss": float(logs[-1]["train_z_visual_loss"]),
        "val_decoder_recon_loss": float(logs[-1].get("val_decoder_recon_loss_reconstructed", "nan")),
    }, out)
    print("wrote", out, f"{os.path.getsize(out) / 1e6:.0f} MB, epoch {ck['epoch']}, train/val episodes {len(train_names)}/{len(val_names)}")


def init(template, env_name, name, out):
    cfg = OmegaConf.load(template)
    env = OmegaConf.load(Path("conf/env") / f"{env_name}.yaml")
    cfg.env = env  # ${normalize_action}, ${img_size}, ${oc.env:DATASET_DIR} resolve against the template's top level
    cfg.saved_folder = f"<not trained>/{name}"
    for k in ("wandb_run_id",):
        cfg.pop(k, None)
    cfg = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))
    train_names, val_names, mean, std = _split(cfg)
    torch.save({
        "model_name": name, "cfg": OmegaConf.to_yaml(cfg), "epoch": 0,
        "predictor": None, "action_encoder": None, "proprio_encoder": None, "predictor_optimizer": None,
        "decoder": None, "decoder_optimizer": None,
        "train_names": train_names, "val_names": val_names, "action_mean": mean, "action_std": std,
        "val_z_visual_loss": float("nan"), "train_z_visual_loss": float("nan"), "val_decoder_recon_loss": float("nan"),
    }, out)
    print("wrote", out, f"env {env_name}, train/val episodes {len(train_names)}/{len(val_names)}, action dim {len(mean)}")


def merge(ckpt_base, name, state_path, new_name, decoder_path=None):
    src, dst = Path(ckpt_base) / "outputs" / name, Path(ckpt_base) / "outputs" / new_name
    ck = torch.load(src / "checkpoints" / "model_latest.pth", map_location="cpu", weights_only=False)
    st = torch.load(state_path, map_location="cpu")
    for k in ("predictor", "action_encoder", "proprio_encoder"):
        ck[k].load_state_dict(st[k])
    if decoder_path:
        ck["decoder"].load_state_dict(torch.load(decoder_path, map_location="cpu")["decoder"])
    ck["epoch"] = int(st["epoch"])
    ck.pop("predictor_optimizer", None)  # not the optimizer of these weights; evaluation does not read it
    ck.pop("decoder_optimizer", None)
    (dst / "checkpoints").mkdir(parents=True, exist_ok=True)
    if not (dst / "hydra.yaml").exists():
        shutil.copy(src / "hydra.yaml", dst / "hydra.yaml")
    out = dst / "checkpoints" / f"model_{ck['epoch']}.pth"
    torch.save(ck, out)
    print("wrote", out)


def build(slim_path, state_path, decoder_path, ckpt_base, new_name):
    """A checkpoint in train.py's format from weights alone (environments that were never trained with train.py)."""
    S = torch.load(slim_path, map_location="cpu")
    cfg = OmegaConf.create(S["cfg"])
    st = torch.load(state_path, map_location="cpu")
    dec = torch.load(decoder_path, map_location="cpu")
    A = len(S["action_mean"]) * cfg.frameskip
    encoder = hydra.utils.instantiate(cfg.encoder)
    proprio_encoder = hydra.utils.instantiate(cfg.proprio_encoder, in_chans=1, emb_dim=cfg.proprio_emb_dim)
    action_encoder = hydra.utils.instantiate(cfg.action_encoder, in_chans=A, emb_dim=cfg.action_emb_dim)
    n_patches = (cfg.img_size // 16) ** 2
    predictor = hydra.utils.instantiate(cfg.predictor, num_patches=n_patches, num_frames=cfg.num_hist,
                                        dim=encoder.emb_dim + (cfg.proprio_emb_dim * cfg.num_proprio_repeat + cfg.action_emb_dim * cfg.num_action_repeat) * cfg.concat_dim)
    decoder = hydra.utils.instantiate(cfg.decoder, emb_dim=encoder.emb_dim)
    predictor.load_state_dict(st["predictor"]); action_encoder.load_state_dict(st["action_encoder"]); proprio_encoder.load_state_dict(st["proprio_encoder"])
    decoder.load_state_dict(dec["decoder"])
    dst = Path(ckpt_base) / "outputs" / new_name
    (dst / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg.saved_folder = str(dst)
    OmegaConf.save(cfg, dst / "hydra.yaml")
    out = dst / "checkpoints" / f"model_{int(st['epoch'])}.pth"
    # same keys as train.py's checkpoints; plan.load_model builds the frozen encoder from the config
    torch.save({"epoch": int(st["epoch"]), "predictor": predictor, "decoder": decoder,
                "action_encoder": action_encoder, "proprio_encoder": proprio_encoder}, out)
    print("wrote", out)


def extract(ckpt_path, out):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    torch.save({"model_name": Path(ckpt_path).parents[1].name, "epoch": int(ck["epoch"]), "predictor": ck["predictor"].state_dict(),
                "action_encoder": ck["action_encoder"].state_dict(), "proprio_encoder": ck["proprio_encoder"].state_dict()}, out)
    print("wrote", out, "epoch", int(ck["epoch"]))


if __name__ == "__main__":
    {"slim": slim, "init": init, "merge": merge, "build": build, "extract": extract}[sys.argv[1]](*sys.argv[2:])
