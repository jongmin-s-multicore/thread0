"""Run a repro/eval script where plan.py cannot be imported (Colab VM: no gym, mujoco_py, submitit, wandb).

The evaluation scripts only need plan.load_model. This puts a module "plan" with the same load_model (and load_ckpt)
into sys.modules, then runs the script with the given arguments. Checkpoints are our own files with pickled modules,
so they are loaded with weights_only=False (torch >= 2.6 defaults to True).

usage (cwd = repo root):  python repro/colab/eval_lite.py repro/eval/eval_acwm_phys.py <its arguments>
"""
import os, sys, runpy, types
import torch
import hydra

sys.path.insert(0, os.getcwd())
ALL_MODEL_KEYS = ["encoder", "predictor", "decoder", "proprio_encoder", "action_encoder"]


def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    result = {k: v.to(device) for k, v in payload.items() if k in ALL_MODEL_KEYS}
    result["epoch"] = payload["epoch"]
    return result


def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    """plan.load_model, unchanged except for the checkpoint loading above"""
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")
    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")
    if train_cfg.has_decoder and "decoder" not in result:
        raise ValueError("Decoder not found in model checkpoint")
    elif not train_cfg.has_decoder:
        result["decoder"] = None
    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model


plan = types.ModuleType("plan")
plan.load_model, plan.load_ckpt = load_model, load_ckpt
sys.modules["plan"] = plan

_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{"weights_only": False, **k})  # the scripts read the epoch from our checkpoints

script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
