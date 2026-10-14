# ACWM-DiT-S baseline: evaluation scripts

These scripts evaluate the released ACWM-DiT-S checkpoints (Hugging Face `t1an/ACWM-Phys-checkpoints`, commit `f24a2a2`) with the official ACWM-Phys code, and write the per-window numbers and predicted frames that the comparison with DINO-WM needs. They run inside a checkout of the ACWM-Phys code, not inside this repository.

| Script | What it does |
|---|---|
| `eval_launch.py` | Runs the official `eval.py` without changing it. It adds a fixed seed, loads every file with `weights_only=True`, writes the metrics of every window to a jsonl file and, with `ACWM_EVAL_SAVE_PRED`, saves the predicted frames as uint8 npz. |
| `analyze.py` | Turns the jsonl and the saved frames into `<env>_summary.json`: the official numbers (all episodes and the first 50), per-episode means with 95 % bootstrap intervals, and the metric of our DINO-WM evaluation on the frames a frameskip-5 model predicts (window frames 5, 10, ..., 35; Pour Water 15, ..., 35). |
| `rollout_chain.py` | Whole-episode rollout by chaining 37-frame windows. Each window starts from the last predicted frame of the previous one. |
| `amount.py` | Water, cloth and sand pixels and the stacked-cube rule on those rollouts, with the colour masks of `repro/eval/eval_acwm_phys_quant.py` at 224 px. |
| `compare.py`, `paper_table.py`, `render_tables.py` | Tables of the comparison: ACWM-DiT-S against the paper, and DINO-WM against ACWM-DiT-S on the same frames. They read `<env>_summary.json` and `<env>_amount.json` from `$DINO_WORK/results/acwm_dit_baseline/summary/` and the DINO-WM result files under `$DINO_WORK/results/`, and write `compare.md`, `paper_table.md`, `paper_table.tex` and `figs/table{1,2,3,4}.png` next to `summary/`. `render_tables.py` needs tectonic and pdftoppm. |

## Setup

```bash
git clone https://github.com/jongmin-s-multicore/acwm acwm   # commit 9a5ce2b, fork of xavihart/ACWM-Phys-dev
cd acwm
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -r requirements.txt
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["Wan2.1_VAE.pth", "VideoDiT_S_stack_cube_240x240/latest.pt", "VideoDiT_S_push_sand_240x400/latest.pt",
          "VideoDiT_S_clothmove_240x240_240x240/latest.pt", "VideoDiT_S_pour_water_240x240/latest.pt"]:
    hf_hub_download("t1an/ACWM-Phys-checkpoints", f, revision="f24a2a2af19b915ba69c5aa309146d35be4080f4", local_dir="checkpoints")
PY
export ACWM_DATA_ROOT=<ACWM-Phys data>   # Hugging Face t1an/ACWM-Phys, commit 4017e5f
export WAN_VAE_PATH=$PWD/checkpoints/Wan2.1_VAE.pth
```

## Run (one environment)

```bash
S=<this folder>; O=results/main; E=stack_cube; C=checkpoints/VideoDiT_S_stack_cube_240x240/latest.pt
ACWM_EVAL_WINDOWS=$O/${E}_windows.jsonl ACWM_EVAL_SAVE_PRED=$O/pred/$E \
  python $S/eval_launch.py --env $E --ckpt $C --steps 50 --split both --max_trajs 0 --batch_size 2 --output_root $O/$E --save_videos
python $S/analyze.py $O $E
python $S/rollout_chain.py $E $O results/chain
python $S/amount.py results/chain $E
```

`--max_trajs 0` evaluates every test episode. The official `scripts/eval_all.sh` uses `--max_trajs 50`, which takes the first 50 entries of `metadata.pt`. `analyze.py` reports both. For Reacher-like 37-frame episodes add `--test_cuts 1`. Speed: about 9.5 s per window on an RTX 3060 and 4 s on an RTX 3090 at 240×240.
