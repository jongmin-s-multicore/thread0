#!/usr/bin/env python
"""Pre-compute DINOv2 patch features for a dataset arm (OPTIONAL).

    python scripts/cache_features.py --data_path $DATASET_DIR/distracting_reacher/clean

Writes features/episode_%04d.pth, each (T, num_patches, emb_dim) float16, and
sets use_cached_features in the env config to pick them up.

READ THIS BEFORE TURNING IT ON
------------------------------
The DINOv2 encoder is frozen in this repo (model.train_encoder=false), so caching
is mathematically a no-op on the result and saves the encoder forward pass. But:

1. It only works with `has_decoder=false`. The decoder reconstructs *pixels*, and
   cached features replace obs['visual'] with features, so there are no pixels to
   reconstruct. Training with has_decoder=true and cached features will fail.
2. It changes nothing at planning time: scripts/eval.py encodes live camera
   frames with the real encoder, as it must.
3. Disk: ~196 patches x 384 dims x 2 bytes = ~150kB per frame, i.e. roughly 10x
   an h264 frame. For 500x100 frames per arm that is ~7.5GB per arm.
4. It bakes in the current encoder config. Change encoder/img_size and the cache
   is silently wrong, which is why the config fingerprint is stored alongside and
   checked on load.

Because of (1) and (3), the default in conf/env/* is use_cached_features: false.
Turn it on only for predictor-only runs where the encoder forward is the
bottleneck.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data_path", required=True, help="one arm, e.g. .../distracting_reacher/clean")
    p.add_argument("--encoder_name", default="dinov2_vits14")
    p.add_argument("--feature_key", default="x_norm_patchtokens")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gpu", default="0")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from einops import rearrange

    from datasets.distracting_reacher_dset import DistractingReacherDataset
    from datasets.img_transforms import default_transform
    from models.dino import DinoV2Encoder

    data_path = Path(args.data_path)
    out_dir = data_path / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = DinoV2Encoder(name=args.encoder_name, feature_key=args.feature_key)
    encoder.to(device).eval()

    # VWorldModel resizes to (img_size // 16) * patch_size before encoding; the
    # cache must match that exactly or the patch grid will differ.
    encoder_image_size = (args.img_size // 16) * encoder.patch_size
    resize = torch.nn.Upsample(
        size=(encoder_image_size, encoder_image_size), mode="bilinear", align_corners=False
    )

    fingerprint = {
        "encoder_name": args.encoder_name,
        "feature_key": args.feature_key,
        "img_size": args.img_size,
        "encoder_image_size": encoder_image_size,
        "emb_dim": encoder.emb_dim,
    }
    fp_path = out_dir / "fingerprint.json"
    if fp_path.exists() and not args.overwrite:
        old = json.loads(fp_path.read_text())
        if old != fingerprint:
            raise SystemExit(
                f"Existing cache was built with a different config:\n  old={old}\n"
                f"  new={fingerprint}\nRe-run with --overwrite."
            )
    fp_path.write_text(json.dumps(fingerprint, indent=2))

    dset = DistractingReacherDataset(
        data_path=str(data_path),
        transform=default_transform(args.img_size),
        normalize_action=True,
    )

    t0 = time.time()
    total_frames = 0
    for idx in range(len(dset)):
        out_path = out_dir / f"episode_{idx:04d}.pth"
        if out_path.exists() and not args.overwrite:
            continue
        obs, _, _, _ = dset[idx]
        visual = obs["visual"]  # (T, 3, H, W), already transformed
        feats = []
        with torch.no_grad():
            for start in range(0, visual.shape[0], args.batch_size):
                chunk = visual[start : start + args.batch_size].to(device)
                chunk = resize(chunk)
                feats.append(encoder(chunk).half().cpu())
        torch.save(torch.cat(feats, dim=0), out_path)
        total_frames += visual.shape[0]
        if (idx + 1) % 10 == 0 or idx == len(dset) - 1:
            el = time.time() - t0
            print(
                f"[cache] {idx+1}/{len(dset)} episodes  "
                f"{total_frames/max(el,1e-9):.1f} frames/s  elapsed {el/60:.1f}m",
                flush=True,
            )

    print(f"[cache] done -> {out_dir}")
    print(
        "[cache] remember: training on these requires has_decoder=false and "
        "use_cached_features=true in the env config."
    )


if __name__ == "__main__":
    main()
