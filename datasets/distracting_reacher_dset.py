"""Dataset for the distracting DMC Reacher experiment.

Layout produced by scripts/collect_data.py (mirrors datasets/pusht_dset.py for mp4
and datasets/point_maze_dset.py for pth):

    <data_path>/
        obses/episode_%04d.mp4          # obs_format=mp4 (default)
        obses/episode_%04d.pth          # obs_format=pth, uint8 (T, H, W, C)
        features/episode_%04d.pth       # optional DINOv2 cache (scripts/cache_features.py)
        states.pth                      # (N, T, 8) float32
        actions.pth                     # (N, T, 2) float32
        seq_lengths.pth                 # (N,) int
        meta.json                       # seed, backgrounds, hashes, provenance

The `clean` and `distract` directories hold the SAME states/actions; only the
rendered pixels differ. Normalization statistics are therefore identical for
models A and B, which is what makes the two arms comparable.
"""

import json
import torch
import numpy as np
from pathlib import Path
from einops import rearrange
from typing import Callable, Optional

from .traj_dset import TrajDataset, get_train_val_sliced

from env.distracting_reacher import PROPRIO_DIM, STATE_DIM


class DistractingReacherDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = "data/distracting_reacher/clean",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = True,
        action_scale: float = 1.0,
        use_cached_features: bool = False,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        self.use_cached_features = use_cached_features

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Generate it with scripts/collect_data.py."
            )

        self.states = torch.load(self.data_path / "states.pth").float()
        self.actions = torch.load(self.data_path / "actions.pth").float()
        self.actions = self.actions / action_scale  # scaled back up in env
        self.seq_lengths = torch.load(self.data_path / "seq_lengths.pth")

        meta_file = self.data_path / "meta.json"
        self.meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        self.obs_format = self.meta.get("obs_format", "mp4")

        n = n_rollout if n_rollout else len(self.states)
        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        # proprio excludes the target position on purpose: the planner must read
        # the target out of the image (see env/distracting_reacher.py).
        self.proprios = self.states[..., :PROPRIO_DIM].clone()
        print(f"Loaded {n} rollouts from {self.data_path}")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]
        assert self.state_dim == STATE_DIM, (
            f"expected {STATE_DIM}-dim states, got {self.state_dim}"
        )

        if normalize_action:
            self.action_mean, self.action_std = self.get_data_mean_std(
                self.actions, self.seq_lengths
            )
            self.state_mean, self.state_std = self.get_data_mean_std(
                self.states, self.seq_lengths
            )
            self.proprio_mean, self.proprio_std = self.get_data_mean_std(
                self.proprios, self.seq_lengths
            )
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def get_data_mean_std(self, data, traj_lengths):
        all_data = []
        for traj in range(len(traj_lengths)):
            all_data.append(data[traj, : traj_lengths[traj]])
        all_data = torch.vstack(all_data)
        std = torch.std(all_data, dim=0)
        std[std < 1e-6] = 1.0  # guard against constant dims (e.g. a static target)
        return torch.mean(all_data, dim=0), std

    def get_seq_length(self, idx):
        return int(self.seq_lengths[idx])

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            result.append(self.actions[i, : self.get_seq_length(i), :])
        return torch.cat(result, dim=0)

    # ---------------- frame loading ----------------

    def _load_images(self, idx, frames):
        if self.obs_format == "pth":
            image = torch.load(self.data_path / "obses" / f"episode_{idx:04d}.pth")
            return image[frames]  # THWC uint8
        import decord

        decord.bridge.set_bridge("torch")
        reader = decord.VideoReader(
            str(self.data_path / "obses" / f"episode_{idx:04d}.mp4"), num_threads=1
        )
        return reader.get_batch(list(frames))  # THWC uint8

    def get_frames(self, idx, frames):
        frames = list(frames)
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]

        if self.use_cached_features:
            # Pre-encoded DINOv2 patch features; see scripts/cache_features.py for
            # the (significant) restrictions on using these for training.
            feat = torch.load(self.data_path / "features" / f"episode_{idx:04d}.pth")
            obs = {"visual": feat[frames].float(), "proprio": proprio}
            return obs, act, state, {}

        image = self._load_images(idx, frames).float() / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        return {"visual": image, "proprio": proprio}, act, state, {}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0


def load_distracting_reacher_slice_train_val(
    transform,
    n_rollout=None,
    data_path="data/distracting_reacher/clean",
    normalize_action=True,
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    use_cached_features=False,
):
    dset = DistractingReacherDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        normalize_action=normalize_action,
        use_cached_features=use_cached_features,
    )
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset,
        train_fraction=split_ratio,
        num_frames=num_hist + num_pred,
        frameskip=frameskip,
    )
    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": dset_train, "valid": dset_val}
    return datasets, traj_dset
