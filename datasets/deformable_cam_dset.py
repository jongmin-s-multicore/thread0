"""Dataset for Rope/Granular frames re-rendered with a different camera.

The frames are produced by scripts/render_camera_data.py. Trajectories, push actions and the
train/val split are the same as in datasets/deformable_env_dset.py.

cam_action=True appends the camera motion pose[t+1] - pose[t] (azimuth, height, radius) to
the push action. cam_action=False uses the push action only.

Particle states are not loaded. `state` is a zero placeholder, like `proprio`.
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from einops import rearrange

from .traj_dset import TrajDataset, TrajSlicerDataset, split_traj_datasets


def load_actions(data_dir, cam_action):
    """Raw actions of a rendered set: (N, T, 4), or (N, T, 7) with the camera pose delta appended."""
    actions = torch.from_numpy(np.load(Path(data_dir) / "actions.npy")).float()
    if cam_action:
        pose = torch.from_numpy(np.load(Path(data_dir) / "cam_pose.npy")).float()
        delta = torch.zeros_like(pose)
        delta[:, :-1] = pose[:, 1:] - pose[:, :-1]
        # no frame after the last step: reuse the previous delta
        delta[:, -1] = delta[:, -2]
        actions = torch.cat([actions, delta], dim=-1)
    return actions


def action_stats(actions):
    """Mean and std used to normalize actions."""
    flat = actions.reshape(-1, actions.shape[-1])
    return flat.mean(dim=0), flat.std(dim=0) + 1e-6


class DeformCamDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = "data/deformable_cam",
        object_name: str = "rope",
        variant: str = "move",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        cam_action: bool = False,
    ):
        self.data_path = Path(data_path) / object_name / variant
        self.transform = transform
        self.normalize_action = normalize_action
        self.cam_action = cam_action

        # (N, T, H, W, 3) uint8, memory-mapped
        self.frames = np.load(self.data_path / "frames.npy", mmap_mode="r")
        self.actions = load_actions(self.data_path, cam_action)

        n = n_rollout if n_rollout else len(self.actions)
        self.actions = self.actions[:n]
        self.n_rollout = n
        n_steps = self.actions.shape[1]
        self.action_dim = self.actions.shape[-1]
        self.seq_lengths = torch.tensor([n_steps] * n)

        self.states = torch.zeros((n, n_steps, 1))  # placeholder
        self.state_dim = 1
        self.proprios = torch.zeros((n, n_steps, 1))  # placeholder, as in DeformDataset
        self.proprio_dim = 1

        if normalize_action:
            self.action_mean, self.action_std = action_stats(self.actions)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
        self.state_mean, self.state_std = torch.zeros(1), torch.ones(1)
        self.proprio_mean, self.proprio_std = torch.zeros(1), torch.ones(1)
        self.actions = (self.actions - self.action_mean) / self.action_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        return self.actions.reshape(-1, self.action_dim)

    def get_frames(self, idx, frames):
        frames = list(frames)
        image = torch.from_numpy(np.ascontiguousarray(self.frames[idx, frames]))  # T H W C uint8
        image = rearrange(image, "T H W C -> T C H W").float() / 255.0
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image, "proprio": self.proprios[idx, frames]}
        return obs, self.actions[idx, frames], self.states[idx, frames], {}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w")


class CamSlicerDataset(TrajSlicerDataset):
    """TrajSlicerDataset that loads only the frames of the window.

    TrajSlicerDataset.__getitem__ loads the full trajectory before slicing. This class
    returns the same samples.
    """

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        base, ep = self.dataset.dataset, self.dataset.indices[i]
        obs, _, state, _ = base.get_frames(ep, range(start, end, self.frameskip))
        act = base.actions[ep, start:end]
        act = rearrange(act, "(n f) d -> n (f d)", n=self.num_frames)  # concat actions
        return tuple([obs, act, state])


def load_deformable_cam_slice_train_val(
    transform,
    n_rollout=None,
    data_path="data/deformable_cam",
    object_name="rope",
    variant="move",
    cam_action=False,
    normalize_action=False,
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=0,
):
    dset = DeformCamDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        object_name=object_name,
        variant=variant,
        cam_action=cam_action,
        normalize_action=normalize_action,
    )
    # same split and slice order as traj_dset.get_train_val_sliced
    dset_train, dset_val = split_traj_datasets(dset, train_fraction=split_ratio)
    train_slices = CamSlicerDataset(dset_train, num_hist + num_pred, frameskip)
    val_slices = CamSlicerDataset(dset_val, num_hist + num_pred, frameskip)
    datasets = {"train": train_slices, "valid": val_slices}
    traj_dset = {"train": dset_train, "valid": dset_val}
    return datasets, traj_dset
