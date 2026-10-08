"""ACWM-Phys Pour Water (HF t1an/ACWM-Phys, particle/pour_water) as a DINO-WM trajectory dataset.

Layout: <data_path>/{ind_train,ind_test,ood_test}/{episode_<i>.mp4, metadata.pt}.
metadata.pt is a pickled list of dicts (video_path, actions [T, 4], length, episode_idx); it is loaded
with weights_only=True. Frame t of an episode and actions[t] are paired the DINO-WM way: actions[t]
takes frame t to t+1. Action dims 0 and 2 are always 0 in this data and are dropped (action_dims).

Unlike DeformDataset, videos are decoded lazily: a slice only reads and transforms the frames it uses
(the episodes have 300 frames, so reading whole trajectories per sample is too slow).
"""
import re
import numpy as np
import torch
import decord
from einops import rearrange
from pathlib import Path
from typing import Callable, Optional, Sequence
from .traj_dset import TrajDataset, TrajSlicerDataset, TrajSubset, split_traj_datasets


def _episode_index(entry):
    return int(re.search(r"(\d+)", Path(entry["video_path"]).name).group(1))


def _to_tensor(batch):
    # decord returns NDArray, or a torch tensor when another dataset module set the torch bridge
    return torch.as_tensor(batch.asnumpy() if hasattr(batch, "asnumpy") else batch)


class ACWMPourWaterDataset(TrajDataset):
    def __init__(
        self,
        data_path: str,
        split: str = "ind_train",
        n_rollout: Optional[int] = None,
        exclude: Sequence[int] = (),
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_dims: Sequence[int] = (1, 3),
        action_stats: Optional[tuple] = None,
    ):
        self.split_path = Path(data_path) / split
        self.transform = transform
        meta = torch.load(self.split_path / "metadata.pt", map_location="cpu", weights_only=True)
        exclude = set(int(i) for i in exclude)
        meta = [e for e in meta if _episode_index(e) not in exclude]
        if n_rollout:
            meta = meta[:n_rollout]
        self.video_names = [e["video_path"] for e in meta]
        self.actions = [e["actions"][:, list(action_dims)].float() for e in meta]
        self.seq_lengths = torch.tensor([len(a) for a in self.actions])
        for e, a in zip(meta, self.actions):
            assert int(e["length"]) == len(a), f"{e['video_path']}: length {e['length']} != {len(a)} actions"
        self.action_dim = len(action_dims)
        self.state_dim = 1  # no state in the data; dummy like the proprio (train.py only slices it)
        self.proprio_dim = 1

        if normalize_action:
            if action_stats is None:
                all_actions = torch.cat(self.actions)
                action_stats = (all_actions.mean(0), all_actions.std(0) + 1e-6)
            self.action_mean, self.action_std = action_stats
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
        self.actions = [(a - self.action_mean) / self.action_std for a in self.actions]

    def get_seq_length(self, idx):
        return int(self.seq_lengths[idx])

    def get_all_actions(self):
        return torch.cat(self.actions)

    def get_frames(self, idx, frames):
        frames = list(frames)
        reader = decord.VideoReader(str(self.split_path / self.video_names[idx]), num_threads=1)
        image = rearrange(_to_tensor(reader.get_batch(frames)), "T H W C -> T C H W").float() / 255.0
        if self.transform:
            image = self.transform(image)
        zeros = torch.zeros(len(frames), 1)
        obs = {"visual": image, "proprio": zeros}
        return obs, self.actions[idx][frames], zeros.clone(), {}

    def get_slice(self, idx, start, end, frameskip):
        """What TrajSlicerDataset.__getitem__ returns for the window [start, end), reading only the frames it needs."""
        obs, _, state, _ = self.get_frames(idx, range(start, end, frameskip))
        act = rearrange(self.actions[idx][start:end], "(n f) d -> n (f d)", f=frameskip)
        return obs, act, state

    def __getitem__(self, idx):
        # whole trajectory (train.py's open-loop rollout plots); transformed in chunks to bound memory
        T = self.get_seq_length(idx)
        chunks = [self.get_frames(idx, range(s, min(s + 50, T)))[0]["visual"] for s in range(0, T, 50)]
        zeros = torch.zeros(T, 1)
        return {"visual": torch.cat(chunks), "proprio": zeros}, self.actions[idx], zeros.clone(), {}

    def __len__(self):
        return len(self.video_names)


class LazyTrajSlicerDataset(TrajSlicerDataset):
    def __getitem__(self, idx):
        i, start, end = (int(x) for x in self.slices[idx])
        if isinstance(self.dataset, TrajSubset):
            dset, i = self.dataset.dataset, self.dataset.indices[i]
        else:
            dset = self.dataset
        return dset.get_slice(i, start, end, self.frameskip)


def load_acwm_pour_water_slice_train_val(
    transform,
    data_path,
    n_rollout=None,
    exclude=(),
    normalize_action=False,
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=0,
):
    dset = ACWMPourWaterDataset(
        data_path=data_path,
        split="ind_train",
        n_rollout=n_rollout,
        exclude=exclude,
        transform=transform,
        normalize_action=normalize_action,
    )
    dset_train, dset_val = split_traj_datasets(dset, train_fraction=split_ratio, random_seed=42)
    num_frames = num_hist + num_pred
    datasets = {
        "train": LazyTrajSlicerDataset(dset_train, num_frames, frameskip),
        "valid": LazyTrajSlicerDataset(dset_val, num_frames, frameskip),
    }
    traj_dset = {"train": dset_train, "valid": dset_val}
    return datasets, traj_dset


def load_acwm_pour_water_split(
    transform,
    data_path,
    split,
    n_rollout=None,
    exclude=(),
    normalize_action=False,
):
    """A whole split (ind_test / ood_test) with the action statistics of the training data (same n_rollout/exclude as training)."""
    train = ACWMPourWaterDataset(
        data_path=data_path,
        split="ind_train",
        n_rollout=n_rollout,
        exclude=exclude,
        normalize_action=normalize_action,
    )
    return ACWMPourWaterDataset(
        data_path=data_path,
        split=split,
        transform=transform,
        normalize_action=normalize_action,
        action_stats=(train.action_mean, train.action_std),
    )
