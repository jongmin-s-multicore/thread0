"""ACWM-Phys Cloth Move, Push Sand and Stack Cube (HF t1an/ACWM-Phys) as DINO-WM trajectory datasets.

Layout: <data_path>/{ind_train,ind_test,ood_test}/{*.mp4, metadata.pt}. metadata.pt is a pickled list of dicts
(video_path, actions [T, A], length, ...); it is loaded with weights_only=True. Videos are decoded lazily like
Pour Water (acwm_pour_water_dset.py): a slice only reads the frames it uses.

The dataset actions follow the DINO-WM convention: action t takes frame t to frame t+1. How the metadata actions
map to that differs per environment (action_format):
  raw        actions[t] already drives frame t -> t+1 (Cloth Move: 65 frames / 64 actions; Stack Cube: T frames / T actions).
             action_dims picks the dimensions that vary (Cloth Move drops the two constant grip flags).
  pose_next  Push Sand. actions[t] is the pusher pose at frame t (x, 0.9, z, quaternion (0, qy, 0, qw)), so the
             transition t -> t+1 is described by the pose at t+1: action t = (x, z, cos yaw, sin yaw) of actions[t+1].
             The quaternion changes sign within episodes (same orientation), which is why the yaw is given as (cos, sin).
  [repro] Formats of the environments added for the baseline plan (issue #4); the frame each action belongs to was
  checked by tracking objects in the videos:
  next       Push Rope. actions[t] is the pusher position (x, z) at frame t: action t = actions[t+1].
  angle_next Reacher. actions[t] are the two joint angles at frame t: action t = (cos, sin) of each angle of actions[t+1].
  pair_next  Push Cube. actions[t] is the unit direction of the pusher velocity; the motion of frame t -> t+1 is
             0.275 actions[t] + 0.725 actions[t+1], so action t = (actions[t], actions[t+1]).
  Robot Arm uses raw: actions[t] is the joint-angle change of frame t -> t+1.

Metadata entries whose video file is missing are skipped (Reacher lists about 30,000 episodes without a video).
exclude_by_split drops further video file names per split (e.g. Reacher episodes in an older video format).
"""
import torch
import decord
from einops import rearrange
from pathlib import Path
from typing import Callable, Optional, Sequence
from torchvision import transforms
from .traj_dset import TrajDataset, split_traj_datasets
from .acwm_pour_water_dset import LazyTrajSlicerDataset, _to_tensor


def resize_transform(img_size=224):
    """default_transform without the centre crop: the whole frame becomes img_size x img_size (Push Sand is 304x512)."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def pose_next_actions(actions):
    nxt = actions[1:].float()
    yaw = 2 * torch.atan2(nxt[:, 4], nxt[:, 6])
    return torch.stack([nxt[:, 0], nxt[:, 2], torch.cos(yaw), torch.sin(yaw)], dim=1)


def angle_next_actions(actions):
    """[repro] (cos, sin) of every angle of the next frame: (T, D) -> (T - 1, 2 D) ordered cos0, sin0, cos1, sin1, ..."""
    nxt = actions[1:].float()
    return torch.stack([torch.cos(nxt), torch.sin(nxt)], dim=2).reshape(len(nxt), -1)


class ACWMPhysDataset(TrajDataset):
    def __init__(
        self,
        data_path: str,
        split: str = "ind_train",
        n_rollout: Optional[int] = None,
        exclude: Sequence[str] = (),
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_dims: Optional[Sequence[int]] = None,
        action_format: str = "raw",
        action_stats: Optional[tuple] = None,
        exclude_by_split: Optional[dict] = None,
    ):
        self.split_path = Path(data_path) / split
        self.transform = transform
        meta = torch.load(self.split_path / "metadata.pt", map_location="cpu", weights_only=True)
        # [repro] skip entries without a video file, and the names listed for this split
        meta = [e for e in meta if (self.split_path / Path(e["video_path"]).name).exists()]
        exclude = set(exclude) | set((exclude_by_split or {}).get(split, ()))  # video file names
        meta = [e for e in meta if Path(e["video_path"]).name not in exclude]
        if n_rollout:
            meta = meta[:n_rollout]
        self.video_names = [Path(e["video_path"]).name for e in meta]  # Stack Cube stores absolute paths of the source machine
        dims = list(action_dims) if action_dims is not None else list(range(meta[0]["actions"].shape[1]))
        if action_format == "pose_next":
            self.actions = [pose_next_actions(e["actions"]) for e in meta]
        elif action_format == "raw":
            self.actions = [e["actions"][:, dims].float() for e in meta]
        elif action_format == "next":  # [repro]
            self.actions = [e["actions"][1:, dims].float() for e in meta]
        elif action_format == "angle_next":  # [repro]
            self.actions = [angle_next_actions(e["actions"][:, dims]) for e in meta]
        elif action_format == "pair_next":  # [repro]
            self.actions = [torch.cat([e["actions"][:-1, dims], e["actions"][1:, dims]], 1).float() for e in meta]
        else:
            raise ValueError(f"unknown action_format {action_format}")
        self.seq_lengths = torch.tensor([len(a) for a in self.actions])
        # frames of the whole-trajectory view: every frame that is the start or the end of a transition
        self.n_obs = [min(int(e["length"]), len(a) + 1) for e, a in zip(meta, self.actions)]
        for e, a, n in zip(meta, self.actions, self.n_obs):
            assert n >= len(a), f"{e['video_path']}: {e['length']} frames for {len(a)} actions"
        self.action_dim = self.actions[0].shape[1]
        self.state_dim = 1  # dummy like the proprio (train.py only slices it)
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
        return {"visual": image, "proprio": zeros}, zeros.clone()

    def get_slice(self, idx, start, end, frameskip):
        """What TrajSlicerDataset.__getitem__ returns for the window [start, end), reading only the frames it needs."""
        obs, state = self.get_frames(idx, range(start, end, frameskip))
        act = rearrange(self.actions[idx][start:end], "(n f) d -> n (f d)", f=frameskip)
        return obs, act, state

    def __getitem__(self, idx):
        # whole trajectory (train.py's open-loop rollout plots)
        obs, state = self.get_frames(idx, range(self.n_obs[idx]))
        return obs, self.actions[idx], state, {}

    def __len__(self):
        return len(self.video_names)


def load_acwm_phys_slice_train_val(
    transform,
    data_path,
    n_rollout=None,
    exclude=(),
    normalize_action=False,
    action_dims=None,
    action_format="raw",
    split_ratio=0.9,
    num_hist=0,
    num_pred=0,
    frameskip=0,
    exclude_by_split=None,
):
    dset = ACWMPhysDataset(
        data_path=data_path,
        split="ind_train",
        n_rollout=n_rollout,
        exclude=exclude,
        transform=transform,
        normalize_action=normalize_action,
        action_dims=action_dims,
        action_format=action_format,
        exclude_by_split=exclude_by_split,
    )
    dset_train, dset_val = split_traj_datasets(dset, train_fraction=split_ratio, random_seed=42)
    num_frames = num_hist + num_pred
    datasets = {
        "train": LazyTrajSlicerDataset(dset_train, num_frames, frameskip),
        "valid": LazyTrajSlicerDataset(dset_val, num_frames, frameskip),
    }
    traj_dset = {"train": dset_train, "valid": dset_val}
    return datasets, traj_dset


def load_acwm_phys_split(
    transform,
    data_path,
    split,
    n_rollout=None,
    exclude=(),
    normalize_action=False,
    action_dims=None,
    action_format="raw",
    exclude_by_split=None,
):
    """A whole split (ind_test / ood_test) with the action statistics of the training data (same n_rollout/exclude as training)."""
    train = ACWMPhysDataset(
        data_path=data_path,
        split="ind_train",
        n_rollout=n_rollout,
        exclude=exclude,
        normalize_action=normalize_action,
        action_dims=action_dims,
        action_format=action_format,
        exclude_by_split=exclude_by_split,
    )
    return ACWMPhysDataset(
        data_path=data_path,
        split=split,
        transform=transform,
        normalize_action=normalize_action,
        action_dims=action_dims,
        action_format=action_format,
        action_stats=(train.action_mean, train.action_std),
        exclude_by_split=exclude_by_split,
    )
