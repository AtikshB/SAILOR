"""Simple dataset utilities for humanoid-bench rollouts.

Provides a tiny `EpisodicNPZDataset` that reads `.npz` episodes saved by
`scripts/collect_humanoid_demo.py` and exposes timestep samples for training.
"""
from pathlib import Path
import os
import numpy as np
from torch.utils.data import Dataset


class EpisodicNPZDataset(Dataset):
    """Treat a set of `.npz` episode files as a flat dataset of timesteps.

    Each `.npz` is expected to contain arrays: `obs` (object array), `actions`,
    `rewards`, `dones`.
    """

    def __init__(self, npz_files, transform=None):
        self.files = list(npz_files)
        self.transform = transform
        # Build an index mapping from global idx -> (file_idx, step_idx)
        self.index_map = []
        self.lengths = []
        for i, f in enumerate(self.files):
            data = np.load(f, allow_pickle=True)
            L = len(data['rewards'])
            self.lengths.append(L)
            for t in range(L):
                self.index_map.append((i, t))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        fidx, t = self.index_map[idx]
        data = np.load(self.files[fidx], allow_pickle=True)
        obs = data['obs'][t]
        action = data['actions'][t]
        reward = float(data['rewards'][t])
        done = bool(data['dones'][t])
        sample = {'obs': obs, 'action': action, 'reward': reward, 'done': done}
        if self.transform:
            sample = self.transform(sample)
        return sample


def make_dataset_from_dir(dir_path):
    p = Path(dir_path)
    files = sorted([str(x) for x in p.glob("**/*.npz")])
    if not files:
        raise FileNotFoundError(f"No .npz files found in {dir_path}")
    return EpisodicNPZDataset(files)
