from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from codex_zbot.data import SequentialEpisodeDataset

from .config import OccupancyDriftConfig


class OccupancyEpisodeDataset(Dataset):
    """Attach padded GT volumes for loss computation, never model conditioning."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        cfg: OccupancyDriftConfig,
        paths: list[Path] | None = None,
    ):
        self.cfg = cfg
        self.base = SequentialEpisodeDataset(
            data_dir,
            split,
            sequence_length=cfg.sequence_length,
            frame_stride=cfg.frame_stride,
            num_gt_points=cfg.num_points,
            image_size=cfg.image_size,
            seed=cfg.seed,
            min_first_valid_ratio=0.0,
            max_windows_per_episode=cfg.windows_per_episode_per_epoch if split == "train" else None,
            paths=paths,
        )
        # Schema-v2 episodes contain a GT-derived visibility audit signal.  It
        # is used only to select learnable training/evaluation windows; neither
        # the signal nor object boxes are exposed to the generator at inference.
        filtered_index = []
        visibility_cache: dict[int, np.ndarray] = {}
        per_object_visibility_cache: dict[int, np.ndarray] = {}
        for episode_id, path in enumerate(self.base.paths):
            with np.load(path, allow_pickle=False) as archive:
                if "object_visibility" in archive.files:
                    visibility_cache[episode_id] = archive["object_visibility"].astype(np.float32)
                if "object_visibility_per_object" in archive.files:
                    per_object_visibility_cache[episode_id] = archive["object_visibility_per_object"].astype(np.float32)
        for episode_id, start in self.base._candidate_index:
            if episode_id not in visibility_cache:
                filtered_index.append((episode_id, start))
                continue
            ids = start + np.arange(cfg.sequence_length) * cfg.frame_stride
            window_visibility = visibility_cache[episode_id][ids]
            if float(window_visibility.max()) < cfg.min_sequence_object_visibility:
                continue
            empty_fraction = float(
                (window_visibility <= cfg.empty_frame_visibility_threshold).mean()
            )
            if empty_fraction > cfg.max_sequence_empty_frame_fraction:
                continue
            if cfg.min_sequence_per_object_visibility > 0.0:
                if episode_id not in per_object_visibility_cache:
                    continue
                maximum_per_object = per_object_visibility_cache[episode_id][ids].max(axis=0)
                if float(maximum_per_object.min()) < cfg.min_sequence_per_object_visibility:
                    continue
            filtered_index.append((episode_id, start))
        if not filtered_index:
            raise RuntimeError("No windows satisfy min_sequence_object_visibility")
        self.base._candidate_index = filtered_index
        self.base.set_epoch(0)
        self.specs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for episode_id, path in enumerate(self.base.paths):
            with np.load(path, allow_pickle=False) as archive:
                self.specs[episode_id] = (
                    archive["object_centers"].astype(np.float32),
                    archive["object_sizes"].astype(np.float32),
                    (
                        archive["object_quaternions_wxyz"].astype(np.float32)
                        if "object_quaternions_wxyz" in archive.files
                        else archive["object_yaws"].astype(np.float32)
                    ),
                )

    @property
    def paths(self):
        return self.base.paths

    def set_epoch(self, epoch: int) -> None:
        self.base.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.base[index]
        centers_np, sizes_np, yaws_np = self.specs[int(sample["episode_id"])]
        count = min(len(centers_np), self.cfg.max_objects)
        centers = torch.zeros(self.cfg.max_objects, 3)
        sizes = torch.zeros(self.cfg.max_objects, 3)
        yaws = torch.zeros((self.cfg.max_objects, 4)) if yaws_np.ndim == 2 else torch.zeros(self.cfg.max_objects)
        if yaws_np.ndim == 2:
            yaws[:, 0] = 1.0
        mask = torch.zeros(self.cfg.max_objects, dtype=torch.bool)
        centers[:count] = torch.from_numpy(centers_np[:count])
        sizes[:count] = torch.from_numpy(sizes_np[:count])
        yaws[:count] = torch.from_numpy(yaws_np[:count])
        mask[:count] = True
        sample.update({"object_centers": centers, "object_sizes": sizes, "object_yaws": yaws, "object_mask": mask})
        return sample
