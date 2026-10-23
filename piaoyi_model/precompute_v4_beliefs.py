from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from codex_zbot.data import SequentialEpisodeDataset
from codex_zbot.utils.checkpoint import load_checkpoint
from codex_zbot.utils.metrics import chamfer_distance
from codex_zbot.utils.seed import seed_everything
from piaoyi_model.config import load_drift_config
from piaoyi_model.geometry import sample_occupied_volume
from piaoyi_model.inference_v3 import reconstruct_occupancy_v3
from piaoyi_model.model_v3 import OccupancyDriftGeneratorV3
from piaoyi_model.policy_adapter_v3 import v3_policy_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache deterministic V4 belief transitions for World Model training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--window_interval", type=int, default=15)
    parser.add_argument("--ensemble_samples", type=int, default=2)
    parser.add_argument("--action_scale_rad", type=float, default=0.015)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_drift_config(args.config)
    generator = seed_everything(cfg.seed + 404)
    device = torch.device(args.device)
    model = OccupancyDriftGeneratorV3(cfg).to(device).eval()
    payload = load_checkpoint(args.checkpoint, model, map_location=device)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(args.data_dir).glob("ep_*.npz"))
    for episode_index, path in enumerate(paths):
        target_path = output / path.name
        if target_path.exists():
            continue
        dataset = SequentialEpisodeDataset(
            path.parent,
            "train",
            sequence_length=cfg.sequence_length,
            frame_stride=cfg.frame_stride,
            window_stride=args.window_interval,
            num_gt_points=cfg.num_points,
            image_size=cfg.image_size,
            seed=cfg.seed,
            paths=[path],
        )
        with np.load(path, allow_pickle=False) as raw:
            centers_np = raw["object_centers"].astype(np.float32)
            sizes_np = raw["object_sizes"].astype(np.float32)
            yaws_np = (
                raw["object_quaternions_wxyz"].astype(np.float32)
                if "object_quaternions_wxyz" in raw.files
                else raw["object_yaws"].astype(np.float32)
            )
            visibility_np = raw["object_visibility_per_object"].astype(np.float32)
        max_objects = cfg.max_objects
        centers = torch.zeros(1, max_objects, 3, device=device)
        sizes = torch.zeros(1, max_objects, 3, device=device)
        yaws = (
            torch.zeros(1, max_objects, 4, device=device)
            if yaws_np.ndim == 2
            else torch.zeros(1, max_objects, device=device)
        )
        if yaws_np.ndim == 2:
            yaws[..., 0] = 1.0
        mask = torch.zeros(1, max_objects, dtype=torch.bool, device=device)
        count = len(centers_np)
        centers[0, :count] = torch.from_numpy(centers_np).to(device)
        sizes[0, :count] = torch.from_numpy(sizes_np).to(device)
        yaws[0, :count] = torch.from_numpy(yaws_np).to(device)
        mask[0, :count] = True
        target = sample_occupied_volume(centers, sizes, yaws, mask, cfg.num_points, 1, generator)[:, 0]

        records: dict[str, list[np.ndarray | float | bool]] = {
            key: []
            for key in [
                "position", "tri_state_probability", "policy_features", "robot_state", "coverage",
                "reconstruction_error", "collision", "source_start_frame",
            ]
        }
        previous = None
        with torch.inference_mode():
            for window_index in range(len(dataset)):
                sample = dataset[window_index]
                batch = {key: value.unsqueeze(0).to(device) for key, value in sample.items()}
                state = reconstruct_occupancy_v3(
                    model, batch, cfg, generator, ensemble_samples=args.ensemble_samples
                )
                policy = v3_policy_features(state, previous)
                raw_ids = int(sample["start_frame"]) + np.arange(cfg.sequence_length) * cfg.frame_stride
                # PPO coverage is cumulative within an episode.  The dataset
                # stores per-frame per-object visible ratios rather than
                # point identities, so a cumulative per-object maximum is the
                # closest monotonic target available offline.
                coverage = float(visibility_np[: int(raw_ids[-1]) + 1].max(axis=0).mean())
                q = sample["joint_pos"][-1]
                qdot = sample["joint_vel"][-1]
                pose = sample["pose"][-1]
                records["position"].append(state.position[0].cpu().numpy())
                records["tri_state_probability"].append(state.tri_state_probability[0].cpu().numpy())
                records["policy_features"].append(policy[0].cpu().numpy())
                records["robot_state"].append(torch.cat([q, qdot, pose]).cpu().numpy())
                records["coverage"].append(coverage)
                records["reconstruction_error"].append(float(chamfer_distance(state.position, target)[0]))
                records["collision"].append(bool(sample["collision"].any()))
                records["source_start_frame"].append(int(sample["start_frame"]))
                previous = state
        robot_state = np.asarray(records["robot_state"], dtype=np.float32)
        joint = robot_state[:, :8]
        actions = np.clip(np.diff(joint, axis=0) / args.action_scale_rad, -1.0, 1.0).astype(np.float32)
        arrays = {key: np.asarray(value) for key, value in records.items()}
        arrays["actions"] = actions
        arrays["source_episode_index"] = np.asarray(episode_index, dtype=np.int64)
        np.savez_compressed(target_path, **arrays)
        print(f"[V4-CACHE] {episode_index + 1}/{len(paths)} {target_path.name} windows={len(dataset)}", flush=True)

    manifest = {
        "source_data": str(Path(args.data_dir).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "episodes": len(paths),
        "window_interval": args.window_interval,
        "sequence_length": cfg.sequence_length,
        "frame_stride": cfg.frame_stride,
        "ensemble_samples": args.ensemble_samples,
        "seed": cfg.seed + 404,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
