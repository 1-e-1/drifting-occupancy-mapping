from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from codex_zbot.utils.checkpoint import load_checkpoint
from codex_zbot.utils.seed import seed_everything
from piaoyi_model.config import load_drift_config
from piaoyi_model.data import OccupancyEpisodeDataset
from piaoyi_model.geometry import signed_distance_to_occupied_boxes
from piaoyi_model.inference_v3 import reconstruct_occupancy_v3
from piaoyi_model.model_v3 import OccupancyDriftGeneratorV3
from piaoyi_model.outline import OutlineConfig, extract_environment_outline, outline_metrics
from piaoyi_model.visual_servo_guidance import build_visual_servo_guidance


def target_grid(batch: dict[str, torch.Tensor], cfg, resolution: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lower = torch.as_tensor(cfg.workspace_min, device=batch["rgb"].device, dtype=batch["rgb"].dtype)
    upper = torch.as_tensor(cfg.workspace_max, device=lower.device, dtype=lower.dtype)
    axes = [torch.linspace(lower[i], upper[i], resolution + 1, device=lower.device) for i in range(3)]
    axes = [0.5 * (axis[:-1] + axis[1:]) for axis in axes]
    centers = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(1, -1, 3)
    centers = centers.expand(batch["rgb"].shape[0], -1, -1)
    sdf = signed_distance_to_occupied_boxes(
        centers, batch["object_centers"], batch["object_sizes"], batch["object_yaws"], batch["object_mask"]
    )
    return (sdf <= 0).reshape(-1, resolution, resolution, resolution), lower, upper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "piaoyi_model/configs/occupancy_drifting_v7_partial_visibility.json"))
    parser.add_argument(
        "--checkpoint", default=str(ROOT / "checkpoints/drifting_v7/best.pt")
    )
    parser.add_argument("--data_dir", default=str(ROOT / "data/sequences_v7_partial_visibility_formal"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--max_batches", type=int, default=96)
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--particle_splat_mode", choices=["fixed", "adaptive"], default="adaptive")
    parser.add_argument("--particle_sigma", type=float, default=1.0)
    parser.add_argument("--particle_probability_weight", type=float, default=1.0)
    parser.add_argument("--free_space_suppression", type=float, default=0.80)
    parser.add_argument("--implicit_field_weight", type=float, default=0.80)
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--temporal_new_weight", type=float, default=0.30)
    parser.add_argument("--occupancy_exit_threshold", type=float, default=0.60)
    parser.add_argument("--component_merge_distance_m", type=float, default=0.10)
    parser.add_argument("--minimum_outline_confidence", type=float, default=0.85)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_drift_config(args.config)
    seed_everything(cfg.seed)
    device = torch.device(args.device)
    dataset = OccupancyEpisodeDataset(args.data_dir, args.split, cfg)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = OccupancyDriftGeneratorV3(cfg).to(device).eval()
    payload = load_checkpoint(args.checkpoint, model, map_location=device)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 44_011)
    outline_cfg = OutlineConfig(
        resolution=args.resolution,
        occupancy_threshold=args.threshold,
        particle_splat_mode=args.particle_splat_mode,
        particle_sigma_voxels=args.particle_sigma,
        particle_probability_weight=args.particle_probability_weight,
        free_space_suppression=args.free_space_suppression,
        implicit_field_weight=args.implicit_field_weight,
        temporal_new_weight=args.temporal_new_weight,
        occupancy_exit_threshold=args.occupancy_exit_threshold,
        component_merge_distance_m=args.component_merge_distance_m,
    )
    totals: dict[str, float] = defaultdict(float)
    count = 0
    outline_elapsed = 0.0
    guidance_elapsed = 0.0
    previous_outline = None
    previous_episode_id: int | None = None
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            episode_id = int(batch["episode_id"][0])
            if episode_id != previous_episode_id:
                previous_outline = None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            guidance_start = time.perf_counter()
            state = reconstruct_occupancy_v3(model, batch, cfg, generator, ensemble_samples=4)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            outline = extract_environment_outline(
                state,
                cfg,
                outline_cfg,
                model=model,
                previous_outline=previous_outline if args.temporal else None,
            )
            if args.temporal:
                previous_outline = outline
                previous_episode_id = episode_id
            guidance = build_visual_servo_guidance(
                state,
                outline,
                batch["pose"][:, -1],
                minimum_outline_confidence=args.minimum_outline_confidence,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            outline_elapsed += time.perf_counter() - start
            guidance_elapsed += time.perf_counter() - guidance_start
            truth, lower, upper = target_grid(batch, cfg, args.resolution)
            metrics = outline_metrics(outline, truth, lower, upper)
            for key, value in metrics.items():
                totals[key] += float(value.mean())
            shell = outline.shell_points[0][None]
            shell_sdf = signed_distance_to_occupied_boxes(
                shell, batch["object_centers"], batch["object_sizes"], batch["object_yaws"], batch["object_mask"]
            ).abs()
            cell_diagonal = float(((upper - lower) / args.resolution).norm())
            totals["shell_precision"] += float((shell_sdf <= cell_diagonal).float().mean())
            target_index = torch.argwhere(truth[0])
            target_centroid = lower + (target_index.float().mean(dim=0) + 0.5) * (upper - lower) / args.resolution
            totals["centroid_error"] += float((outline.centroid[0] - target_centroid).norm())
            totals["outline_confidence"] += float(outline.outline_confidence.mean())
            totals["components"] += float(outline.component_count.float().mean())
            box_centers = outline.component_centers[0]
            box_extents = outline.component_extents[0]
            box_yaws = outline.component_yaws[0]
            totals["component_boxes"] += float(box_centers.shape[0])
            if box_centers.shape[0] > 0:
                box_volume = box_extents.prod(dim=1).sum().clamp_min(1.0e-8)
                voxel_volume = float(((upper - lower) / args.resolution).prod())
                occupied_volume = outline.occupancy[0].sum().float() * voxel_volume
                totals["occupied_box_fill_ratio"] += float((occupied_volume / box_volume).clamp_max(1.0))
                object_centers = batch["object_centers"][0, batch["object_mask"][0]]
                delta = object_centers[:, None, :] - box_centers[None]
                cosine = torch.cos(box_yaws)[None]
                sine = torch.sin(box_yaws)[None]
                local_x = cosine * delta[..., 0] + sine * delta[..., 1]
                local_y = -sine * delta[..., 0] + cosine * delta[..., 1]
                local = torch.stack([local_x, local_y, delta[..., 2]], dim=-1)
                covered = (local.abs() <= 0.5 * box_extents[None]).all(dim=-1).any(dim=1)
                totals["object_center_box_recall"] += float(covered.float().mean())
            else:
                totals["occupied_box_fill_ratio"] += 0.0
                totals["object_center_box_recall"] += 0.0
            totals["shell_points"] += float(outline.shell_points[0].shape[0])
            totals["frontier_points"] += float(outline.frontier_points[0].shape[0])
            totals["guidance_valid"] += float(guidance.valid.float().mean())
            totals["guidance_distance"] += float(guidance.distance.mean())
            count += 1
            if count >= args.max_batches:
                break
    summary = {key: value / count for key, value in totals.items()}
    summary.update({
        "samples": count,
        "resolution": args.resolution,
        "threshold": args.threshold,
        "particle_splat_mode": args.particle_splat_mode,
        "particle_sigma": args.particle_sigma,
        "particle_probability_weight": args.particle_probability_weight,
        "free_space_suppression": args.free_space_suppression,
        "implicit_field_weight": args.implicit_field_weight,
        "temporal": args.temporal,
        "temporal_new_weight": args.temporal_new_weight,
        "occupancy_exit_threshold": args.occupancy_exit_threshold,
        "minimum_outline_confidence": args.minimum_outline_confidence,
        "outline_latency_ms": 1000 * outline_elapsed / count,
        "end_to_end_guidance_latency_ms": 1000 * guidance_elapsed / count,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "seed": cfg.seed,
    })
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
