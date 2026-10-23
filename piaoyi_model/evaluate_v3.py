from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from codex_zbot.utils.checkpoint import load_checkpoint
from codex_zbot.utils.seed import seed_everything
from piaoyi_model.config import apply_v3_ablation, load_drift_config
from piaoyi_model.data import OccupancyEpisodeDataset
from piaoyi_model.geometry import sample_occupied_volume, volumetric_metrics
from piaoyi_model.inference_v3 import reconstruct_occupancy_v3
from piaoyi_model.model_v3 import OccupancyDriftGeneratorV3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--max_batches", type=int, default=96)
    parser.add_argument("--ensemble_samples", type=int, default=4)
    parser.add_argument("--selection_temperature", type=float, default=None)
    parser.add_argument(
        "--ablation", choices=["full", "no_temporal", "no_free_space", "no_implicit", "local_only"],
        default="full",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = apply_v3_ablation(load_drift_config(args.config), args.ablation)
    seed_everything(cfg.seed)
    device = torch.device(args.device)
    dataset = OccupancyEpisodeDataset(args.data_dir, args.split, cfg)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = OccupancyDriftGeneratorV3(cfg).to(device)
    payload = load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 1777)
    totals: dict[str, float] = defaultdict(float)
    count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            state = reconstruct_occupancy_v3(
                model, batch, cfg, generator, args.ensemble_samples, args.selection_temperature
            )
            target = sample_occupied_volume(
                batch["object_centers"], batch["object_sizes"], batch["object_yaws"], batch["object_mask"],
                cfg.num_points, 1, generator,
            )[:, 0]
            metrics = volumetric_metrics(
                state.position, target, batch["object_centers"], batch["object_sizes"], batch["object_yaws"],
                batch["object_mask"], cfg.occupancy_threshold_m, cfg.interior_margin_m,
            )
            for key, value in metrics.items():
                totals[key] += float(value.mean())
            totals["ensemble_uncertainty"] += float(state.ensemble_uncertainty.mean())
            totals["unknown_fraction"] += float(state.unknown_fraction.mean())
            totals["free_space_conflict"] += float(state.free_space_conflict.mean())
            count += 1
            if count >= args.max_batches:
                break
    summary = {key: value / count for key, value in totals.items()}
    summary.update({
        "samples": count,
        "nfe": 1,
        "ensemble_samples": args.ensemble_samples,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "seed": cfg.seed,
        "ablation": args.ablation,
        "selection_temperature": (
            cfg.field_selection_temperature if args.selection_temperature is None else args.selection_temperature
        ),
    })
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
