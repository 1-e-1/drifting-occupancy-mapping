"""Measure the retained surface-CFM checkpoint against occupied-volume targets."""
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

from codex_zbot.cfm import SecondOrderCFM
from codex_zbot.cfm.rollout import rollout_sequence
from codex_zbot.config import load_config
from codex_zbot.utils.checkpoint import load_checkpoint
from piaoyi_model.config import load_drift_config
from piaoyi_model.data import OccupancyEpisodeDataset
from piaoyi_model.geometry import sample_occupied_volume, volumetric_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfm_config", required=True)
    parser.add_argument("--drift_config", required=True)
    parser.add_argument("--cfm_checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--max_batches", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfm_cfg = load_config(args.cfm_config)
    drift_cfg = load_drift_config(args.drift_config)
    cfm_cfg.perception.num_points = cfm_cfg.cfm.num_points
    cfm_cfg.perception.observed_points = cfm_cfg.cfm.observed_points
    dataset = OccupancyEpisodeDataset(args.data_dir, "test", drift_cfg)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device(args.device)
    model = SecondOrderCFM(cfm_cfg.cfm).to(device)
    load_checkpoint(args.cfm_checkpoint, model, map_location=device)
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(drift_cfg.seed + 1777)
    totals: dict[str, float] = defaultdict(float)
    count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            state = rollout_sequence(model, batch, cfm_cfg.perception, generator=generator).states[-1]
            target = sample_occupied_volume(
                batch["object_centers"], batch["object_sizes"], batch["object_yaws"], batch["object_mask"],
                drift_cfg.num_points, 1, generator,
            )[:, 0]
            metrics = volumetric_metrics(
                state.position, target, batch["object_centers"], batch["object_sizes"], batch["object_yaws"],
                batch["object_mask"], drift_cfg.occupancy_threshold_m, drift_cfg.interior_margin_m,
            )
            for key, value in metrics.items():
                totals[key] += float(value.mean())
            count += 1
            if count >= args.max_batches:
                break
    summary = {key: value / count for key, value in totals.items()}
    summary.update({"samples": count, "nfe": cfm_cfg.cfm.ode_steps, "target": "occupied_volume"})
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
