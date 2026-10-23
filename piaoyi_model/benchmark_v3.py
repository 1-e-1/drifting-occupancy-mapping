from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from codex_zbot.utils.checkpoint import load_checkpoint
from codex_zbot.utils.seed import seed_everything
from piaoyi_model.config import apply_v3_ablation, load_drift_config
from piaoyi_model.data import OccupancyEpisodeDataset
from piaoyi_model.inference_v3 import reconstruct_occupancy_v3
from piaoyi_model.model_v3 import OccupancyDriftGeneratorV3
from piaoyi_model.policy_adapter_v3 import v3_policy_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ensemble_samples", type=int, default=4)
    parser.add_argument("--selection_temperature", type=float, default=None)
    parser.add_argument(
        "--ablation", choices=["full", "no_temporal", "no_free_space", "no_implicit", "local_only"],
        default="full",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = apply_v3_ablation(load_drift_config(args.config), args.ablation)
    seed_everything(cfg.seed)
    device = torch.device(args.device)
    dataset = OccupancyEpisodeDataset(args.data_dir, "test", cfg)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    batch = {key: value.to(device) for key, value in batch.items()}
    model = OccupancyDriftGeneratorV3(cfg).to(device).eval()
    load_checkpoint(args.checkpoint, model, map_location=device)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 6001)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            state = reconstruct_occupancy_v3(
                model, batch, cfg, generator, args.ensemble_samples, args.selection_temperature
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(args.repeats):
            state = reconstruct_occupancy_v3(
                model, batch, cfg, generator, args.ensemble_samples, args.selection_temperature
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        policy = v3_policy_features(state)
    result = {
        "mean_latency_ms": 1000.0 * elapsed / args.repeats,
        "nfe": 1,
        "ensemble_samples": args.ensemble_samples,
        "points": cfg.num_points,
        "observation_frames": cfg.sequence_length,
        "policy_feature_shape": list(policy.shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
        ),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "ablation": args.ablation,
        "selection_temperature": (
            cfg.field_selection_temperature if args.selection_temperature is None else args.selection_temperature
        ),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
