from __future__ import annotations

import argparse
import copy
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[no-redef]
        """No-op fallback keeps training runnable in minimal environments."""
        def __init__(self, *_args, **_kwargs): pass
        def add_scalar(self, *_args, **_kwargs): pass
        def close(self): pass

from codex_zbot.utils.checkpoint import (
    load_checkpoint,
    load_checkpoint_resized_temporal_position,
    save_checkpoint,
)
from codex_zbot.data import split_episode_paths
from codex_zbot.utils.seed import seed_everything
from piaoyi_model.config import apply_v3_ablation, load_drift_config
from piaoyi_model.data import OccupancyEpisodeDataset
from piaoyi_model.features import MultiScaleOccupancyFeatures
from piaoyi_model.model_v3 import OccupancyDriftGeneratorV3
from piaoyi_model.system_v3 import OccupancyV3LossOutput, compute_occupancy_v3_loss


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def configure_stage(model: OccupancyDriftGeneratorV3, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(stage == "joint")
    if stage == "memory":
        for parameter in model.geometry_memory.parameters():
            parameter.requires_grad_(True)
        for parameter in model.field_head.parameters():
            parameter.requires_grad_(True)
    elif stage == "completion":
        for name, parameter in model.named_parameters():
            if not name.startswith("geometry_memory."):
                parameter.requires_grad_(True)
    elif stage != "joint":
        raise ValueError("stage must be memory, completion, or joint")


def stage_loss(output: OccupancyV3LossOutput, cfg, stage: str) -> torch.Tensor:
    if stage == "memory":
        return (
            cfg.tri_state_weight * output.tri_state
            + cfg.implicit_occupancy_weight * output.implicit_occupancy
            + cfg.implicit_sdf_weight * output.implicit_sdf
        )
    if stage == "completion":
        return (
            cfg.drift_weight * output.drift
            + cfg.volume_chamfer_weight * output.volume_chamfer
            + cfg.outside_weight * output.outside
            + cfg.observation_weight * output.observation
            + cfg.repulsion_weight * output.repulsion
            + cfg.free_space_weight * output.free_space
        )
    return output.total


@torch.no_grad()
def update_ema(ema_model, online_model, decay: float) -> None:
    for ema_parameter, parameter in zip(ema_model.parameters(), online_model.parameters(), strict=True):
        ema_parameter.lerp_(parameter.detach(), 1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), online_model.buffers(), strict=True):
        ema_buffer.copy_(buffer)


@torch.no_grad()
def validate(model, features, loader, cfg, device, max_batches: int, stage: str) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 91_003)
    batches = 0
    for batch in loader:
        output, _ = compute_occupancy_v3_loss(
            model, features, move_batch(batch, device), cfg, generator, epoch=cfg.coarse_warmup_epochs,
            enable_teacher_forcing=False,
        )
        values = output.detached_metrics()
        values["stage_loss"] = float(stage_loss(output, cfg, stage))
        for key, value in values.items():
            totals[key] += value
        batches += 1
        if batches >= max_batches:
            break
    return {key: value / batches for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "piaoyi_model" / "configs" / "occupancy_drifting_v7_partial_visibility.json"))
    parser.add_argument("--data_dir", default=str(ROOT / "data" / "sequences_v2_occupancy"))
    parser.add_argument(
        "--extra_data_dir", action="append", default=[],
        help="Additional on-policy replay directory; may be repeated and is used only for training",
    )
    parser.add_argument("--run_name", default="occupancy_v7_joint")
    parser.add_argument("--stage", choices=["memory", "completion", "joint"], default="joint")
    parser.add_argument(
        "--ablation", choices=["full", "no_temporal", "no_free_space", "no_implicit", "local_only"],
        default="full",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--val_batches", type=int, default=None)
    parser.add_argument("--windows_per_episode_per_epoch", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", default=None, help="Resume model and optimizer state")
    parser.add_argument("--init_checkpoint", default=None, help="Load model weights only for the next stage")
    parser.add_argument("--ema_checkpoint", default=None, help="Restore EMA weights separately when resuming")
    args = parser.parse_args()

    cfg = apply_v3_ablation(load_drift_config(args.config), args.ablation)
    if args.max_epochs is not None:
        cfg.epochs = args.max_epochs
    if args.val_batches is not None:
        cfg.val_batches = args.val_batches
    if args.windows_per_episode_per_epoch is not None:
        cfg.windows_per_episode_per_epoch = args.windows_per_episode_per_epoch
    if args.early_stopping_patience is not None:
        cfg.early_stopping_patience = args.early_stopping_patience
    if args.lr is not None:
        cfg.lr = args.lr
    generator = seed_everything(cfg.seed)
    device = torch.device(args.device)
    train_paths = split_episode_paths(args.data_dir, "train", cfg.seed)
    for replay_dir in args.extra_data_dir:
        train_paths.extend(split_episode_paths(replay_dir, "train", cfg.seed))
    train_data = OccupancyEpisodeDataset(args.data_dir, "train", cfg, paths=train_paths)
    val_data = OccupancyEpisodeDataset(args.data_dir, "val", cfg)
    train_loader = DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=args.workers,
        generator=generator, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False, num_workers=args.workers)
    model = OccupancyDriftGeneratorV3(cfg).to(device)
    if args.init_checkpoint:
        payload = load_checkpoint_resized_temporal_position(
            args.init_checkpoint, model, map_location=device
        )
        if payload["resized_temporal_position"]:
            print(
                "[V3:MIGRATION] interpolated temporal position: "
                + ", ".join(payload["resized_temporal_position"]),
                flush=True,
            )
    configure_stage(model, args.stage)
    features = MultiScaleOccupancyFeatures(cfg).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    start_epoch = 0
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer, map_location=device)
        start_epoch = int(payload.get("epoch", -1)) + 1
    ema_model = copy.deepcopy(model).requires_grad_(False).eval()
    if args.ema_checkpoint:
        load_checkpoint(args.ema_checkpoint, ema_model, map_location=device)

    run_dir = ROOT / "checkpoints" / args.run_name
    log_dir = ROOT / "piaoyi_model" / "logs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    best_score = float("inf")
    global_step = 0
    epochs_without_improvement = 0
    for epoch in range(start_epoch, start_epoch + cfg.epochs):
        train_data.set_epoch(epoch)
        model.train()
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_id, batch in enumerate(train_loader):
            output, _ = compute_occupancy_v3_loss(
                model, features, move_batch(batch, device), cfg, generator, epoch=epoch
            )
            loss = stage_loss(output, cfg, args.stage)
            (loss / cfg.gradient_accumulation_steps).backward()
            if (batch_id + 1) % cfg.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                optimizer.step()
                update_ema(ema_model, model, cfg.ema_decay)
                optimizer.zero_grad(set_to_none=True)
                writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                global_step += 1
            values = output.detached_metrics()
            values["stage_loss"] = float(loss.detach())
            for key, value in values.items():
                totals[key] += value
            batches += 1
            if args.max_steps is not None and batches >= args.max_steps:
                break
        if batches % cfg.gradient_accumulation_steps:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            update_ema(ema_model, model, cfg.ema_decay)
            optimizer.zero_grad(set_to_none=True)
            writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
            global_step += 1
        train_metrics = {key: value / batches for key, value in totals.items()}
        val_metrics = validate(ema_model, features, val_loader, cfg, device, cfg.val_batches, args.stage)
        if args.stage == "memory":
            selection_score = val_metrics["tri_state"] + val_metrics["implicit_occupancy"] + val_metrics["implicit_sdf"]
        else:
            selection_score = (
                val_metrics["volume_chamfer"] + (1.0 - val_metrics["volume_coverage"])
                + (1.0 - val_metrics["occupancy_precision"]) + 0.5 * (1.0 - val_metrics["interior_fraction"])
                + 0.25 * val_metrics["free_space"]
            )
        metrics = {
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "selection_score": selection_score,
            "stage": args.stage,
            "ablation": args.ablation,
        }
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key.replace("_", "/", 1), value, epoch)
        save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, cfg.to_dict(), metrics)
        if selection_score < best_score:
            best_score = selection_score
            epochs_without_improvement = 0
            save_checkpoint(run_dir / "best.pt", ema_model, None, epoch, cfg.to_dict(), metrics)
        else:
            epochs_without_improvement += 1
        print(
            f"[V3:{args.stage}] epoch={epoch:03d} train={train_metrics['stage_loss']:.5f} "
            f"val={val_metrics['stage_loss']:.5f} coarse_iou={val_metrics['coarse_occupied_iou']:.4f} "
            f"coverage={val_metrics['volume_coverage']:.4f} precision={val_metrics['occupancy_precision']:.4f}",
            flush=True,
        )
        if epochs_without_improvement >= cfg.early_stopping_patience:
            print(f"[V3:{args.stage}] early_stop epoch={epoch:03d} best={best_score:.6f}", flush=True)
            break
    writer.close()


if __name__ == "__main__":
    main()
