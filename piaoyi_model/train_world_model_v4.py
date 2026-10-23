from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from codex_zbot.config import load_config
from codex_zbot.data import split_episode_paths
from codex_zbot.utils.checkpoint import load_checkpoint, save_checkpoint
from codex_zbot.utils.seed import seed_everything
from codex_zbot.world_model.model import world_model_loss
from piaoyi_model.belief_v4 import V4BeliefLearningSystem
from piaoyi_model.config import load_drift_config


class CachedBeliefDataset(Dataset):
    def __init__(self, paths: list[Path], window_count: int):
        self.paths = paths
        self.window_count = window_count

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with np.load(self.paths[index], allow_pickle=False) as archive:
            available = int(archive["position"].shape[0])
            if available < self.window_count:
                raise RuntimeError(f"{self.paths[index]} has {available} windows, expected {self.window_count}")
            start = index % (available - self.window_count + 1)
            stop = start + self.window_count
            return {
                "position": torch.from_numpy(archive["position"][start:stop].astype(np.float32)),
                "tri_state_probability": torch.from_numpy(archive["tri_state_probability"][start:stop].astype(np.float32)),
                "policy_features": torch.from_numpy(archive["policy_features"][start:stop].astype(np.float32)),
                "robot_state": torch.from_numpy(archive["robot_state"][start:stop].astype(np.float32)),
                "actions": torch.from_numpy(archive["actions"][start : stop - 1].astype(np.float32)),
                "coverage": torch.from_numpy(archive["coverage"][start:stop].astype(np.float32)),
                "reconstruction_error": torch.from_numpy(archive["reconstruction_error"][start:stop].astype(np.float32)),
                "collision": torch.from_numpy(archive["collision"][start:stop].astype(np.bool_)),
            }


def compute_loss(system: V4BeliefLearningSystem, batch: dict[str, torch.Tensor]):
    batch_size, windows = batch["position"].shape[:2]
    latent = system.encoder(
        batch["position"].flatten(0, 1),
        batch["tri_state_probability"].flatten(0, 1),
        batch["policy_features"].flatten(0, 1),
    ).reshape(batch_size, windows, -1)
    predictions = system.dynamics.rollout(
        latent[:, 0], batch["robot_state"][:, :-1], batch["actions"]
    )
    coverage_gain = batch["coverage"][:, 1:] - batch["coverage"][:, :-1]
    reconstruction_gain = batch["reconstruction_error"][:, :-1] - batch["reconstruction_error"][:, 1:]
    dynamics_loss, terms = world_model_loss(
        predictions,
        latent[:, 1:].detach(),
        coverage_gain,
        reconstruction_gain,
        batch["collision"][:, 1:],
    )
    decoded = system.metric_head(latent)
    metric_target = torch.stack(
        [batch["reconstruction_error"], batch["coverage"], batch["policy_features"][..., 0]], dim=-1
    )
    metric_loss = F.huber_loss(decoded, metric_target)
    latent_std = latent.flatten(0, 1).std(dim=0)
    variance_loss = F.relu(0.1 - latent_std).mean()
    total = dynamics_loss + metric_loss + 0.1 * variance_loss
    return total, terms, metric_loss, variance_loss


@torch.no_grad()
def evaluate(system, loader, device, max_batches: int) -> float:
    system.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        loss, _, _, _ = compute_loss(system, batch)
        total += float(loss)
        count += 1
        if count >= max_batches:
            break
    if count == 0:
        raise RuntimeError("V4 World Model validation split is empty")
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the V4 action-conditioned occupancy belief model")
    parser.add_argument("--drift_config", required=True)
    parser.add_argument("--algorithm_config", default=str(ROOT / "configs/formal_v4.json"))
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--run_name", default="formal_world_v4")
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--val_batches", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init_checkpoint", default=None, help="Load weights only for a new fine-tuning run")
    args = parser.parse_args()

    drift_cfg = load_drift_config(args.drift_config)
    algorithm_cfg = load_config(args.algorithm_config)
    world_cfg = algorithm_cfg.world_model
    generator = seed_everything(world_cfg.seed)
    device = torch.device(args.device)
    train_paths = split_episode_paths(args.cache_dir, "train", world_cfg.seed)
    val_paths = split_episode_paths(args.cache_dir, "val", world_cfg.seed)
    all_paths = train_paths + val_paths
    minimum_windows = min(
        int(np.load(path, allow_pickle=False)["position"].shape[0]) for path in all_paths
    )
    window_count = min(world_cfg.sequence_length, minimum_windows)
    if window_count < 2:
        raise RuntimeError("V4 World Model needs at least two cached belief windows per episode")
    train_loader = DataLoader(
        CachedBeliefDataset(train_paths, window_count), batch_size=world_cfg.batch_size, shuffle=True, drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(CachedBeliefDataset(val_paths, window_count), batch_size=world_cfg.batch_size, shuffle=False)
    system = V4BeliefLearningSystem(drift_cfg, world_cfg).to(device)
    if args.init_checkpoint:
        load_checkpoint(args.init_checkpoint, system, map_location=device)
    optimizer = torch.optim.AdamW(system.parameters(), lr=world_cfg.lr, weight_decay=1.0e-5)
    start_epoch = 0
    if args.resume:
        payload = load_checkpoint(args.resume, system, optimizer, map_location=device)
        start_epoch = int(payload.get("epoch", -1)) + 1
    checkpoint_dir = ROOT / "checkpoints" / args.run_name
    log_dir = ROOT / "piaoyi_model" / "logs" / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)
    resolved = {
        "drift": drift_cfg.to_dict(), "world": algorithm_cfg.to_dict(),
        "cache_dir": str(Path(args.cache_dir).resolve()), "window_count": window_count,
    }
    (log_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    best_val = float("inf")
    for epoch in range(start_epoch, start_epoch + args.max_epochs):
        system.train()
        total = 0.0
        count = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss, terms, metric_loss, variance_loss = compute_loss(system, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(system.parameters(), world_cfg.grad_clip)
            optimizer.step()
            total += float(loss.detach())
            count += 1
            step = epoch * len(train_loader) + count
            writer.add_scalar("train/loss", loss, step)
            writer.add_scalar("train/metric", metric_loss, step)
            writer.add_scalar("train/variance", variance_loss, step)
            writer.add_scalar("train/grad_norm", grad_norm, step)
            for key, value in terms.items():
                writer.add_scalar(f"train/{key}", value, step)
        train_loss = total / max(1, count)
        val_loss = evaluate(system, val_loader, device, args.val_batches)
        metrics = {"train_loss": train_loss, "val_loss": val_loss}
        save_checkpoint(checkpoint_dir / "last.pt", system, optimizer, epoch, resolved, metrics)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(checkpoint_dir / "best.pt", system, None, epoch, resolved, metrics)
        writer.add_scalar("epoch/train", train_loss, epoch)
        writer.add_scalar("epoch/val", val_loss, epoch)
        print(f"[WORLD-V4] epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f}", flush=True)
    writer.close()


if __name__ == "__main__":
    main()
