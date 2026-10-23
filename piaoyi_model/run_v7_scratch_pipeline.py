"""Run the schema-v7 perception pipeline from random initialization."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, phase: str, state: str, **extra: object) -> None:
    path.write_text(
        json.dumps({"phase": phase, "state": state, "updated_at": utc_now(), **extra}, indent=2),
        encoding="utf-8",
    )


def run_command(name: str, command: list[str], log_dir: Path, status_path: Path) -> None:
    log_path = log_dir / f"{name}.log"
    write_status(status_path, name, "running", command=command, log=str(log_path))
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        write_status(status_path, name, "failed", returncode=result.returncode, log=str(log_path))
        raise RuntimeError(f"{name} failed with exit code {result.returncode}; inspect {log_path}")
    write_status(status_path, name, "complete", log=str(log_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V7 RGB-D drifting reconstruction from scratch")
    parser.add_argument(
        "--data_dir", default=str(ROOT / "data" / "sequences_v7_partial_visibility_formal")
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "piaoyi_model" / "configs" / "occupancy_drifting_v7_partial_visibility.json"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    config = Path(args.config).resolve()
    if len(list(data_dir.glob("ep_*.npz"))) != 80:
        raise RuntimeError(f"expected exactly 80 v7 episodes in {data_dir}")

    log_dir = ROOT / "piaoyi_model" / "logs" / "v7_scratch_pipeline"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.json"
    write_status(status_path, "initialization", "running", initialization="random", transfer=False)

    audit_output = log_dir / "dataset_audit.json"
    if not audit_output.exists():
        run_command(
            "audit_dataset",
            [
                str(PYTHON), "-m", "piaoyi_model.audit_dataset",
                "--data_dir", str(data_dir),
                "--sequence_length", "16",
                "--frame_stride", "2",
                "--min_window_visibility", "0.002",
                "--max_empty_frame_fraction", "0.20",
                "--max_consecutive_empty_frames", "15",
                "--min_episode_per_object_visibility", "0.03",
                "--output", str(audit_output),
            ],
            log_dir,
            status_path,
        )

    memory = ROOT / "checkpoints" / "occupancy_v7_scratch_memory" / "best.pt"
    completion = ROOT / "checkpoints" / "occupancy_v7_scratch_completion" / "best.pt"
    joint = ROOT / "checkpoints" / "occupancy_v7_scratch_joint" / "best.pt"
    phases = [
        (
            "train_memory",
            memory,
            [
                str(PYTHON), "-u", "piaoyi_model/train_v3.py",
                "--config", str(config), "--data_dir", str(data_dir),
                "--run_name", "occupancy_v7_scratch_memory", "--stage", "memory",
                "--max_epochs", "15", "--workers", str(args.workers), "--device", args.device,
            ],
        ),
        (
            "train_completion",
            completion,
            [
                str(PYTHON), "-u", "piaoyi_model/train_v3.py",
                "--config", str(config), "--data_dir", str(data_dir),
                "--run_name", "occupancy_v7_scratch_completion", "--stage", "completion",
                "--max_epochs", "20", "--workers", str(args.workers), "--device", args.device,
                "--init_checkpoint", str(memory),
            ],
        ),
        (
            "train_joint",
            joint,
            [
                str(PYTHON), "-u", "piaoyi_model/train_v3.py",
                "--config", str(config), "--data_dir", str(data_dir),
                "--run_name", "occupancy_v7_scratch_joint", "--stage", "joint",
                "--max_epochs", "40", "--workers", str(args.workers), "--device", args.device,
                "--init_checkpoint", str(completion),
            ],
        ),
    ]
    for name, artifact, command in phases:
        if not artifact.exists():
            run_command(name, command, log_dir, status_path)

    evaluation = log_dir / "evaluation_joint.json"
    if not evaluation.exists():
        run_command(
            "evaluate_joint",
            [
                str(PYTHON), "-u", "piaoyi_model/evaluate_v3.py",
                "--config", str(config), "--checkpoint", str(joint),
                "--data_dir", str(data_dir), "--split", "test",
                "--max_batches", "96", "--ensemble_samples", "4",
                "--device", args.device, "--output", str(evaluation),
            ],
            log_dir,
            status_path,
        )

    outline_evaluation = log_dir / "outline_joint.json"
    if not outline_evaluation.exists():
        run_command(
            "evaluate_outline",
            [
                str(PYTHON), "-u", "piaoyi_model/evaluate_outline.py",
                "--config", str(config), "--checkpoint", str(joint),
                "--data_dir", str(data_dir), "--split", "test", "--max_batches", "96",
                "--particle_splat_mode", "adaptive", "--component_merge_distance_m", "0.10",
                "--device", args.device, "--output", str(outline_evaluation),
            ],
            log_dir,
            status_path,
        )

    best_dir = ROOT / "checkpoints" / "occupancy_v7_partial_visibility_scratch_best"
    best_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(joint, best_dir / "best.pt")
    selection = {
        "initialization": "random",
        "transfer": False,
        "stages": {"memory_epochs": 15, "completion_epochs": 20, "joint_epochs": 40},
        "source_checkpoint": str(joint),
        "selected_checkpoint": str((best_dir / "best.pt").resolve()),
        "evaluation": json.loads(evaluation.read_text(encoding="utf-8")),
        "outline_evaluation": json.loads(outline_evaluation.read_text(encoding="utf-8")),
        "completed_at": utc_now(),
    }
    (best_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    write_status(status_path, "complete", "complete", selected_checkpoint=selection["selected_checkpoint"])
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
