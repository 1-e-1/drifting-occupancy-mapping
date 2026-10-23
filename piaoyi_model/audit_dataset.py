from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(mask, dtype=np.bool_).tolist():
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit object-visible RGB-D sequences, including schema-v3 per-object coverage.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--sequence_length", type=int, default=12)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--min_initial_visibility", type=float, default=0.02)
    parser.add_argument("--min_window_visibility", type=float, default=0.005)
    parser.add_argument("--min_per_object_visibility", type=float, default=0.01)
    parser.add_argument("--require_per_object_visibility", action="store_true")
    parser.add_argument("--empty_visibility_threshold", type=float, default=1.0e-4)
    parser.add_argument("--max_empty_frame_fraction", type=float, default=0.20)
    parser.add_argument("--max_consecutive_empty_frames", type=int, default=15)
    parser.add_argument("--max_window_empty_fraction", type=float, default=0.50)
    parser.add_argument("--min_episode_per_object_visibility", type=float, default=0.03)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    paths = sorted(Path(args.data_dir).glob("ep_*.npz"))
    failures: list[str] = []
    initial_visibility: list[float] = []
    frame_visibility: list[float] = []
    initial_per_object_visibility: list[float] = []
    all_objects_visible_frames = 0
    audited_frames = 0
    empty_frames = 0
    maximum_empty_run = 0
    episode_empty_fractions: list[float] = []
    minimum_cumulative_per_object_visibility: list[float] = []
    episodes_with_complete_cumulative_evidence = 0
    usable_windows = 0
    total_windows = 0
    span = (args.sequence_length - 1) * args.frame_stride + 1
    for path in paths:
        with np.load(path, allow_pickle=False) as episode:
            if "object_visibility" not in episode.files:
                failures.append(f"{path.name}: missing object_visibility (schema-v1 data)")
                continue
            visibility = episode["object_visibility"].astype(np.float32)
            if visibility.ndim != 1 or len(visibility) != episode["rgb"].shape[0]:
                failures.append(f"{path.name}: invalid object_visibility shape")
                continue
            initial_visibility.append(float(visibility[0]))
            frame_visibility.extend(visibility.tolist())
            if float(visibility[0]) < args.min_initial_visibility:
                failures.append(f"{path.name}: first object visibility {visibility[0]:.5f}")
            if "object_visibility_per_object" in episode.files:
                per_object = episode["object_visibility_per_object"].astype(np.float32)
                if per_object.ndim != 2 or per_object.shape[0] != len(visibility):
                    failures.append(f"{path.name}: invalid object_visibility_per_object shape")
                    continue
                initial_per_object_visibility.extend(per_object[0].tolist())
                if (
                    args.require_per_object_visibility
                    and float(per_object[0].min()) < args.min_per_object_visibility
                ):
                    failures.append(
                        f"{path.name}: first-frame minimum per-object visibility {per_object[0].min():.5f}"
                    )
                all_objects_visible_frames += int((per_object >= args.min_per_object_visibility).all(axis=1).sum())
                audited_frames += len(per_object)
                cumulative_minimum = float(per_object.max(axis=0).min())
                minimum_cumulative_per_object_visibility.append(cumulative_minimum)
                if cumulative_minimum >= args.min_episode_per_object_visibility:
                    episodes_with_complete_cumulative_evidence += 1
                else:
                    failures.append(
                        f"{path.name}: cumulative per-object visibility {cumulative_minimum:.5f}"
                    )
            elif args.require_per_object_visibility:
                failures.append(f"{path.name}: missing object_visibility_per_object (schema-v3 required)")
            if "objects_in_view" in episode.files:
                empty = episode["objects_in_view"].astype(np.int32) == 0
            else:
                empty = visibility <= args.empty_visibility_threshold
            empty_fraction = float(empty.mean())
            longest_empty = longest_true_run(empty)
            empty_frames += int(empty.sum())
            maximum_empty_run = max(maximum_empty_run, longest_empty)
            episode_empty_fractions.append(empty_fraction)
            if empty_fraction > args.max_empty_frame_fraction:
                failures.append(f"{path.name}: empty-frame fraction {empty_fraction:.5f}")
            if longest_empty > args.max_consecutive_empty_frames:
                failures.append(f"{path.name}: consecutive empty frames {longest_empty}")
            for start in range(0, max(0, len(visibility) - span + 1), 4):
                ids = start + np.arange(args.sequence_length) * args.frame_stride
                total_windows += 1
                usable_windows += int(
                    float(visibility[ids].max()) >= args.min_window_visibility
                    and float(empty[ids].mean()) <= args.max_window_empty_fraction
                )

    summary = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "episodes": len(paths),
        "mean_initial_object_visibility": float(np.mean(initial_visibility)) if initial_visibility else 0.0,
        "minimum_initial_object_visibility": float(np.min(initial_visibility)) if initial_visibility else 0.0,
        "mean_frame_object_visibility": float(np.mean(frame_visibility)) if frame_visibility else 0.0,
        "minimum_initial_per_object_visibility": (
            float(np.min(initial_per_object_visibility)) if initial_per_object_visibility else 0.0
        ),
        "all_objects_visible_frame_fraction": all_objects_visible_frames / max(audited_frames, 1),
        "episodes_with_complete_cumulative_evidence_fraction": (
            episodes_with_complete_cumulative_evidence / max(len(paths), 1)
        ),
        "minimum_cumulative_per_object_visibility": (
            min(minimum_cumulative_per_object_visibility)
            if minimum_cumulative_per_object_visibility else 0.0
        ),
        "empty_frame_fraction": empty_frames / max(len(frame_visibility), 1),
        "maximum_episode_empty_frame_fraction": (
            max(episode_empty_fractions) if episode_empty_fractions else 0.0
        ),
        "maximum_consecutive_empty_frames": maximum_empty_run,
        "usable_windows": usable_windows,
        "total_windows": total_windows,
        "usable_window_fraction": usable_windows / max(total_windows, 1),
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not paths or failures or usable_windows == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
