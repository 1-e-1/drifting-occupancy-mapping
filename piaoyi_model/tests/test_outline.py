from __future__ import annotations

import torch

from piaoyi_model.config import OccupancyDriftConfig
from piaoyi_model.inference_v3 import OccupancyDriftV3State
from piaoyi_model.outline import (
    OutlineConfig,
    _filter_components,
    extract_environment_outline,
    inflate_occupancy_for_clearance,
)


def test_nearby_components_share_one_inferred_box_without_filling_the_gap() -> None:
    occupancy = torch.zeros(20, 20, 20, dtype=torch.bool).numpy()
    occupancy[3:7, 5:10, 5:10] = True
    occupancy[10:14, 5:10, 5:10] = True
    cell_size = torch.tensor([0.02, 0.02, 0.02]).numpy()
    separate, separate_count, _ = _filter_components(
        occupancy, OutlineConfig(min_component_voxels=1), cell_size
    )
    merged, merged_count, labels = _filter_components(
        occupancy,
        OutlineConfig(min_component_voxels=1, component_merge_distance_m=0.10),
        cell_size,
    )
    assert separate_count == 2
    assert merged_count == 1
    assert (merged == occupancy).all()
    assert not merged[7:10].any()
    assert set(labels[occupancy].tolist()) == {1}


def test_outline_extracts_a_finite_servo_target() -> None:
    cfg = OccupancyDriftConfig(memory_resolution=4, workspace_min=[-1, -1, 0], workspace_max=[1, 1, 1])
    generator = torch.Generator().manual_seed(3)
    particles = torch.rand(1, 256, 3, generator=generator) * torch.tensor([0.3, 0.4, 0.2]) + torch.tensor([0.2, -0.2, 0.1])
    observed = particles[:, :64].clone()
    tri = torch.zeros(1, 64, 3)
    tri[..., 0] = 0.4
    tri[..., 1] = 0.1
    tri[..., 2] = 0.5
    state = OccupancyDriftV3State(
        particles, torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1),
        observed, torch.ones(1, 64, dtype=torch.bool), tri,
    )
    outline = extract_environment_outline(
        state, cfg, OutlineConfig(resolution=24, occupancy_threshold=0.18, min_component_voxels=4)
    )
    assert outline.occupancy.shape == (1, 24, 24, 24)
    assert outline.shell_points[0].shape[1] == 3
    assert outline.shell_points[0].shape[0] > 0
    assert outline.component_centers[0].shape[1] == 3
    assert outline.component_extents[0].shape == outline.component_centers[0].shape
    assert outline.component_yaws[0].shape[0] == outline.component_centers[0].shape[0]
    assert outline.component_corners[0].shape == (outline.component_centers[0].shape[0], 8, 3)
    assert (outline.component_extents[0] > 0).all()
    assert torch.isfinite(outline.guidance_target).all()
    assert outline.policy_features().shape == (1, 9)


def test_adaptive_splat_rejects_sparse_low_confidence_outlier() -> None:
    cfg = OccupancyDriftConfig(memory_resolution=4, workspace_min=[-1, -1, 0], workspace_max=[1, 1, 1])
    generator = torch.Generator().manual_seed(11)
    cluster = torch.rand(1, 255, 3, generator=generator) * torch.tensor([0.24, 0.24, 0.16]) + torch.tensor([0.15, -0.12, 0.08])
    outlier = torch.tensor([[[-0.85, 0.85, 0.75]]])
    particles = torch.cat([cluster, outlier], dim=1)
    tri = torch.zeros(1, 64, 3)
    tri[..., 2] = 1.0
    state = OccupancyDriftV3State(
        particles, torch.zeros(1, 1), torch.ones(1, 1), torch.zeros(1, 1),
        cluster[:, :64], torch.ones(1, 64, dtype=torch.bool), tri,
    )
    outline = extract_environment_outline(
        state,
        cfg,
        OutlineConfig(
            resolution=24,
            particle_splat_mode="adaptive",
            occupancy_threshold=0.20,
            min_component_voxels=4,
            implicit_field_weight=0.0,
            tri_occupied_weight=0.0,
        ),
    )
    shell = outline.shell_points[0]
    assert shell.shape[0] > 0
    assert not (((shell - outlier[0, 0]).norm(dim=1)) < 0.15).any()


def test_temporal_hysteresis_retains_previous_supported_occupancy() -> None:
    cfg = OccupancyDriftConfig(memory_resolution=4, workspace_min=[-1, -1, 0], workspace_max=[1, 1, 1])
    generator = torch.Generator().manual_seed(17)
    particles = torch.rand(1, 256, 3, generator=generator) * torch.tensor([0.22, 0.22, 0.16]) + torch.tensor([0.1, -0.1, 0.08])
    tri = torch.zeros(1, 64, 3)
    tri[..., 2] = 1.0
    state = OccupancyDriftV3State(
        particles, torch.zeros(1, 1), torch.ones(1, 1), torch.zeros(1, 1),
        particles[:, :64], torch.ones(1, 64, dtype=torch.bool), tri,
    )
    first_cfg = OutlineConfig(
        resolution=24, occupancy_threshold=0.20, min_component_voxels=4,
        implicit_field_weight=0.0, tri_occupied_weight=0.0,
    )
    first = extract_environment_outline(state, cfg, first_cfg)
    second_cfg = OutlineConfig(
        resolution=24, occupancy_threshold=0.95, occupancy_exit_threshold=0.10,
        temporal_new_weight=0.1, min_component_voxels=4,
        implicit_field_weight=0.0, tri_occupied_weight=0.0,
    )
    second = extract_environment_outline(state, cfg, second_cfg, previous_outline=first)
    assert second.occupancy.sum() > 0


def test_clearance_inflation_closes_a_narrow_channel_without_using_a_box() -> None:
    occupancy = torch.zeros(24, 24, 24, dtype=torch.bool).numpy()
    occupancy[5:10, 4:20, 5:15] = True
    occupancy[13:18, 4:20, 5:15] = True
    blocked, distance = inflate_occupancy_for_clearance(
        occupancy, torch.tensor([0.02, 0.02, 0.02]).numpy(), clearance_radius_m=0.04
    )
    # The original 3-voxel (0.06 m centre-to-centre) free band is narrower
    # than 2r and therefore unavailable to a point robot with radius r.
    assert blocked[10:13, 8:16, 8:12].all()
    # A remote concavity stays free, demonstrating that this is a complex
    # voxel envelope rather than one global axis-aligned bounding box.
    assert not blocked[0, 0, 0]
    assert distance[0, 0, 0] > 0.04
