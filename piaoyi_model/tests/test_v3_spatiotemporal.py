from __future__ import annotations

import torch

from piaoyi_model.conditioning import collect_observed_object_points, image_condition
from piaoyi_model.config import OccupancyDriftConfig
from piaoyi_model.evidence import build_ray_evidence, evidence_grid
from piaoyi_model.model_v3 import OccupancyDriftGeneratorV3


def _cfg() -> OccupancyDriftConfig:
    return OccupancyDriftConfig(
        architecture="v3_spatiotemporal", num_points=32, observed_points=64,
        sequence_length=2, image_size=32, ray_grid_size=4, memory_resolution=4,
        dim=32, heads=4, ffn_dim=64, blocks=1, context_tokens=4,
        spatial_transformer_layers=1, temporal_transformer_layers=1,
        memory_cross_attention_layers=1, set_latent_tokens=4,
    )


def _batch() -> dict[str, torch.Tensor]:
    pose = torch.tensor([[[-0.5, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
                          [-0.4, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0]]])
    return {
        "rgb": torch.rand(1, 2, 3, 32, 32),
        "depth": torch.full((1, 2, 1, 32, 32), 0.8),
        "pose": pose,
        "intrinsics": torch.tensor([[[24.0, 0.0, 16.0], [0.0, 24.0, 16.0], [0.0, 0.0, 1.0]]]),
    }


def test_ray_evidence_has_surface_and_free_space() -> None:
    cfg = _cfg()
    evidence = build_ray_evidence(_batch(), cfg)
    grid, surface, free = evidence_grid(evidence, cfg)
    assert evidence.surface_points.shape == (1, 2, 16, 3)
    assert evidence.free_points.shape == (1, 2, 48, 3)
    assert grid.shape == (1, 64, 2)
    assert surface.any() and free.any()


def test_v3_forward_and_deterministic_field_backward() -> None:
    torch.manual_seed(7)
    cfg = _cfg()
    batch = _batch()
    evidence = build_ray_evidence(batch, cfg)
    observed, observed_mask = collect_observed_object_points(batch, cfg)
    model = OccupancyDriftGeneratorV3(cfg)
    noise = torch.randn(1, 2, cfg.num_points, 3)
    output = model(
        noise, image_condition(batch, cfg.max_depth_m), batch["pose"], evidence,
        observed, observed_mask, return_aux=True,
    )
    assert output.points.shape == (1, 2, cfg.num_points, 3)
    assert output.memory.tri_state_logits.shape == (1, cfg.memory_resolution**3, 3)
    query = torch.rand(1, 12, 3)
    occupancy, sdf = model.decode_field(output.memory, query)
    (occupancy.mean() + sdf.mean() + output.points.mean()).backward()
    assert occupancy.shape == (1, 12)
    assert sdf.shape == (1, 12)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
