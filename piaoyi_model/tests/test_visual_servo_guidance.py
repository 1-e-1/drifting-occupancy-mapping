from __future__ import annotations

import torch

from piaoyi_model.inference_v3 import OccupancyDriftV3State
from piaoyi_model.outline import EnvironmentOutline
from piaoyi_model.visual_servo_guidance import build_visual_servo_guidance


def test_visual_servo_guidance_contract() -> None:
    state = OccupancyDriftV3State(
        torch.rand(1, 32, 3), torch.zeros(1, 1), torch.ones(1, 1) * 0.5,
        torch.zeros(1, 1), torch.rand(1, 16, 3), torch.ones(1, 16, dtype=torch.bool),
        torch.softmax(torch.rand(1, 64, 3), dim=-1),
    )
    outline = EnvironmentOutline(
        torch.ones(1, 4, 4, 4, dtype=torch.bool), torch.ones(1, 4, 4, 4),
        [torch.rand(20, 3)], [torch.rand(4, 3)], torch.tensor([[0.5, 0.0, 0.2]]),
        torch.tensor([[0.3, 0.4, 0.2]]), torch.tensor([[0.55, 0.05, 0.25]]),
        torch.tensor([[0.95]]), torch.tensor([[2]]),
    )
    pose = torch.tensor([[0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]])
    guidance = build_visual_servo_guidance(state, outline, pose)
    assert guidance.observation.shape == (1, 22)
    assert guidance.valid.item()
    assert torch.allclose(guidance.direction_world.norm(dim=1), torch.ones(1), atol=1.0e-5)
