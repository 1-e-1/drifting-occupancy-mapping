from __future__ import annotations

from dataclasses import dataclass

import torch

from .inference_v3 import OccupancyDriftV3State
from .outline import EnvironmentOutline
from .policy_adapter_v3 import v3_policy_features


@dataclass
class VisualServoGuidance:
    observation: torch.Tensor  # (B,22)
    target_world: torch.Tensor  # (B,3)
    direction_world: torch.Tensor  # (B,3), unit vector from camera to target
    distance: torch.Tensor  # (B,1)
    valid: torch.Tensor  # (B,1) bool


def build_visual_servo_guidance(
    reconstruction: OccupancyDriftV3State,
    outline: EnvironmentOutline,
    camera_pose_world: torch.Tensor,
    previous_reconstruction: OccupancyDriftV3State | None = None,
    minimum_outline_confidence: float = 0.85,
) -> VisualServoGuidance:
    """Build a compact visual-guidance observation without issuing joint commands.

    Joint-delta clipping, collision gating and emergency stop remain the
    responsibility of the robot-control environment.
    """
    if camera_pose_world.ndim != 2 or camera_pose_world.shape[-1] != 7:
        raise ValueError("camera_pose_world must have shape (B,7) as xyz+wxyz")
    displacement = outline.guidance_target - camera_pose_world[:, :3]
    distance = displacement.norm(dim=1, keepdim=True)
    direction = displacement / distance.clamp_min(1.0e-6)
    reconstruction_features = v3_policy_features(reconstruction, previous_reconstruction)
    outline_features = outline.policy_features()
    observation = torch.cat([reconstruction_features, outline_features, direction, distance], dim=1)
    finite = torch.isfinite(observation).all(dim=1, keepdim=True)
    nonempty = torch.tensor(
        [[points.shape[0] > 0] for points in outline.shell_points],
        device=observation.device,
        dtype=torch.bool,
    )
    valid = finite & nonempty & (outline.outline_confidence >= minimum_outline_confidence)
    return VisualServoGuidance(observation, outline.guidance_target, direction, distance, valid)
