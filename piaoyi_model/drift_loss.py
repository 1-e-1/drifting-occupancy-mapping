"""First- and second-order Drifting Model objectives in PyTorch."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DriftLossOutput:
    loss: torch.Tensor
    scale: torch.Tensor
    force_norms: dict[float, torch.Tensor]
    velocity_norm: torch.Tensor
    target_displacement: torch.Tensor


def _drifting_field(
    generated: torch.Tensor,
    positives: torch.Tensor,
    temperatures: list[float] | tuple[float, ...],
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """Evaluate the stopped anti-symmetric attraction/repulsion field."""
    generated_count = generated.shape[1]
    targets = torch.cat([generated, positives], dim=1)
    distance = torch.cdist(generated, targets)
    scale = distance.mean(dim=(1, 2), keepdim=True).clamp_min(1.0e-3)
    normalized_distance = distance / scale
    diagonal = torch.eye(generated_count, device=generated.device, dtype=generated.dtype)
    normalized_distance[:, :, :generated_count] += diagonal.unsqueeze(0) * 100.0

    total_force = torch.zeros_like(generated)
    force_norms: dict[float, torch.Tensor] = {}
    for temperature in temperatures:
        logits = -normalized_distance / float(temperature)
        row = logits.softmax(dim=-1)
        column = logits.softmax(dim=-2)
        affinity = (row * column).clamp_min(1.0e-6).sqrt()
        negative_affinity = affinity[:, :, :generated_count]
        positive_affinity = affinity[:, :, generated_count:]
        positive_sum = positive_affinity.sum(dim=-1, keepdim=True)
        negative_sum = negative_affinity.sum(dim=-1, keepdim=True)
        coefficients = torch.cat(
            [-negative_affinity * positive_sum, positive_affinity * negative_sum], dim=-1
        )
        force = torch.bmm(coefficients, targets)
        force = force - coefficients.sum(dim=-1, keepdim=True) * generated
        force_norm = force.square().mean(dim=(1, 2), keepdim=True).clamp_min(1.0e-8).sqrt()
        force_norms[float(temperature)] = force_norm.mean().detach()
        total_force = total_force + force / force_norm
    return total_force, force_norms


def _scaled_features(
    generated_features: torch.Tensor, positive_features: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generated = generated_features.float()
    positives = positive_features.detach().float()
    initial = generated.detach()
    targets = torch.cat([initial, positives], dim=1)
    scale = torch.cdist(initial, targets).mean(dim=(1, 2), keepdim=True).clamp_min(1.0e-3)
    scale_inputs = (scale / generated.shape[-1] ** 0.5).clamp_min(1.0e-3)
    return generated / scale_inputs.detach(), positives / scale_inputs, scale


def drifting_loss(
    generated_features: torch.Tensor,
    positive_features: torch.Tensor,
    temperatures: list[float] | tuple[float, ...] = (0.02, 0.05, 0.2),
    steps: int = 1,
    step_size: float = 1.0,
) -> DriftLossOutput:
    """Iterative first-order drifting target.

    ``steps=1, step_size=1`` exactly preserves the original Algorithm 2
    target.  For additional Euler updates, the caller should normally scale
    ``step_size`` inversely with ``steps`` so the target displacement does not
    grow merely because the discretization is finer.
    """
    if steps < 1:
        raise ValueError("first-order drifting needs at least one update")
    if step_size <= 0.0:
        raise ValueError("first-order step_size must be positive")
    prediction, positives, scale = _scaled_features(generated_features, positive_features)
    initial = prediction.detach()
    position = initial
    force_norms: dict[float, torch.Tensor] = {}
    force = torch.zeros_like(position)
    for _ in range(steps):
        force, force_norms = _drifting_field(position, positives, temperatures)
        position = position + float(step_size) * force
    force_norm = force.square().mean().sqrt().detach()
    displacement = (position - initial).square().mean().sqrt().detach()
    return DriftLossOutput(
        F.mse_loss(prediction, position.detach()),
        scale.mean().detach(),
        force_norms,
        force_norm,
        displacement,
    )


def second_order_drifting_loss(
    generated_features: torch.Tensor,
    positive_features: torch.Tensor,
    temperatures: list[float] | tuple[float, ...] = (0.02, 0.05, 0.2),
    eta1: float = 0.25,
    eta2: float = 0.25,
    steps: int = 4,
    alpha: float = 3.0,
) -> DriftLossOutput:
    """Semi-implicit phase-space target from Brown et al. (2026), Eq. 9-10.

    Artificial velocity is used only to build a stopped training target;
    inference remains a single generator evaluation.
    """
    if steps < 1:
        raise ValueError("second-order drifting needs at least one inertial step")
    prediction, positives, scale = _scaled_features(generated_features, positive_features)
    initial = prediction.detach()
    position = initial
    velocity = torch.zeros_like(position)
    force_norms: dict[float, torch.Tensor] = {}
    for step in range(steps):
        force, force_norms = _drifting_field(position, positives, temperatures)
        damping = 1.0 - float(alpha) / float(step + 1)
        velocity = damping * velocity + float(eta1) * force
        position = position + float(eta2) * velocity
    return DriftLossOutput(
        F.mse_loss(prediction, position.detach()),
        scale.mean().detach(),
        force_norms,
        velocity.square().mean().sqrt().detach(),
        (position - initial).square().mean().sqrt().detach(),
    )
