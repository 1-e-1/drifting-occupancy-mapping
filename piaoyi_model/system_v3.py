from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .conditioning import collect_observed_object_points, image_condition
from .config import OccupancyDriftConfig
from .drift_loss import drifting_loss, second_order_drifting_loss
from .evidence import RayEvidence, build_ray_evidence, fixed_grid_centers
from .features import MultiScaleOccupancyFeatures
from .geometry import sample_occupied_volume, signed_distance_to_occupied_boxes, volumetric_metrics
from .model_v3 import OccupancyDriftGeneratorV3, OccupancyDriftV3Output


@dataclass
class OccupancyV3LossOutput:
    total: torch.Tensor
    drift: torch.Tensor
    volume_chamfer: torch.Tensor
    outside: torch.Tensor
    observation: torch.Tensor
    repulsion: torch.Tensor
    tri_state: torch.Tensor
    free_space: torch.Tensor
    implicit_occupancy: torch.Tensor
    implicit_sdf: torch.Tensor
    volume_coverage: torch.Tensor
    occupancy_precision: torch.Tensor
    interior_fraction: torch.Tensor
    coarse_occupied_iou: torch.Tensor
    coarse_free_accuracy: torch.Tensor
    implicit_occupancy_accuracy: torch.Tensor
    drift_scale: torch.Tensor
    phase_velocity: torch.Tensor
    completion_scale: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {key: float(value.detach()) for key, value in self.__dict__.items() if isinstance(value, torch.Tensor)}


def _subsample(points: torch.Tensor, count: int, generator: torch.Generator | None) -> torch.Tensor:
    count = min(count, points.shape[-2])
    ids = torch.randperm(points.shape[-2], generator=generator, device="cpu")[:count].to(points.device)
    return points[..., ids, :]


def _masked_sample(
    points: torch.Tensor,
    mask: torch.Tensor,
    count: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, output_mask = [], []
    for batch_id in range(points.shape[0]):
        valid = points[batch_id, mask[batch_id]]
        selected = torch.zeros(count, 3, device=points.device, dtype=points.dtype)
        selected_mask = torch.zeros(count, device=points.device, dtype=torch.bool)
        actual = min(count, valid.shape[0])
        if actual:
            ids = torch.randperm(valid.shape[0], generator=generator, device="cpu")[:actual].to(points.device)
            selected[:actual] = valid[ids]
            selected_mask[:actual] = True
        output.append(selected)
        output_mask.append(selected_mask)
    return torch.stack(output), torch.stack(output_mask)


def _tri_state_targets(
    batch: dict[str, torch.Tensor],
    evidence: RayEvidence,
    free_cells: torch.Tensor,
    cfg: OccupancyDriftConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    centers = fixed_grid_centers(cfg, batch["rgb"].device, batch["rgb"].dtype)
    centers = centers[None].expand(batch["rgb"].shape[0], -1, -1)
    sdf = signed_distance_to_occupied_boxes(
        centers, batch["object_centers"], batch["object_sizes"], batch["object_yaws"], batch["object_mask"]
    )
    cell_size = (torch.as_tensor(cfg.workspace_max, device=centers.device) - torch.as_tensor(
        cfg.workspace_min, device=centers.device
    )) / cfg.memory_resolution
    intersection_radius = 0.5 * cell_size.norm()
    occupied = sdf <= intersection_radius
    labels = torch.full_like(sdf, 2, dtype=torch.long)  # unknown
    labels[free_cells & ~occupied] = 1
    labels[occupied] = 0
    return labels, occupied


def _sample_true_occupied_cells(occupied: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    probabilities = occupied.to(noise.dtype)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1)
    cdf = probabilities.cumsum(dim=1).contiguous()
    uniform = 0.5 * (1.0 + torch.erf(noise[..., 2] / (2.0**0.5)))
    ids = torch.searchsorted(cdf, uniform.flatten(1), right=False)
    return ids.clamp_max(cdf.shape[1] - 1).reshape(uniform.shape)


def compute_occupancy_v3_loss(
    model: OccupancyDriftGeneratorV3,
    features: MultiScaleOccupancyFeatures,
    batch: dict[str, torch.Tensor],
    cfg: OccupancyDriftConfig,
    generator: torch.Generator | None = None,
    epoch: int = 0,
    enable_teacher_forcing: bool = True,
) -> tuple[OccupancyV3LossOutput, torch.Tensor]:
    observed, observed_mask = collect_observed_object_points(batch, cfg, generator)
    images = image_condition(batch, cfg.max_depth_m)
    evidence = build_ray_evidence(batch, cfg)
    noise = torch.randn(
        batch["rgb"].shape[0], cfg.generated_samples, cfg.num_points, 3, generator=generator, device="cpu"
    ).to(batch["rgb"].device)
    memory = model.encode_memory(images, batch["pose"], evidence)
    tri_labels, occupied_cells = _tri_state_targets(batch, evidence, memory.free_cells, cfg)
    teacher_ratio = 0.0
    teacher_ids = None
    if enable_teacher_forcing and cfg.coarse_teacher_forcing_epochs > 0:
        teacher_ratio = max(0.0, 1.0 - epoch / cfg.coarse_teacher_forcing_epochs)
        teacher_ids = _sample_true_occupied_cells(occupied_cells, noise)
    generated, coarse_ids = model.generate(
        noise, memory, observed, observed_mask, teacher_cell_ids=teacher_ids,
        teacher_forcing_ratio=teacher_ratio,
    )
    output = OccupancyDriftV3Output(generated, memory, coarse_ids)
    generated = output.points
    positives = sample_occupied_volume(
        batch["object_centers"], batch["object_sizes"], batch["object_yaws"], batch["object_mask"],
        cfg.num_points, cfg.positive_samples, generator,
    )

    generated_features = features(generated)
    positive_features = features(positives)
    drift_total = generated.new_zeros(())
    scales, phase_velocities = [], []
    for weight, generated_feature, positive_feature in zip(
        cfg.feature_weights, generated_features, positive_features, strict=True
    ):
        if cfg.drift_order == 2:
            drift = second_order_drifting_loss(
                generated_feature, positive_feature, cfg.temperatures, cfg.second_order_eta1,
                cfg.second_order_eta2, cfg.second_order_steps, cfg.second_order_alpha,
            )
        elif cfg.drift_order == 1:
            drift = drifting_loss(
                generated_feature,
                positive_feature,
                cfg.temperatures,
                cfg.first_order_steps,
                cfg.first_order_step_size,
            )
        else:
            raise ValueError("drift_order must be 1 or 2")
        drift_total = drift_total + float(weight) * drift.loss
        scales.append(drift.scale)
        phase_velocities.append(drift.velocity_norm)

    generated_match = _subsample(generated, cfg.match_points, generator)
    positive_match = _subsample(positives, cfg.match_points, generator)
    paired_count = min(cfg.generated_samples, cfg.positive_samples)
    generated_pair = generated_match[:, :paired_count].flatten(0, 1)
    positive_pair = positive_match[:, :paired_count].flatten(0, 1)
    distances = torch.cdist(generated_pair, positive_pair)
    volume_chamfer = distances.amin(dim=2).mean() + distances.amin(dim=1).mean()

    sdf = signed_distance_to_occupied_boxes(
        generated, batch["object_centers"], batch["object_sizes"], batch["object_yaws"], batch["object_mask"]
    )
    outside = (sdf + cfg.interior_margin_m).clamp_min(0).mean()
    generated_all = generated.flatten(1, 2)
    observed_sub = observed[:, : min(cfg.match_points, observed.shape[1])]
    observed_sub_mask = observed_mask[:, : observed_sub.shape[1]]
    observed_distance = torch.cdist(observed_sub, generated_all).amin(dim=2)
    observation = (observed_distance * observed_sub_mask).sum() / observed_sub_mask.sum().clamp_min(1)

    repulsion_points = generated_match.flatten(0, 1)
    repulsion_distance = torch.cdist(repulsion_points, repulsion_points)
    diagonal = torch.eye(repulsion_points.shape[1], device=generated.device, dtype=torch.bool).unsqueeze(0)
    nearest = repulsion_distance.masked_fill(diagonal, float("inf")).amin(dim=2)
    repulsion = F.relu(0.015 - nearest).square().mean()

    class_weights = torch.as_tensor(cfg.tri_state_class_weights, device=generated.device, dtype=generated.dtype)
    flat_logits = output.memory.tri_state_logits.flatten(0, 1)
    flat_labels = tri_labels.flatten()
    tri_ce = F.cross_entropy(flat_logits, flat_labels, weight=class_weights, reduction="none")
    tri_probability = flat_logits.softmax(dim=-1).gather(1, flat_labels[:, None]).squeeze(1)
    tri_state = ((1.0 - tri_probability).pow(cfg.tri_state_focal_gamma) * tri_ce).mean()
    tri_prediction = output.memory.tri_state_logits.argmax(dim=-1)
    predicted_occupied = tri_prediction == 0
    intersection = (predicted_occupied & occupied_cells).sum(dim=1).float()
    union = (predicted_occupied | occupied_cells).sum(dim=1).clamp_min(1).float()
    coarse_iou = (intersection / union).mean()
    free_target = tri_labels == 1
    coarse_free_accuracy = ((tri_prediction == 1) & free_target).sum().float() / free_target.sum().clamp_min(1)

    free_points, free_mask = _masked_sample(
        evidence.free_points.flatten(1, 2), evidence.free_mask.flatten(1, 2), cfg.match_points, generator
    )
    free_distance = torch.cdist(generated_all, free_points).masked_fill(~free_mask[:, None], float("inf")).amin(dim=2)
    free_space = torch.exp(-free_distance / cfg.free_space_radius_m).mean()

    query_count = cfg.implicit_query_points
    occupied_query = sample_occupied_volume(
        batch["object_centers"], batch["object_sizes"], batch["object_yaws"], batch["object_mask"],
        query_count // 2, 1, generator,
    )[:, 0]
    random_query = torch.rand(
        batch["rgb"].shape[0], query_count - occupied_query.shape[1], 3, generator=generator, device="cpu"
    ).to(generated.device)
    lower = torch.as_tensor(cfg.workspace_min, device=generated.device, dtype=generated.dtype)
    upper = torch.as_tensor(cfg.workspace_max, device=generated.device, dtype=generated.dtype)
    random_query = lower + random_query * (upper - lower)
    field_query = torch.cat([occupied_query, random_query], dim=1)
    target_sdf = signed_distance_to_occupied_boxes(
        field_query, batch["object_centers"], batch["object_sizes"], batch["object_yaws"], batch["object_mask"]
    ).clamp(-cfg.sdf_truncation_m, cfg.sdf_truncation_m)
    occupancy_target = (target_sdf <= 0).to(generated.dtype)
    occupancy_logit, predicted_sdf = model.decode_field(output.memory, field_query)
    implicit_occupancy = F.binary_cross_entropy_with_logits(occupancy_logit, occupancy_target)
    implicit_sdf = F.smooth_l1_loss(predicted_sdf, target_sdf, beta=0.02)
    implicit_accuracy = ((occupancy_logit > 0) == (occupancy_target > 0.5)).float().mean()

    metric_target = positives[:, 0]
    metric_generated = generated[:, 0]
    metrics = volumetric_metrics(
        metric_generated, metric_target, batch["object_centers"], batch["object_sizes"],
        batch["object_yaws"], batch["object_mask"], cfg.occupancy_threshold_m, cfg.interior_margin_m,
    )
    completion_scale = generated.new_tensor(min(1.0, (epoch + 1) / max(1, cfg.coarse_warmup_epochs)))
    completion_loss = (
        cfg.drift_weight * drift_total
        + cfg.volume_chamfer_weight * volume_chamfer
        + cfg.outside_weight * outside
        + cfg.observation_weight * observation
        + cfg.repulsion_weight * repulsion
        + cfg.free_space_weight * free_space
    )
    total = (
        cfg.tri_state_weight * tri_state
        + cfg.implicit_occupancy_weight * implicit_occupancy
        + cfg.implicit_sdf_weight * implicit_sdf
        + completion_scale * completion_loss
    )
    return OccupancyV3LossOutput(
        total, drift_total, volume_chamfer, outside, observation, repulsion, tri_state, free_space,
        implicit_occupancy, implicit_sdf, metrics["volume_coverage"].mean(),
        metrics["occupancy_precision"].mean(), metrics["interior_fraction"].mean(), coarse_iou,
        coarse_free_accuracy, implicit_accuracy, torch.stack(scales).mean(),
        torch.stack(phase_velocities).mean(), completion_scale,
    ), generated
