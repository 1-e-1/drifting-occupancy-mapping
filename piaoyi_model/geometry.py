from __future__ import annotations

import torch


def _quaternion_rotate(quaternion_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quaternion_wxyz = quaternion_wxyz / quaternion_wxyz.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
    scalar = quaternion_wxyz[..., :1]
    imaginary = quaternion_wxyz[..., 1:]
    return vector + 2.0 * torch.cross(
        imaginary, torch.cross(imaginary, vector, dim=-1) + scalar * vector, dim=-1
    )


def _local_to_world(local: torch.Tensor, orientations: torch.Tensor) -> torch.Tensor:
    if orientations.ndim == local.ndim and orientations.shape[-1] == 4:
        return _quaternion_rotate(orientations, local)
    cosine, sine = orientations.cos(), orientations.sin()
    x = cosine * local[..., 0] - sine * local[..., 1]
    y = sine * local[..., 0] + cosine * local[..., 1]
    return torch.stack([x, y, local[..., 2]], dim=-1)


def _world_to_local(relative: torch.Tensor, orientations: torch.Tensor) -> torch.Tensor:
    if orientations.shape[-1:] == (4,):
        inverse = torch.cat([orientations[..., :1], -orientations[..., 1:]], dim=-1)
        return _quaternion_rotate(inverse, relative)
    cosine, sine = orientations.cos(), orientations.sin()
    local_x = cosine * relative[..., 0] + sine * relative[..., 1]
    local_y = -sine * relative[..., 0] + cosine * relative[..., 1]
    return torch.stack([local_x, local_y, relative[..., 2]], dim=-1)


def sample_occupied_volume(
    centers: torch.Tensor,
    sizes: torch.Tensor,
    yaws: torch.Tensor,
    object_mask: torch.Tensor,
    sample_count: int,
    set_count: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample uniformly from the union of oriented cuboid interiors.

    Objects are selected proportional to volume, so the resulting particle
    distribution represents occupied volume instead of visible surfaces.
    """
    batch, max_objects, _ = centers.shape
    output = []
    for batch_id in range(batch):
        volumes = sizes[batch_id].prod(dim=-1) * object_mask[batch_id].to(sizes.dtype)
        if float(volumes.sum()) <= 0:
            raise ValueError("Every scene needs at least one valid object")
        probabilities = volumes / volumes.sum()
        object_ids = torch.multinomial(
            probabilities.cpu(), sample_count * set_count, replacement=True, generator=generator
        ).to(centers.device)
        local = torch.rand(sample_count * set_count, 3, generator=generator, device="cpu").to(centers.device) - 0.5
        local = local * sizes[batch_id, object_ids]
        orientation = yaws[batch_id, object_ids]
        world = _local_to_world(local, orientation) + centers[batch_id, object_ids]
        output.append(world.reshape(set_count, sample_count, 3))
    return torch.stack(output)


def signed_distance_to_occupied_boxes(
    points: torch.Tensor,
    centers: torch.Tensor,
    sizes: torch.Tensor,
    yaws: torch.Tensor,
    object_mask: torch.Tensor,
) -> torch.Tensor:
    """Signed distance to the union of oriented boxes; negative means occupied."""
    original_shape = points.shape
    if points.ndim == 4:
        points = points.flatten(1, 2)
    relative = points[:, :, None, :] - centers[:, None, :, :]
    orientation = yaws[:, None, ...]
    local = _world_to_local(relative, orientation)
    q = local.abs() - 0.5 * sizes[:, None, :, :]
    outside = q.clamp_min(0).norm(dim=-1)
    inside = q.amax(dim=-1).clamp_max(0)
    distance = outside + inside
    distance = distance.masked_fill(~object_mask[:, None, :], float("inf"))
    union = distance.amin(dim=-1)
    if len(original_shape) == 4:
        union = union.reshape(original_shape[:3])
    return union


def volumetric_metrics(
    generated: torch.Tensor,
    target: torch.Tensor,
    centers: torch.Tensor,
    sizes: torch.Tensor,
    yaws: torch.Tensor,
    object_mask: torch.Tensor,
    threshold: float = 0.05,
    interior_margin: float = 0.01,
) -> dict[str, torch.Tensor]:
    distance = torch.cdist(generated, target)
    sdf = signed_distance_to_occupied_boxes(generated, centers, sizes, yaws, object_mask)
    return {
        "volume_chamfer": distance.amin(dim=2).mean(dim=1) + distance.amin(dim=1).mean(dim=1),
        "volume_coverage": (distance.amin(dim=1) < threshold).float().mean(dim=1),
        "occupancy_precision": (sdf <= 0).float().mean(dim=1),
        "interior_fraction": (sdf < -interior_margin).float().mean(dim=1),
        "outside_distance": sdf.clamp_min(0).mean(dim=1),
    }
