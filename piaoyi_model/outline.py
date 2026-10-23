from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy import ndimage

from .config import OccupancyDriftConfig
from .inference_v3 import OccupancyDriftV3State

if TYPE_CHECKING:
    from .model_v3 import OccupancyDriftGeneratorV3


@dataclass
class OutlineConfig:
    resolution: int = 48
    particle_sigma_voxels: float = 1.0
    observed_sigma_voxels: float = 0.75
    tri_occupied_weight: float = 0.45
    implicit_field_weight: float = 0.80
    observed_weight: float = 0.25
    particle_probability_weight: float = 1.0
    free_space_suppression: float = 0.80
    occupancy_threshold: float = 0.85
    opening_iterations: int = 0
    closing_iterations: int = 1
    min_component_voxels: int = 10
    max_components: int = 8
    component_connectivity: int = 1
    # Components whose occupied surfaces are closer than this are grouped for
    # box extraction.  The occupancy itself stays non-convex; downstream path
    # planning uses ``build_clearance_envelope`` for tool-specific inflation.
    component_merge_distance_m: float = 0.0
    min_outline_height_m: float = 0.015
    component_box_padding_voxels: float = 0.5
    unknown_threshold: float = 0.30
    max_shell_points: int = 4096
    # ``fixed`` preserves the original implementation. ``adaptive`` performs
    # confidence/density-aware Gaussian splatting: sparse uncertain particles
    # receive a wider but lower-amplitude kernel instead of becoming hard
    # occupied voxels.
    particle_splat_mode: str = "fixed"
    adaptive_knn: int = 16
    adaptive_sigma_min_voxels: float = 0.65
    adaptive_sigma_max_voxels: float = 1.80
    adaptive_sparse_amplitude_floor: float = 0.20
    adaptive_field_weight_floor: float = 0.15
    adaptive_outlier_mad: float = 3.0
    adaptive_outlier_field_threshold: float = 0.50
    adaptive_density_tail_fraction: float = 0.02
    occupancy_exit_threshold: float = 0.60
    temporal_new_weight: float = 0.30


@dataclass
class EnvironmentOutline:
    occupancy: torch.Tensor  # (B,R,R,R) bool
    confidence_grid: torch.Tensor  # (B,R,R,R)
    shell_points: list[torch.Tensor]
    frontier_points: list[torch.Tensor]
    centroid: torch.Tensor  # (B,3)
    extent: torch.Tensor  # (B,3)
    guidance_target: torch.Tensor  # (B,3), frontier centroid or shell centroid
    outline_confidence: torch.Tensor  # (B,1)
    component_count: torch.Tensor  # (B,1)
    component_centers: list[torch.Tensor] = field(default_factory=list)  # each (K,3)
    component_extents: list[torch.Tensor] = field(default_factory=list)  # each (K,3), PCA-aligned XY
    component_yaws: list[torch.Tensor] = field(default_factory=list)  # each (K,)
    component_corners: list[torch.Tensor] = field(default_factory=list)  # each (K,8,3)

    def policy_features(self) -> torch.Tensor:
        frontier_fraction = []
        for shell, frontier in zip(self.shell_points, self.frontier_points, strict=True):
            frontier_fraction.append(shell.new_tensor([frontier.shape[0] / max(1, shell.shape[0])]))
        return torch.cat(
            [
                self.centroid,
                self.extent,
                self.outline_confidence,
                torch.stack(frontier_fraction),
                self.component_count.to(self.centroid.dtype),
            ],
            dim=1,
        )


@dataclass
class ClearanceEnvelope:
    """Complex obstacle envelope after configuration-space inflation."""

    blocked: torch.Tensor  # (B,R,R,R) bool
    clearance_grid: torch.Tensor  # distance from every free voxel to occupancy, metres
    shell_points: list[torch.Tensor]
    blocked_fraction: torch.Tensor  # (B,1)
    component_count: torch.Tensor  # (B,1)
    clearance_radius_m: float


def inflate_occupancy_for_clearance(
    occupancy: np.ndarray,
    cell_size_m: np.ndarray,
    clearance_radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Minkowski-inflate occupancy; corridors narrower than 2r disappear."""
    if occupancy.ndim != 3:
        raise ValueError("occupancy must be a 3-D grid")
    if clearance_radius_m < 0:
        raise ValueError("clearance_radius_m must be non-negative")
    occupancy = occupancy.astype(bool, copy=False)
    if occupancy.any():
        clearance = ndimage.distance_transform_edt(~occupancy, sampling=cell_size_m).astype(np.float32)
        blocked = occupancy | (clearance <= clearance_radius_m)
    else:
        clearance = np.full(occupancy.shape, np.inf, dtype=np.float32)
        blocked = occupancy.copy()
    return blocked, clearance


def build_clearance_envelope(
    outline: EnvironmentOutline,
    drift_cfg: OccupancyDriftConfig,
    clearance_radius_m: float,
    max_shell_points: int = 8192,
) -> ClearanceEnvelope:
    """Build a non-convex servo/planning obstacle boundary.

    Unlike component OBBs, this retains concavities and unions nearby objects
    only when their free corridor violates the requested physical clearance.
    """
    resolution = outline.occupancy.shape[-1]
    lower = np.asarray(drift_cfg.workspace_min, dtype=np.float32)
    upper = np.asarray(drift_cfg.workspace_max, dtype=np.float32)
    cell_size = (upper - lower) / resolution
    structure = np.ones((3, 3, 3), dtype=bool)
    blocked_batch, clearance_batch = [], []
    shell_batch: list[torch.Tensor] = []
    fractions, counts = [], []
    device = outline.occupancy.device
    dtype = outline.confidence_grid.dtype
    for occupancy_tensor in outline.occupancy:
        occupancy = occupancy_tensor.detach().cpu().numpy().astype(bool)
        blocked, clearance = inflate_occupancy_for_clearance(
            occupancy, cell_size, clearance_radius_m
        )
        labels, component_count = ndimage.label(
            blocked, structure=ndimage.generate_binary_structure(3, 1)
        )
        del labels
        eroded = ndimage.binary_erosion(blocked, structure=structure, border_value=0)
        shell_index = np.argwhere(blocked & ~eroded)
        if len(shell_index) > max_shell_points:
            keep = np.linspace(0, len(shell_index) - 1, max_shell_points, dtype=np.int64)
            shell_index = shell_index[keep]
        shell_world = lower + (shell_index.astype(np.float32) + 0.5) * cell_size
        blocked_batch.append(torch.from_numpy(blocked.copy()).to(device))
        clearance_batch.append(torch.from_numpy(clearance).to(device=device, dtype=dtype))
        shell_batch.append(torch.from_numpy(shell_world).to(device=device, dtype=dtype))
        fractions.append(torch.tensor([blocked.mean()], device=device, dtype=dtype))
        counts.append(torch.tensor([component_count], device=device, dtype=torch.long))
    return ClearanceEnvelope(
        blocked=torch.stack(blocked_batch),
        clearance_grid=torch.stack(clearance_batch),
        shell_points=shell_batch,
        blocked_fraction=torch.stack(fractions),
        component_count=torch.stack(counts),
        clearance_radius_m=float(clearance_radius_m),
    )


def _splat(points: np.ndarray, lower: np.ndarray, upper: np.ndarray, resolution: int) -> np.ndarray:
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    if points.size == 0:
        return grid
    normalized = np.clip((points - lower) / np.maximum(upper - lower, 1.0e-6), 0, 1 - 1.0e-6)
    index = np.floor(normalized * resolution).astype(np.int64)
    np.add.at(grid, (index[:, 0], index[:, 1], index[:, 2]), 1.0)
    return grid


def _density_probability(density: np.ndarray) -> np.ndarray:
    positive = density[density > 0]
    if positive.size == 0:
        return density
    scale = max(float(np.quantile(positive, 0.75)), 1.0e-6)
    return 1.0 - np.exp(-density / scale)


def _sample_grid_at_points(
    grid: np.ndarray, points: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Trilinearly sample a voxel-centred scalar grid at world points."""
    if points.size == 0:
        return np.empty((0,), dtype=np.float32)
    resolution = np.asarray(grid.shape, dtype=np.float32)
    coordinates = (points - lower) / np.maximum(upper - lower, 1.0e-6) * resolution - 0.5
    coordinates = np.clip(coordinates, 0.0, resolution - 1.0)
    return ndimage.map_coordinates(
        grid, coordinates.T, order=1, mode="nearest", prefilter=False
    ).astype(np.float32)


def _adaptive_particle_probability(
    points: np.ndarray,
    field_probability: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    cfg: OutlineConfig,
) -> np.ndarray:
    """Build a soft density field using binned adaptive Gaussian kernels.

    The implementation deliberately splats into a *soft* field.  A sparse
    particle gets a larger support, but its amplitude is reduced so that it
    cannot turn an unobserved gap into confident occupied space by itself.
    Three sigma bins keep this deterministic and substantially cheaper than a
    Python loop that rasterizes one full covariance kernel per particle.
    """
    resolution = cfg.resolution
    if points.size == 0:
        return np.zeros((resolution, resolution, resolution), dtype=np.float32)
    inside = np.logical_and(points >= lower, points < upper).all(axis=1)
    points = points[inside]
    if points.size == 0:
        return np.zeros((resolution, resolution, resolution), dtype=np.float32)

    # cKDTree is intentionally imported lazily: the fixed baseline remains
    # byte-for-byte equivalent and pays no tree construction cost.
    from scipy.spatial import cKDTree

    neighbours = min(max(2, cfg.adaptive_knn + 1), len(points))
    distance, _ = cKDTree(points).query(points, k=neighbours, workers=1)
    kth = distance[:, -1] if distance.ndim == 2 else distance
    median = max(float(np.median(kth)), 1.0e-6)
    mad = max(float(np.median(np.abs(kth - median))), 1.0e-6)
    field_weight = np.clip(
        _sample_grid_at_points(field_probability, points, lower, upper), 0.0, 1.0
    )
    keep = ~(
        (kth > median + cfg.adaptive_outlier_mad * 1.4826 * mad)
        & (field_weight < cfg.adaptive_outlier_field_threshold)
    )
    points, kth, field_weight = points[keep], kth[keep], field_weight[keep]
    if points.size == 0:
        return np.zeros((resolution, resolution, resolution), dtype=np.float32)

    relative_spacing = np.clip(kth / median, 0.65, 2.25)
    sigma = np.clip(
        relative_spacing,
        cfg.adaptive_sigma_min_voxels,
        cfg.adaptive_sigma_max_voxels,
    )
    sparse_amplitude = np.clip(
        median / np.maximum(kth, 1.0e-6), cfg.adaptive_sparse_amplitude_floor, 1.0
    )
    amplitude = sparse_amplitude * (
        cfg.adaptive_field_weight_floor
        + (1.0 - cfg.adaptive_field_weight_floor) * field_weight
    )

    # Approximate continuously varying sigma with three Gaussian-filter bins.
    # The kernel is still non-uniform within every bin (maximum at its centre,
    # smoothly fading at the boundary), which is the desired fuzzy prediction.
    bin_edges = np.linspace(
        cfg.adaptive_sigma_min_voxels, cfg.adaptive_sigma_max_voxels, 4
    )
    density = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    normalized = np.clip(
        (points - lower) / np.maximum(upper - lower, 1.0e-6), 0, 1 - 1.0e-6
    )
    index = np.floor(normalized * resolution).astype(np.int64)
    for bin_id in range(3):
        if bin_id == 2:
            selected = (sigma >= bin_edges[bin_id]) & (sigma <= bin_edges[bin_id + 1])
        else:
            selected = (sigma >= bin_edges[bin_id]) & (sigma < bin_edges[bin_id + 1])
        if not selected.any():
            continue
        impulse = np.zeros_like(density)
        np.add.at(
            impulse,
            tuple(index[selected].T),
            amplitude[selected].astype(np.float32),
        )
        sigma_i = float(np.average(sigma[selected], weights=np.maximum(amplitude[selected], 1.0e-6)))
        density += ndimage.gaussian_filter(impulse, sigma_i, mode="constant")
    # Ignore numerically non-zero Gaussian tails when calibrating the density
    # scale.  Otherwise a wider kernel lowers the 75th percentile so much that
    # its intended low-confidence fringe is promoted to hard occupancy.
    peak = float(density.max())
    supported = density[density >= cfg.adaptive_density_tail_fraction * peak]
    if supported.size == 0:
        return np.zeros_like(density)
    scale = max(float(np.quantile(supported, 0.75)), 1.0e-6)
    return 1.0 - np.exp(-density / scale)


def _resize_tri_state(probability: np.ndarray, resolution: int) -> np.ndarray:
    source_resolution = round(probability.shape[0] ** (1 / 3))
    if source_resolution**3 != probability.shape[0]:
        raise ValueError("tri-state memory must be a flattened cubic grid")
    grid = probability.reshape(source_resolution, source_resolution, source_resolution, 3)
    zoom = (resolution / source_resolution,) * 3 + (1,)
    resized = ndimage.zoom(grid, zoom, order=1, mode="nearest", prefilter=False)
    return resized[:resolution, :resolution, :resolution]


def _filter_components(
    binary: np.ndarray,
    cfg: OutlineConfig,
    cell_size_m: np.ndarray | None = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    structure = ndimage.generate_binary_structure(3, cfg.component_connectivity)
    grouping_mask = binary
    if cfg.component_merge_distance_m > 0.0:
        if cell_size_m is None:
            raise ValueError("cell_size_m is required when component merging is enabled")
        # Two surfaces less than d apart meet after each is expanded by d/2.
        distance = ndimage.distance_transform_edt(~binary, sampling=cell_size_m)
        grouping_mask = binary | (distance <= 0.5 * cfg.component_merge_distance_m)
    labels, count = ndimage.label(grouping_mask, structure=structure)
    if count == 0:
        return binary, 0, np.zeros_like(labels, dtype=np.int32)
    # Rank/filter using real occupied voxels, not the temporary grouping halo.
    sizes = np.bincount(labels[binary].ravel(), minlength=count + 1)
    candidates = [index for index in range(1, len(sizes)) if sizes[index] >= cfg.min_component_voxels]
    candidates.sort(key=lambda index: int(sizes[index]), reverse=True)
    kept = candidates[: cfg.max_components]
    relabeled = np.zeros_like(labels, dtype=np.int32)
    for output_id, source_id in enumerate(kept, start=1):
        relabeled[(labels == source_id) & binary] = output_id
    return relabeled > 0, len(kept), relabeled


def _oriented_component_box(
    points: np.ndarray, padding: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Fit a deterministic yaw-only PCA box to one occupied component."""
    xy_mean = points[:, :2].mean(axis=0)
    centered = points[:, :2] - xy_mean
    if points.shape[0] >= 3 and np.linalg.norm(centered) > 1.0e-8:
        covariance = centered.T @ centered / max(1, points.shape[0] - 1)
        values, vectors = np.linalg.eigh(covariance)
        primary = vectors[:, int(np.argmax(values))]
        if primary[0] < 0 or (abs(primary[0]) < 1.0e-8 and primary[1] < 0):
            primary = -primary
        secondary = np.array([-primary[1], primary[0]], dtype=np.float32)
        rotation = np.stack([primary, secondary], axis=1).astype(np.float32)
    else:
        rotation = np.eye(2, dtype=np.float32)
    local = centered @ rotation
    local_min = local.min(axis=0) - padding[:2]
    local_max = local.max(axis=0) + padding[:2]
    local_center = 0.5 * (local_min + local_max)
    center_xy = xy_mean + local_center @ rotation.T
    z_min = float(points[:, 2].min() - padding[2])
    z_max = float(points[:, 2].max() + padding[2])
    center = np.array([center_xy[0], center_xy[1], 0.5 * (z_min + z_max)], dtype=np.float32)
    extent = np.array([*(local_max - local_min), z_max - z_min], dtype=np.float32)
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    corners = []
    for x in (local_min[0], local_max[0]):
        for y in (local_min[1], local_max[1]):
            world_xy = xy_mean + np.array([x, y], dtype=np.float32) @ rotation.T
            for z in (z_min, z_max):
                corners.append([world_xy[0], world_xy[1], z])
    return center, extent, yaw, np.asarray(corners, dtype=np.float32)


def extract_environment_outline(
    state: OccupancyDriftV3State,
    drift_cfg: OccupancyDriftConfig,
    outline_cfg: OutlineConfig | None = None,
    model: OccupancyDriftGeneratorV3 | None = None,
    previous_outline: EnvironmentOutline | None = None,
) -> EnvironmentOutline:
    """Convert stochastic volume particles into a stable servo-facing shell.

    The method is non-learned and deterministic for a fixed reconstruction:
    density fusion -> tri-state free-space carving -> morphology -> boundary.
    """
    cfg = outline_cfg or OutlineConfig()
    device = state.position.device
    dtype = state.position.dtype
    lower = np.asarray(drift_cfg.workspace_min, dtype=np.float32)
    upper = np.asarray(drift_cfg.workspace_max, dtype=np.float32)
    cell_size = (upper - lower) / cfg.resolution
    shell_batch: list[torch.Tensor] = []
    frontier_batch: list[torch.Tensor] = []
    occupancy_batch, confidence_batch = [], []
    centroids, extents, targets, outline_confidences, component_counts = [], [], [], [], []
    component_centers_batch: list[torch.Tensor] = []
    component_extents_batch: list[torch.Tensor] = []
    component_yaws_batch: list[torch.Tensor] = []
    component_corners_batch: list[torch.Tensor] = []

    field_probability_batch: np.ndarray | None = None
    if model is not None and state.memory is not None and cfg.implicit_field_weight > 0:
        axes = [
            torch.linspace(
                drift_cfg.workspace_min[i], drift_cfg.workspace_max[i], cfg.resolution + 1,
                device=device, dtype=dtype,
            )
            for i in range(3)
        ]
        axes = [0.5 * (axis[:-1] + axis[1:]) for axis in axes]
        query = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        field_parts = []
        with torch.no_grad():
            for start in range(0, query.shape[0], 16_384):
                chunk = query[start : start + 16_384][None].expand(state.position.shape[0], -1, -1)
                logit, _ = model.decode_field(state.memory, chunk)
                field_parts.append(logit.sigmoid())
        field_probability_batch = torch.cat(field_parts, dim=1).reshape(
            state.position.shape[0], cfg.resolution, cfg.resolution, cfg.resolution
        ).detach().cpu().numpy()

    for batch_id in range(state.position.shape[0]):
        particles = state.position[batch_id].detach().cpu().numpy()
        observed = state.observed_points[batch_id, state.observed_mask[batch_id]].detach().cpu().numpy()
        observed_density = ndimage.gaussian_filter(
            _splat(observed, lower, upper, cfg.resolution), cfg.observed_sigma_voxels, mode="constant"
        )
        observed_probability = _density_probability(observed_density)
        tri = _resize_tri_state(state.tri_state_probability[batch_id].detach().cpu().numpy(), cfg.resolution)
        occupied_prior = np.clip(tri[..., 0], 0, 1)
        free_probability = np.clip(tri[..., 1], 0, 1)
        unknown_probability = np.clip(tri[..., 2], 0, 1)
        field_probability = (
            field_probability_batch[batch_id] if field_probability_batch is not None else np.zeros_like(occupied_prior)
        )
        if cfg.particle_splat_mode == "adaptive":
            particle_probability = _adaptive_particle_probability(
                particles, field_probability, lower, upper, cfg
            )
        elif cfg.particle_splat_mode == "fixed":
            particle_density = ndimage.gaussian_filter(
                _splat(particles, lower, upper, cfg.resolution),
                cfg.particle_sigma_voxels,
                mode="constant",
            )
            particle_probability = _density_probability(particle_density)
        else:
            raise ValueError(f"unsupported particle_splat_mode={cfg.particle_splat_mode!r}")

        confidence = 1.0 - (
            (1.0 - cfg.particle_probability_weight * particle_probability)
            * (1.0 - cfg.tri_occupied_weight * occupied_prior)
            * (1.0 - cfg.observed_weight * observed_probability)
            * (1.0 - cfg.implicit_field_weight * field_probability)
        )
        confidence *= np.clip(1.0 - cfg.free_space_suppression * free_probability, 0, 1)
        if previous_outline is not None:
            previous_confidence = previous_outline.confidence_grid[batch_id].detach().cpu().numpy()
            previous_occupancy = previous_outline.occupancy[batch_id].detach().cpu().numpy().astype(bool)
            alpha = float(np.clip(cfg.temporal_new_weight, 0.0, 1.0))
            confidence = (1.0 - alpha) * previous_confidence + alpha * confidence
            binary = (confidence >= cfg.occupancy_threshold) | (
                previous_occupancy & (confidence >= cfg.occupancy_exit_threshold)
            )
        else:
            binary = confidence >= cfg.occupancy_threshold
        structure = np.ones((3, 3, 3), dtype=bool)
        z_centers = lower[2] + (np.arange(cfg.resolution, dtype=np.float32) + 0.5) * cell_size[2]
        binary[:, :, z_centers < cfg.min_outline_height_m] = False
        if cfg.opening_iterations > 0:
            opening_structure = ndimage.generate_binary_structure(3, cfg.component_connectivity)
            binary = ndimage.binary_opening(
                binary, structure=opening_structure, iterations=cfg.opening_iterations
            )
        if cfg.closing_iterations > 0:
            binary = ndimage.binary_closing(binary, structure=structure, iterations=cfg.closing_iterations)
        binary = ndimage.binary_fill_holes(binary)
        binary, component_count, component_labels = _filter_components(binary, cfg, cell_size)
        if not binary.any():
            # Deterministic connected fallback for a very uncertain
            # reconstruction.  Selecting independent top-k voxels can leave
            # only sub-threshold singletons after component filtering.
            seed_index = np.unravel_index(int(np.argmax(confidence)), confidence.shape)
            binary[seed_index] = True
            fallback_structure = ndimage.generate_binary_structure(3, cfg.component_connectivity)
            while int(binary.sum()) < cfg.min_component_voxels:
                binary = ndimage.binary_dilation(binary, structure=fallback_structure, iterations=1)
            binary, component_count, component_labels = _filter_components(binary, cfg, cell_size)

        eroded = ndimage.binary_erosion(binary, structure=structure, border_value=0)
        shell = binary & ~eroded
        unknown_nearby = ndimage.maximum_filter(unknown_probability, size=3, mode="nearest") >= cfg.unknown_threshold
        frontier = shell & unknown_nearby
        shell_index = np.argwhere(shell)
        frontier_index = np.argwhere(frontier)
        if shell_index.shape[0] > cfg.max_shell_points:
            # Highest-confidence shell voxels are stable and deterministic.
            values = confidence[tuple(shell_index.T)]
            keep = np.argsort(values)[-cfg.max_shell_points:]
            shell_index = shell_index[keep]
        shell_world = lower + (shell_index.astype(np.float32) + 0.5) * cell_size
        frontier_world = lower + (frontier_index.astype(np.float32) + 0.5) * cell_size
        occupied_index = np.argwhere(binary)
        occupied_world = lower + (occupied_index.astype(np.float32) + 0.5) * cell_size
        box_centers, box_extents, box_yaws, box_corners = [], [], [], []
        padding = cfg.component_box_padding_voxels * cell_size
        for component_id in range(1, component_count + 1):
            component_index = np.argwhere(component_labels == component_id)
            component_world = lower + (component_index.astype(np.float32) + 0.5) * cell_size
            center_i, extent_i, yaw_i, corners_i = _oriented_component_box(component_world, padding)
            box_centers.append(center_i)
            box_extents.append(extent_i)
            box_yaws.append(yaw_i)
            box_corners.append(corners_i)
        centroid = occupied_world.mean(axis=0)
        extent = occupied_world.max(axis=0) - occupied_world.min(axis=0)
        guidance = frontier_world.mean(axis=0) if len(frontier_world) else centroid
        outline_confidence = float(confidence[shell].mean()) if shell.any() else 0.0

        occupancy_batch.append(torch.from_numpy(binary.copy()).to(device))
        confidence_batch.append(torch.from_numpy(confidence.astype(np.float32)).to(device=device, dtype=dtype))
        shell_batch.append(torch.from_numpy(shell_world).to(device=device, dtype=dtype))
        frontier_batch.append(torch.from_numpy(frontier_world).to(device=device, dtype=dtype))
        centroids.append(torch.from_numpy(centroid).to(device=device, dtype=dtype))
        extents.append(torch.from_numpy(extent).to(device=device, dtype=dtype))
        targets.append(torch.from_numpy(guidance).to(device=device, dtype=dtype))
        outline_confidences.append(torch.tensor([outline_confidence], device=device, dtype=dtype))
        component_counts.append(torch.tensor([component_count], device=device, dtype=torch.long))
        component_centers_batch.append(torch.as_tensor(np.asarray(box_centers), device=device, dtype=dtype).reshape(-1, 3))
        component_extents_batch.append(torch.as_tensor(np.asarray(box_extents), device=device, dtype=dtype).reshape(-1, 3))
        component_yaws_batch.append(torch.as_tensor(np.asarray(box_yaws), device=device, dtype=dtype).reshape(-1))
        component_corners_batch.append(torch.as_tensor(np.asarray(box_corners), device=device, dtype=dtype).reshape(-1, 8, 3))

    return EnvironmentOutline(
        torch.stack(occupancy_batch),
        torch.stack(confidence_batch),
        shell_batch,
        frontier_batch,
        torch.stack(centroids),
        torch.stack(extents),
        torch.stack(targets),
        torch.stack(outline_confidences),
        torch.stack(component_counts),
        component_centers_batch,
        component_extents_batch,
        component_yaws_batch,
        component_corners_batch,
    )


def outline_metrics(
    outline: EnvironmentOutline,
    target_occupancy: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute voxel, silhouette and boundary metrics for one batch."""
    predicted = outline.occupancy.bool()
    target = target_occupancy.bool()
    intersection = (predicted & target).flatten(1).sum(dim=1).float()
    union = (predicted | target).flatten(1).sum(dim=1).clamp_min(1).float()
    predicted_count = predicted.flatten(1).sum(dim=1).clamp_min(1).float()
    target_count = target.flatten(1).sum(dim=1).clamp_min(1).float()
    result: dict[str, torch.Tensor] = {
        "voxel_iou": intersection / union,
        "voxel_precision": intersection / predicted_count,
        "voxel_recall": intersection / target_count,
    }
    silhouettes = []
    for axis in range(1, 4):
        predicted_silhouette = predicted.any(dim=axis)
        target_silhouette = target.any(dim=axis)
        intersection_2d = (predicted_silhouette & target_silhouette).flatten(1).sum(dim=1).float()
        union_2d = (predicted_silhouette | target_silhouette).flatten(1).sum(dim=1).clamp_min(1).float()
        silhouettes.append(intersection_2d / union_2d)
    result["silhouette_iou"] = torch.stack(silhouettes, dim=1).mean(dim=1)

    structure = torch.ones(1, 1, 3, 3, 3, device=target.device)
    eroded_target = -torch.nn.functional.max_pool3d(
        -target[:, None].float(), kernel_size=3, stride=1, padding=1
    )[:, 0] > 0.5
    target_shell = target & ~eroded_target
    resolution = target.shape[-1]
    axes = [
        torch.linspace(lower[i], upper[i], resolution + 1, device=target.device, dtype=lower.dtype)
        for i in range(3)
    ]
    axes = [0.5 * (axis[:-1] + axis[1:]) for axis in axes]
    centers = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    chamfer, coverage = [], []
    threshold = float(((upper - lower) / resolution).norm())
    for batch_id, predicted_shell in enumerate(outline.shell_points):
        target_points = centers[target_shell[batch_id]]
        if predicted_shell.numel() == 0 or target_points.numel() == 0:
            chamfer.append(target.new_tensor(float("inf"), dtype=torch.float32))
            coverage.append(target.new_tensor(0.0, dtype=torch.float32))
            continue
        if target_points.shape[0] > 4096:
            target_points = target_points[:: max(1, target_points.shape[0] // 4096)][:4096]
        distance = torch.cdist(predicted_shell[None], target_points[None])[0]
        chamfer.append(distance.amin(dim=1).mean() + distance.amin(dim=0).mean())
        coverage.append((distance.amin(dim=0) <= threshold).float().mean())
    result["boundary_chamfer"] = torch.stack(chamfer)
    result["boundary_coverage"] = torch.stack(coverage)
    return result
