import torch

from piaoyi_model.config import OccupancyDriftConfig
from piaoyi_model.features import MultiScaleOccupancyFeatures


def test_occupancy_features_are_permutation_invariant():
    cfg = OccupancyDriftConfig(num_points=64, voxel_resolutions=[4, 6])
    encoder = MultiScaleOccupancyFeatures(cfg)
    points = torch.rand(1, 2, 64, 3)
    permutation = torch.randperm(64)
    before = encoder(points)
    after = encoder(points[:, :, permutation])
    for first, second in zip(before, after, strict=True):
        torch.testing.assert_close(first, second, atol=1.0e-6, rtol=1.0e-6)
