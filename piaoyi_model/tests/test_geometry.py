import torch

from piaoyi_model.geometry import sample_occupied_volume, signed_distance_to_occupied_boxes


def test_volume_samples_are_inside_oriented_box():
    centers = torch.tensor([[[0.2, -0.1, 0.3]]])
    sizes = torch.tensor([[[0.4, 0.2, 0.3]]])
    yaws = torch.tensor([[0.7]])
    mask = torch.tensor([[True]])
    points = sample_occupied_volume(centers, sizes, yaws, mask, 512, 3, torch.Generator().manual_seed(1))
    distance = signed_distance_to_occupied_boxes(points, centers, sizes, yaws, mask)
    assert points.shape == (1, 3, 512, 3)
    assert float(distance.max()) < 1.0e-5


def test_volume_geometry_supports_full_wxyz_orientation():
    centers = torch.tensor([[[0.2, -0.1, 0.35]]])
    sizes = torch.tensor([[[0.4, 0.2, 0.3]]])
    angle = torch.tensor(0.8)
    quaternion = torch.tensor([[[torch.cos(angle / 2), torch.sin(angle / 2), 0.0, 0.0]]])
    mask = torch.tensor([[True]])
    points = sample_occupied_volume(
        centers, sizes, quaternion, mask, 512, 2, torch.Generator().manual_seed(4)
    )
    distance = signed_distance_to_occupied_boxes(points, centers, sizes, quaternion, mask)
    assert points.shape == (1, 2, 512, 3)
    assert float(distance.max()) < 1.0e-5
