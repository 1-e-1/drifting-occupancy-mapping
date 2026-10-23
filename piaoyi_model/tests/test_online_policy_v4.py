import torch

from codex_zbot.config import RewardConfig
from codex_zbot.policy.features_v4 import (
    V4_ACTOR_OBSERVATION_DIM,
    V4_CRITIC_OBSERVATION_DIM,
    build_v4_actor_observation,
    build_v4_critic_observation,
)
from codex_zbot.policy.rewards_v4 import camera_to_oriented_boxes_clearance, normalized_v4_active_vision_reward
from piaoyi_model.config import OccupancyDriftConfig
from piaoyi_model.online_v4 import OnlineV4ObservationBuffer


def test_v4_policy_contract():
    batch = 2
    actor = build_v4_actor_observation(
        torch.zeros(batch, 8), torch.zeros(batch, 8), torch.zeros(batch, 7),
        torch.zeros(batch, 128), torch.zeros(batch, 9), torch.zeros(batch, 8), torch.zeros(batch, 8),
    )
    critic = build_v4_critic_observation(actor, torch.zeros(batch, 7))
    assert actor.shape == (batch, V4_ACTOR_OBSERVATION_DIM)
    assert critic.shape == (batch, V4_CRITIC_OBSERVATION_DIM)


def test_online_buffer_temporal_stride_and_reset_padding():
    cfg = OccupancyDriftConfig(sequence_length=3, frame_stride=2, image_size=8)
    buffer = OnlineV4ObservationBuffer(1, cfg, "cpu")
    intrinsics = torch.eye(3).unsqueeze(0)
    for value in range(5):
        rgb = torch.full((1, 16, 16, 3), value, dtype=torch.uint8)
        depth = torch.full((1, 16, 16, 1), float(value + 1))
        pose = torch.zeros(1, 7)
        pose[:, 0] = value
        if value == 0:
            buffer.reset(torch.tensor([0]), rgb, depth, pose, intrinsics)
        else:
            buffer.append(rgb, depth, pose, intrinsics)
    batch = buffer.batch()
    assert batch["pose"][0, :, 0].tolist() == [0.0, 2.0, 4.0]
    assert batch["rgb"].shape == (1, 3, 3, 8, 8)


def test_v4_reward_prefers_information_gain_and_penalizes_collision():
    cfg = RewardConfig()
    zeros = torch.zeros(2)
    terms = normalized_v4_active_vision_reward(
        cfg,
        torch.tensor([0.02, 0.0]), torch.tensor([0.01, 0.0]), torch.tensor([0.01, 0.0]),
        torch.tensor([0.02, 0.0]), torch.tensor([0.5, 0.0]), torch.tensor([0.8, 0.8]),
        torch.tensor([0.02, 0.02]), torch.tensor([0.005, 0.02]),
        torch.zeros(2, 8), torch.zeros(2, 8), torch.zeros(2, 8),
        torch.zeros(2, dtype=torch.bool), torch.tensor([False, True]),
        torch.zeros(2),
    )
    assert terms.total[0] > terms.total[1]
    assert terms.safety[1] < terms.safety[0]


def test_camera_clearance_uses_oriented_box_surface_and_ignores_inactive_objects():
    camera = torch.tensor([[0.50, 0.00, 0.10], [0.00, 0.00, 0.00]])
    centers = torch.tensor([[[0.00, 0.00, 0.10], [10.0, 0.0, 0.0]],
                            [[0.00, 0.00, 0.00], [0.0, 0.0, 0.0]]])
    sizes = torch.tensor([[0.20, 0.40, 0.20], [1.0, 1.0, 1.0]])
    yaws = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    active = torch.tensor([[True, False], [False, True]])
    clearance = camera_to_oriented_boxes_clearance(camera, centers, sizes, yaws, active)
    assert torch.allclose(clearance, torch.tensor([0.40, -0.50]), atol=1.0e-6)


def test_v4_reward_penalizes_camera_proximity():
    cfg = RewardConfig(camera_proximity=-2.0)
    zeros = torch.zeros(2)
    terms = normalized_v4_active_vision_reward(
        cfg, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros,
        torch.zeros(2, 8), torch.zeros(2, 8), torch.zeros(2, 8),
        torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool),
        torch.tensor([0.0, 1.0]),
    )
    assert terms.total[0] > terms.total[1]
    assert torch.allclose(terms.proximity, torch.tensor([0.0, 1.0]))
