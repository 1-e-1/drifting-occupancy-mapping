import numpy as np
import torch

from codex_zbot.config import WorldModelConfig
from piaoyi_model.belief_v4 import V4BeliefLearningSystem
from piaoyi_model.config import OccupancyDriftConfig
from piaoyi_model.train_world_model_v4 import CachedBeliefDataset


def test_v4_belief_encoder_and_dynamics_shapes():
    drift = OccupancyDriftConfig(num_points=32, memory_resolution=4, dim=32, heads=4, ffn_dim=64)
    world = WorldModelConfig(latent_dim=16, robot_state_dim=23, action_dim=8, hidden_dim=32, recurrent_dim=16)
    system = V4BeliefLearningSystem(drift, world)
    position = torch.randn(2, 32, 3)
    tri_state = torch.softmax(torch.randn(2, 4**3, 3), dim=-1)
    stats = torch.randn(2, 9)
    latent = system.encoder(position, tri_state, stats)
    assert latent.shape == (2, 16)
    prediction = system.dynamics(latent, torch.randn(2, 23), torch.randn(2, 8))
    assert prediction.next_latent.shape == (2, 16)
    assert prediction.hidden.shape == (2, 16)


def test_cached_belief_dataset_crops_variable_episode_lengths(tmp_path):
    paths = []
    for episode, windows in enumerate((7, 19)):
        path = tmp_path / f"ep_{episode:04d}.npz"
        np.savez_compressed(
            path,
            position=np.zeros((windows, 4, 3), np.float32),
            tri_state_probability=np.zeros((windows, 8, 3), np.float32),
            policy_features=np.zeros((windows, 9), np.float32),
            robot_state=np.zeros((windows, 23), np.float32),
            actions=np.zeros((windows - 1, 8), np.float32),
            coverage=np.zeros(windows, np.float32),
            reconstruction_error=np.zeros(windows, np.float32),
            collision=np.zeros(windows, np.bool_),
        )
        paths.append(path)
    dataset = CachedBeliefDataset(paths, window_count=7)
    assert dataset[0]["position"].shape[0] == dataset[1]["position"].shape[0] == 7
    assert dataset[0]["actions"].shape == dataset[1]["actions"].shape == (6, 8)
