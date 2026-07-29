"""Shared feature extractor and policy helpers for transfer-compatible SLAM."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from transfer_config import FEATURES_DIM, LSTM_HIDDEN_SIZE, LSTM_LAYERS, POLICY_HEAD_ARCH


class CoordConv2d(nn.Module):
    """Conv2d that appends normalized x/y coordinate channels."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _channels, height, width = x.shape
        yy = torch.linspace(-1.0, 1.0, height, device=x.device).view(1, 1, height, 1).expand(batch, 1, height, width)
        xx = torch.linspace(-1.0, 1.0, width, device=x.device).view(1, 1, 1, width).expand(batch, 1, height, width)
        return self.conv(torch.cat([x, xx, yy], dim=1))


class TransferSLAMFeatureExtractor(BaseFeaturesExtractor):
    """Transfer-compatible extractor shared across lightweight and Gazebo stacks."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = FEATURES_DIM):
        super().__init__(observation_space, features_dim)
        map_shape = observation_space["map_tensor"].shape
        scalar_dim = observation_space["scalars"].shape[0]

        self.cnn = nn.Sequential(
            CoordConv2d(map_shape[0], 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(6),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out_dim = self.cnn(torch.zeros(1, *map_shape)).shape[1]

        self.scalar_net = nn.Sequential(
            nn.LayerNorm(scalar_dim),
            nn.Linear(scalar_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.combine = nn.Sequential(
            nn.Linear(cnn_out_dim + 32, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        map_tensor = obs["map_tensor"].float()
        scalars = obs["scalars"].float()
        return self.combine(torch.cat([self.cnn(map_tensor), self.scalar_net(scalars)], dim=1))


def build_recurrent_policy_kwargs(features_dim: int = FEATURES_DIM) -> dict:
    return dict(
        features_extractor_class=TransferSLAMFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=features_dim),
        net_arch=dict(pi=list(POLICY_HEAD_ARCH), vf=list(POLICY_HEAD_ARCH)),
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        n_lstm_layers=LSTM_LAYERS,
    )
