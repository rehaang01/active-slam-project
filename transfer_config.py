"""Shared transfer-learning configuration for lightweight and Gazebo stacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


ACTION_DIM = 3
ALTITUDE_LEVELS: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)
N_ALTITUDES = len(ALTITUDE_LEVELS)
ALT_UP_THRESH = 0.3
ALT_DOWN_THRESH = -0.3

MAP_CHANNEL_NAMES: tuple[str, ...] = (
    "current_occupancy",
    "current_visited",
    "current_trajectory",
    "least_explored_occupancy",
    "cross_altitude_visited_fraction",
)
MAP_CHANNELS = len(MAP_CHANNEL_NAMES)

SCALAR_NAMES: tuple[str, ...] = (
    "covariance_norm",
    "frontier_count_norm",
    "total_coverage",
    "nearest_frontier_distance_norm",
    "frontier_direction_rel_yaw_norm",
    "altitude_index_norm",
    "current_altitude_coverage",
    "other_altitude_potential",
    "minimum_other_altitude_coverage",
    "step_fraction",
)
N_SCALARS = len(SCALAR_NAMES)

FEATURES_DIM = 256
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 1
POLICY_HEAD_ARCH = (256, 256)

LIGHTWEIGHT_TRAIN_SEEDS: tuple[int, ...] = tuple(range(100, 120))
LIGHTWEIGHT_EVAL_SEEDS: tuple[int, ...] = tuple(range(200, 300))

LIGHTWEIGHT_TOTAL_TIMESTEPS = 600_000
LIGHTWEIGHT_CHECKPOINT_INTERVAL = 60_000


def linear_schedule(start: float, end: float) -> Callable[[float], float]:
    """Stable-Baselines schedule with progress_remaining in [1, 0]."""

    def schedule(progress_remaining: float) -> float:
        progress_remaining = min(max(progress_remaining, 0.0), 1.0)
        done_frac = 1.0 - progress_remaining
        return start + (end - start) * done_frac

    return schedule


@dataclass(frozen=True)
class ObservationSpec:
    map_channels: int = MAP_CHANNELS
    scalars: int = N_SCALARS
    action_dim: int = ACTION_DIM
    altitude_levels: Sequence[float] = ALTITUDE_LEVELS


OBSERVATION_SPEC = ObservationSpec()
