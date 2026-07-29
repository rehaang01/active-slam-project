"""Transfer-compatible lightweight 3D Active SLAM environment."""

from __future__ import annotations

import math
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from transfer_config import (
    ACTION_DIM,
    ALTITUDE_LEVELS,
    ALT_DOWN_THRESH,
    ALT_UP_THRESH,
    MAP_CHANNELS,
    N_ALTITUDES,
    N_SCALARS,
)

GRID_SIZE = 48
SENSOR_RANGE = 14
ADJACENT_SENSOR_RANGE = SENSOR_RANGE // 2
STEP_SIZE = 3.0
MAX_STEPS = int(os.environ.get("MAX_STEPS_OVERRIDE", 400))
VISIT_RADIUS = 3

COV_MOTION_RATE = 0.03
COV_ALT_CHANGE_RATE = 0.15
COV_LOOP_DECAY = 0.70
COV_MAX = 4.0
MIN_ALT_DWELL = 20

MAX_LC_PER_EPISODE = 30
LC_LANDMARK_SPACING = 20
LC_MIN_GAP = 40
LC_DIST = 4
LC_MIN_TRAVEL = 12.0

_VANILLA = os.environ.get("VANILLA_MODE", "0") == "1"

W_NEW_CELL = 1.0
W_PROGRESS = 0.0
W_FRONTIER_DIR = 0.0 if _VANILLA else 2.0
W_COLLISION = -1.0
W_STEP = -0.05
W_REVISIT = 0.0 if _VANILLA else -0.15
W_ALT_NEW_VISIT = 0.0 if _VANILLA else 25.0
W_ALT_SWITCH_COST = 0.0 if _VANILLA else -3.0
W_ALT_SWITCH_BAD = 0.0 if _VANILLA else -2.0
W_STAGNATION = 0.0 if _VANILLA else -15.0
W_COVERAGE_BONUS = 0.0 if _VANILLA else 30.0
W_COVERAGE_BONUS_THRESHOLD = float(os.environ.get("W_COVERAGE_BONUS_THRESHOLD_OVERRIDE", 0.40))
W_TERMINAL_COVERAGE = 0.0 if _VANILLA else 300.0
W_COV_PENALTY = 0.0 if _VANILLA else float(os.environ.get("W_COV_PENALTY_OVERRIDE", -0.8))
W_TERMINAL_COV_PENALTY = 0.0 if _VANILLA else float(os.environ.get("W_TERMINAL_COV_PENALTY_OVERRIDE", -120.0))
W_LEAST_EXPLORED_BONUS = 0.0 if _VANILLA else 0.5
W_LOOP_CLOSURE = 0.0 if _VANILLA else float(os.environ.get("W_LOOP_CLOSURE_OVERRIDE", 25.0))
W_BREADTH_BONUS = 0.0 if _VANILLA else float(os.environ.get("W_BREADTH_BONUS_OVERRIDE", 60.0))
W_COV_DELTA = 0.0 if _VANILLA else float(os.environ.get("W_COV_DELTA_OVERRIDE", 0.0))
STAGNATION_STEPS = 25

# --- Sim-to-real transfer knobs (DEFAULT-OFF; do not alter the headline policy) ---
# These are roadmap reward terms for retraining a transfer-friendly policy that better
# matches the Gazebo/RTAB-Map dynamics (see lightweight_train/TRANSFER_DIAGNOSTICS.md).
# All default to no-op values, so re-running evaluation reproduces the published 91.90% headline.
#   W_SMOOTH:         penalize action jerk  -W_SMOOTH * ||a_t - a_{t-1}||^2  (counters the RED
#                     xy-flip / jitter verdict that forces inference-time EMA smoothing on transfer).
#   W_ALT_TRANSITION: per-altitude-switch penalty (negative) discouraging Z-filter thrashing that
#                     stalls RTAB-Map re-slicing on the real stack.
#   FRONTIER_DEF:     'free_adj_unknown' (default, classical clean-grid definition) or
#                     'unknown_adj_known' (matches the RTAB-Map Z-sliced projection used by the
#                     Gazebo transfer env, envs/active_slam_env.py).
W_SMOOTH = 0.0 if _VANILLA else float(os.environ.get("W_SMOOTH_OVERRIDE", 0.0))
W_ALT_TRANSITION = 0.0 if _VANILLA else float(os.environ.get("W_ALT_TRANSITION_OVERRIDE", 0.0))
FRONTIER_DEF = os.environ.get("FRONTIER_DEF", "free_adj_unknown")

CURRICULUM_PHASES = ("easy", "medium", "hard")


def _rand_count(rng: np.random.RandomState, low: int, high: int, difficulty: str, scale: float = 1.0) -> int:
    span = max(int(round(low * scale)), 0), max(int(round(high * scale)), 0)
    lo, hi = span
    hi = max(lo + 1, hi)
    return int(rng.randint(lo, hi))


def generate_3d_warehouse(size: int = GRID_SIZE, n_alt: int = N_ALTITUDES, seed: int | None = None, difficulty: str = "hard") -> np.ndarray:
    """Generate a warehouse map with curriculum-controlled bottlenecks."""

    rng = np.random.RandomState(seed)
    maps = np.zeros((n_alt, size, size), dtype=np.float32)
    obstacle_scale = 1.0 if difficulty == "hard" else 0.5
    include_middle_wall = difficulty in {"medium", "hard"}

    for a in range(n_alt):
        maps[a, 0, :] = 1.0
        maps[a, -1, :] = 1.0
        maps[a, :, 0] = 1.0
        maps[a, :, -1] = 1.0

    shelf_len = size // 4
    top_row = size // 5
    bot_row = size * 3 // 5

    for col in range(6, size - 6, 6):
        if col + 2 < size - 2:
            maps[0, top_row : top_row + shelf_len, col : col + 2] = 1.0

    for col in range(10, size - 10, 10):
        if col + 2 < size - 2:
            for alt in range(min(3, n_alt)):
                maps[alt, bot_row : bot_row + shelf_len, col : col + 2] = 1.0

    if include_middle_wall:
        mid = size // 2
        for alt in range(n_alt):
            maps[alt, mid, 2 : size - 2] = 1.0
            for gap in np.linspace(size // 6, size * 5 // 6, 3, dtype=int):
                lo = max(2, gap - 2)
                hi = min(size - 2, gap + 4)
                maps[alt, mid, lo:hi] = 0.0

    for _ in range(_rand_count(rng, 4, 10, difficulty, obstacle_scale)):
        row, col = rng.randint(2, size - 3), rng.randint(2, size - 3)
        height, width = rng.randint(1, 3), rng.randint(1, 3)
        if maps[0, row, col] == 0.0:
            maps[0, row : min(row + height, size - 1), col : min(col + width, size - 1)] = 1.0

    for _ in range(_rand_count(rng, 8, 15, difficulty, obstacle_scale)):
        row, col = rng.randint(3, size - 4), rng.randint(3, size - 4)
        length = rng.randint(3, 8)
        if rng.rand() > 0.5:
            maps[n_alt - 1, row, col : min(col + length, size - 1)] = 1.0
        else:
            maps[n_alt - 1, row : min(row + length, size - 1), col] = 1.0

    for _ in range(_rand_count(rng, 4, 8, difficulty, obstacle_scale)):
        row, col = rng.randint(4, size - 5), rng.randint(4, size - 5)
        length = rng.randint(5, 12)
        alt = rng.choice([2, 3]) if n_alt > 3 else 2
        if alt < n_alt:
            if rng.rand() > 0.5:
                maps[alt, row, col : min(col + length, size - 1)] = 1.0
            else:
                maps[alt, row : min(row + length, size - 1), col] = 1.0

    for _ in range(_rand_count(rng, 6, 12, difficulty, obstacle_scale)):
        row, col = rng.randint(2, size - 3), rng.randint(2, size - 3)
        if maps[1, row, col] == 0.0:
            for alt in (1, 2):
                if alt < n_alt:
                    maps[alt, row : min(row + 2, size - 1), col : min(col + 2, size - 1)] = 1.0

    for col in range(8, size - 8, 12):
        if col + 2 < size - 2:
            row_start = rng.randint(3, size // 4)
            row_len = rng.randint(4, size // 5)
            maps[1, row_start : row_start + row_len, col : col + 2] = 1.0

    return maps


class AltitudeActiveSLAMEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        render_mode: str | None = None,
        map_seed: int | None = None,
        randomize_start: bool = True,
        seed_suite: tuple[int, ...] | list[int] | None = None,
        difficulty: str = "hard",
        sensor_bleed: bool = True,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.map_seed = map_seed
        self.randomize_start = randomize_start
        self.seed_suite = tuple(seed_suite) if seed_suite else None
        self.difficulty = difficulty
        self.sensor_bleed = sensor_bleed

        self.observation_space = spaces.Dict(
            {
                "map_tensor": spaces.Box(0.0, 1.0, (MAP_CHANNELS, GRID_SIZE, GRID_SIZE), np.float32),
                "scalars": spaces.Box(0.0, 1.0, (N_SCALARS,), np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)

        self.true_maps: np.ndarray | None = None
        self.known_maps: np.ndarray | None = None
        self.visited_maps: np.ndarray | None = None
        self.traj_maps: np.ndarray | None = None
        self.alt_free_counts = np.ones(N_ALTITUDES, dtype=np.int32)
        self.total_free = 1
        self.pos = np.zeros(2, dtype=np.float32)
        self.alt_idx = 0
        self.prev_pos = np.zeros(2, dtype=np.float32)
        self.prev_alt = 0
        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.step_count = 0
        self.episode_count = 0
        self.current_map_seed: int | None = None

        self.cov_trace = 0.1
        self.prev_cov = 0.1
        self.total_loops = 0
        self.alt_changes = 0
        self.prev_frontier_dist: float | None = None
        self.coverage_bonus_given = False
        self.steps_at_current_alt = 0
        self.coverage_at_alt_arrival = 0.0
        self.prev_coverage = 0.0
        self.landmarks: list[tuple[np.ndarray, int, int, float]] = []
        self.path_length = 0.0
        self.alt_visit_bonus_given = [False] * N_ALTITUDES
        self.alt_breadth_bonus_given = [False] * N_ALTITUDES
        self.last_landmark_path_length = 0.0

        self._frontier_cache_step = -1
        self._frontier_cache_alt = -1
        self._frontier_cache: list[np.ndarray] = []

    def set_curriculum_phase(self, difficulty: str) -> None:
        if difficulty not in CURRICULUM_PHASES:
            raise ValueError(f"Unsupported curriculum phase: {difficulty}")
        self.difficulty = difficulty

    def set_seed_suite(self, seed_suite: tuple[int, ...] | list[int] | None) -> None:
        self.seed_suite = tuple(seed_suite) if seed_suite else None

    def _sample_map_seed(self, seed: int | None, options: dict | None) -> int:
        if options and "map_seed" in options:
            return int(options["map_seed"])
        if self.map_seed is not None:
            return int(self.map_seed)
        if self.seed_suite:
            idx = int(self.np_random.integers(0, len(self.seed_suite)))
            return int(self.seed_suite[idx])
        if seed is not None:
            return int(seed)
        return int(self.np_random.integers(0, 10000))

    def _invalidate_frontiers(self) -> None:
        self._frontier_cache_step = -1
        self._frontier_cache_alt = -1
        self._frontier_cache = []

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        difficulty = options.get("difficulty", self.difficulty)
        self.set_curriculum_phase(difficulty)

        self.current_map_seed = self._sample_map_seed(seed, options)
        self.true_maps = generate_3d_warehouse(GRID_SIZE, N_ALTITUDES, seed=self.current_map_seed, difficulty=self.difficulty)
        self.alt_free_counts = np.sum(self.true_maps == 0.0, axis=(1, 2)).astype(np.int32)
        self.total_free = int(np.sum(self.alt_free_counts))

        self.known_maps = np.full((N_ALTITUDES, GRID_SIZE, GRID_SIZE), 0.5, dtype=np.float32)
        self.visited_maps = np.zeros((N_ALTITUDES, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.traj_maps = np.zeros((N_ALTITUDES, GRID_SIZE, GRID_SIZE), dtype=np.float32)

        self.alt_idx = 0
        if self.randomize_start:
            free = np.argwhere(self.true_maps[0] == 0.0)
            self.pos = free[self.np_random.integers(0, len(free))].astype(np.float32)
        else:
            self.pos = np.array([5.0, 5.0], dtype=np.float32)
            if self.true_maps[0, int(self.pos[0]), int(self.pos[1])] == 1.0:
                free = np.argwhere(self.true_maps[0] == 0.0)
                self.pos = free[0].astype(np.float32)

        self.prev_pos = self.pos.copy()
        self.prev_alt = 0
        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.step_count = 0
        self.episode_count += 1
        self.cov_trace = 0.1
        self.prev_cov = 0.1
        self.total_loops = 0
        self.alt_changes = 0
        self.prev_frontier_dist = None
        self.coverage_bonus_given = False
        self.steps_at_current_alt = 0
        self.coverage_at_alt_arrival = 0.0
        self.prev_coverage = 0.0
        self.landmarks = []
        self.path_length = 0.0
        self.last_landmark_path_length = 0.0
        self.alt_visit_bonus_given = [False] * N_ALTITUDES
        self.alt_visit_bonus_given[0] = True
        self.alt_breadth_bonus_given = [False] * N_ALTITUDES
        self._invalidate_frontiers()

        self._sense()
        self.prev_coverage = self._total_cov()
        return self._obs(), self._info()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.step_count += 1
        self._invalidate_frontiers()

        dx, dy, dz = float(action[0]), float(action[1]), float(action[2])
        alt_changed = False
        new_alt = self.alt_idx
        if dz > ALT_UP_THRESH and self.alt_idx < N_ALTITUDES - 1:
            new_alt = self.alt_idx + 1
        elif dz < ALT_DOWN_THRESH and self.alt_idx > 0:
            new_alt = self.alt_idx - 1

        reward_altitude = 0.0

        if new_alt != self.alt_idx:
            # Enforce minimum dwell time: cannot switch altitudes more often than MIN_ALT_DWELL steps.
            if self.steps_at_current_alt < MIN_ALT_DWELL:
                new_alt = self.alt_idx
            else:
                row, col = int(self.pos[0]), int(self.pos[1])
                if self.true_maps[new_alt, row, col] == 0.0:
                    alt_changed = True
                    self.alt_changes += 1
                    reward_altitude += W_ALT_SWITCH_COST
                else:
                    new_alt = self.alt_idx

        self.prev_alt = self.alt_idx
        self.alt_idx = new_alt

        proposed_pos = np.clip(self.pos + np.array([dx, dy], dtype=np.float32) * STEP_SIZE, 1.0, GRID_SIZE - 2.0)
        collision = self._collides(self.pos, proposed_pos, self.alt_idx)
        if collision:
            reward_collision = W_COLLISION
            proposed_pos = self.pos.copy()
        else:
            reward_collision = 0.0
            self.pos = proposed_pos

        dist = float(np.linalg.norm(self.pos - self.prev_pos))
        self.path_length += dist
        uncapped_cov = self.cov_trace + COV_MOTION_RATE * dist + (COV_ALT_CHANGE_RATE if alt_changed else 0.0)

        loop_closure = self._check_loop_closure()
        if loop_closure:
            uncapped_cov *= COV_LOOP_DECAY
            self.total_loops += 1

        self.cov_delta = uncapped_cov - self.cov_trace
        self.prev_cov = self.cov_trace
        self.cov_trace = min(uncapped_cov, COV_MAX)

        if (
            self.step_count % LC_LANDMARK_SPACING == 0
            and (self.path_length - self.last_landmark_path_length) >= LC_MIN_TRAVEL
        ):
            self.landmarks.append((self.pos.copy(), self.alt_idx, self.step_count, self.path_length))
            self.last_landmark_path_length = self.path_length

        known_before = int(np.sum(self.known_maps != 0.5))
        self._sense()
        known_after = int(np.sum(self.known_maps != 0.5))
        new_cells = max(0, known_after - known_before)

        row, col = int(self.pos[0]), int(self.pos[1])
        radius = VISIT_RADIUS
        row_lo, row_hi = max(0, row - radius), min(GRID_SIZE, row + radius + 1)
        col_lo, col_hi = max(0, col - radius), min(GRID_SIZE, col + radius + 1)
        self.visited_maps[self.alt_idx, row_lo:row_hi, col_lo:col_hi] = 1.0
        self.traj_maps *= 0.95
        self.traj_maps[self.alt_idx, row, col] = 1.0

        alt_changed_now = alt_changed
        steps_at_prev = self.steps_at_current_alt
        coverage_at_prev_arrival = self.coverage_at_alt_arrival

        if alt_changed_now:
            self.steps_at_current_alt = 0
            self.coverage_at_alt_arrival = self._alt_cov(self.alt_idx)
        else:
            self.steps_at_current_alt += 1

        # --- Primary signal: coverage increase ---
        current_coverage = self._total_cov()
        coverage_delta = current_coverage - self.prev_coverage
        reward_progress = W_PROGRESS * max(0.0, coverage_delta) + W_NEW_CELL * float(new_cells)

        # --- Frontier direction guidance ---
        frontiers = self._frontiers(self.alt_idx)
        frontier_count = len(frontiers)
        reward_frontier = 0.0
        nearest_frontier_dist = 0.0
        if frontier_count > 0:
            distances = [float(np.linalg.norm(frontier - self.pos)) for frontier in frontiers]
            nearest_frontier_dist = min(distances)
            if self.prev_frontier_dist is not None and not alt_changed_now:
                delta = self.prev_frontier_dist - nearest_frontier_dist
                if abs(delta) <= 3.0:
                    reward_frontier = W_FRONTIER_DIR * delta
            self.prev_frontier_dist = nearest_frontier_dist
        else:
            self.prev_frontier_dist = None

        # --- Small per-step cost and revisit penalty ---
        visit_ratio = float(np.mean(self.visited_maps[self.alt_idx, row_lo:row_hi, col_lo:col_hi]))
        reward_step = W_STEP + W_REVISIT * visit_ratio

        # --- Altitude switching: encourage rotation ---
        if alt_changed_now:
            new_alt_cov = self._alt_cov(self.alt_idx)
            prev_alt_cov = self._alt_cov(self.prev_alt)

            # Big bonus the *first* time we enter an altitude (encourage breadth).
            if not self.alt_visit_bonus_given[self.alt_idx]:
                reward_altitude += W_ALT_NEW_VISIT
                self.alt_visit_bonus_given[self.alt_idx] = True

            # Mild discouragement from thrashing back and forth.
            if new_alt_cov > prev_alt_cov + 0.05:
                reward_altitude += W_ALT_SWITCH_BAD

        if self.steps_at_current_alt >= STAGNATION_STEPS:
            cov_gain = self._alt_cov(self.alt_idx) - self.coverage_at_alt_arrival
            if cov_gain < 0.02:
                reward_altitude += W_STAGNATION
            self.steps_at_current_alt = 0
            self.coverage_at_alt_arrival = self._alt_cov(self.alt_idx)

        all_alt_covs = [self._alt_cov(alt) for alt in range(N_ALTITUDES)]
        least_explored = int(np.argmin(all_alt_covs))
        if self.alt_idx == least_explored:
            reward_altitude += W_LEAST_EXPLORED_BONUS

        # Per-altitude breadth milestones (40% per altitude = full map explored uniformly).
        reward_breadth = 0.0
        for a, cov_a in enumerate(all_alt_covs):
            if not self.alt_breadth_bonus_given[a] and cov_a >= 0.40:
                reward_breadth += W_BREADTH_BONUS
                self.alt_breadth_bonus_given[a] = True

        # --- Milestone bonus ---
        reward_bonus = 0.0
        if current_coverage >= W_COVERAGE_BONUS_THRESHOLD and not self.coverage_bonus_given:
            reward_bonus = W_COVERAGE_BONUS
            self.coverage_bonus_given = True

        # --- Covariance penalty tied to absolute level (agent must keep it low) ---
        cov_fraction = min(self.cov_trace / COV_MAX, 1.0)
        reward_covariance = W_COV_PENALTY * cov_fraction
        reward_lc = W_LOOP_CLOSURE if loop_closure else 0.0
        reward_cov_delta = -W_COV_DELTA * self.cov_delta if W_COV_DELTA != 0.0 else 0.0

        # --- Sim-to-real roadmap terms (default-OFF: both weights are 0.0 unless overridden) ---
        reward_smooth = (
            -W_SMOOTH * float(np.sum((action - self.prev_action) ** 2)) if W_SMOOTH != 0.0 else 0.0
        )
        reward_alt_trans = W_ALT_TRANSITION if (W_ALT_TRANSITION != 0.0 and alt_changed) else 0.0

        reward = (
            reward_progress
            + reward_frontier
            + reward_collision
            + reward_step
            + reward_altitude
            + reward_bonus
            + reward_breadth
            + reward_covariance
            + reward_lc
            + reward_cov_delta
            + reward_smooth
            + reward_alt_trans
        )

        self.prev_pos = self.pos.copy()
        self.prev_action = action.copy()
        self.prev_coverage = current_coverage

        terminated = current_coverage >= 0.95
        truncated = self.step_count >= MAX_STEPS

        if terminated or truncated:
            # Weighted terminal bonus: coverage * (fraction of altitudes explored >= 40%) - covariance penalty.
            breadth_fraction = sum(self.alt_breadth_bonus_given) / max(N_ALTITUDES, 1)
            reward += W_TERMINAL_COVERAGE * current_coverage * (0.5 + 0.5 * breadth_fraction)
            reward += W_TERMINAL_COV_PENALTY * cov_fraction

        info = self._info()
        info.update(
            {
                "new_cells": new_cells,
                "loop_closure": loop_closure,
                "collision": collision,
                "alt_changed": alt_changed,
                "difficulty": self.difficulty,
                "map_seed": self.current_map_seed,
            }
        )
        return self._obs(), float(reward), terminated, truncated, info

    def _sense_altitude(self, alt: int, max_range: int) -> None:
        row, col = int(self.pos[0]), int(self.pos[1])
        for ray_idx in range(36):
            angle = 2.0 * math.pi * ray_idx / 36.0
            d_row, d_col = math.cos(angle), math.sin(angle)
            for distance in range(1, max_range + 1):
                rr = int(row + d_row * distance)
                cc = int(col + d_col * distance)
                if not (0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE):
                    break
                if self.true_maps[alt, rr, cc] == 1.0:
                    self.known_maps[alt, rr, cc] = 1.0
                    break
                self.known_maps[alt, rr, cc] = 0.0

    def _sense(self) -> None:
        self._sense_altitude(self.alt_idx, SENSOR_RANGE)
        if not self.sensor_bleed:
            return
        for alt in (self.alt_idx - 1, self.alt_idx + 1):
            if 0 <= alt < N_ALTITUDES:
                self._sense_altitude(alt, ADJACENT_SENSOR_RANGE)
        self._invalidate_frontiers()

    def _collides(self, start: np.ndarray, end: np.ndarray, alt: int) -> bool:
        n_steps = max(int(np.linalg.norm(end - start) * 2), 2)
        for idx in range(1, n_steps + 1):
            point = start + (idx / n_steps) * (end - start)
            row, col = int(point[0]), int(point[1])
            if row < 0 or row >= GRID_SIZE or col < 0 or col >= GRID_SIZE:
                return True
            if self.true_maps[alt, row, col] == 1.0:
                return True
        return False

    def _check_loop_closure(self) -> bool:
        if self.total_loops >= MAX_LC_PER_EPISODE or len(self.landmarks) < 3:
            return False
        for idx, (landmark_pos, landmark_alt, landmark_step, landmark_path) in enumerate(self.landmarks):
            if landmark_alt != self.alt_idx:
                continue
            if (self.step_count - landmark_step) < LC_MIN_GAP:
                continue
            # Require the agent to have actually travelled since visiting this landmark,
            # otherwise a stationary agent would farm free loop closures.
            if (self.path_length - landmark_path) < LC_MIN_TRAVEL * 2:
                continue
            if np.linalg.norm(self.pos - landmark_pos) < LC_DIST:
                self.landmarks.pop(idx)
                return True
        return False

    def _frontiers(self, alt: int) -> list[np.ndarray]:
        if self._frontier_cache_step == self.step_count and self._frontier_cache_alt == alt:
            return self._frontier_cache
        known_map = self.known_maps[alt]
        unknown_mask = (known_map > 0.4) & (known_map < 0.6)
        if FRONTIER_DEF == "unknown_adj_known":
            # Transfer-aligned definition: unknown cell adjacent to a known cell. On the real
            # RTAB-Map Z-sliced grid almost every observed cell reads occupied, so the classical
            # 'free adjacent to unknown' definition yields zero frontiers (see active_slam_env.py).
            seed_mask = unknown_mask
            nbr_mask = ~unknown_mask
        else:
            # Default classical definition: free cell adjacent to unknown (clean grid).
            seed_mask = known_map < 0.25
            nbr_mask = unknown_mask
        frontier_mask = np.zeros_like(seed_mask)
        frontier_mask[1:] |= seed_mask[1:] & nbr_mask[:-1]
        frontier_mask[:-1] |= seed_mask[:-1] & nbr_mask[1:]
        frontier_mask[:, 1:] |= seed_mask[:, 1:] & nbr_mask[:, :-1]
        frontier_mask[:, :-1] |= seed_mask[:, :-1] & nbr_mask[:, 1:]
        coords = np.argwhere(frontier_mask)
        if len(coords) > 20:
            coords = coords[np.linspace(0, len(coords) - 1, 20, dtype=int)]
        self._frontier_cache_step = self.step_count
        self._frontier_cache_alt = alt
        self._frontier_cache = [coord.astype(np.float32) for coord in coords]
        return self._frontier_cache

    def _alt_cov(self, alt: int) -> float:
        return float(np.sum(self.known_maps[alt] == 0.0)) / max(int(self.alt_free_counts[alt]), 1)

    def _total_cov(self) -> float:
        return float(np.sum(self.known_maps == 0.0)) / max(self.total_free, 1)

    def _obs(self) -> dict[str, np.ndarray]:
        least_explored_alt = int(np.argmin([self._alt_cov(alt) for alt in range(N_ALTITUDES)]))
        map_tensor = np.stack(
            [
                self.known_maps[self.alt_idx],
                self.visited_maps[self.alt_idx],
                self.traj_maps[self.alt_idx],
                self.known_maps[least_explored_alt],
                np.mean(self.visited_maps, axis=0),
            ]
        ).astype(np.float32)

        total_cov = self._total_cov()
        current_alt_cov = self._alt_cov(self.alt_idx)
        frontiers = self._frontiers(self.alt_idx)
        frontier_count = len(frontiers)
        nearest_frontier_dist = 0.0
        relative_frontier_dir = 0.0
        if frontier_count > 0:
            distances = [float(np.linalg.norm(frontier - self.pos)) for frontier in frontiers]
            nearest_idx = int(np.argmin(distances))
            nearest_frontier_dist = distances[nearest_idx]
            nearest_frontier = frontiers[nearest_idx]
            frontier_dir = math.atan2(nearest_frontier[0] - self.pos[0], nearest_frontier[1] - self.pos[1])
        else:
            frontier_dir = 0.0

        if np.linalg.norm(self.pos - self.prev_pos) > 0.01:
            yaw = math.atan2(self.pos[0] - self.prev_pos[0], self.pos[1] - self.prev_pos[1])
        else:
            yaw = 0.0
        relative_frontier_dir = (frontier_dir - yaw + math.pi) % (2 * math.pi) - math.pi

        other_covs = [self._alt_cov(alt) for alt in range(N_ALTITUDES) if alt != self.alt_idx]
        other_potential = max(1.0 - cov for cov in other_covs) if other_covs else 0.0
        min_other_cov = min(other_covs) if other_covs else 1.0

        scalars = np.array(
            [
                np.clip(self.cov_trace / COV_MAX, 0.0, 1.0),
                min(frontier_count / 50.0, 1.0),
                total_cov,
                min(nearest_frontier_dist / (GRID_SIZE / 2.0), 1.0),
                (relative_frontier_dir + math.pi) / (2.0 * math.pi),
                self.alt_idx / max(N_ALTITUDES - 1, 1),
                current_alt_cov,
                other_potential,
                min_other_cov,
                self.step_count / MAX_STEPS,
            ],
            dtype=np.float32,
        )
        return {"map_tensor": map_tensor, "scalars": scalars}

    def _info(self) -> dict[str, float | int | str]:
        total_cov = self._total_cov()
        alt_coverages = [self._alt_cov(alt) for alt in range(N_ALTITUDES)]
        frontiers = self._frontiers(self.alt_idx)
        info = {
            "step": self.step_count,
            "coverage": total_cov,
            "known_cells": int(np.sum(self.known_maps != 0.5)),
            "total_cells": self.total_free,
            "cov_trace": self.cov_trace,
            "pos_x": float(self.pos[0]),
            "pos_y": float(self.pos[1]),
            "altitude": ALTITUDE_LEVELS[self.alt_idx],
            "alt_idx": self.alt_idx,
            "frontier_count": len(frontiers),
            "loop_closures": self.total_loops,
            "altitude_changes": self.alt_changes,
            "map_seed": self.current_map_seed if self.current_map_seed is not None else -1,
            "difficulty": self.difficulty,
            "episode_index": self.episode_count,
        }
        for alt, coverage in enumerate(alt_coverages):
            info[f"cov_alt{alt}"] = coverage
        return info
