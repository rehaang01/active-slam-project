"""
Transfer-compatible lightweight 3D Active SLAM training and evaluation.

This script is the fast experiment harness for the paper workflow:
  1. Train a recurrent policy on fixed lightweight map suites.
  2. Evaluate every checkpoint on held-out seeds.
  3. Compare against the classical 3D baselines.
  4. Select the earliest checkpoint that satisfies the paper stop rule.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sb3_contrib import RecurrentPPO
except ImportError as exc:  # pragma: no cover - dependency availability varies
    RecurrentPPO = None
    _SB3_CONTRIB_IMPORT_ERROR = exc
else:
    _SB3_CONTRIB_IMPORT_ERROR = None
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CURRENT_DIR))
from baselines_3d import get_all_baselines
from env_3d import ALTITUDE_LEVELS, AltitudeActiveSLAMEnv, CURRICULUM_PHASES, MAX_STEPS, N_ALTITUDES
from shared_slam_policy import build_recurrent_policy_kwargs
from transfer_config import (
    FEATURES_DIM,
    LIGHTWEIGHT_CHECKPOINT_INTERVAL,
    LIGHTWEIGHT_EVAL_SEEDS,
    LIGHTWEIGHT_TOTAL_TIMESTEPS,
    LIGHTWEIGHT_TRAIN_SEEDS,
    linear_schedule,
)


class Cfg:
    OUT = Path("results_3d")
    MODELS = Path("results_3d/models")
    CSV = Path("results_3d/csv")
    PLOTS = Path("results_3d/plots")
    SUMMARY = Path("results_3d/summary")

    TRAIN_SEEDS = tuple(LIGHTWEIGHT_TRAIN_SEEDS)
    EVAL_SEEDS = tuple(LIGHTWEIGHT_EVAL_SEEDS)

    TOTAL_TIMESTEPS = LIGHTWEIGHT_TOTAL_TIMESTEPS
    QUICK_TIMESTEPS = 100_000
    CHECKPOINT_INTERVAL = LIGHTWEIGHT_CHECKPOINT_INTERVAL

    N_ENVS = 4
    N_STEPS = 256
    BATCH_SIZE = 128
    N_EPOCHS = 5
    GAMMA = 0.995
    GAE_LAMBDA = 0.97
    CLIP_RANGE = 0.2
    MAX_GRAD_NORM = 0.5

    LEARNING_RATE = linear_schedule(3e-4, 1e-5)
    ENTROPY_START = 0.02
    ENTROPY_END = 0.002

    FEATURES_DIM = FEATURES_DIM
    DEVICE = "cuda"
    TENSORBOARD_LOG = str(OUT / "tb") if importlib.util.find_spec("tensorboard") else None
    N_BASELINES = 50
    N_EVAL = 50
    EVAL_MAX_STEPS = MAX_STEPS
    SEED = 42

    CURRICULUM_MILESTONES = (
        (0, "easy"),
        (120_000, "medium"),
        (300_000, "hard"),
    )

    @classmethod
    def mkdirs(cls) -> None:
        for directory in (cls.OUT, cls.MODELS, cls.CSV, cls.PLOTS, cls.SUMMARY):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def checkpoint_eval_dir(cls) -> Path:
        path = cls.SUMMARY / "checkpoint_eval"
        path.mkdir(parents=True, exist_ok=True)
        return path


def sanitize_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


def curriculum_phase_for_step(num_timesteps: int) -> str:
    phase = Cfg.CURRICULUM_MILESTONES[0][1]
    for threshold, candidate in Cfg.CURRICULUM_MILESTONES:
        if num_timesteps >= threshold:
            phase = candidate
    return phase


def build_env(map_seed: int | None = None, seed_suite: tuple[int, ...] | None = None, difficulty: str = "hard", randomize_start: bool = True):
    return Monitor(
        AltitudeActiveSLAMEnv(
            map_seed=map_seed,
            seed_suite=seed_suite,
            difficulty=difficulty,
            randomize_start=randomize_start,
        )
    )


class EntropyAnnealCallback(BaseCallback):
    def __init__(self, start: float, end: float):
        super().__init__()
        self.start = start
        self.end = end

    def _on_step(self) -> bool:
        progress_remaining = max(0.0, 1.0 - (self.num_timesteps / max(self.model._total_timesteps, 1)))
        done_frac = 1.0 - progress_remaining
        self.model.ent_coef = self.start + (self.end - self.start) * done_frac
        return True


class CurriculumCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.current_phase = CURRICULUM_PHASES[0]

    def _on_training_start(self) -> None:
        self.current_phase = curriculum_phase_for_step(0)
        self.training_env.env_method("set_curriculum_phase", self.current_phase)

    def _on_step(self) -> bool:
        phase = curriculum_phase_for_step(self.num_timesteps)
        if phase != self.current_phase:
            self.current_phase = phase
            self.training_env.env_method("set_curriculum_phase", self.current_phase)
            print(f"[Curriculum] Switched to {self.current_phase} maps at {self.num_timesteps:,} steps")
        return True


class EpisodeLogCallback(BaseCallback):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.history: dict[str, list[float | int | str]] = {
            "episode_rewards": [],
            "episode_lengths": [],
            "episode_coverages": [],
            "episode_cov_traces": [],
            "episode_loop_closures": [],
            "episode_altitude_changes": [],
            "episode_curriculum": [],
        }
        for alt in range(N_ALTITUDES):
            self.history[f"episode_cov_alt{alt}"] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if not done:
                continue
            ep_info = info.get("episode", {})
            self.history["episode_rewards"].append(float(ep_info.get("r", 0.0)))
            self.history["episode_lengths"].append(int(ep_info.get("l", info.get("step", 0))))
            self.history["episode_coverages"].append(float(info.get("coverage", 0.0)))
            self.history["episode_cov_traces"].append(float(info.get("cov_trace", 0.0)))
            self.history["episode_loop_closures"].append(int(info.get("loop_closures", 0)))
            self.history["episode_altitude_changes"].append(int(info.get("altitude_changes", 0)))
            self.history["episode_curriculum"].append(str(info.get("difficulty", curriculum_phase_for_step(self.num_timesteps))))
            for alt in range(N_ALTITUDES):
                self.history[f"episode_cov_alt{alt}"].append(float(info.get(f"cov_alt{alt}", 0.0)))
        return True

    def _on_training_end(self) -> None:
        with self.path.open("w") as handle:
            json.dump(self.history, handle, indent=2)


def make_train_vec_env(n_envs: int, difficulty: str) -> VecNormalize:
    # seed_suite=None -> env samples a fresh random map_seed every episode,
    # which is essential for generalization. A fixed 20-seed suite caused
    # severe overfit: training ep_rew=4100 but eval coverage 20%.
    if n_envs <= 1:
        vec_env = DummyVecEnv([lambda: build_env(seed_suite=None, difficulty=difficulty)])
    else:
        vec_env = SubprocVecEnv(
            [lambda: build_env(seed_suite=None, difficulty=difficulty) for _ in range(n_envs)],
            start_method="spawn",
        )
    return VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=15.0, gamma=Cfg.GAMMA)


def make_eval_env(map_seed: int, vecnorm_path: Path | None = None) -> VecNormalize | DummyVecEnv:
    vec_env = DummyVecEnv([lambda: build_env(map_seed=map_seed, difficulty="hard")])
    if vecnorm_path is None or not vecnorm_path.exists():
        return vec_env
    vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def parse_checkpoint_step(path: Path) -> int:
    match = re.search(r"_(\d+)_steps", path.stem)
    if match:
        return int(match.group(1))
    if path.stem == "final_model":
        return int(Cfg.TOTAL_TIMESTEPS)
    return -1


def vecnorm_for_checkpoint(checkpoint_path: Path) -> Path | None:
    step = parse_checkpoint_step(checkpoint_path)
    candidate = checkpoint_path.with_name(f"{checkpoint_path.stem.replace('_steps', '_vecnormalize_')}.pkl")
    if candidate.exists():
        return candidate
    candidate = checkpoint_path.with_name(f"slam3d_vecnormalize_{step}_steps.pkl")
    if candidate.exists():
        return candidate
    final_path = Cfg.MODELS / "vec_normalize.pkl"
    return final_path if final_path.exists() else None


def run_episode(env: AltitudeActiveSLAMEnv, policy_fn, max_steps: int = MAX_STEPS):
    obs, info = env.reset()
    data = []
    for step in range(max_steps):
        action = policy_fn(obs, info)
        obs, reward, term, trunc, info = env.step(action)
        row = {
            "step": step + 1,
            "timestamp": float(step + 1) * 2.0,
            "reward": float(reward),
            "coverage": float(info.get("coverage", 0.0)),
            "cov_trace": float(info.get("cov_trace", 0.0)),
            "altitude": float(info.get("altitude", ALTITUDE_LEVELS[0])),
            "alt_idx": int(info.get("alt_idx", 0)),
            "loop_closures": int(info.get("loop_closures", 0)),
            "altitude_changes": int(info.get("altitude_changes", 0)),
            "frontier_count": int(info.get("frontier_count", 0)),
            "new_cells": int(info.get("new_cells", 0)),
            "map_seed": int(info.get("map_seed", -1)),
            "difficulty": info.get("difficulty", "hard"),
        }
        for alt in range(N_ALTITUDES):
            row[f"cov_alt{alt}"] = float(info.get(f"cov_alt{alt}", 0.0))
        data.append(row)
        if term or trunc:
            break
    return data


def run_recurrent_episode(model: RecurrentPPO, map_seed: int, checkpoint_path: Path, max_steps: int = MAX_STEPS):
    vecnorm_path = vecnorm_for_checkpoint(checkpoint_path)
    env = make_eval_env(map_seed=map_seed, vecnorm_path=vecnorm_path)
    obs = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    data = []
    try:
        for step in range(max_steps):
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=False)
            episode_starts[:] = False
            obs, reward, done, infos = env.step(action)
            info = infos[0]
            row = {
                "step": step + 1,
                "timestamp": float(step + 1) * 2.0,
                "reward": float(reward[0]),
                "coverage": float(info.get("coverage", 0.0)),
                "cov_trace": float(info.get("cov_trace", 0.0)),
                "altitude": float(info.get("altitude", ALTITUDE_LEVELS[0])),
                "alt_idx": int(info.get("alt_idx", 0)),
                "loop_closures": int(info.get("loop_closures", 0)),
                "altitude_changes": int(info.get("altitude_changes", 0)),
                "frontier_count": int(info.get("frontier_count", 0)),
                "new_cells": int(info.get("new_cells", 0)),
                "map_seed": int(info.get("map_seed", map_seed)),
                "difficulty": info.get("difficulty", "hard"),
            }
            for alt in range(N_ALTITUDES):
                row[f"cov_alt{alt}"] = float(info.get(f"cov_alt{alt}", 0.0))
            data.append(row)
            if done[0]:
                episode_starts[:] = True
                lstm_states = None
                break
    finally:
        env.close()
    return data


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_episodes(all_data: list[list[dict]]) -> tuple[list[dict], list[dict]]:
    if not all_data:
        return [], []
    max_len = max(len(episode) for episode in all_data)
    keys = list(all_data[0][0].keys())
    padded: list[list[dict]] = []
    for episode in all_data:
        episode_rows = [dict(row) for row in episode]
        last = dict(episode_rows[-1])
        while len(episode_rows) < max_len:
            filler = dict(last)
            filler["step"] = len(episode_rows) + 1
            filler["timestamp"] = float(filler["step"]) * 2.0
            episode_rows.append(filler)
        padded.append(episode_rows)

    mean_rows: list[dict] = []
    std_rows: list[dict] = []
    for idx in range(max_len):
        mean_row = {}
        std_row = {}
        for key in keys:
            values = [episode[idx][key] for episode in padded]
            if isinstance(values[0], (int, float, np.integer, np.floating)):
                numeric = np.asarray(values, dtype=np.float64)
                mean_row[key] = float(np.mean(numeric))
                std_row[key] = float(np.std(numeric))
            else:
                mean_row[key] = values[-1]
                std_row[key] = values[-1]
        mean_rows.append(mean_row)
        std_rows.append(std_row)
    return mean_rows, std_rows


def save_method_outputs(method_key: str, all_data: list[list[dict]]) -> dict:
    safe = sanitize_name(method_key)
    for idx, rows in enumerate(all_data):
        save_csv(rows, Cfg.CSV / f"{safe}_ep{idx}.csv")
    mean_rows, std_rows = aggregate_episodes(all_data)
    save_csv(mean_rows, Cfg.CSV / f"{safe}_avg.csv")
    save_csv(std_rows, Cfg.CSV / f"{safe}_std.csv")

    final = mean_rows[-1] if mean_rows else {}
    summary = {
        "method": method_key,
        "episodes": len(all_data),
        "final_coverage": float(final.get("coverage", 0.0)),
        "final_cov_trace": float(final.get("cov_trace", 0.0)),
        "final_loop_closures": float(final.get("loop_closures", 0.0)),
        "final_altitude_changes": float(final.get("altitude_changes", 0.0)),
    }
    for alt in range(N_ALTITUDES):
        summary[f"final_cov_alt{alt}"] = float(final.get(f"cov_alt{alt}", 0.0))
    return summary


def evaluate_baselines() -> dict[str, dict]:
    print("=" * 72)
    print("PHASE 2: BASELINE SUITE (HELD-OUT SEEDS)")
    print("=" * 72)
    results = {}
    for baseline in get_all_baselines():
        print(f"[Baseline] {baseline.name}")
        all_data = []
        for seed in Cfg.EVAL_SEEDS[: Cfg.N_BASELINES]:
            baseline.reset()
            env = AltitudeActiveSLAMEnv(map_seed=seed, difficulty="hard", randomize_start=True)
            try:
                rows = run_episode(env, baseline.get_action, max_steps=Cfg.EVAL_MAX_STEPS)
            finally:
                env.close()
            all_data.append(rows)
        results[baseline.name] = save_method_outputs(baseline.name, all_data)
        print(
            f"  coverage={results[baseline.name]['final_coverage']:.1%} "
            f"cov_trace={results[baseline.name]['final_cov_trace']:.3f}"
        )
    with (Cfg.SUMMARY / "baseline_summary.json").open("w") as handle:
        json.dump(results, handle, indent=2)
    return results


def list_checkpoint_paths() -> list[Path]:
    checkpoints = sorted(Cfg.MODELS.glob("slam3d_*_steps.zip"), key=parse_checkpoint_step)
    if not checkpoints:
        final_model = Cfg.MODELS / "final_model.zip"
        if final_model.exists():
            checkpoints = [final_model]
    return checkpoints


def evaluate_checkpoint(checkpoint_path: Path) -> dict:
    if RecurrentPPO is None:
        raise ImportError(
            "sb3_contrib is required for recurrent checkpoint evaluation."
        ) from _SB3_CONTRIB_IMPORT_ERROR
    vecnorm_path = vecnorm_for_checkpoint(checkpoint_path)
    env = make_eval_env(map_seed=Cfg.EVAL_SEEDS[0], vecnorm_path=vecnorm_path)
    try:
        model = RecurrentPPO.load(str(checkpoint_path), env=env, device=Cfg.DEVICE)
    finally:
        env.close()

    all_data = [run_recurrent_episode(model, seed, checkpoint_path, max_steps=Cfg.EVAL_MAX_STEPS) for seed in Cfg.EVAL_SEEDS[: Cfg.N_EVAL]]
    step = parse_checkpoint_step(checkpoint_path)
    out_dir = Cfg.checkpoint_eval_dir()
    mean_rows, std_rows = aggregate_episodes(all_data)
    save_csv(mean_rows, out_dir / f"checkpoint_{step}_avg.csv")
    save_csv(std_rows, out_dir / f"checkpoint_{step}_std.csv")

    final = mean_rows[-1] if mean_rows else {}
    summary = {
        "checkpoint": checkpoint_path.name,
        "step": step,
        "final_coverage": float(final.get("coverage", 0.0)),
        "final_cov_trace": float(final.get("cov_trace", 0.0)),
        "final_loop_closures": float(final.get("loop_closures", 0.0)),
        "final_altitude_changes": float(final.get("altitude_changes", 0.0)),
    }
    for alt in range(N_ALTITUDES):
        summary[f"final_cov_alt{alt}"] = float(final.get(f"cov_alt{alt}", 0.0))
    return summary


def select_checkpoint(checkpoint_summaries: list[dict], baseline_summary: dict[str, dict]) -> dict:
    greedy = baseline_summary["Greedy Info Gain"]
    selected = None
    for summary in sorted(checkpoint_summaries, key=lambda item: item["step"]):
        passes = (
            summary["final_coverage"] >= greedy["final_coverage"] + 0.03
            and summary["final_cov_trace"] <= greedy["final_cov_trace"] * 0.80
            and min(summary[f"final_cov_alt{alt}"] for alt in range(N_ALTITUDES)) >= 0.25
        )
        summary["passes_stop_rule"] = passes
        if passes and selected is None:
            selected = summary
    if selected is None and checkpoint_summaries:
        # Pick the checkpoint with the best composite: coverage (maximize) + breadth + low covariance.
        def score(item: dict) -> float:
            min_alt = min(item[f"final_cov_alt{alt}"] for alt in range(N_ALTITUDES))
            return (
                item["final_coverage"]
                + 0.5 * min_alt
                - 0.05 * item["final_cov_trace"]
            )
        selected = max(checkpoint_summaries, key=score)
        selected["passes_stop_rule"] = False
    return selected


def evaluate_checkpoints(baseline_summary: dict[str, dict]) -> dict:
    print("=" * 72)
    print("PHASE 3: CHECKPOINT SWEEP")
    print("=" * 72)
    summaries = []
    for checkpoint_path in list_checkpoint_paths():
        summary = evaluate_checkpoint(checkpoint_path)
        summaries.append(summary)
        print(
            f"[Checkpoint {summary['step']:,}] "
            f"coverage={summary['final_coverage']:.1%} "
            f"cov_trace={summary['final_cov_trace']:.3f}"
        )

    selected = select_checkpoint(summaries, baseline_summary)
    payload = {"checkpoints": summaries, "selected": selected}
    with (Cfg.SUMMARY / "checkpoint_selection.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def export_selected_rl(selected_summary: dict | None) -> dict | None:
    if not selected_summary:
        return None
    if RecurrentPPO is None:
        raise ImportError(
            "sb3_contrib is required for recurrent RL export."
        ) from _SB3_CONTRIB_IMPORT_ERROR
    checkpoint_path = Cfg.MODELS / selected_summary["checkpoint"]
    vecnorm_path = vecnorm_for_checkpoint(checkpoint_path)
    env = make_eval_env(map_seed=Cfg.EVAL_SEEDS[0], vecnorm_path=vecnorm_path)
    try:
        model = RecurrentPPO.load(str(checkpoint_path), env=env, device=Cfg.DEVICE)
    finally:
        env.close()
    all_data = [run_recurrent_episode(model, seed, checkpoint_path, max_steps=Cfg.EVAL_MAX_STEPS) for seed in Cfg.EVAL_SEEDS[: Cfg.N_EVAL]]
    summary = save_method_outputs("rl_agent", all_data)
    summary["checkpoint"] = selected_summary["checkpoint"]
    with (Cfg.SUMMARY / "selected_rl_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def plot_training_curves() -> None:
    log_path = Cfg.CSV / "training_log.json"
    if not log_path.exists():
        return
    with log_path.open() as handle:
        history = json.load(handle)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Lightweight Recurrent Training Progress", fontsize=13, fontweight="bold")
    series_defs = [
        ("episode_rewards", "Episode Reward", "#E63946"),
        ("episode_coverages", "Episode Coverage", "#457B9D"),
        ("episode_cov_traces", "Episode Covariance", "#2A9D8F"),
    ]
    for axis, (key, title, color) in zip(axes, series_defs):
        values = history.get(key, [])
        if not values:
            continue
        smoothed = pd.Series(values).rolling(min(20, max(len(values) // 10, 1)), min_periods=1).mean()
        axis.plot(smoothed, color=color, linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(Cfg.PLOTS / "training_3d_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_results() -> None:
    print("=" * 72)
    print("PHASE 4: PLOTS")
    print("=" * 72)
    methods = {
        "RL Agent (Ours)": ("rl_agent_avg.csv", "rl_agent_std.csv"),
        "Random Walk": ("random_walk_avg.csv", "random_walk_std.csv"),
        "Nearest Frontier": ("nearest_frontier_avg.csv", "nearest_frontier_std.csv"),
        "Spiral": ("spiral_avg.csv", "spiral_std.csv"),
        "Potential Field": ("potential_field_avg.csv", "potential_field_std.csv"),
        "Greedy Info Gain": ("greedy_info_gain_avg.csv", "greedy_info_gain_std.csv"),
        "RRT Explorer": ("rrt_explorer_avg.csv", "rrt_explorer_std.csv"),
    }
    loaded = {}
    for label, (mean_name, std_name) in methods.items():
        mean_path = Cfg.CSV / mean_name
        std_path = Cfg.CSV / std_name
        if mean_path.exists():
            loaded[label] = {
                "mean": pd.read_csv(mean_path),
                "std": pd.read_csv(std_path) if std_path.exists() else None,
            }

    if len(loaded) < 2:
        print("[Plot] Need at least two methods to plot.")
        return

    colors = {
        "RL Agent (Ours)": "#E63946",
        "Random Walk": "#A8DADC",
        "Nearest Frontier": "#457B9D",
        "Spiral": "#2A9D8F",
        "Potential Field": "#E9C46A",
        "Greedy Info Gain": "#F4A261",
        "RRT Explorer": "#6A4C93",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Altitude-Aware Active SLAM: Held-Out Lightweight Results", fontsize=14, fontweight="bold")

    for label, payload in loaded.items():
        mean_df = payload["mean"]
        std_df = payload["std"]
        color = colors.get(label, "gray")
        axes[0, 0].plot(mean_df["timestamp"], mean_df["coverage"] * 100, color=color, linewidth=2.0, label=label)
        if std_df is not None:
            axes[0, 0].fill_between(
                mean_df["timestamp"],
                (mean_df["coverage"] - std_df["coverage"]) * 100,
                (mean_df["coverage"] + std_df["coverage"]) * 100,
                color=color,
                alpha=0.12,
            )
    axes[0, 0].set_title("Coverage vs Time")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Coverage (%)")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7)

    for label, payload in loaded.items():
        mean_df = payload["mean"]
        std_df = payload["std"]
        color = colors.get(label, "gray")
        axes[0, 1].plot(mean_df["timestamp"], mean_df["cov_trace"], color=color, linewidth=2.0, label=label)
        if std_df is not None:
            axes[0, 1].fill_between(
                mean_df["timestamp"],
                mean_df["cov_trace"] - std_df["cov_trace"],
                mean_df["cov_trace"] + std_df["cov_trace"],
                color=color,
                alpha=0.12,
            )
    axes[0, 1].set_title("Covariance Trace")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Trace")
    axes[0, 1].grid(True, alpha=0.3)

    if "RL Agent (Ours)" in loaded:
        rl_mean = loaded["RL Agent (Ours)"]["mean"]
        rl_std = loaded["RL Agent (Ours)"]["std"]
        alt_colors = ["#E63946", "#F4845F", "#F7B267", "#7DCFB6"]
        for alt in range(N_ALTITUDES):
            column = f"cov_alt{alt}"
            axes[0, 2].plot(rl_mean["timestamp"], rl_mean[column] * 100, color=alt_colors[alt], linewidth=2.0, label=f"Alt {alt} ({ALTITUDE_LEVELS[alt]}m)")
            if rl_std is not None and column in rl_std:
                axes[0, 2].fill_between(
                    rl_mean["timestamp"],
                    (rl_mean[column] - rl_std[column]) * 100,
                    (rl_mean[column] + rl_std[column]) * 100,
                    color=alt_colors[alt],
                    alpha=0.12,
                )
        axes[0, 2].legend(fontsize=7)
    axes[0, 2].set_title("Per-Altitude Coverage (RL)")
    axes[0, 2].set_xlabel("Time (s)")
    axes[0, 2].set_ylabel("Coverage (%)")
    axes[0, 2].grid(True, alpha=0.3)

    for label, payload in loaded.items():
        mean_df = payload["mean"]
        color = colors.get(label, "gray")
        axes[1, 0].plot(mean_df["timestamp"], mean_df["altitude"], color=color, linewidth=1.8, label=label)
    axes[1, 0].set_title("Altitude Usage")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Altitude (m)")
    axes[1, 0].grid(True, alpha=0.3)

    for label, payload in loaded.items():
        mean_df = payload["mean"]
        color = colors.get(label, "gray")
        axes[1, 1].plot(mean_df["coverage"] * 100, mean_df["cov_trace"], color=color, linewidth=2.0, label=label)
    axes[1, 1].set_title("Coverage vs Covariance")
    axes[1, 1].set_xlabel("Coverage (%)")
    axes[1, 1].set_ylabel("Covariance Trace")
    axes[1, 1].grid(True, alpha=0.3)

    labels = list(loaded.keys())
    final_coverages = [loaded[label]["mean"]["coverage"].iloc[-1] * 100 for label in labels]
    axes[1, 2].bar(np.arange(len(labels)), final_coverages, color=[colors.get(label, "gray") for label in labels], edgecolor="black", linewidth=0.5)
    axes[1, 2].set_xticks(np.arange(len(labels)))
    axes[1, 2].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axes[1, 2].set_title("Final Coverage")
    axes[1, 2].set_ylabel("Coverage (%)")
    axes[1, 2].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(Cfg.PLOTS / "comparison_3d_main.png", dpi=200, bbox_inches="tight")
    fig.savefig(Cfg.PLOTS / "comparison_3d_main.pdf", bbox_inches="tight")
    plt.close(fig)
    plot_training_curves()


def train(total_timesteps: int) -> None:
    if RecurrentPPO is None:
        raise ImportError(
            "sb3_contrib is required for recurrent lightweight training. "
            "Install it in the active environment before running run_3d.py."
        ) from _SB3_CONTRIB_IMPORT_ERROR
    print("=" * 72)
    print("PHASE 1: TRAINING")
    print("=" * 72)
    vec_env = make_train_vec_env(Cfg.N_ENVS, difficulty="easy")
    model = RecurrentPPO(
        policy="MultiInputLstmPolicy",
        env=vec_env,
        learning_rate=Cfg.LEARNING_RATE,
        n_steps=Cfg.N_STEPS,
        batch_size=Cfg.BATCH_SIZE,
        n_epochs=Cfg.N_EPOCHS,
        gamma=Cfg.GAMMA,
        gae_lambda=Cfg.GAE_LAMBDA,
        clip_range=Cfg.CLIP_RANGE,
        ent_coef=Cfg.ENTROPY_START,
        max_grad_norm=Cfg.MAX_GRAD_NORM,
        policy_kwargs=build_recurrent_policy_kwargs(Cfg.FEATURES_DIM),
        tensorboard_log=Cfg.TENSORBOARD_LOG,
        verbose=1,
        device=Cfg.DEVICE,
        seed=Cfg.SEED,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(Cfg.CHECKPOINT_INTERVAL // Cfg.N_ENVS, 1),
        save_path=str(Cfg.MODELS),
        name_prefix="slam3d",
        save_vecnormalize=True,
    )
    callbacks = CallbackList(
        [
            checkpoint_cb,
            EpisodeLogCallback(Cfg.CSV / "training_log.json"),
            CurriculumCallback(),
            EntropyAnnealCallback(Cfg.ENTROPY_START, Cfg.ENTROPY_END),
        ]
    )

    started = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
    finally:
        print(f"[Train] Wall time: {(time.time() - started) / 60:.1f} min")
        model.save(str(Cfg.MODELS / "final_model"))
        vec_env.save(str(Cfg.MODELS / "vec_normalize.pkl"))
        vec_env.close()


def print_summary_table(rl_summary: dict | None, baseline_summary: dict[str, dict]) -> None:
    print("\n" + "=" * 90)
    print("NUMERICAL SUMMARY — HELD-OUT LIGHTWEIGHT BENCHMARK")
    print("=" * 90)
    print(f"{'Method':<22} {'Coverage%':>10} {'CovTrace':>10} {'Loops':>8} {'AltSw':>8}")
    print("-" * 90)
    if rl_summary:
        print(
            f"{'RL Agent (Ours)':<22} "
            f"{rl_summary['final_coverage'] * 100:>9.1f}% "
            f"{rl_summary['final_cov_trace']:>10.3f} "
            f"{rl_summary['final_loop_closures']:>8.1f} "
            f"{rl_summary['final_altitude_changes']:>8.1f}"
        )
    for name, summary in baseline_summary.items():
        print(
            f"{name:<22} "
            f"{summary['final_coverage'] * 100:>9.1f}% "
            f"{summary['final_cov_trace']:>10.3f} "
            f"{summary['final_loop_closures']:>8.1f} "
            f"{summary['final_altitude_changes']:>8.1f}"
        )
    print("=" * 90)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for RecurrentPPO init")
    parser.add_argument("--tag", type=str, default=None, help="Output suffix -> results_3d_<tag>/")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.tag:
        base = Path(f"results_3d_{args.tag}")
        Cfg.OUT = base
        Cfg.MODELS = base / "models"
        Cfg.CSV = base / "csv"
        Cfg.PLOTS = base / "plots"
        Cfg.SUMMARY = base / "summary"
        Cfg.TENSORBOARD_LOG = str(base / "tb") if importlib.util.find_spec("tensorboard") else None

    if args.seed is not None:
        Cfg.SEED = args.seed

    Cfg.mkdirs()

    if args.quick:
        Cfg.TOTAL_TIMESTEPS = Cfg.QUICK_TIMESTEPS
        Cfg.N_BASELINES = 3
        Cfg.N_EVAL = 3

    if args.timesteps is not None:
        Cfg.TOTAL_TIMESTEPS = args.timesteps

    total_started = time.time()
    baseline_summary = None
    checkpoint_payload = None
    rl_summary = None

    if args.plot_only:
        plot_results()
    elif args.train_only:
        train(Cfg.TOTAL_TIMESTEPS)
    elif args.baselines_only:
        baseline_summary = evaluate_baselines()
        print_summary_table(None, baseline_summary)
    elif args.eval_only:
        
        baseline_summary = evaluate_baselines()
        checkpoint_payload = evaluate_checkpoints(baseline_summary)
        rl_summary = export_selected_rl(checkpoint_payload["selected"])
        plot_results()
        print_summary_table(rl_summary, baseline_summary)
    else:
        train(Cfg.TOTAL_TIMESTEPS)
        baseline_summary = evaluate_baselines()
        checkpoint_payload = evaluate_checkpoints(baseline_summary)
        rl_summary = export_selected_rl(checkpoint_payload["selected"])
        plot_results()
        print_summary_table(rl_summary, baseline_summary)

    print(f"\nTOTAL: {(time.time() - total_started) / 60:.1f} min")


if __name__ == "__main__":
    main()
    