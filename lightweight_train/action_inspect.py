"""Tier 1: policy-action inspector for sim-to-sim transfer.

Question: does the lightweight-trained policy emit motion patterns that would
corrupt RTAB-Map in Gazebo? Specifically we care about:
  (A) (dx, dy) sign-flip rate — each flip = deceleration + reversal
  (B) dz altitude-change cadence — Gazebo has no MIN_ALT_DWELL
  (C) action jitter (RMS of consecutive diffs) — proxy for stop-start motion
  (D) action magnitude distribution — saturated actions stress the path planner

Runs each MS600-selected checkpoint (180k) through the lightweight env for
N episodes and prints a red/yellow/green verdict per criterion.

Thresholds are calibrated to the Gazebo env's defenses:
  - MAX_YAW_CHANGE = 11°/step absorbs most (dx,dy) oscillation
  - STEP_DURATION = 2s means a sign flip costs ~2s of deceleration + 2s reversal
  - ALT_UP_THRESH/ALT_DOWN_THRESH = ±0.3 — any dz beyond that triggers Z-filter reconfig

Usage:
    python3 action_inspect.py                        # all 3 MS600 seeds × 5 episodes, stochastic
    python3 action_inspect.py --deterministic        # compare deterministic action mean
    python3 action_inspect.py --episodes 10          # more episodes
    python3 action_inspect.py --seeds 42             # single seed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

LIGHTWEIGHT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LIGHTWEIGHT_DIR.parent))
sys.path.insert(0, str(LIGHTWEIGHT_DIR))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_3d import AltitudeActiveSLAMEnv as SLAM3DEnv, MAX_STEPS
from transfer_config import LIGHTWEIGHT_EVAL_SEEDS

MS600_RUNS = {
    42:  "results_3d_shape_delta_800k_maxsteps600",
    100: "results_3d_shape_delta_ms600_s100",
    200: "results_3d_shape_delta_ms600_s200",
}
CKPT_STEPS = 180_000

ALT_UP_THRESH = 0.3   # from transfer_config
ALT_DOWN_THRESH = -0.3


def build_env(map_seed: int, vecnorm_path: Path) -> VecNormalize:
    def make():
        return SLAM3DEnv(map_seed=map_seed, difficulty="hard", seed_suite=None)
    vec = DummyVecEnv([make])
    vec = VecNormalize.load(str(vecnorm_path), vec)
    vec.training = False
    vec.norm_reward = False
    return vec


def run_episode(model, env, deterministic: bool, max_steps: int):
    obs = env.reset()
    lstm_states = None
    ep_start = np.ones((1,), dtype=bool)
    actions = []
    alt_changes_steps = []  # step indices where altitude actually changed
    for step in range(max_steps):
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=ep_start, deterministic=deterministic,
        )
        ep_start[:] = False
        obs, _reward, done, infos = env.step(action)
        actions.append(action[0].copy())
        if infos[0].get("alt_changed", False):
            alt_changes_steps.append(step)
        if done[0]:
            break
    return np.asarray(actions), alt_changes_steps


def analyze_actions(actions: np.ndarray, alt_changes_steps: list[int]):
    """Compute motion-smoothness metrics for a single rollout."""
    n = len(actions)
    dx, dy, dz = actions[:, 0], actions[:, 1], actions[:, 2]

    # (A) sign-flip rate for xy components
    dx_flips = int(np.sum(np.diff(np.sign(dx)) != 0))
    dy_flips = int(np.sum(np.diff(np.sign(dy)) != 0))
    xy_flip_rate = (dx_flips + dy_flips) / max(2 * (n - 1), 1)

    # (B) altitude cadence
    alt_change_count = len(alt_changes_steps)
    if alt_change_count >= 2:
        alt_intervals = np.diff(alt_changes_steps)
        min_alt_interval = int(alt_intervals.min())
        mean_alt_interval = float(alt_intervals.mean())
    else:
        min_alt_interval = n
        mean_alt_interval = float(n)

    dz_above_thresh = float(np.mean(
        (dz > ALT_UP_THRESH) | (dz < ALT_DOWN_THRESH)
    ))

    # (C) jitter: RMS of consecutive action deltas (all 3 dims)
    action_deltas = np.diff(actions, axis=0)
    jitter_rms = float(np.sqrt(np.mean(action_deltas ** 2)))

    # (D) magnitude
    xy_magnitude = float(np.mean(np.sqrt(dx ** 2 + dy ** 2)))
    xy_saturation = float(np.mean(
        (np.abs(dx) > 0.9) | (np.abs(dy) > 0.9)
    ))

    return {
        "steps": n,
        "xy_flip_rate": xy_flip_rate,
        "alt_change_count": alt_change_count,
        "min_alt_interval": min_alt_interval,
        "mean_alt_interval": mean_alt_interval,
        "dz_triggered_frac": dz_above_thresh,
        "jitter_rms": jitter_rms,
        "xy_magnitude": xy_magnitude,
        "xy_saturation": xy_saturation,
    }


def verdict(label: str, value: float, green_max: float, yellow_max: float,
            lower_is_better: bool = True) -> str:
    if lower_is_better:
        if value <= green_max:
            tag = "GREEN"
        elif value <= yellow_max:
            tag = "YELLOW"
        else:
            tag = "RED"
    else:
        if value >= green_max:
            tag = "GREEN"
        elif value >= yellow_max:
            tag = "YELLOW"
        else:
            tag = "RED"
    return f"  {label:<28s} {value:8.3f}   [{tag}]"


def print_verdict(seed: int, mode: str, stats_per_ep: list[dict]):
    print(f"\n--- seed {seed} ({mode}) ---")
    mean = {k: float(np.mean([s[k] for s in stats_per_ep])) for k in stats_per_ep[0]}
    mean["min_alt_interval"] = float(np.min([s["min_alt_interval"] for s in stats_per_ep]))

    print(f"  episodes={len(stats_per_ep)}, mean_steps={mean['steps']:.0f}")
    # thresholds informed by gazebo env defenses
    print(verdict("xy_flip_rate (lo=smooth)",    mean["xy_flip_rate"],    0.15, 0.30))
    print(verdict("jitter_rms (lo=smooth)",      mean["jitter_rms"],      0.35, 0.70))
    print(verdict("xy_saturation (lo=smooth)",   mean["xy_saturation"],   0.40, 0.70))
    print(verdict("dz_triggered_frac (lo=ok)",   mean["dz_triggered_frac"], 0.05, 0.10))
    print(verdict("alt_change_count (lo=ok)",    mean["alt_change_count"], 10, 25))
    print(verdict("min_alt_interval (hi=safe)", mean["min_alt_interval"], 20, 10,
                  lower_is_better=False))
    print(f"  xy_magnitude: {mean['xy_magnitude']:.3f}")


def run_seed(seed: int, episodes: int, deterministic: bool, map_seeds: list[int]):
    run_dir = LIGHTWEIGHT_DIR / MS600_RUNS[seed]
    ckpt = run_dir / "models" / f"slam3d_{CKPT_STEPS}_steps.zip"
    vec = run_dir / "models" / f"slam3d_vecnormalize_{CKPT_STEPS}_steps.pkl"
    if not ckpt.exists() or not vec.exists():
        print(f"[skip] seed={seed}: missing {ckpt} or {vec}")
        return None

    print(f"\n=== loading seed {seed} from {ckpt.name} ===")
    stats_per_ep = []
    for i, ms in enumerate(map_seeds[:episodes]):
        env = build_env(ms, vec)
        model = RecurrentPPO.load(str(ckpt), env=env, device="auto")
        actions, alt_steps = run_episode(model, env, deterministic, MAX_STEPS)
        s = analyze_actions(actions, alt_steps)
        stats_per_ep.append(s)
        env.close()
    return stats_per_ep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 200])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--deterministic", action="store_true",
                   help="Also compare deterministic action mean")
    args = p.parse_args()

    map_seeds = list(LIGHTWEIGHT_EVAL_SEEDS[:args.episodes])

    print("=" * 70)
    print(" Tier 1 policy-action inspector — RTAB-Map motion-safety proxy")
    print("=" * 70)
    print(f" MS600 checkpoints @ {CKPT_STEPS:,} steps")
    print(f" Episodes/seed: {args.episodes}   map_seeds: {map_seeds}")
    print(f" MAX_STEPS per ep: {MAX_STEPS}")
    print("=" * 70)

    results = {}
    for seed in args.seeds:
        results[(seed, "stoch")] = run_seed(seed, args.episodes, False, map_seeds)
        if args.deterministic:
            results[(seed, "determ")] = run_seed(seed, args.episodes, True, map_seeds)

    print("\n" + "=" * 70)
    print(" VERDICT")
    print("=" * 70)
    print(
        " Thresholds (GREEN → YELLOW → RED), informed by ActiveSLAMEnv defenses:\n"
        "   xy_flip_rate     0.15 / 0.30    — MAX_YAW_CHANGE=11°/step partially absorbs\n"
        "   jitter_rms       0.35 / 0.70    — STEP_DURATION=2s, so high jitter = stop-start\n"
        "   xy_saturation    0.40 / 0.70    — saturated actions overwhelm A* smoothing\n"
        "   dz_triggered_frac 0.05 / 0.10   — triggers Z-filter reconfig\n"
        "   alt_change_count 10 / 25 per ep — RTAB-Map hates frequent alt switches\n"
        "   min_alt_interval 20 / 10 steps  — lightweight has MIN_ALT_DWELL=20, Gazebo has NONE\n"
    )
    for (seed, mode), stats in results.items():
        if stats:
            print_verdict(seed, mode, stats)

    any_red = False
    for stats in results.values():
        if not stats:
            continue
        for s in stats:
            if s["xy_flip_rate"] > 0.30 or s["jitter_rms"] > 0.70 or \
                    s["alt_change_count"] > 25 or s["min_alt_interval"] < 10:
                any_red = True
    print("\n" + ("!!! RED VERDICT — add MIN_ALT_DWELL guard + action low-pass before Gazebo !!!"
                  if any_red else
                  "No RED thresholds tripped. Proceed to Tier 2 (200-step Gazebo smoke test)."))


if __name__ == "__main__":
    main()
