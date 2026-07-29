"""Tier 1b: EMA low-pass α sweep — does action smoothing preserve exploration?

Wraps model inference with an exponential moving average:
    a_t = α · a_raw + (1-α) · a_{t-1}

α=1.0 is no filter (baseline). Lower α = more smoothing. The policy's LSTM still
sees the real obs trajectory, but the actions the env (and the drone) see are
low-passed. For Gazebo transfer we need BOTH:
  - motion metrics drop to GREEN (RTAB-Map safety)
  - coverage does not collapse (policy still useful)

Usage:
    python3 ema_sweep.py                                # full sweep
    python3 ema_sweep.py --seeds 42 --episodes 3        # quick smoke
    python3 ema_sweep.py --alphas 1.0 0.3               # single-α compare
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

from action_inspect import analyze_actions, MS600_RUNS, CKPT_STEPS


def build_env(map_seed: int, vecnorm_path: Path) -> VecNormalize:
    def make():
        return SLAM3DEnv(map_seed=map_seed, difficulty="hard", seed_suite=None)
    vec = DummyVecEnv([make])
    vec = VecNormalize.load(str(vecnorm_path), vec)
    vec.training = False
    vec.norm_reward = False
    return vec


def run_episode_ema(model, env, alpha: float, max_steps: int):
    obs = env.reset()
    lstm_states = None
    ep_start = np.ones((1,), dtype=bool)
    actions = []
    alt_changes_steps = []
    prev_action = None
    final_info = {}
    for step in range(max_steps):
        raw_action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=ep_start, deterministic=False,
        )
        if prev_action is None or alpha >= 0.999:
            action = raw_action
        else:
            action = alpha * raw_action + (1.0 - alpha) * prev_action
        prev_action = action
        ep_start[:] = False
        obs, _reward, done, infos = env.step(action)
        actions.append(action[0].copy())
        final_info = infos[0]
        if final_info.get("alt_changed", False):
            alt_changes_steps.append(step)
        if done[0]:
            break
    return (
        np.asarray(actions),
        alt_changes_steps,
        float(final_info.get("coverage", 0.0)),
        float(final_info.get("cov_trace", 0.0)),
    )


def run_cell(seed: int, alpha: float, episodes: int, map_seeds: list[int]):
    run_dir = LIGHTWEIGHT_DIR / MS600_RUNS[seed]
    ckpt = run_dir / "models" / f"slam3d_{CKPT_STEPS}_steps.zip"
    vec = run_dir / "models" / f"slam3d_vecnormalize_{CKPT_STEPS}_steps.pkl"

    stats_list = []
    coverages = []
    traces = []
    for ms in map_seeds[:episodes]:
        env = build_env(ms, vec)
        model = RecurrentPPO.load(str(ckpt), env=env, device="auto")
        actions, alt_steps, cov, trace = run_episode_ema(model, env, alpha, MAX_STEPS)
        stats_list.append(analyze_actions(actions, alt_steps))
        coverages.append(cov)
        traces.append(trace)
        env.close()

    agg = {k: float(np.mean([s[k] for s in stats_list])) for k in stats_list[0]}
    agg["coverage_mean"] = float(np.mean(coverages))
    agg["coverage_std"] = float(np.std(coverages))
    agg["cov_trace_mean"] = float(np.mean(traces))
    return agg


def fmt(val, fmt_str="{:.2f}"):
    return fmt_str.format(val)


def print_table(results: dict):
    """results[(seed, alpha)] = dict of aggregated stats."""
    seeds = sorted({k[0] for k in results})
    alphas = sorted({k[1] for k in results}, reverse=True)

    print("\n" + "=" * 100)
    print(" RESULTS — EMA α sweep")
    print("=" * 100)
    print(" α = 1.0 is baseline (no filter). Lower α = more aggressive smoothing.")
    print()

    # Per-seed breakdown
    for seed in seeds:
        print(f"--- seed {seed} ---")
        print(f"  {'α':<6}{'cov_mean':<11}{'cov_std':<10}{'xy_flip':<10}"
              f"{'jitter':<10}{'xy_sat':<10}{'dz_trig':<10}{'alt_chg':<10}")
        for a in alphas:
            r = results.get((seed, a))
            if r is None:
                continue
            print(f"  {a:<6.2f}"
                  f"{fmt(r['coverage_mean']):<11}"
                  f"{fmt(r['coverage_std']):<10}"
                  f"{fmt(r['xy_flip_rate']):<10}"
                  f"{fmt(r['jitter_rms']):<10}"
                  f"{fmt(r['xy_saturation']):<10}"
                  f"{fmt(r['dz_triggered_frac']):<10}"
                  f"{fmt(r['alt_change_count'], '{:.1f}'):<10}")
        print()

    # Cross-seed aggregate per α
    print("--- aggregate across seeds (mean ± std) ---")
    print(f"  {'α':<6}{'cov_mean':<18}{'xy_flip':<12}{'jitter':<12}{'dz_trig':<12}{'alt_chg':<12}")
    for a in alphas:
        cells = [results[(s, a)] for s in seeds if (s, a) in results]
        if not cells:
            continue
        cov_m = np.mean([c['coverage_mean'] for c in cells])
        cov_s = np.std([c['coverage_mean'] for c in cells])
        xy_flip = np.mean([c['xy_flip_rate'] for c in cells])
        jitter = np.mean([c['jitter_rms'] for c in cells])
        dz = np.mean([c['dz_triggered_frac'] for c in cells])
        alt_chg = np.mean([c['alt_change_count'] for c in cells])
        print(f"  {a:<6.2f}"
              f"{cov_m:.3f} ± {cov_s:.3f}    "
              f"{xy_flip:<12.2f}{jitter:<12.2f}{dz:<12.2f}{alt_chg:<12.1f}")
    print()

    # Verdict per α
    print("--- safety verdict per α (aggregate) ---")
    for a in alphas:
        cells = [results[(s, a)] for s in seeds if (s, a) in results]
        if not cells:
            continue
        xy_flip = np.mean([c['xy_flip_rate'] for c in cells])
        jitter = np.mean([c['jitter_rms'] for c in cells])
        cov = np.mean([c['coverage_mean'] for c in cells])
        motion_ok = xy_flip <= 0.30 and jitter <= 0.70
        cov_ok = cov >= 0.60  # don't collapse below 60%
        tag = "GREEN  (safe + explores)" if (motion_ok and cov_ok) else \
              "YELLOW (safe but coverage degraded)" if (motion_ok and not cov_ok) else \
              "YELLOW (explores but motion unsafe)" if (cov_ok and not motion_ok) else \
              "RED    (unsafe AND bad coverage)"
        print(f"  α={a:.2f}  →  {tag}   "
              f"[flip={xy_flip:.2f}, jitter={jitter:.2f}, cov={cov:.3f}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 200])
    p.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.5, 0.3, 0.2])
    p.add_argument("--episodes", type=int, default=3)
    args = p.parse_args()

    map_seeds = list(LIGHTWEIGHT_EVAL_SEEDS[:args.episodes])

    print("=" * 70)
    print(" Tier 1b EMA α sweep — motion safety vs exploration preservation")
    print("=" * 70)
    print(f" α values: {args.alphas}")
    print(f" Seeds: {args.seeds}   Episodes: {args.episodes}   MAX_STEPS: {MAX_STEPS}")
    print(f" Map seeds: {map_seeds}")
    print(f" Total rollouts: {len(args.seeds) * len(args.alphas) * args.episodes}")
    print("=" * 70)

    results = {}
    total = len(args.seeds) * len(args.alphas)
    done = 0
    for seed in args.seeds:
        for alpha in args.alphas:
            done += 1
            print(f"\n[{done}/{total}] seed={seed}  α={alpha:.2f} ...")
            results[(seed, alpha)] = run_cell(seed, alpha, args.episodes, map_seeds)

    print_table(results)


if __name__ == "__main__":
    main()
