"""Diagnostic: compare deterministic vs stochastic eval for the main checkpoint.

Also dumps action statistics (mean, std, how often argmax matches sampled action)
so we can decide whether the policy needs more training, annealed entropy, or
action-squashing to make deterministic eval work at test time.

Usage:
    python diagnose_eval.py --tag seed42 --checkpoint slam3d_360000_steps.zip --n-episodes 10
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path
import numpy as np

LIGHTWEIGHT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LIGHTWEIGHT_DIR.parent))
sys.path.insert(0, str(LIGHTWEIGHT_DIR))

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_3d import AltitudeActiveSLAMEnv as SLAM3DEnv, MAX_STEPS
from transfer_config import ALTITUDE_LEVELS, LIGHTWEIGHT_EVAL_SEEDS


def build_eval_env(map_seed: int, vecnorm_path: Path) -> VecNormalize:
    def make():
        return SLAM3DEnv(map_seed=map_seed, difficulty="hard", seed_suite=None)
    vec = DummyVecEnv([make])
    vec = VecNormalize.load(str(vecnorm_path), vec)
    vec.training = False
    vec.norm_reward = False
    return vec


def run_episode(model, env, deterministic: bool, max_steps: int = MAX_STEPS):
    obs = env.reset()
    lstm_states = None
    ep_start = np.ones((1,), dtype=bool)
    total_reward = 0.0
    final_cov = 0.0
    final_cov_trace = 0.0
    action_log = []
    for step in range(max_steps):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=ep_start, deterministic=deterministic)
        ep_start[:] = False
        obs, reward, done, infos = env.step(action)
        total_reward += float(reward[0])
        info = infos[0]
        final_cov = float(info.get("coverage", 0.0))
        final_cov_trace = float(info.get("cov_trace", 0.0))
        action_log.append(action[0].copy())
        if done[0]:
            break
    return {
        "reward": total_reward,
        "final_coverage": final_cov,
        "final_cov_trace": final_cov_trace,
        "actions": np.array(action_log),
        "episode_len": step + 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-episodes", type=int, default=10)
    args = parser.parse_args()

    results_dir = LIGHTWEIGHT_DIR / f"results_3d_{args.tag}"
    model_path = results_dir / "models" / args.checkpoint
    step_tag = args.checkpoint.replace("slam3d_", "").replace("_steps.zip", "")
    vecnorm_path = results_dir / "models" / f"slam3d_vecnormalize_{step_tag}_steps.pkl"
    if not vecnorm_path.exists():
        vecnorm_path = results_dir / "models" / "vec_normalize.pkl"
    print(f"Loading model: {model_path}")
    print(f"Loading vecnorm: {vecnorm_path}")

    model = RecurrentPPO.load(str(model_path), device="cuda")

    seeds = LIGHTWEIGHT_EVAL_SEEDS[: args.n_episodes]

    for mode_name, deterministic in [("STOCHASTIC", False), ("DETERMINISTIC", True)]:
        print(f"\n=== {mode_name} ===")
        covs, traces, rews, action_stats = [], [], [], []
        for s in seeds:
            env = build_eval_env(int(s), vecnorm_path)
            res = run_episode(model, env, deterministic)
            covs.append(res["final_coverage"])
            traces.append(res["final_cov_trace"])
            rews.append(res["reward"])
            action_stats.append(res["actions"])
            env.close()
            print(f"  seed {s}: cov={res['final_coverage']:.2%} trace={res['final_cov_trace']:.3f} rew={res['reward']:.1f} steps={res['episode_len']}")
        all_actions = np.concatenate(action_stats, axis=0)
        print(f"  Mean cov: {np.mean(covs):.2%} ± {np.std(covs):.2%}")
        print(f"  Mean trace: {np.mean(traces):.3f} ± {np.std(traces):.3f}")
        print(f"  Action[0] (dx) stats: mean={all_actions[:,0].mean():.3f} std={all_actions[:,0].std():.3f}  min={all_actions[:,0].min():.3f} max={all_actions[:,0].max():.3f}")
        print(f"  Action[1] (dy) stats: mean={all_actions[:,1].mean():.3f} std={all_actions[:,1].std():.3f}  min={all_actions[:,1].min():.3f} max={all_actions[:,1].max():.3f}")
        print(f"  Action[2] (dz) stats: mean={all_actions[:,2].mean():.3f} std={all_actions[:,2].std():.3f}  min={all_actions[:,2].min():.3f} max={all_actions[:,2].max():.3f}")


if __name__ == "__main__":
    main()
