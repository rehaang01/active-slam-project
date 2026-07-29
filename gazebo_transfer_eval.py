#!/usr/bin/env python3
"""gazebo_transfer_eval.py — evaluate a lightweight-trained RL checkpoint inside Gazebo.

This is the **headline sim-to-sim transfer experiment** for the ICRA/IROS paper:
    1. Load a RecurrentPPO checkpoint that was trained on the lightweight 48×48 grid env
       (lightweight_train/results_3d_seedXX/models/slam3d_XXXXXX_steps.zip).
    2. Instantiate the Gazebo/RTAB-Map env (envs/active_slam_env.py).
    3. Run N evaluation episodes (no fine-tuning — zero-shot transfer).
    4. Save per-episode CSVs compatible with plot_comparison.py schema.

Usage
-----
  # Zero-shot transfer (no fine-tuning):
  python3 gazebo_transfer_eval.py \\
      --model lightweight_train/results_3d_seed42/models/slam3d_360000_steps.zip \\
      --vecnorm lightweight_train/results_3d_seed42/models/slam3d_vecnormalize_360000_steps.pkl \\
      --episodes 5 \\
      --max-steps 1500

Requires the ROS2/Gazebo/PX4/RTAB-Map stack to be launched (see launch/full_simulation.launch.py).
Runs on whatever Gazebo world the env is configured for (warehouse, default).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------
# numpy 2.x -> numpy 1.x compatibility shim for cross-version pickles.
# Host (training) runs numpy 2.x where internals live under numpy._core.
# Docker (ROS2) pins numpy<2.0 where they live under numpy.core.
# The submodule contents are identical; only the path renamed.
# Register sys.modules aliases so cloudpickle's `import numpy._core.*`
# resolves to the 1.x equivalents.
# ------------------------------------------------------------------
if not hasattr(np, "_core"):
    import numpy.core  # noqa: F401
    import numpy.core.numeric  # noqa: F401
    import numpy.core.multiarray  # noqa: F401
    import numpy.core.umath  # noqa: F401
    for _modname in list(sys.modules):
        if _modname == "numpy.core" or _modname.startswith("numpy.core."):
            sys.modules[_modname.replace("numpy.core", "numpy._core")] = sys.modules[_modname]

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to lightweight-trained RecurrentPPO .zip")
    p.add_argument("--vecnorm", required=True, help="Path to matching VecNormalize .pkl")
    p.add_argument("--episodes", type=int, default=5, help="Number of Gazebo eval episodes")
    p.add_argument("--max-steps", type=int, default=1500, help="Max steps per Gazebo episode")
    p.add_argument("--output-dir", default="logs/gazebo_transfer", help="Where to save per-episode CSVs")
    p.add_argument("--deterministic", action="store_true",
                   help="Use argmax actions (NOT recommended — lightweight policy has biased action mean, "
                        "see diagnose_eval.py; default uses stochastic sampling).")
    p.add_argument("--ema-alpha", type=float, default=0.5,
                   help="Legacy single-axis EMA (applied to all 3 dims when --ema-alpha-xy/-dz "
                        "are not provided). α=1.0 disables filter.")
    p.add_argument("--ema-alpha-xy", type=float, default=None,
                   help="Per-axis EMA coefficient for (dx, dy). Higher α = less smoothing, "
                        "preserves sustained lateral motion. Recommended 0.7–0.9 to counter "
                        "the symmetric EMA collapsing ±-flipping xy actions to ~0.")
    p.add_argument("--ema-alpha-dz", type=float, default=None,
                   help="Per-axis EMA coefficient for dz. Lower α = heavier smoothing, "
                        "dampens the policy's upward altitude bias. Recommended 0.2–0.4.")
    p.add_argument("--episode-tag", default="rl_transfer", help="Filename prefix for CSVs")
    return p.parse_args()


def main():
    args = parse_args()

    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from envs.active_slam_env import ActiveSLAMEnv

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve per-axis EMA. If the per-axis flags are provided, use them;
    # otherwise fall back to the legacy single --ema-alpha on all 3 dims.
    alpha_xy = args.ema_alpha_xy if args.ema_alpha_xy is not None else args.ema_alpha
    alpha_dz = args.ema_alpha_dz if args.ema_alpha_dz is not None else args.ema_alpha
    alpha_vec = np.array([alpha_xy, alpha_xy, alpha_dz], dtype=np.float32)

    print(f"[Transfer] Model:     {args.model}")
    print(f"[Transfer] VecNorm:   {args.vecnorm}")
    print(f"[Transfer] Episodes:  {args.episodes}")
    print(f"[Transfer] EMA α:     xy={alpha_xy:.2f}  dz={alpha_dz:.2f}")
    print(f"[Transfer] Output:    {out_dir}")

    CSV_FIELDS = [
        "step", "timestamp", "coverage", "known_cells", "total_cells",
        "cov_trace", "pos_x", "pos_y", "altitude", "frontier_count",
        "nearest_frontier_dist", "loop_closures", "tracking_lost_events",
        "new_cells", "alt_idx", "altitude_changes",
        "cov_alt0", "cov_alt1", "cov_alt2", "cov_alt3", "note",
    ]

    # We want a single env instance that persists across eval runs (Gazebo map cannot be reset).
    # Following the train.py convention: instantiate once, reset between eval episodes.
    env = DummyVecEnv([lambda: ActiveSLAMEnv()])
    if os.path.isfile(args.vecnorm):
        print(f"[Transfer] Loading VecNormalize stats...")
        env = VecNormalize.load(args.vecnorm, env)
        env.training = False
        env.norm_reward = False
    else:
        print(f"[Transfer] WARNING: no VecNormalize file at {args.vecnorm} — obs will be unnormalized.")

    print(f"[Transfer] Loading model...")
    # Model was saved on host (numpy 2.x); Docker runs numpy<2.0 for ROS2.
    # cloudpickled schedules / spaces in the zip reference numpy._core (2.x path),
    # which numpy<2.0 doesn't expose. custom_objects overrides those fields at
    # load time — weights in the .pth come through torch and are safe.
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
        "observation_space": env.observation_space,
        "action_space": env.action_space,
    }
    model = RecurrentPPO.load(args.model, env=env, device="auto",
                              custom_objects=custom_objects)
    print(f"[Transfer] Model loaded.")

    for ep in range(args.episodes):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = out_dir / f"{args.episode_tag}_ep{ep}_{ts}.csv"
        print(f"\n[Transfer] === Episode {ep}/{args.episodes} -> {out_csv.name} ===")

        obs = env.reset()
        lstm_states = None
        ep_start = np.ones((1,), dtype=bool)
        rows = []
        prev_known = 0
        total_loop_closures = 0
        total_tracking_lost = 0
        prev_trk_lost = False
        ema_prev_action = None

        start = time.time()
        try:
            for step in range(1, args.max_steps + 1):
                raw_action, lstm_states = model.predict(
                    obs, state=lstm_states, episode_start=ep_start, deterministic=args.deterministic,
                )
                if ema_prev_action is None:
                    action = raw_action
                else:
                    # Per-axis EMA: a_t = α·raw + (1-α)·a_{t-1}, broadcast over (dx,dy,dz).
                    # alpha_vec = [α_xy, α_xy, α_dz] resolved above.
                    action = alpha_vec * raw_action + (1.0 - alpha_vec) * ema_prev_action
                ema_prev_action = action
                ep_start[:] = False
                obs, reward, done, infos = env.step(action)
                info = infos[0]

                known_cells = info.get("known_cells", 0)
                new_cells = int(max(0, known_cells - prev_known))
                prev_known = known_cells

                lc_count = info.get("loop_closures", 0)
                if isinstance(lc_count, (int, float)):
                    total_loop_closures = int(lc_count)

                trk = bool(info.get("tracking_lost", False))
                if trk and not prev_trk_lost:
                    total_tracking_lost += 1
                prev_trk_lost = trk

                row = {
                    "step": step,
                    "timestamp": float(time.time() - start),
                    "coverage": float(info.get("coverage", 0.0)),
                    "known_cells": int(known_cells),
                    "total_cells": int(info.get("total_cells", 0)),
                    "cov_trace": float(info.get("cov_trace", 0.0)),
                    "pos_x": float(info.get("pos_x", 0.0)),
                    "pos_y": float(info.get("pos_y", 0.0)),
                    "altitude": float(info.get("altitude", 0.0)),
                    "frontier_count": int(info.get("frontier_count", 0)),
                    "nearest_frontier_dist": float(info.get("nearest_frontier_dist", 0.0)),
                    "loop_closures": total_loop_closures,
                    "tracking_lost_events": total_tracking_lost,
                    "new_cells": new_cells,
                    "alt_idx": int(info.get("alt_idx", 0)),
                    "altitude_changes": int(info.get("altitude_changes", 0)),
                    "cov_alt0": float(info.get("cov_alt0", 0.0)),
                    "cov_alt1": float(info.get("cov_alt1", 0.0)),
                    "cov_alt2": float(info.get("cov_alt2", 0.0)),
                    "cov_alt3": float(info.get("cov_alt3", 0.0)),
                    "note": "",
                }
                rows.append(row)

                if step % 100 == 0:
                    print(f"  step {step}: cov={row['coverage']:.2%} trace={row['cov_trace']:.3f} "
                          f"loops={total_loop_closures} trk_lost={total_tracking_lost}")
                if done[0]:
                    row["note"] = "done"
                    break
        except KeyboardInterrupt:
            print("[Transfer] Interrupted — saving partial log.")

        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"[Transfer] Wrote {out_csv} ({len(rows)} steps, final cov={rows[-1]['coverage']:.2%})")

    print("\n[Transfer] All episodes complete.")


if __name__ == "__main__":
    main()
