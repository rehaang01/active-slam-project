#!/usr/bin/env python3
"""Shared Gazebo baseline runner using the transfer-compatible gym env."""

from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from envs.active_slam_env import ActiveSLAMEnv, MAX_STEPS


CSV_FIELDS = [
    "step",
    "timestamp",
    "coverage",
    "known_cells",
    "total_cells",
    "cov_trace",
    "pos_x",
    "pos_y",
    "altitude",
    "alt_idx",
    "frontier_count",
    "nearest_frontier_dist",
    "loop_closures",
    "tracking_lost_events",
    "new_cells",
    "altitude_changes",
    "cov_alt0",
    "cov_alt1",
    "cov_alt2",
    "cov_alt3",
    "note",
]


def run_gazebo_baseline(policy, prefix: str, title: str) -> str:
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(log_dir, f"{prefix}_{timestamp}.csv")

    env = ActiveSLAMEnv()
    metrics = []
    tracking_lost_events = 0
    prev_tracking_lost = False

    try:
        obs, info = env.reset()
        policy.reset()

        print(f"[{title}] Starting run...")
        for step in range(1, MAX_STEPS + 1):
            action = policy.get_action(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)

            tracking_lost = bool(info.get("tracking_lost", False))
            if tracking_lost and not prev_tracking_lost:
                tracking_lost_events += 1
            prev_tracking_lost = tracking_lost

            note_parts = []
            if info.get("panic_mode"):
                note_parts.append("panic")
            if tracking_lost:
                note_parts.append("trk_lost")
            if info.get("r_coverage_bonus", 0) > 0:
                note_parts.append("cov_bonus")

            row = {
                "step": step,
                "timestamp": time.time(),
                "coverage": round(float(info.get("coverage", 0.0)), 4),
                "known_cells": int(info.get("known_cells", 0)),
                "total_cells": int(info.get("total_cells", 0)),
                "cov_trace": round(float(info.get("cov_trace", 0.0)), 6),
                "pos_x": round(float(info.get("pos_x", 0.0)), 3),
                "pos_y": round(float(info.get("pos_y", 0.0)), 3),
                "altitude": round(float(info.get("altitude", 0.0)), 3),
                "alt_idx": int(info.get("alt_idx", 0)),
                "frontier_count": int(info.get("frontier_count", 0)),
                "nearest_frontier_dist": round(float(info.get("nearest_frontier_dist", -1.0)), 3),
                "loop_closures": int(info.get("loop_closures", 0)),
                "tracking_lost_events": tracking_lost_events,
                "new_cells": int(info.get("new_cells", 0)),
                "altitude_changes": int(info.get("altitude_changes", 0)),
                "cov_alt0": round(float(info.get("cov_alt0", 0.0)), 4),
                "cov_alt1": round(float(info.get("cov_alt1", 0.0)), 4),
                "cov_alt2": round(float(info.get("cov_alt2", 0.0)), 4),
                "cov_alt3": round(float(info.get("cov_alt3", 0.0)), 4),
                "note": ",".join(note_parts),
            }
            metrics.append(row)

            if step % 50 == 0 or step == 1:
                print(
                    f"[{title}] Step {step:4d}/{MAX_STEPS} "
                    f"cov={row['coverage']:.1%} cov_trace={row['cov_trace']:.4f} "
                    f"alt={row['altitude']:.1f}m new_cells={row['new_cells']}"
                )

            if terminated or truncated:
                print(f"[{title}] Episode ended at step {step} with coverage {row['coverage']:.1%}")
                break

    finally:
        try:
            env.close()
        except Exception:
            pass

    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(metrics)

    print(f"[{title}] CSV saved -> {output_path}")
    return output_path
