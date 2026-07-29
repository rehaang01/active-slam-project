#!/usr/bin/env python3
"""plot_trajectory_altitudes.py - qualitative trajectory visualization across altitudes.

Reviewer item (iii): "at least one qualitative trajectory visualization across altitudes."

Standalone deterministic rollout of the headline policy on a single held-out evaluation
map. Logs the agent's (x, y, altitude) at every step and the per-altitude occupancy maps,
then renders a 1x4 panel figure (one per altitude layer, 1-4 m) with the agent's path while
it was at that layer, colour-coded by time. Grayscale-safe (a thin black path line backs the
viridis time-colour scatter) with SI-unit axes (1 grid cell = 1 m).

Does NOT modify run_3d.py / env reward knobs -> headline behaviour unchanged.

Usage
-----
  MAX_STEPS_OVERRIDE=600 .venv/bin/python lightweight_train/plot_trajectory_altitudes.py \
    --model   lightweight_train/results_3d_shape_delta_800k_maxsteps600/models/slam3d_180000_steps.zip \
    --vecnorm lightweight_train/results_3d_shape_delta_800k_maxsteps600/models/slam3d_vecnormalize_180000_steps.pkl \
    --seed 207 --deterministic
"""
from __future__ import annotations

import os

os.environ.setdefault("MAX_STEPS_OVERRIDE", "600")  # headline 600-step regime

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "_core"):  # numpy 2.x<->1.x pickle shim (no-op on host)
    import numpy.core  # noqa: F401
    import numpy.core.multiarray  # noqa: F401
    import numpy.core.numeric  # noqa: F401
    import numpy.core.umath  # noqa: F401

    for _m in list(sys.modules):
        if _m == "numpy.core" or _m.startswith("numpy.core."):
            sys.modules[_m.replace("numpy.core", "numpy._core")] = sys.modules[_m]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def rollout(model, env, raw, max_steps, deterministic):
    obs = env.reset()
    states = None
    starts = np.ones((1,), dtype=bool)
    log = []
    snap = {
        "true_maps": raw.true_maps.copy(),
        "known_maps": raw.known_maps.copy(),
        "visited_maps": raw.visited_maps.copy(),
    }
    final_cov = 0.0
    for step in range(1, max_steps + 1):
        action, states = model.predict(obs, state=states, episode_start=starts, deterministic=deterministic)
        starts[:] = False
        obs, _reward, done, infos = env.step(action)
        info = infos[0]
        final_cov = float(info.get("coverage", final_cov))
        if not done[0]:
            log.append(
                {
                    "step": step,
                    "pos_x": float(raw.pos[0]),  # row
                    "pos_y": float(raw.pos[1]),  # col
                    "alt_idx": int(raw.alt_idx),
                    "coverage": float(info.get("coverage", 0.0)),
                    "cov_trace": float(info.get("cov_trace", 0.0)),
                }
            )
            snap = {
                "true_maps": raw.true_maps.copy(),
                "known_maps": raw.known_maps.copy(),
                "visited_maps": raw.visited_maps.copy(),
            }
        else:
            break
    return log, snap, final_cov


def make_figure(log, snap, final_cov, altitude_levels, out_base, seed):
    from transfer_config import N_ALTITUDES

    n_alt = N_ALTITUDES
    last_step = log[-1]["step"] if log else 1
    norm = Normalize(vmin=1, vmax=last_step)
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, n_alt, figsize=(4.0 * n_alt, 4.3))
    if n_alt == 1:
        axes = [axes]

    for a in range(n_alt):
        ax = axes[a]
        obstacles = snap["true_maps"][a]
        # light-gray obstacle backdrop (dark = obstacle) - readable in grayscale print
        ax.imshow(obstacles, cmap="Greys", origin="lower", vmin=0.0, vmax=1.0, alpha=0.55, interpolation="nearest")

        pts = [(r["step"], r["pos_y"], r["pos_x"]) for r in log if r["alt_idx"] == a]  # (step, x=col, y=row)
        if pts:
            steps = np.array([p[0] for p in pts])
            xs = np.array([p[1] for p in pts])
            ys = np.array([p[2] for p in pts])
            # thin black connecting line for contiguous same-altitude segments (grayscale legibility)
            seg_x, seg_y = [xs[0]], [ys[0]]
            for i in range(1, len(steps)):
                if steps[i] - steps[i - 1] == 1:
                    seg_x.append(xs[i])
                    seg_y.append(ys[i])
                else:
                    ax.plot(seg_x, seg_y, color="black", lw=0.5, alpha=0.5, zorder=2)
                    seg_x, seg_y = [xs[i]], [ys[i]]
            ax.plot(seg_x, seg_y, color="black", lw=0.5, alpha=0.5, zorder=2)
            # time-coloured path
            ax.scatter(xs, ys, c=steps, cmap=cmap, norm=norm, s=12, zorder=3, edgecolors="none")
            # enter / exit markers
            ax.scatter([xs[0]], [ys[0]], marker="o", s=90, facecolors="none",
                       edgecolors="lime", linewidths=2.0, zorder=4)
            ax.scatter([xs[-1]], [ys[-1]], marker="s", s=90, facecolors="none",
                       edgecolors="red", linewidths=2.0, zorder=4)
            dwell = len(pts)
        else:
            dwell = 0

        ax.set_title(f"Altitude {altitude_levels[a]:.0f} m  (layer {a}, {dwell} steps)", fontsize=10)
        ax.set_xlabel("x (m)")
        if a == 0:
            ax.set_ylabel("y (m)")
        ax.set_xlim(0, obstacles.shape[1] - 1)
        ax.set_ylim(0, obstacles.shape[0] - 1)
        ax.set_aspect("equal")

    # shared time colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("episode step")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="lime", markeredgewidth=2, markersize=10, label="enter layer"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
               markeredgecolor="red", markeredgewidth=2, markersize=10, label="exit layer"),
        Line2D([0], [0], color="black", lw=1.0, alpha=0.6, label="flight path"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle(
        f"Representative multi-altitude exploration trajectory (held-out map seed {seed}, "
        f"final coverage {final_cov*100:.1f}%); 1 grid cell = 1 m",
        fontsize=11,
    )
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_base}.png", dpi=200, bbox_inches="tight")
    print(f"[traj] wrote {out_base}.pdf / .png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vecnorm", required=True)
    ap.add_argument("--seed", type=int, default=207, help="held-out eval map seed (>=200)")
    ap.add_argument("--deterministic", action="store_true",
                    help="Use argmax actions. NOT recommended: the policy was trained/evaluated with "
                         "stochastic sampling and its deterministic action mean is biased/degenerate "
                         "(see diagnose_eval.py). Default uses stochastic sampling to match the 91.90%% eval.")
    ap.add_argument("--rng-seed", type=int, default=0, help="torch/numpy seed -> reproducible stochastic rollout")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    torch.manual_seed(args.rng_seed)
    np.random.seed(args.rng_seed)

    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from env_3d import MAX_STEPS, AltitudeActiveSLAMEnv
    from transfer_config import ALTITUDE_LEVELS

    env = DummyVecEnv([lambda: AltitudeActiveSLAMEnv(map_seed=args.seed, difficulty="hard")])
    env = VecNormalize.load(args.vecnorm, env)
    env.training = False
    env.norm_reward = False
    raw = env.venv.envs[0]

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
        "observation_space": env.observation_space,
        "action_space": env.action_space,
    }
    model = RecurrentPPO.load(args.model, env=env, device="cpu", custom_objects=custom_objects)
    print(f"[traj] model loaded; MAX_STEPS={MAX_STEPS}; seed={args.seed}; deterministic={args.deterministic}")

    log, snap, final_cov = rollout(model, env, raw, MAX_STEPS, args.deterministic)
    alt_visits = [sum(1 for r in log if r["alt_idx"] == a) for a in range(len(ALTITUDE_LEVELS))]
    print(f"[traj] steps={len(log)}  final_cov={final_cov*100:.1f}%  per-altitude steps={alt_visits}")

    out_dir = Path(args.model).resolve().parent.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = args.out or str(out_dir / "trajectory_altitudes")

    # dump per-step log for reproducibility
    log_csv = Path(out_base).with_name(Path(out_base).name + "_log.csv")
    with log_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "pos_x", "pos_y", "alt_idx", "coverage", "cov_trace"])
        w.writeheader()
        w.writerows(log)
    print(f"[traj] wrote {log_csv}")

    make_figure(log, snap, final_cov, ALTITUDE_LEVELS, out_base, args.seed)


if __name__ == "__main__":
    main()
