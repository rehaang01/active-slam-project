"""Time-series comparison plots: coverage(step) and cov_trace(step) with shaded std bands.

For each method (RL, Greedy, Vanilla-PPO if present), aggregate per-episode CSVs
into mean ± std curves over step, then plot them side-by-side. Standard
ICRA/IROS figure format for exploration/SLAM comparisons.

Inputs:
    results_3d_seed42/csv/{rl_agent,greedy_info_gain,random_walk,...}_ep*.csv
    results_3d_vanilla_s42/csv/rl_agent_ep*.csv  (if present)

Outputs:
    results_3d_seed42/plots/timeseries_coverage.{pdf,png}
    results_3d_seed42/plots/timeseries_covtrace.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIGHTWEIGHT_DIR = Path(__file__).resolve().parent
OUT_DIR = LIGHTWEIGHT_DIR / "results_3d_seed42" / "plots"


def _stack_episodes(csv_dir: Path, prefix: str, col: str) -> np.ndarray | None:
    files = sorted(csv_dir.glob(f"{prefix}_ep*.csv"))
    files = [f for f in files if not (f.stem.endswith("_avg") or f.stem.endswith("_std"))]
    if not files:
        return None
    series = []
    for f in files:
        df = pd.read_csv(f)
        if col not in df.columns or len(df) == 0:
            continue
        series.append(df[col].to_numpy())
    if not series:
        return None
    L = min(len(s) for s in series)
    return np.stack([s[:L] for s in series], axis=0)


def _plot_ts(ax, steps, mean, std, color, label, ls="-", marker=None):
    # Distinct linestyle + sparse marker keeps curves separable in grayscale print.
    ax.plot(steps, mean, color=color, linestyle=ls, marker=marker, markevery=60, ms=5,
            lw=2, label=label)
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms600", action="store_true", help="Use MS600 (final recipe) RL CSVs")
    args = ap.parse_args()

    if args.ms600:
        rl_label = "RL Ours + Δcov-trace + MS600 (seed42)"
        rl_dir = LIGHTWEIGHT_DIR / "results_3d_shape_delta_800k_maxsteps600" / "csv"
        out_dir = LIGHTWEIGHT_DIR / "results_3d_shape_delta_800k_maxsteps600" / "plots"
    else:
        rl_label = "RL (Ours, seed42)"
        rl_dir = LIGHTWEIGHT_DIR / "results_3d_seed42" / "csv"
        out_dir = OUT_DIR

    out_dir.mkdir(parents=True, exist_ok=True)
    # Baselines CSVs are seed-independent (same eval seeds, deterministic methods).
    baseline_dir = LIGHTWEIGHT_DIR / "results_3d_seed42" / "csv"
    vanilla_dir = LIGHTWEIGHT_DIR / "results_3d_vanilla_s42" / "csv"

    METHODS = [
        ("rl_agent", rl_dir, rl_label, "tab:blue", "-", "o"),
        ("greedy_info_gain", baseline_dir, "Greedy Info Gain", "tab:orange", "--", "s"),
        ("nearest_frontier", baseline_dir, "Nearest Frontier", "tab:green", ":", "^"),
        ("rrt_explorer", baseline_dir, "RRT Explorer", "tab:red", "-.", "D"),
    ]
    if (vanilla_dir / "rl_agent_ep0.csv").exists():
        METHODS.append(("rl_agent", vanilla_dir, "Vanilla PPO (seed42)", "tab:purple", (0, (3, 1, 1, 1)), "v"))

    for metric_name, ylabel, scale, fname, is_pct in [
        ("coverage", "Coverage (%)", 100.0, "timeseries_coverage", True),
        ("cov_trace", "Covariance trace (dimensionless, lower is better)", 1.0, "timeseries_covtrace", False),
    ]:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        for prefix, csv_dir, label, color, ls, marker in METHODS:
            data = _stack_episodes(csv_dir, prefix, metric_name)
            if data is None:
                continue
            steps = np.arange(1, data.shape[1] + 1)
            mean = data.mean(axis=0) * scale
            std = data.std(axis=0) * scale
            _plot_ts(ax, steps, mean, std, color, f"{label} (n={data.shape[0]})", ls, marker)
        ax.set_xlabel("Episode step (number; 1 step ≈ 2 s of flight)")
        ax.set_ylabel(ylabel)

        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"{fname}.pdf")
        fig.savefig(out_dir / f"{fname}.png", dpi=150)
        plt.close(fig)
        print(f"Wrote {out_dir / (fname + '.pdf')}")


if __name__ == "__main__":
    main()
