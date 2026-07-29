"""Plot learning curves: coverage & cov_trace vs training steps, aggregated over seeds.

Usage:
    python3 plot_training_curves.py              # default (results_3d_seed{42,100,200})
    python3 plot_training_curves.py --ms600      # final recipe (results_3d_shape_delta_*)

Reads each seed's summary/checkpoint_selection.json, then plots mean ± std per metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIGHTWEIGHT_DIR = Path(__file__).resolve().parent

PRESETS = {
    "default": {
        "run_dirs": {
            42:  "results_3d_seed42",
            100: "results_3d_seed100",
            200: "results_3d_seed200",
        },
        "out_dir": "results_3d_seed42/plots",
        "baseline_dir": "results_3d_seed42",
        "label": "RL (Ours, n=3 seeds)",
    },
    "ms600": {
        "run_dirs": {
            42:  "results_3d_shape_delta_800k_maxsteps600",
            100: "results_3d_shape_delta_ms600_s100",
            200: "results_3d_shape_delta_ms600_s200",
        },
        "out_dir": "results_3d_shape_delta_800k_maxsteps600/plots",
        "baseline_dir": "results_3d_seed42",  # baselines are seed-independent
        "label": "RL Ours + Δcov-trace + MAX_STEPS=600 (n=3 seeds)",
    },
}


def load_curves(run_dirs):
    curves = {}
    for s, rundir in run_dirs.items():
        path = LIGHTWEIGHT_DIR / rundir / "summary" / "checkpoint_selection.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        curves[s] = d.get("checkpoints", [])
    return curves


def load_baseline_curves(baseline_dir):
    path = LIGHTWEIGHT_DIR / baseline_dir / "summary" / "baseline_summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _collect(curves, metric):
    # Intersect step sets across seeds — not all seeds ran to the same step count.
    # (MS600: s42 went to 780k, s100/s200 stopped at 360k.) Plot only the range
    # where all seeds have data so mean±std is always over n=len(seeds).
    seeds = list(curves.keys())
    step_sets = [set(c["step"] for c in curves[s]) for s in seeds]
    common = sorted(set.intersection(*step_sets))
    vals = []
    for s in seeds:
        by_step = {c["step"]: c[metric] for c in curves[s]}
        vals.append([by_step[st] for st in common])
    vals = np.array(vals)
    return np.array(common), vals.mean(axis=0), vals.std(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms600", action="store_true", help="Use the Δcov-trace + MAX_STEPS=600 final recipe runs")
    args = ap.parse_args()
    preset = PRESETS["ms600" if args.ms600 else "default"]
    out_dir = LIGHTWEIGHT_DIR / preset["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = load_curves(preset["run_dirs"])
    baselines = load_baseline_curves(preset["baseline_dir"])
    label = preset["label"]
    greedy_cov = baselines.get("Greedy Info Gain", {}).get("final_coverage", None)
    greedy_trace = baselines.get("Greedy Info Gain", {}).get("final_cov_trace", None)

    # Figure 1: coverage vs steps
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    steps, mean, std = _collect(curves, "final_coverage")
    ax.plot(steps, mean * 100, color="tab:blue", linestyle="-", marker="o", label=label, lw=2, ms=5)
    ax.fill_between(steps, (mean - std) * 100, (mean + std) * 100, color="tab:blue", alpha=0.2)
    if greedy_cov is not None:
        ax.axhline(greedy_cov * 100, color="tab:orange", linestyle="--", lw=1.8, label=f"Greedy Info Gain baseline ({greedy_cov*100:.1f}%)")
    ax.set_xlabel("Training steps (number)")
    ax.set_ylabel("Final coverage (%)")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "training_curve_coverage.pdf")
    fig.savefig(out_dir / "training_curve_coverage.png", dpi=150)
    print("Wrote coverage curve")

    # Figure 2: cov_trace vs steps
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    steps, mean, std = _collect(curves, "final_cov_trace")
    ax.plot(steps, mean, color="tab:green", linestyle="-", marker="o", label="RL (Ours, n=3 seeds)", lw=2, ms=5)
    ax.fill_between(steps, mean - std, mean + std, color="tab:green", alpha=0.2)
    if greedy_trace is not None:
        ax.axhline(greedy_trace, color="tab:orange", linestyle="--", lw=1.8, label=f"Greedy Info Gain baseline ({greedy_trace:.2f})")
    ax.set_xlabel("Training steps (number)")
    ax.set_ylabel("Final covariance trace (dimensionless, lower is better)")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "training_curve_covtrace.pdf")
    fig.savefig(out_dir / "training_curve_covtrace.png", dpi=150)
    print("Wrote cov_trace curve")

    # Figure 3: per-altitude coverage curves.
    # Distinct linestyle + marker per altitude so the four curves remain distinguishable
    # in grayscale print (solid/dashed/dotted/dash-dot + circle/square/triangle/diamond).
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    # (colour, linestyle, marker, altitude in metres)
    styles = [
        ("tab:blue", "-", "o", 1),
        ("tab:green", "--", "s", 2),
        ("tab:red", ":", "^", 3),
        ("tab:purple", "-.", "D", 4),
    ]
    for alt, (c, ls, mk, metres) in enumerate(styles):
        steps, mean, std = _collect(curves, f"final_cov_alt{alt}")
        ax.plot(steps, mean * 100, color=c, linestyle=ls, marker=mk, markevery=2,
                label=f"Altitude {metres} m (RL)", lw=1.8, ms=5)
        ax.fill_between(steps, (mean - std) * 100, (mean + std) * 100, color=c, alpha=0.15)
    if baselines:
        g = baselines.get("Greedy Info Gain", {})
        for alt, (c, ls, mk, metres) in enumerate(styles):
            v = g.get(f"final_cov_alt{alt}")
            if v is not None:
                ax.axhline(v * 100, color=c, linestyle=(0, (1, 1)), lw=1.0, alpha=0.7)
        ax.plot([], [], color="black", linestyle=(0, (1, 1)), lw=1.0, label="Greedy (per-altitude final)")
    ax.set_xlabel("Training steps (number)")
    ax.set_ylabel("Per-altitude coverage (%)")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curve_per_altitude.pdf")
    fig.savefig(out_dir / "training_curve_per_altitude.png", dpi=150)
    print("Wrote per-altitude curves")


if __name__ == "__main__":
    main()
