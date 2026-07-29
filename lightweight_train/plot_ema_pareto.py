"""Transfer-diagnostic figure: EMA α vs (motion safety, exploration preservation).

Visualizes the sim-to-sim transfer trade-off discovered in Tier 1b (ema_sweep.py).
Lower α = more smoothing = safer for RTAB-Map but less exploration.
α=0.5 sits at the Pareto sweet-spot: motion metrics drop into GREEN while
coverage only degrades from 90% to 80%.

Output: lightweight_train/results_3d_shape_delta_800k_maxsteps600/plots/ema_pareto.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIGHTWEIGHT_DIR = Path(__file__).resolve().parent
OUT_DIR = LIGHTWEIGHT_DIR / "results_3d_shape_delta_800k_maxsteps600" / "plots"

# Aggregated EMA sweep results across 3 MS600 seeds × 3 episodes
# (from ema_sweep.py output).
SWEEP = [
    # alpha, xy_flip_rate, jitter_rms, coverage_mean, coverage_std
    (1.00, 0.40, 0.99, 0.904, 0.025),
    (0.50, 0.25, 0.40, 0.800, 0.053),
    (0.30, 0.11, 0.22, 0.638, 0.098),
    (0.20, 0.07, 0.14, 0.521, 0.128),
]

# Safety thresholds from action_inspect.py (calibrated to Gazebo env defenses).
XY_FLIP_GREEN = 0.15   # GREEN if xy_flip <= 0.15
XY_FLIP_YELLOW = 0.30  # YELLOW between; RED above
JITTER_GREEN = 0.35
JITTER_YELLOW = 0.70


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    alphas = [r[0] for r in SWEEP]
    xy_flip = [r[1] for r in SWEEP]
    jitter = [r[2] for r in SWEEP]
    cov = [r[3] for r in SWEEP]
    cov_std = [r[4] for r in SWEEP]

    fig, (ax_motion, ax_cov) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # Left: motion safety metrics vs α
    ax_motion.plot(alphas, xy_flip, "-o", color="tab:red",
                   lw=2, ms=7, label="xy_flip_rate")
    ax_motion.plot(alphas, jitter, "-s", color="tab:purple",
                   lw=2, ms=7, label="jitter_rms")
    ax_motion.axhspan(0, XY_FLIP_GREEN, alpha=0.10, color="green", label="GREEN (flip rate)")
    ax_motion.axhspan(XY_FLIP_GREEN, XY_FLIP_YELLOW, alpha=0.08, color="gold")
    ax_motion.axhspan(XY_FLIP_YELLOW, 1.2, alpha=0.08, color="red")
    ax_motion.set_xlabel("EMA α (1.0 = no filter, 0.0 = infinite smoothing)")
    ax_motion.set_ylabel("Motion metric (lower = safer)")
    ax_motion.set_title("Motion safety — RTAB-Map compatibility")
    ax_motion.invert_xaxis()
    ax_motion.set_ylim(0, 1.1)
    ax_motion.grid(True, alpha=0.3)
    ax_motion.legend(loc="upper right", fontsize=9)

    # Right: coverage vs α
    cov_pct = [c * 100 for c in cov]
    cov_std_pct = [s * 100 for s in cov_std]
    ax_cov.errorbar(alphas, cov_pct, yerr=cov_std_pct,
                    fmt="-o", color="tab:blue", lw=2, ms=7,
                    capsize=4, label="Coverage (n=3 seeds × 3 eps)")
    ax_cov.axhline(60, color="tab:gray", linestyle="--", lw=1.2,
                   label="Coverage floor (60%)")
    ax_cov.set_xlabel("EMA α")
    ax_cov.set_ylabel("Final coverage (%)")
    ax_cov.set_title("Exploration preservation under action smoothing")
    ax_cov.invert_xaxis()
    ax_cov.set_ylim(40, 100)
    ax_cov.grid(True, alpha=0.3)
    ax_cov.legend(loc="lower right", fontsize=9)

    # Annotate Pareto sweet spot
    for ax in (ax_motion, ax_cov):
        ax.axvline(0.5, color="black", linestyle=":", lw=1.2, alpha=0.6)
    ax_cov.annotate("α=0.5 (Pareto)\nmotion GREEN\ncoverage 80%",
                    xy=(0.5, 80), xytext=(0.35, 90),
                    fontsize=9, ha="center",
                    arrowprops=dict(arrowstyle="->", color="black", alpha=0.6))

    fig.suptitle("Transfer diagnostic: action-EMA α sweep", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ema_pareto.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "ema_pareto.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'ema_pareto.pdf'}")


if __name__ == "__main__":
    main()
