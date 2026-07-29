"""Pooled Wilcoxon signed-rank tests of MS600 (final headline recipe, 91.90%)
vs every classical baseline AND vs Vanilla PPO.

Writes summary/statistical_tests_ms600.json for the paper Table III.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

LIGHTWEIGHT_DIR = Path(__file__).resolve().parent
SEED_TAGS = ("seed42", "seed100", "seed200")
MS600_TAGS = ("shape_delta_800k_maxsteps600", "shape_delta_ms600_s100", "shape_delta_ms600_s200")
VANILLA_TAGS = ("vanilla_s42", "vanilla_s100", "vanilla_s200")
BASELINE_METHODS = (
    "random_walk",
    "nearest_frontier",
    "spiral",
    "potential_field",
    "greedy_info_gain",
    "rrt_explorer",
)


def _final_values(csv_dir: Path, prefix: str, metric: str):
    vals = []
    files = sorted(csv_dir.glob(f"{prefix}_ep*.csv"))
    for f in files:
        if f.stem.endswith("_avg") or f.stem.endswith("_std"):
            continue
        df = pd.read_csv(f)
        if metric in df.columns and len(df):
            vals.append(float(df[metric].iloc[-1]))
    return np.array(vals)


def _cliffs_delta(a, b):
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return float("nan")
    gt = sum(int(np.sum(b > x)) for x in a)
    lt = sum(int(np.sum(b < x)) for x in a)
    return (gt - lt) / (n_a * n_b)


def main():
    pooled = {}
    for bl in BASELINE_METHODS:
        cov_m, cov_b, trace_m, trace_b = [], [], [], []
        for seed_tag, ms_tag in zip(SEED_TAGS, MS600_TAGS):
            base_csv = LIGHTWEIGHT_DIR / f"results_3d_{seed_tag}" / "csv"  # baselines live here
            ms_csv = LIGHTWEIGHT_DIR / f"results_3d_{ms_tag}" / "csv"
            if not (base_csv.exists() and ms_csv.exists()):
                continue
            mc = _final_values(ms_csv, "rl_agent", "coverage")
            bc = _final_values(base_csv, bl, "coverage")
            mt = _final_values(ms_csv, "rl_agent", "cov_trace")
            bt = _final_values(base_csv, bl, "cov_trace")
            n = min(len(mc), len(bc), len(mt), len(bt))
            if n == 0:
                continue
            cov_m.extend(mc[:n]); cov_b.extend(bc[:n])
            trace_m.extend(mt[:n]); trace_b.extend(bt[:n])
        if not cov_m:
            continue
        mc, bc = np.array(cov_m), np.array(cov_b)
        mt, bt = np.array(trace_m), np.array(trace_b)
        # Coverage: H1 RL > BL
        res_cov = stats.wilcoxon(mc, bc, alternative="greater")
        # Trace: lower is better, H1 BL > RL
        res_trace = stats.wilcoxon(bt, mt, alternative="greater")
        pooled[bl] = {
            "n": int(len(mc)),
            "cov_rl_mean": float(mc.mean()), "cov_bl_mean": float(bc.mean()),
            "cov_p": float(res_cov.pvalue), "cov_cliffs_delta": float(_cliffs_delta(bc, mc)),
            "trace_rl_mean": float(mt.mean()), "trace_bl_mean": float(bt.mean()),
            "trace_p": float(res_trace.pvalue), "trace_cliffs_delta": float(_cliffs_delta(mt, bt)),
        }

    # MS600 vs Vanilla PPO
    cov_m, cov_v, trace_m, trace_v = [], [], [], []
    for ms_tag, van_tag in zip(MS600_TAGS, VANILLA_TAGS):
        ms_csv = LIGHTWEIGHT_DIR / f"results_3d_{ms_tag}" / "csv"
        van_csv = LIGHTWEIGHT_DIR / f"results_3d_{van_tag}" / "csv"
        if not (ms_csv.exists() and van_csv.exists()):
            continue
        mc = _final_values(ms_csv, "rl_agent", "coverage")
        vc = _final_values(van_csv, "rl_agent", "coverage")
        mt = _final_values(ms_csv, "rl_agent", "cov_trace")
        vt = _final_values(van_csv, "rl_agent", "cov_trace")
        n = min(len(mc), len(vc), len(mt), len(vt))
        if n == 0:
            continue
        cov_m.extend(mc[:n]); cov_v.extend(vc[:n])
        trace_m.extend(mt[:n]); trace_v.extend(vt[:n])
    if cov_m:
        mc, vc = np.array(cov_m), np.array(cov_v)
        mt, vt = np.array(trace_m), np.array(trace_v)
        res_cov = stats.wilcoxon(mc, vc, alternative="greater")
        res_trace = stats.wilcoxon(vt, mt, alternative="greater")
        pooled["vanilla_ppo"] = {
            "n": int(len(mc)),
            "cov_rl_mean": float(mc.mean()), "cov_bl_mean": float(vc.mean()),
            "cov_p": float(res_cov.pvalue), "cov_cliffs_delta": float(_cliffs_delta(vc, mc)),
            "trace_rl_mean": float(mt.mean()), "trace_bl_mean": float(vt.mean()),
            "trace_p": float(res_trace.pvalue), "trace_cliffs_delta": float(_cliffs_delta(mt, vt)),
        }

    out_dir = LIGHTWEIGHT_DIR / "results_3d_seed42" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "statistical_tests_ms600.json"
    json_path.write_text(json.dumps(pooled, indent=2))

    print("=" * 100)
    print("MS600 (91.90%) vs each baseline — POOLED across 3 seeds (n=150)")
    print("=" * 100)
    print(f"{'Comparison':<26} {'CovRL':>7} {'CovBL':>7} {'Cov p':>11} {'Cov delta':>10} {'TrRL':>6} {'TrBL':>6} {'Tr p':>11} {'Tr delta':>10}")
    print("-" * 100)
    for k, s in pooled.items():
        print(
            f"{k:<26} "
            f"{s['cov_rl_mean']*100:>7.2f} "
            f"{s['cov_bl_mean']*100:>7.2f} "
            f"{s['cov_p']:>11.2e} "
            f"{s['cov_cliffs_delta']:>+10.3f} "
            f"{s['trace_rl_mean']:>6.3f} "
            f"{s['trace_bl_mean']:>6.3f} "
            f"{s['trace_p']:>11.2e} "
            f"{s['trace_cliffs_delta']:>+10.3f}"
        )
    print(f"\nWrote {json_path}")


if __name__ == "__main__":
    main()
