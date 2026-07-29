"""Pairwise statistical significance tests (Wilcoxon signed-rank) between RL and each baseline.

For each (method, metric), we look up the per-episode final value from the corresponding
{method}_ep*.csv file, pair it with the RL agent's value on the same held-out eval seed,
and run a Wilcoxon test.

Writes summary/statistical_tests.md with p-values and effect sizes (Cliff's delta).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

LIGHTWEIGHT_DIR = Path(__file__).resolve().parent
SEED_TAGS = ("seed42", "seed100", "seed200")
VANILLA_TAGS = ("vanilla_s42", "vanilla_s100", "vanilla_s200")
SHAPE_DELTA_TAGS = ("shape_delta", "shape_delta_s100", "shape_delta_s200")
SHAPE_DELTA_MS600_TAGS = ("shape_delta_800k_maxsteps600", "shape_delta_ms600_s100", "shape_delta_ms600_s200")
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
    s_a = np.sort(a)
    ranks_a = np.searchsorted(s_a, b, side="right") - np.searchsorted(s_a, b, side="left") / 2
    # Count pairs: #(a<b), #(a>b)
    gt = 0
    lt = 0
    for x in a:
        gt += np.sum(b > x)
        lt += np.sum(b < x)
    return (gt - lt) / (n_a * n_b)


def main():
    results = {}
    for seed_tag in SEED_TAGS:
        csv_dir = LIGHTWEIGHT_DIR / f"results_3d_{seed_tag}" / "csv"
        if not csv_dir.exists():
            continue
        rl_cov = _final_values(csv_dir, "rl_agent", "coverage")
        rl_trace = _final_values(csv_dir, "rl_agent", "cov_trace")
        if len(rl_cov) == 0:
            continue
        results[seed_tag] = {"rl_cov": rl_cov, "rl_trace": rl_trace, "baselines": {}}
        for bl in BASELINE_METHODS:
            bl_cov = _final_values(csv_dir, bl, "coverage")
            bl_trace = _final_values(csv_dir, bl, "cov_trace")
            # Align lengths (pairing assumes same eval-seed order — both come from identical LIGHTWEIGHT_EVAL_SEEDS)
            n = min(len(rl_cov), len(bl_cov))
            if n < 5:
                continue
            res_cov = stats.wilcoxon(rl_cov[:n], bl_cov[:n], alternative="greater")
            res_trace = stats.wilcoxon(bl_trace[:n], rl_trace[:n], alternative="greater")
            # For trace, lower is better so we test baseline > rl (i.e. baseline is worse)
            results[seed_tag]["baselines"][bl] = {
                "n": n,
                "cov_rl_mean": float(np.mean(rl_cov[:n])),
                "cov_bl_mean": float(np.mean(bl_cov[:n])),
                "cov_p": float(res_cov.pvalue),
                "cov_cliffs_delta": float(_cliffs_delta(bl_cov[:n], rl_cov[:n])),
                "trace_rl_mean": float(np.mean(rl_trace[:n])),
                "trace_bl_mean": float(np.mean(bl_trace[:n])),
                "trace_p": float(res_trace.pvalue),
                "trace_cliffs_delta": float(_cliffs_delta(rl_trace[:n], bl_trace[:n])),
            }

    out_dir = LIGHTWEIGHT_DIR / "results_3d_seed42" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "statistical_tests.json"
    json_path.write_text(json.dumps(results, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))

    # Pooled across seeds: concatenate per-seed paired observations
    pooled = {}
    for bl in BASELINE_METHODS:
        cov_pairs_rl, cov_pairs_bl, trace_pairs_rl, trace_pairs_bl = [], [], [], []
        for seed_tag, rec in results.items():
            csv_dir = LIGHTWEIGHT_DIR / f"results_3d_{seed_tag}" / "csv"
            rl_c = _final_values(csv_dir, "rl_agent", "coverage")
            bl_c = _final_values(csv_dir, bl, "coverage")
            rl_t = _final_values(csv_dir, "rl_agent", "cov_trace")
            bl_t = _final_values(csv_dir, bl, "cov_trace")
            n = min(len(rl_c), len(bl_c), len(rl_t), len(bl_t))
            if n == 0:
                continue
            cov_pairs_rl.extend(rl_c[:n]); cov_pairs_bl.extend(bl_c[:n])
            trace_pairs_rl.extend(rl_t[:n]); trace_pairs_bl.extend(bl_t[:n])
        if not cov_pairs_rl:
            continue
        rl_c = np.array(cov_pairs_rl); bl_c = np.array(cov_pairs_bl)
        rl_t = np.array(trace_pairs_rl); bl_t = np.array(trace_pairs_bl)
        res_cov = stats.wilcoxon(rl_c, bl_c, alternative="greater")
        res_trace = stats.wilcoxon(bl_t, rl_t, alternative="greater")
        pooled[bl] = {
            "n": int(len(rl_c)),
            "cov_rl_mean": float(rl_c.mean()), "cov_bl_mean": float(bl_c.mean()),
            "cov_p": float(res_cov.pvalue), "cov_cliffs_delta": float(_cliffs_delta(bl_c, rl_c)),
            "trace_rl_mean": float(rl_t.mean()), "trace_bl_mean": float(bl_t.mean()),
            "trace_p": float(res_trace.pvalue), "trace_cliffs_delta": float(_cliffs_delta(rl_t, bl_t)),
        }
    results["pooled_3seeds"] = {"baselines": pooled}

    # Pooled RL vs Vanilla PPO (learned baseline)
    vanilla_pooled = None
    cov_rl, cov_v, trace_rl, trace_v = [], [], [], []
    for seed_tag, van_tag in zip(SEED_TAGS, VANILLA_TAGS):
        main_csv = LIGHTWEIGHT_DIR / f"results_3d_{seed_tag}" / "csv"
        van_csv = LIGHTWEIGHT_DIR / f"results_3d_{van_tag}" / "csv"
        if not (main_csv.exists() and van_csv.exists()):
            continue
        rc = _final_values(main_csv, "rl_agent", "coverage")
        vc = _final_values(van_csv, "rl_agent", "coverage")
        rt = _final_values(main_csv, "rl_agent", "cov_trace")
        vt = _final_values(van_csv, "rl_agent", "cov_trace")
        n = min(len(rc), len(vc), len(rt), len(vt))
        if n == 0:
            continue
        cov_rl.extend(rc[:n]); cov_v.extend(vc[:n])
        trace_rl.extend(rt[:n]); trace_v.extend(vt[:n])
    if cov_rl:
        rc, vc = np.array(cov_rl), np.array(cov_v)
        rt, vt = np.array(trace_rl), np.array(trace_v)
        res_cov = stats.wilcoxon(rc, vc, alternative="greater")
        res_trace = stats.wilcoxon(vt, rt, alternative="greater")
        vanilla_pooled = {
            "n": int(len(rc)),
            "cov_rl_mean": float(rc.mean()), "cov_bl_mean": float(vc.mean()),
            "cov_p": float(res_cov.pvalue), "cov_cliffs_delta": float(_cliffs_delta(vc, rc)),
            "trace_rl_mean": float(rt.mean()), "trace_bl_mean": float(vt.mean()),
            "trace_p": float(res_trace.pvalue), "trace_cliffs_delta": float(_cliffs_delta(rt, vt)),
        }
        results["pooled_3seeds"]["baselines"]["vanilla_ppo"] = vanilla_pooled
        pooled["vanilla_ppo"] = vanilla_pooled

    # Vanilla PPO vs Greedy Info Gain (learned baseline beats hand-crafted)
    vanilla_vs_greedy = None
    cov_v, cov_g, trace_v, trace_g = [], [], [], []
    for seed_tag, van_tag in zip(SEED_TAGS, VANILLA_TAGS):
        main_csv = LIGHTWEIGHT_DIR / f"results_3d_{seed_tag}" / "csv"
        van_csv = LIGHTWEIGHT_DIR / f"results_3d_{van_tag}" / "csv"
        if not (main_csv.exists() and van_csv.exists()):
            continue
        vc = _final_values(van_csv, "rl_agent", "coverage")
        gc = _final_values(main_csv, "greedy_info_gain", "coverage")
        vt = _final_values(van_csv, "rl_agent", "cov_trace")
        gt = _final_values(main_csv, "greedy_info_gain", "cov_trace")
        n = min(len(vc), len(gc), len(vt), len(gt))
        if n == 0:
            continue
        cov_v.extend(vc[:n]); cov_g.extend(gc[:n])
        trace_v.extend(vt[:n]); trace_g.extend(gt[:n])
    if cov_v:
        vc, gc = np.array(cov_v), np.array(cov_g)
        vt, gt = np.array(trace_v), np.array(trace_g)
        res_cov = stats.wilcoxon(vc, gc, alternative="greater")
        res_trace = stats.wilcoxon(gt, vt, alternative="greater")
        vanilla_vs_greedy = {
            "n": int(len(vc)),
            "cov_v_mean": float(vc.mean()), "cov_g_mean": float(gc.mean()),
            "cov_p": float(res_cov.pvalue), "cov_cliffs_delta": float(_cliffs_delta(gc, vc)),
            "trace_v_mean": float(vt.mean()), "trace_g_mean": float(gt.mean()),
            "trace_p": float(res_trace.pvalue), "trace_cliffs_delta": float(_cliffs_delta(vt, gt)),
        }
        results["vanilla_vs_greedy_pooled"] = vanilla_vs_greedy

    # Shape-Delta (new reward term winner) vs Vanilla PPO — the KEY comparison for the paper
    shape_delta_vs_vanilla = None
    cov_d, cov_v, trace_d, trace_v = [], [], [], []
    for van_tag, delta_tag in zip(VANILLA_TAGS, SHAPE_DELTA_TAGS):
        van_csv = LIGHTWEIGHT_DIR / f"results_3d_{van_tag}" / "csv"
        delta_csv = LIGHTWEIGHT_DIR / f"results_3d_{delta_tag}" / "csv"
        if not (van_csv.exists() and delta_csv.exists()):
            continue
        dc = _final_values(delta_csv, "rl_agent", "coverage")
        vc = _final_values(van_csv, "rl_agent", "coverage")
        dt = _final_values(delta_csv, "rl_agent", "cov_trace")
        vt = _final_values(van_csv, "rl_agent", "cov_trace")
        n = min(len(dc), len(vc), len(dt), len(vt))
        if n == 0:
            continue
        cov_d.extend(dc[:n]); cov_v.extend(vc[:n])
        trace_d.extend(dt[:n]); trace_v.extend(vt[:n])
    if cov_d:
        dc, vc = np.array(cov_d), np.array(cov_v)
        dt, vt = np.array(trace_d), np.array(trace_v)
        res_cov = stats.wilcoxon(dc, vc, alternative="greater")
        res_trace = stats.wilcoxon(vt, dt, alternative="greater")
        shape_delta_vs_vanilla = {
            "n": int(len(dc)),
            "cov_d_mean": float(dc.mean()), "cov_v_mean": float(vc.mean()),
            "cov_p": float(res_cov.pvalue), "cov_cliffs_delta": float(_cliffs_delta(vc, dc)),
            "trace_d_mean": float(dt.mean()), "trace_v_mean": float(vt.mean()),
            "trace_p": float(res_trace.pvalue), "trace_cliffs_delta": float(_cliffs_delta(dt, vt)),
        }
        results["shape_delta_vs_vanilla_pooled"] = shape_delta_vs_vanilla

    # MS600 (Δcov-trace reward + MAX_STEPS=600, our final recipe) vs Vanilla PPO — pooled
    ms600_vs_vanilla = None
    cov_m, cov_v, trace_m, trace_v = [], [], [], []
    for van_tag, ms_tag in zip(VANILLA_TAGS, SHAPE_DELTA_MS600_TAGS):
        van_csv = LIGHTWEIGHT_DIR / f"results_3d_{van_tag}" / "csv"
        ms_csv = LIGHTWEIGHT_DIR / f"results_3d_{ms_tag}" / "csv"
        if not (van_csv.exists() and ms_csv.exists()):
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
        ms600_vs_vanilla = {
            "n": int(len(mc)),
            "cov_m_mean": float(mc.mean()), "cov_v_mean": float(vc.mean()),
            "cov_p": float(res_cov.pvalue), "cov_cliffs_delta": float(_cliffs_delta(vc, mc)),
            "trace_m_mean": float(mt.mean()), "trace_v_mean": float(vt.mean()),
            "trace_p": float(res_trace.pvalue), "trace_cliffs_delta": float(_cliffs_delta(mt, vt)),
        }
        results["ms600_vs_vanilla_pooled"] = ms600_vs_vanilla

    # MS600 vs Shape-Delta (400 step) — shows MAX_STEPS=600 is the additional win
    ms600_vs_shape_delta = None
    cov_m, cov_d, trace_m, trace_d = [], [], [], []
    for delta_tag, ms_tag in zip(SHAPE_DELTA_TAGS, SHAPE_DELTA_MS600_TAGS):
        delta_csv = LIGHTWEIGHT_DIR / f"results_3d_{delta_tag}" / "csv"
        ms_csv = LIGHTWEIGHT_DIR / f"results_3d_{ms_tag}" / "csv"
        if not (delta_csv.exists() and ms_csv.exists()):
            continue
        mc = _final_values(ms_csv, "rl_agent", "coverage")
        dc = _final_values(delta_csv, "rl_agent", "coverage")
        mt = _final_values(ms_csv, "rl_agent", "cov_trace")
        dt = _final_values(delta_csv, "rl_agent", "cov_trace")
        n = min(len(mc), len(dc), len(mt), len(dt))
        if n == 0:
            continue
        cov_m.extend(mc[:n]); cov_d.extend(dc[:n])
        trace_m.extend(mt[:n]); trace_d.extend(dt[:n])
    if cov_m:
        mc, dc = np.array(cov_m), np.array(cov_d)
        mt, dt = np.array(trace_m), np.array(trace_d)
        res_cov = stats.wilcoxon(mc, dc, alternative="greater")
        res_trace = stats.wilcoxon(dt, mt, alternative="greater")
        ms600_vs_shape_delta = {
            "n": int(len(mc)),
            "cov_m_mean": float(mc.mean()), "cov_d_mean": float(dc.mean()),
            "cov_p": float(res_cov.pvalue), "cov_cliffs_delta": float(_cliffs_delta(dc, mc)),
            "trace_m_mean": float(mt.mean()), "trace_d_mean": float(dt.mean()),
            "trace_p": float(res_trace.pvalue), "trace_cliffs_delta": float(_cliffs_delta(mt, dt)),
        }
        results["ms600_vs_shape_delta_pooled"] = ms600_vs_shape_delta

    # MS600 vs Greedy Info Gain — final headline comparison for the paper
    ms600_vs_greedy = None
    cov_m, cov_g, trace_m, trace_g = [], [], [], []
    for seed_tag, ms_tag in zip(SEED_TAGS, SHAPE_DELTA_MS600_TAGS):
        main_csv = LIGHTWEIGHT_DIR / f"results_3d_{seed_tag}" / "csv"
        ms_csv = LIGHTWEIGHT_DIR / f"results_3d_{ms_tag}" / "csv"
        if not (main_csv.exists() and ms_csv.exists()):
            continue
        mc = _final_values(ms_csv, "rl_agent", "coverage")
        gc = _final_values(main_csv, "greedy_info_gain", "coverage")
        mt = _final_values(ms_csv, "rl_agent", "cov_trace")
        gt = _final_values(main_csv, "greedy_info_gain", "cov_trace")
        n = min(len(mc), len(gc), len(mt), len(gt))
        if n == 0:
            continue
        cov_m.extend(mc[:n]); cov_g.extend(gc[:n])
        trace_m.extend(mt[:n]); trace_g.extend(gt[:n])
    if cov_m:
        mc, gc = np.array(cov_m), np.array(cov_g)
        mt, gt = np.array(trace_m), np.array(trace_g)
        res_cov = stats.wilcoxon(mc, gc, alternative="greater")
        res_trace = stats.wilcoxon(gt, mt, alternative="greater")
        ms600_vs_greedy = {
            "n": int(len(mc)),
            "cov_m_mean": float(mc.mean()), "cov_g_mean": float(gc.mean()),
            "cov_p": float(res_cov.pvalue), "cov_cliffs_delta": float(_cliffs_delta(gc, mc)),
            "trace_m_mean": float(mt.mean()), "trace_g_mean": float(gt.mean()),
            "trace_p": float(res_trace.pvalue), "trace_cliffs_delta": float(_cliffs_delta(mt, gt)),
        }
        results["ms600_vs_greedy_pooled"] = ms600_vs_greedy

    md_path = out_dir / "statistical_tests.md"
    with md_path.open("w") as f:
        f.write("# Wilcoxon signed-rank tests (RL vs each baseline)\n\n")
        f.write("Hypothesis: RL dominates on coverage (higher) and on cov_trace (lower).\n")
        f.write("Cliff's delta ∈ [-1,1]: positive means RL favored; |δ|>0.47 is 'large' effect.\n\n")
        f.write("## POOLED across 3 seeds\n\n")
        f.write("| Baseline | n | Cov RL | Cov BL | Cov p | Cov δ | Trace RL | Trace BL | Trace p | Trace δ |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        def _star(p):
            return "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 5e-2 else ""))
        for bl, s in pooled.items():
            f.write(
                f"| {bl} | {s['n']} | {s['cov_rl_mean']*100:.2f} | {s['cov_bl_mean']*100:.2f} | "
                f"{s['cov_p']:.2e}{_star(s['cov_p'])} | {s['cov_cliffs_delta']:+.3f} | "
                f"{s['trace_rl_mean']:.3f} | {s['trace_bl_mean']:.3f} | "
                f"{s['trace_p']:.2e}{_star(s['trace_p'])} | {s['trace_cliffs_delta']:+.3f} |\n"
            )
        f.write("\n")
        if shape_delta_vs_vanilla:
            s = shape_delta_vs_vanilla
            f.write("## Shape-Delta (ours, proposed) vs Vanilla PPO, pooled\n\n")
            f.write("Hypothesis: the new W_COV_DELTA step-wise trace-reduction reward beats vanilla PPO.\n\n")
            f.write("| Comparison | n | Cov Shape-Delta | Cov Vanilla | Cov p | Cov \u03b4 | Trace Shape-Delta | Trace Vanilla | Trace p | Trace \u03b4 |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
            f.write(
                f"| Shape-Delta vs Vanilla | {s['n']} | {s['cov_d_mean']*100:.2f} | {s['cov_v_mean']*100:.2f} | "
                f"{s['cov_p']:.2e}{_star(s['cov_p'])} | {s['cov_cliffs_delta']:+.3f} | "
                f"{s['trace_d_mean']:.3f} | {s['trace_v_mean']:.3f} | "
                f"{s['trace_p']:.2e}{_star(s['trace_p'])} | {s['trace_cliffs_delta']:+.3f} |\n\n"
            )
        if vanilla_vs_greedy:
            s = vanilla_vs_greedy
            f.write("## Learned baseline (Vanilla PPO) vs hand-crafted best (Greedy Info Gain), pooled\n\n")
            f.write("Hypothesis: PPO on raw coverage + collision beats hand-crafted greedy.\n\n")
            f.write("| Comparison | n | Cov Vanilla | Cov Greedy | Cov p | Cov δ | Trace Vanilla | Trace Greedy | Trace p | Trace δ |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
            f.write(
                f"| Vanilla PPO vs Greedy | {s['n']} | {s['cov_v_mean']*100:.2f} | {s['cov_g_mean']*100:.2f} | "
                f"{s['cov_p']:.2e}{_star(s['cov_p'])} | {s['cov_cliffs_delta']:+.3f} | "
                f"{s['trace_v_mean']:.3f} | {s['trace_g_mean']:.3f} | "
                f"{s['trace_p']:.2e}{_star(s['trace_p'])} | {s['trace_cliffs_delta']:+.3f} |\n\n"
            )
        if ms600_vs_vanilla:
            s = ms600_vs_vanilla
            f.write("## MS600 (Δcov-trace + MAX_STEPS=600, final recipe) vs Vanilla PPO, pooled\n\n")
            f.write("Hypothesis: the final recipe (reward shaping + longer episode horizon) beats vanilla PPO.\n\n")
            f.write("| Comparison | n | Cov MS600 | Cov Vanilla | Cov p | Cov δ | Trace MS600 | Trace Vanilla | Trace p | Trace δ |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
            f.write(
                f"| MS600 vs Vanilla | {s['n']} | {s['cov_m_mean']*100:.2f} | {s['cov_v_mean']*100:.2f} | "
                f"{s['cov_p']:.2e}{_star(s['cov_p'])} | {s['cov_cliffs_delta']:+.3f} | "
                f"{s['trace_m_mean']:.3f} | {s['trace_v_mean']:.3f} | "
                f"{s['trace_p']:.2e}{_star(s['trace_p'])} | {s['trace_cliffs_delta']:+.3f} |\n\n"
            )
        if ms600_vs_shape_delta:
            s = ms600_vs_shape_delta
            f.write("## MS600 vs Shape-Delta (ablate horizon), pooled\n\n")
            f.write("Hypothesis: extending episode horizon from 400 to 600 steps strictly improves the shape-delta policy.\n\n")
            f.write("| Comparison | n | Cov MS600 | Cov 400-step | Cov p | Cov δ | Trace MS600 | Trace 400-step | Trace p | Trace δ |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
            f.write(
                f"| MS600 vs 400-step | {s['n']} | {s['cov_m_mean']*100:.2f} | {s['cov_d_mean']*100:.2f} | "
                f"{s['cov_p']:.2e}{_star(s['cov_p'])} | {s['cov_cliffs_delta']:+.3f} | "
                f"{s['trace_m_mean']:.3f} | {s['trace_d_mean']:.3f} | "
                f"{s['trace_p']:.2e}{_star(s['trace_p'])} | {s['trace_cliffs_delta']:+.3f} |\n\n"
            )
        if ms600_vs_greedy:
            s = ms600_vs_greedy
            f.write("## MS600 vs Greedy Info Gain (hand-crafted best), pooled\n\n")
            f.write("Hypothesis: our learned policy beats the strongest classical active-SLAM planner.\n\n")
            f.write("| Comparison | n | Cov MS600 | Cov Greedy | Cov p | Cov δ | Trace MS600 | Trace Greedy | Trace p | Trace δ |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
            f.write(
                f"| MS600 vs Greedy | {s['n']} | {s['cov_m_mean']*100:.2f} | {s['cov_g_mean']*100:.2f} | "
                f"{s['cov_p']:.2e}{_star(s['cov_p'])} | {s['cov_cliffs_delta']:+.3f} | "
                f"{s['trace_m_mean']:.3f} | {s['trace_g_mean']:.3f} | "
                f"{s['trace_p']:.2e}{_star(s['trace_p'])} | {s['trace_cliffs_delta']:+.3f} |\n\n"
            )
        f.write("## Per-seed breakdown\n\n")
        for seed_tag, rec in results.items():
            if seed_tag in (
                "pooled_3seeds",
                "vanilla_vs_greedy_pooled",
                "shape_delta_vs_vanilla_pooled",
                "ms600_vs_vanilla_pooled",
                "ms600_vs_shape_delta_pooled",
                "ms600_vs_greedy_pooled",
            ):
                continue
            f.write(f"### {seed_tag}\n\n")
            f.write("| Baseline | n | Cov RL | Cov BL | Cov p | Cov δ | Trace RL | Trace BL | Trace p | Trace δ |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
            for bl, s in rec["baselines"].items():
                f.write(
                    f"| {bl} | {s['n']} | {s['cov_rl_mean']*100:.2f} | {s['cov_bl_mean']*100:.2f} | "
                    f"{s['cov_p']:.2e}{_star(s['cov_p'])} | {s['cov_cliffs_delta']:+.3f} | "
                    f"{s['trace_rl_mean']:.3f} | {s['trace_bl_mean']:.3f} | "
                    f"{s['trace_p']:.2e}{_star(s['trace_p'])} | {s['trace_cliffs_delta']:+.3f} |\n"
                )
            f.write("\n")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print("\n=== Preview ===\n")
    print(md_path.read_text())


if __name__ == "__main__":
    main()
