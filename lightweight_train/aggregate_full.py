"""Publication-ready table: baselines + 3-seed RL (mean±std) + ablations.

Reads:
  results_3d_seed{42,100,200}/summary/selected_rl_summary.json -> RL mean±std
  results_3d_seed42/summary/baseline_summary.json              -> baselines
  results_3d_abl_noloop/summary/selected_rl_summary.json       -> ablation (no loop-closure reward)
  results_3d_abl_nobreadth/summary/selected_rl_summary.json    -> ablation (no breadth bonus)

Emits summary/full_table.{md,csv,json} with RL main row aggregated across seeds and
ablation rows as single-seed points (highlighted as such).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

LIGHTWEIGHT_DIR = Path(__file__).resolve().parent
SEEDS = (42, 100, 200)
VANILLA_TAGS = ("vanilla_s42", "vanilla_s100", "vanilla_s200")
SHAPE_DELTA_TAGS = ("shape_delta", "shape_delta_s100", "shape_delta_s200")
SHAPE_DELTA_MS600_TAGS = ("shape_delta_800k_maxsteps600", "shape_delta_ms600_s100", "shape_delta_ms600_s200")
ABLATIONS = (
    ("noloop", "RL w/o LoopClosure reward", ("abl_noloop", "abl_noloop_s100", "abl_noloop_s200")),
    ("nobreadth", "RL w/o BreadthBonus", ("abl_nobreadth", "abl_nobreadth_s100", "abl_nobreadth_s200")),
)
METRICS = (
    "final_coverage",
    "final_cov_trace",
    "final_cov_alt0",
    "final_cov_alt1",
    "final_cov_alt2",
    "final_cov_alt3",
    "final_loop_closures",
    "final_altitude_changes",
)


def _mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return (float("nan"), float("nan"))
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return (m, 0.0)
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return (m, math.sqrt(var))


def _load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    rl_per_seed = {
        s: _load_json(LIGHTWEIGHT_DIR / f"results_3d_seed{s}" / "summary" / "selected_rl_summary.json")
        for s in SEEDS
    }
    rl_per_seed = {s: d for s, d in rl_per_seed.items() if d}

    rl_agg = {m: _mean_std([rl_per_seed[s].get(m) for s in rl_per_seed]) for m in METRICS}

    baselines = _load_json(LIGHTWEIGHT_DIR / "results_3d_seed42" / "summary" / "baseline_summary.json")

    ablation_rows = []
    ablation_raw = {}
    for key, label, tags in ABLATIONS:
        per_seed_data = [_load_json(LIGHTWEIGHT_DIR / f"results_3d_{t}" / "summary" / "selected_rl_summary.json") for t in tags]
        per_seed_data = [d for d in per_seed_data if d]
        if not per_seed_data:
            continue
        ablation_raw[key] = per_seed_data
        agg = {m: _mean_std([d.get(m) for d in per_seed_data]) for m in METRICS}
        label_with_n = f"{label} (n={len(per_seed_data)})"
        ablation_rows.append((label_with_n, agg))

    vanilla_per_seed = [
        _load_json(LIGHTWEIGHT_DIR / f"results_3d_{t}" / "summary" / "selected_rl_summary.json")
        for t in VANILLA_TAGS
    ]
    vanilla_per_seed = [d for d in vanilla_per_seed if d]
    vanilla_agg = (
        {m: _mean_std([d.get(m) for d in vanilla_per_seed]) for m in METRICS}
        if vanilla_per_seed
        else None
    )

    shape_delta_per_seed = [
        _load_json(LIGHTWEIGHT_DIR / f"results_3d_{t}" / "summary" / "selected_rl_summary.json")
        for t in SHAPE_DELTA_TAGS
    ]
    shape_delta_per_seed = [d for d in shape_delta_per_seed if d]
    shape_delta_agg = (
        {m: _mean_std([d.get(m) for d in shape_delta_per_seed]) for m in METRICS}
        if shape_delta_per_seed
        else None
    )

    shape_delta_ms600_per_seed = [
        _load_json(LIGHTWEIGHT_DIR / f"results_3d_{t}" / "summary" / "selected_rl_summary.json")
        for t in SHAPE_DELTA_MS600_TAGS
    ]
    shape_delta_ms600_per_seed = [d for d in shape_delta_ms600_per_seed if d]
    shape_delta_ms600_agg = (
        {m: _mean_std([d.get(m) for d in shape_delta_ms600_per_seed]) for m in METRICS}
        if shape_delta_ms600_per_seed
        else None
    )

    rows = []
    for name, data in baselines.items():
        rows.append((name, {m: (data.get(m), 0.0) for m in METRICS}))
    if vanilla_agg is not None:
        rows.append((f"Vanilla PPO (learned baseline, n={len(vanilla_per_seed)})", vanilla_agg))
    rows.append((f"RL Ours (main, n={len(rl_per_seed)})", rl_agg))
    if shape_delta_agg is not None:
        rows.append((f"RL Ours + \u0394cov-trace reward (n={len(shape_delta_per_seed)})", shape_delta_agg))
    if shape_delta_ms600_agg is not None:
        rows.append((f"RL Ours + Δcov-trace + MAX_STEPS=600 (n={len(shape_delta_ms600_per_seed)})", shape_delta_ms600_agg))
    rows.extend(ablation_rows)

    out_dir = LIGHTWEIGHT_DIR / "results_3d_seed42" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Markdown
    md_path = out_dir / "full_table.md"
    with md_path.open("w") as f:
        f.write("| Method | Coverage | CovTrace | Alt0 | Alt1 | Alt2 | Alt3 | LoopClos | AltChg |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for name, data in rows:
            def fmt(metric, pct=False):
                m, s = data[metric]
                scale = 100.0 if pct else 1.0
                if s > 0:
                    return f"{m*scale:.2f} ± {s*scale:.2f}"
                return f"{m*scale:.2f}"
            f.write(
                f"| {name} | {fmt('final_coverage', pct=True)} | "
                f"{fmt('final_cov_trace')} | {fmt('final_cov_alt0', pct=True)} | "
                f"{fmt('final_cov_alt1', pct=True)} | {fmt('final_cov_alt2', pct=True)} | "
                f"{fmt('final_cov_alt3', pct=True)} | {fmt('final_loop_closures')} | "
                f"{fmt('final_altitude_changes')} |\n"
            )

    # CSV
    csv_path = out_dir / "full_table.csv"
    header = ["method"] + [f"{m}_mean" for m in METRICS] + [f"{m}_std" for m in METRICS]
    with csv_path.open("w") as f:
        f.write(",".join(header) + "\n")
        for name, data in rows:
            vals = [f"{data[m][0]:.4f}" for m in METRICS] + [f"{data[m][1]:.4f}" for m in METRICS]
            f.write(",".join([name] + vals) + "\n")

    # JSON
    json_path = out_dir / "full_table.json"
    json_path.write_text(
        json.dumps(
            {
                "seeds_present": list(rl_per_seed.keys()),
                "rl_per_seed": rl_per_seed,
                "rl_mean_std": {m: list(rl_agg[m]) for m in METRICS},
                "baselines": baselines,
                "vanilla_per_seed": vanilla_per_seed,
                "vanilla_mean_std": {m: list(vanilla_agg[m]) for m in METRICS} if vanilla_agg else {},
                "shape_delta_per_seed": shape_delta_per_seed,
                "shape_delta_mean_std": {m: list(shape_delta_agg[m]) for m in METRICS} if shape_delta_agg else {},
                "shape_delta_ms600_per_seed": shape_delta_ms600_per_seed,
                "shape_delta_ms600_mean_std": {m: list(shape_delta_ms600_agg[m]) for m in METRICS} if shape_delta_ms600_agg else {},
                "ablations_raw": ablation_raw,
            },
            indent=2,
        )
    )

    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print("\n=== Markdown preview ===\n")
    print(md_path.read_text())


if __name__ == "__main__":
    main()
