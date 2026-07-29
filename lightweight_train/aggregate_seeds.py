"""Aggregate multi-seed RL results + baselines into a publication-ready table.

Reads results_3d_seed{42,100,200}/summary/selected_rl_summary.json for RL stats and
results_3d_seed42/summary/baseline_summary.json for baselines (baselines are
seed-independent — they use the same held-out eval seeds, so one run is enough).
Emits:
  - summary/multi_seed_table.csv    (methods x metric cols, mean ± std)
  - summary/multi_seed_table.md     (same, markdown)
  - summary/multi_seed_stats.json   (raw per-seed + aggregated)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

LIGHTWEIGHT_DIR = Path(__file__).resolve().parent
SEEDS = (42, 100, 200)
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


def load_rl_per_seed():
    per_seed = {}
    for seed in SEEDS:
        path = LIGHTWEIGHT_DIR / f"results_3d_seed{seed}" / "summary" / "selected_rl_summary.json"
        if path.exists():
            per_seed[seed] = json.loads(path.read_text())
    return per_seed


def load_baselines():
    path = LIGHTWEIGHT_DIR / "results_3d_seed42" / "summary" / "baseline_summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    rl_per_seed = load_rl_per_seed()
    baselines = load_baselines()

    rl_agg = {}
    for metric in METRICS:
        vals = [rl_per_seed[s].get(metric) for s in rl_per_seed]
        rl_agg[metric] = _mean_std(vals)

    rows = []
    for name, data in baselines.items():
        rows.append((name, {m: (data.get(m), 0.0) for m in METRICS}))
    rows.append((f"RL (n={len(rl_per_seed)} seeds)", rl_agg))

    out_dir = LIGHTWEIGHT_DIR / "results_3d_seed42" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "multi_seed_table.csv"
    header = ["method"] + [f"{m}_mean" for m in METRICS] + [f"{m}_std" for m in METRICS]
    with csv_path.open("w") as f:
        f.write(",".join(header) + "\n")
        for name, data in rows:
            vals = [f"{data[m][0]:.4f}" for m in METRICS] + [f"{data[m][1]:.4f}" for m in METRICS]
            f.write(",".join([name] + vals) + "\n")

    # Markdown
    md_path = out_dir / "multi_seed_table.md"
    with md_path.open("w") as f:
        f.write("| Method | Coverage | Cov Trace | Alt0 | Alt1 | Alt2 | Alt3 | LoopClos | AltChg |\n")
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

    # JSON dump
    stats_path = out_dir / "multi_seed_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "seeds_present": list(rl_per_seed.keys()),
                "per_seed_rl": rl_per_seed,
                "rl_mean_std": {m: list(rl_agg[m]) for m in METRICS},
                "baselines": baselines,
            },
            indent=2,
        )
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {stats_path}")
    print("\n=== Markdown preview ===\n")
    print(md_path.read_text())


if __name__ == "__main__":
    main()
