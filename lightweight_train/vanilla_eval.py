"""Evaluate Vanilla-PPO (ablation=all shaping OFF) checkpoints at N=50.

For each vanilla seed dir, reuse the already-computed baseline_summary.json from
results_3d_seed42/summary/ (baselines are seed-independent at the shaping level
since only map_seed varies — same map seeds used everywhere). Select the best
checkpoint by the composite score rule in run_3d.py and export N=50 eval CSVs
and selected_rl_summary.json so aggregate_full.py can pick it up as a learned
baseline row.

Usage:
    python vanilla_eval.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

LIGHTWEIGHT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LIGHTWEIGHT_DIR))

import run_3d

VANILLA_TAGS = ("vanilla_s42", "vanilla_s100", "vanilla_s200")
MAIN_BASELINE_JSON = LIGHTWEIGHT_DIR / "results_3d_seed42" / "summary" / "baseline_summary.json"


def main():
    if not MAIN_BASELINE_JSON.exists():
        raise FileNotFoundError(
            f"Missing {MAIN_BASELINE_JSON}. Run main seed42 eval first to generate baseline_summary.json."
        )
    baseline_summary = json.loads(MAIN_BASELINE_JSON.read_text())

    for tag in VANILLA_TAGS:
        base = LIGHTWEIGHT_DIR / f"results_3d_{tag}"
        if not base.exists():
            print(f"[vanilla_eval] SKIP {tag}: dir missing")
            continue
        final_marker = base / "models" / "final_model.zip"
        if not final_marker.exists():
            print(f"[vanilla_eval] SKIP {tag}: training not complete ({final_marker} missing)")
            continue
        sel_summary = base / "summary" / "selected_rl_summary.json"
        if sel_summary.exists():
            print(f"[vanilla_eval] SKIP {tag}: already evaluated ({sel_summary} present)")
            continue

        print("=" * 72)
        print(f"[vanilla_eval] {tag}")
        print("=" * 72)

        run_3d.Cfg.BASE = base
        run_3d.Cfg.MODELS = base / "models"
        run_3d.Cfg.CSV = base / "csv"
        run_3d.Cfg.PLOTS = base / "plots"
        run_3d.Cfg.SUMMARY = base / "summary"
        run_3d.Cfg.TENSORBOARD_LOG = None
        run_3d.Cfg.mkdirs()

        # Cache baseline_summary in this tag's summary dir so downstream code finds it.
        dst_baseline = base / "summary" / "baseline_summary.json"
        if not dst_baseline.exists():
            shutil.copy2(MAIN_BASELINE_JSON, dst_baseline)
            print(f"[vanilla_eval] Copied main baseline_summary.json -> {dst_baseline}")

        checkpoint_payload = run_3d.evaluate_checkpoints(baseline_summary)
        rl_summary = run_3d.export_selected_rl(checkpoint_payload["selected"])
        if rl_summary:
            print(
                f"[vanilla_eval] {tag} selected={rl_summary.get('checkpoint')} "
                f"cov={rl_summary['final_coverage']:.2%} trace={rl_summary['final_cov_trace']:.3f}"
            )

    print("\n[vanilla_eval] All vanilla-PPO seeds evaluated.")


if __name__ == "__main__":
    main()
