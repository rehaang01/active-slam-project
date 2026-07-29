"""Evaluate arbitrary training tags at N=50, reusing main-seed42 baselines.

Usage:
    python sweep_eval.py shape_covmax shape_delta shape_breadth2x
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

LIGHTWEIGHT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LIGHTWEIGHT_DIR))

import run_3d

MAIN_BASELINE_JSON = LIGHTWEIGHT_DIR / "results_3d_seed42" / "summary" / "baseline_summary.json"


def main():
    if len(sys.argv) < 2:
        print("Usage: python sweep_eval.py TAG1 [TAG2 ...]")
        sys.exit(1)
    tags = sys.argv[1:]
    if not MAIN_BASELINE_JSON.exists():
        raise FileNotFoundError(f"Missing {MAIN_BASELINE_JSON}")
    baseline_summary = json.loads(MAIN_BASELINE_JSON.read_text())

    for tag in tags:
        base = LIGHTWEIGHT_DIR / f"results_3d_{tag}"
        if not base.exists():
            print(f"[sweep_eval] SKIP {tag}: dir missing")
            continue
        final_marker = base / "models" / "final_model.zip"
        if not final_marker.exists():
            print(f"[sweep_eval] SKIP {tag}: training not complete")
            continue
        sel = base / "summary" / "selected_rl_summary.json"
        if sel.exists():
            print(f"[sweep_eval] SKIP {tag}: already evaluated")
            continue

        print("=" * 72)
        print(f"[sweep_eval] {tag}")
        print("=" * 72)

        run_3d.Cfg.BASE = base
        run_3d.Cfg.MODELS = base / "models"
        run_3d.Cfg.CSV = base / "csv"
        run_3d.Cfg.PLOTS = base / "plots"
        run_3d.Cfg.SUMMARY = base / "summary"
        run_3d.Cfg.TENSORBOARD_LOG = None
        run_3d.Cfg.mkdirs()

        dst = base / "summary" / "baseline_summary.json"
        if not dst.exists():
            shutil.copy2(MAIN_BASELINE_JSON, dst)

        payload = run_3d.evaluate_checkpoints(baseline_summary)
        rl_summary = run_3d.export_selected_rl(payload["selected"])
        if rl_summary:
            print(
                f"[sweep_eval] {tag} selected={rl_summary.get('checkpoint')} "
                f"cov={rl_summary['final_coverage']:.2%} trace={rl_summary['final_cov_trace']:.3f}"
            )

    print("\n[sweep_eval] done.")


if __name__ == "__main__":
    main()
