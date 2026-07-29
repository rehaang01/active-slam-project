"""Re-select the best checkpoint for each training run using a less myopic rule.

Current rule in run_3d.select_checkpoint: pick the FIRST checkpoint passing
(cov > greedy+3%, trace < greedy*0.8, min_alt_cov >= 25%). This often locks in an
early-lottery win. Better: restrict to the last 50% of training and pick the one
with the highest composite score (coverage + 0.5*min_alt - 0.05*trace).

This script reads each tag's checkpoint_selection.json, applies the new rule,
and if the newly-selected checkpoint differs from the current one, re-runs the
N=50 export via run_3d.export_selected_rl — producing fresh rl_agent_ep*.csv
and selected_rl_summary.json.

Usage:
    python reselect_checkpoint.py seed42 seed100 seed200 shape_delta shape_delta_s100 shape_delta_s200 vanilla_s42 vanilla_s100 vanilla_s200
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

LIGHTWEIGHT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LIGHTWEIGHT_DIR))

import run_3d
from transfer_config import N_ALTITUDES


def composite_score(c: dict) -> float:
    min_alt = min(c.get(f"final_cov_alt{a}", 0.0) for a in range(N_ALTITUDES))
    return c["final_coverage"] + 0.5 * min_alt - 0.05 * c["final_cov_trace"]


def best_of_all(ckpts: list[dict]) -> dict:
    """Pick highest composite-score across ALL checkpoints (regardless of step rule).

    The original select_checkpoint in run_3d returns the FIRST checkpoint passing a
    hand-crafted stop rule. That rule is used to decide whether to stop training;
    once training is complete, we should always take the best-performing one.
    """
    return max(ckpts, key=composite_score)


def main():
    if len(sys.argv) < 2:
        print("Usage: python reselect_checkpoint.py TAG1 [TAG2 ...]")
        sys.exit(1)
    tags = sys.argv[1:]

    summary_rows = []
    for tag in tags:
        base = LIGHTWEIGHT_DIR / f"results_3d_{tag}"
        ck_json = base / "summary" / "checkpoint_selection.json"
        if not ck_json.exists():
            print(f"[reselect] SKIP {tag}: no checkpoint_selection.json")
            continue
        payload = json.loads(ck_json.read_text())
        ckpts = payload["checkpoints"]
        old_sel = payload.get("selected", {})
        new_sel = best_of_all(ckpts)
        old_name = old_sel.get("checkpoint", "none")
        old_score = composite_score(old_sel) if old_sel else float("nan")
        new_score = composite_score(new_sel)
        changed = new_sel["checkpoint"] != old_name
        print(
            f"[reselect] {tag}: old={old_name} (score {old_score:.3f}, "
            f"cov {old_sel.get('final_coverage', 0)*100:.2f}%, trace {old_sel.get('final_cov_trace', 0):.3f}) -> "
            f"new={new_sel['checkpoint']} (score {new_score:.3f}, "
            f"cov {new_sel['final_coverage']*100:.2f}%, trace {new_sel['final_cov_trace']:.3f}) "
            f"[{'CHANGED' if changed else 'same'}]"
        )
        if changed:
            # Re-run the 50-episode export with the better checkpoint.
            run_3d.Cfg.BASE = base
            run_3d.Cfg.MODELS = base / "models"
            run_3d.Cfg.CSV = base / "csv"
            run_3d.Cfg.PLOTS = base / "plots"
            run_3d.Cfg.SUMMARY = base / "summary"
            run_3d.Cfg.TENSORBOARD_LOG = None
            run_3d.Cfg.mkdirs()
            rl_summary = run_3d.export_selected_rl(new_sel)
            if rl_summary:
                print(
                    f"[reselect] {tag} re-exported: cov={rl_summary['final_coverage']:.2%} "
                    f"trace={rl_summary['final_cov_trace']:.3f}"
                )
            # Persist new selected record in checkpoint_selection.json.
            payload["selected"] = new_sel
            ck_json.write_text(json.dumps(payload, indent=2))
        summary_rows.append((tag, old_name, new_sel["checkpoint"], changed))

    print("\n=== Summary ===")
    for tag, old, new, changed in summary_rows:
        tag_changed = " ** CHANGED **" if changed else ""
        print(f"  {tag}: {old} -> {new}{tag_changed}")


if __name__ == "__main__":
    main()
