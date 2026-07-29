# ICAAV 2026 Submission — Deliverables & Reviewer Response

This directory contains the revised submission, produced from the original IEEE paper in
`final_paper/` and reformatted into the **official Springer Nature LNCS proceedings template
(`llncs`)**, anonymized for double-blind review.

## What to submit (PDF only, v1.4+, fonts embedded, searchable, no password — all verified)

| Deliverable | Primary file (submit this) | Editable Word backup |
|---|---|---|
| **Full-length paper** (8–12 pp → **12 pp**) | `paper_full_springer/main.pdf` | `paper_full_springer/main.docx` |
| **Modified extended abstract** (4 pp, 500–1500 words → **1197 words**) | `abstract_springer/main.pdf` | `abstract_springer/main.docx` |

`main_word_render.pdf` in each folder is a LibreOffice render of the `.docx` (reference only).
The `.docx` were generated with `tools/pandoc` (equations → OMML, numbered IEEE-style citations).

## How the reviewer comments were addressed (full paper)

| Reviewer ask | Where addressed |
|---|---|
| (i) sim-to-real discussion / roadmap | **Sec. 7 "Sim-to-Real Transfer: Findings and Roadmap"** — reports the working zero-shot Gazebo/PX4/RTAB-Map pipeline, the smooth-motion constraint, and a forward roadmap (framed positively). |
| (ii) inference compute & latency on a UAV-class platform | **Sec. 6 "Computational Cost and Real-Time Feasibility"** — 2.02 M params, 10.16 M MACs, **1.20 ms** mean inference (836 Hz) on a single CPU thread; embedded projection. |
| (iii) ≥1 qualitative trajectory across altitudes | **Fig. 2** — 4-panel multi-altitude trajectory (seed 221, 95.1 % coverage). |
| (iv) smooth notation in problem formulation | Rewrote Sec. 3.2 + added the **symbol table (Table 1)**; every symbol defined on first use. |

Plus: DOIs added to all references; SI units throughout; "UAV" used consistently (no "drone");
grayscale-safe figures with SI-unit axis labels and in-plot legends; authors/affiliations commented
out and acknowledgments removed for double-blind; neutral self-referencing.

## New code (no retraining — headline 91.90 ± 1.72 % unchanged, verified)

- `lightweight_train/bench_inference.py` — measures inference latency, params, and MACs/FLOPs.
- `lightweight_train/plot_trajectory_altitudes.py` — standalone rollout + the cross-altitude figure.
- `lightweight_train/plot_training_curves.py`, `plot_timeseries.py` — made grayscale-/SI-compliant.
- `lightweight_train/env_3d.py` — **default-OFF** sim-to-real knobs (`W_SMOOTH_OVERRIDE`,
  `W_ALT_TRANSITION_OVERRIDE`, `FRONTIER_DEF=unknown_adj_known`) described as roadmap; defaults
  reproduce the published results exactly.

## Reproduce

```bash
# 1. Inference benchmark (numbers used in Sec. 6)
OMP_NUM_THREADS=1 MAX_STEPS_OVERRIDE=600 .venv/bin/python lightweight_train/bench_inference.py \
  --model lightweight_train/results_3d_shape_delta_800k_maxsteps600/models/slam3d_180000_steps.zip \
  --vecnorm lightweight_train/results_3d_shape_delta_800k_maxsteps600/models/slam3d_vecnormalize_180000_steps.pkl

# 2. Trajectory figure (Fig. 2)
MAX_STEPS_OVERRIDE=600 .venv/bin/python lightweight_train/plot_trajectory_altitudes.py \
  --model .../slam3d_180000_steps.zip --vecnorm .../slam3d_vecnormalize_180000_steps.pkl --seed 221

# 3. Recompile a PDF (no sudo; tectonic auto-fetches packages)
cd paper_full_springer && ../tools/tectonic main.tex      # -> main.pdf (12 pp)
cd ../abstract_springer && ../tools/tectonic main.tex     # -> main.pdf (4 pp)
```

Headline configuration: `W_COV_DELTA_OVERRIDE=0.8` (the Δσ term) + `W_COV_PENALTY=-0.8` (absolute σ
penalty) + `MAX_STEPS_OVERRIDE=600`; both −0.8 terms appear in the reward table (Table 2).
