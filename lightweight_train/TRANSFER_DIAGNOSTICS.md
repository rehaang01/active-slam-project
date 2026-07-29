# Three-Tier Sim-to-Sim Transfer Diagnostic Framework

A cheap-first evaluation pipeline for screening RL policies trained in a lightweight 2D grid environment before committing them to a heavyweight photorealistic environment (Gazebo + RTAB-Map + PX4). Each tier gates the next: failure at a lower tier predicts failure at a higher tier at a fraction of the compute cost.

| Tier | Script | Cost (wall-clock) | What it measures |
|---|---|---|---|
| 1 | [`action_inspect.py`](action_inspect.py) | ~1 minute per seed | Raw action-distribution statistics (no Gazebo, no sensor stack) |
| 1b | [`ema_sweep.py`](ema_sweep.py) | ~3 minutes per (seed, α) cell | Action-smoothing α vs exploration-preservation trade-off |
| 2 | `../gazebo_transfer_eval.py` | ~7–10 minutes per 200-step smoke, ~50 minutes per full 1500-step run | End-to-end zero-shot transfer, with motion-safety guards |

All three tools share the lightweight env's `AltitudeActiveSLAMEnv` or its Gazebo counterpart `envs/active_slam_env.py` — no retraining required to run any of them.

---

## Tier 1 — Action-distribution inspector

Runs each checkpoint through its native lightweight env for N episodes and computes six motion-smoothness metrics, each scored GREEN / YELLOW / RED against thresholds calibrated to the Gazebo env's defenses.

**Thresholds** (informed by `envs/active_slam_env.py` parameters):

| Metric | GREEN | YELLOW | Rationale |
|---|---|---|---|
| `xy_flip_rate` | ≤ 0.15 | ≤ 0.30 | `MAX_YAW_CHANGE = 11°/step` absorbs modest oscillation, not sign-flipping |
| `jitter_rms` | ≤ 0.35 | ≤ 0.70 | `STEP_DURATION = 2 s` — high jitter = stop-start motion |
| `xy_saturation` | ≤ 0.40 | ≤ 0.70 | Saturated commands overwhelm the planner |
| `dz_triggered_frac` | ≤ 0.05 | ≤ 0.10 | Any dz > ALT_UP/DOWN_THRESH triggers Z-filter reconfig |
| `alt_change_count` | ≤ 10 | ≤ 25 | RTAB-Map can't re-converge with frequent layer swaps |
| `min_alt_interval` | ≥ 20 | ≥ 10 | Must respect `MIN_ALT_DWELL = 20` in Gazebo env |

**Usage:**

```bash
python3 action_inspect.py                 # 3 MS600 seeds × 5 episodes, stochastic
python3 action_inspect.py --deterministic # also compare argmax actions
python3 action_inspect.py --seeds 42      # single seed
```

Verdict output:

```
--- seed 42 (stoch) ---
  xy_flip_rate (lo=smooth)      0.398   [RED]
  jitter_rms  (lo=smooth)       0.988   [RED]
  ...
!!! RED VERDICT — add MIN_ALT_DWELL guard + action low-pass before Gazebo !!!
```

A RED verdict is a go/no-go gate — do not commit Gazebo compute until mitigated.

---

## Tier 1b — EMA α sweep

Wraps inference with an exponential moving average `a_t = α·raw + (1−α)·a_{t−1}` and measures both motion metrics AND coverage at each α. Produces the Pareto curve in [results_3d_shape_delta_800k_maxsteps600/plots/ema_pareto.png](results_3d_shape_delta_800k_maxsteps600/plots/ema_pareto.png).

**Usage:**

```bash
python3 ema_sweep.py                              # full sweep (3 seeds × 4 α × 3 eps)
python3 ema_sweep.py --alphas 1.0 0.5             # compare two α values
python3 ema_sweep.py --seeds 42 --episodes 3      # quick smoke
```

**Aggregate results (this project, MS600 checkpoints):**

| α | xy_flip | jitter | coverage | verdict |
|---|---|---|---|---|
| 1.00 | 0.40 | 0.99 | 0.904 ± 0.025 | RED motion / GREEN exploration (unsafe) |
| 0.50 | 0.25 | 0.40 | 0.800 ± 0.053 | **GREEN motion / GREEN exploration (Pareto sweet spot)** |
| 0.30 | 0.11 | 0.22 | 0.638 ± 0.098 | GREEN motion / YELLOW exploration |
| 0.20 | 0.07 | 0.14 | 0.521 ± 0.128 | GREEN motion / RED exploration (over-smoothed) |

---

## Tier 2 — End-to-end Gazebo transfer evaluator

Loads a lightweight checkpoint into the Gazebo env with optional motion-safety guards:

- `--ema-alpha-xy` / `--ema-alpha-dz` — per-axis EMA (decouples the policy's lateral pathology from its altitude pathology)
- `--ema-alpha` — legacy single-axis EMA (applied to all dims)
- `MIN_ALT_DWELL` + `MAX_ALT_IDX` are env-side constants, not CLI flags ([envs/active_slam_env.py](../envs/active_slam_env.py))

**Usage:**

```bash
# Minimum-risk smoke (200 steps, ~7 min)
python3 gazebo_transfer_eval.py \
  --model   lightweight_train/results_3d_shape_delta_800k_maxsteps600/models/slam3d_180000_steps.zip \
  --vecnorm lightweight_train/results_3d_shape_delta_800k_maxsteps600/models/slam3d_vecnormalize_180000_steps.compat.pkl \
  --episodes 1 --max-steps 200 \
  --ema-alpha-xy 0.8 --ema-alpha-dz 0.3 \
  --episode-tag smoke

# Full run (1500 steps, ~50 min)
python3 gazebo_transfer_eval.py ... --max-steps 1500 --episode-tag tier3
```

Per-step CSV output (21 columns) matches the lightweight env's eval CSVs, so plotting / aggregation scripts work unchanged.

---

## Known cross-version compatibility shims

| Problem | Where | Fix |
|---|---|---|
| Host numpy 2.x pickles `numpy._core.*` refs; Docker numpy<2.0 can't deserialize | `VecNormalize.load(...)` | [`convert_vecnorm_to_compat.py`](../convert_vecnorm_to_compat.py) — strips `numpy.random.Generator` refs, repickles as protocol 4 |
| `RecurrentPPO.load(...)` same issue | cloudpickle deserialization in SB3 | `sys.modules["numpy._core.*"]` shim in `gazebo_transfer_eval.py` + `custom_objects={...}` kwarg |

---

## Env-side Gazebo guards (one-time patches to `envs/active_slam_env.py`)

| Guard | Motivation | Default |
|---|---|---|
| `MIN_ALT_DWELL = 20` | Absorbs policy's high dz-vote rate that would otherwise reconfig the Z-filter every step | mirrors lightweight env's lock |
| `MAX_ALT_IDX = 2` | Limits altitude to ≤ 3 m so camera stays in feature-dense zone (prevents 331 s tracking loss at 4 m ceiling) | discovered during dry run |
| Frontier definition: `unknown ∧ adj-to-known` | RTAB-Map's Z-sliced 2D projection marks nearly every observed cell as occupied; the classical `free ∧ adj-to-unknown` yields zero frontiers | replaces classical definition |
| Min frontier cluster size: 8 cells | Filters sensor-shadow holes near the drone that would otherwise appear as `near_f ≈ 0` noise | raised from 2 |

Each guard is a ~5 line change reflecting a specific reality-gap we identified — see [`../paper_drafts/gap_analysis.md`](../paper_drafts/gap_analysis.md).
