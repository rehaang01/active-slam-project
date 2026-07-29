# Sim-to-Sim Reality Gap Analysis

This document identifies four specific factors that degrade zero-shot transfer of a 2D-lightweight-trained active SLAM policy to a Gazebo / RTAB-Map / PX4 stack. Each factor is stated as a gap in the training-env model, with the empirical evidence that surfaced it and a proposed lightweight-env modification for future work.

The overarching finding: the policy is **optimal for its training domain** (achieves 91.9 ± 1.7% coverage over 50 eval episodes × 3 seeds, matching or exceeding every classical baseline with Wilcoxon p < 1e-25 against all non-learned methods). But several unmodeled physical constraints cause the behaviors that maximize lightweight reward to actively harm a physics-grounded SLAM stack.

---

## Gap 1 — Motion smoothness is free in lightweight; costly in Gazebo

**Lightweight assumption.** Each step, the policy emits `(dx, dy)` ∈ [−1, 1]²; the agent's position updates instantly by `pos ← pos + STEP_SIZE · action`. Sign-flipping `(dx, dy)` from one step to the next carries no penalty — the policy effectively teleports between adjacent cells.

**Gazebo reality.** Identical `(dx, dy)` actions are issued to a quadrotor flight controller that must physically decelerate, reverse thrust direction, and reaccelerate. A sign flip costs ≈ 2 × `STEP_DURATION = 4 s` of net-zero motion. The OakD-Lite's 70° FOV + RTAB-Map visual odometry also requires smooth motion for feature tracking; aggressive reversals produce motion-blur and degenerate baselines that cause loop-closure rejections.

**Evidence.**
- Tier 1 ([action_inspect.py](../lightweight_train/action_inspect.py)) measured `xy_flip_rate = 0.40` on stochastic inference (policy flips sign 40% of steps) — well into the RED band.
- Applying a symmetric EMA low-pass `α = 0.5` passes the motion-safety checks but drops lightweight coverage from 90.4% → 80.0% ([ema_pareto.png](../lightweight_train/results_3d_shape_delta_800k_maxsteps600/plots/ema_pareto.png)).
- In the Gazebo transfer dry run, even with α = 0.8 / 0.3 per-axis EMA and aggressive smoothing, the drone's net xy displacement over 13 minutes of flight was ≈ 0.7 m — the policy's sign flipping makes the EMA output converge to ≈ 0 on net.

**Proposed lightweight-env fix.** Add a smoothness penalty to the reward:

```
r_smooth = −λ · ‖a_t − a_{t−1}‖²
```

with `λ ≈ 0.1`, which would make sustained directional motion instantaneously rewarding compared to sign-flipping. This re-shapes the policy's action distribution during training rather than papering over it at inference.

---

## Gap 2 — Altitude-dependent feature density

**Lightweight assumption.** All four altitude levels (1 m, 2 m, 3 m, 4 m) are modeled as equally informative 2D occupancy grids. Switching altitude costs one step and provides instant access to a different slice of the same 3D world.

**Gazebo reality.** The warehouse environment is shelf-dense between 0.5–3.0 m (boxes, rack frames, floor pallets) but feature-sparse above 3.5 m (the ceiling is a uniform corrugated surface). At altitude 4 m, the downward-facing OakD-Lite sees mostly the tops of shelves — low-texture and at the sensor's depth-range limit. RTAB-Map's visual odometry loses tracking within ≈ 50 s of sustained high-altitude flight.

**Evidence.**
- Tier 3 dry-run console log at step ≈ 950: `RTAB-Map tracking LOST — no odometry for 47.9 s` immediately after the drone spent >100 s at alt = 4 m. A second tracking-lost warning at 331 s coincided with the laptop freezing under the backlog of failed re-localizations.
- With `MAX_ALT_IDX = 2` enforced (cap at 3 m), a subsequent 300-step run held `trk_lost = 0` throughout.

**Proposed lightweight-env fix.** Parameterize each altitude layer with a per-layer "feature-loss probability" or an information-gain multiplier < 1 for the top layer. The policy would learn to avoid tracking-failure-prone altitudes. Alternatively, include per-altitude `expected_tracking_loss_rate` as a scalar observation — trained from a small Gazebo-labeled dataset.

---

## Gap 3 — Z-filter transition cost

**Lightweight assumption.** The active altitude slice is a pure logical index (`alt_idx`); switching altitudes swaps in a different pre-rendered 2D map at zero cost.

**Gazebo reality.** Every altitude switch triggers an RTAB-Map `SetParameters` service call reconfiguring the octomap's Z-filter (e.g. [2.5 m, 3.5 m] → [3.5 m, 4.5 m]). This causes the 2D projected grid to be re-derived from the 3D octomap — typically a 500–2000 ms stall during which `cov_trace` spikes and coverage temporarily drops as previously-known cells re-enter "unknown" state under the new Z band.

**Evidence.**
- Across 300 steps of the altitude-cap run, six Z-filter reconfigs were observed, each producing a transient `cov_trace` spike (0.20 → 0.66 between steps 100 and 150 in one instance).
- The lightweight policy's trained dz-vote rate is 65–80% of steps (Tier 1). Without `MIN_ALT_DWELL = 20` enforcement in the Gazebo env, this would trigger a Z-filter reconfig almost every step, thrashing RTAB-Map's map.

**Proposed lightweight-env fix.** Add a per-step cost for altitude transitions:

```
r_alt_transition = −κ when alt_idx changes
```

with `κ ≈ 0.3` (roughly the magnitude of a loop-closure reward). Combined with the existing `MIN_ALT_DWELL = 20` constraint in the lightweight env, this would make the policy committed to an altitude long enough for any realistic mapping system to absorb the reconfig.

---

## Gap 4 — Frontier definition divergence

**Lightweight assumption.** Each 2D altitude slice has a clean ternary occupancy: `{free, occupied, unknown}`. Frontiers are cells where `free ∧ adjacent-to-unknown` — the canonical definition from Yamauchi 1997. This definition yields dozens of well-distributed exploration targets per step.

**Gazebo reality.** RTAB-Map's Z-sliced 2D projection (`/rtabmap/octomap_grid` with `Grid/RangeMin`/`RangeMax` set to the current altitude band) marks a 2D cell *occupied* if **any voxel** in its 1 m-thick Z column contains an obstacle. In a warehouse with shelves every few meters, nearly every observed 2D cell ends up occupied, and the classical `free ∧ adjacent-to-unknown` definition yields **zero frontiers** in practice.

**Evidence.**
- Instrumented diagnostic ([envs/active_slam_env.py:475 debug block](../envs/active_slam_env.py)) at step 100 of a transfer run: `free = 7 cells, occupied = 1173 cells, unknown = 9320 cells, free ∧ adj-to-unknown = 0`.
- Redefining the frontier mask to `unknown ∧ adj-to-known` (where `known = free ∨ occupied`) restored 13–19 clusters per step and enabled the policy to receive non-zero `frontier_count_norm`, `frontier_dist_norm`, `frontier_dir_norm` in its observation.

**Proposed lightweight-env fix.** Change the lightweight env's frontier definition to also use `unknown ∧ adj-to-known`, matching the real-world projection semantics. This is a 3-line change in [env_3d.py](../lightweight_train/env_3d.py) and does not alter the policy's training signal substantively in the lightweight env (where free-space is abundant) but makes the observation semantics consistent across sim-to-sim transfer.

---

## Summary table

| Gap | Lightweight assumption | Gazebo reality | Observable symptom | Evidence figure/script |
|---|---|---|---|---|
| 1. Motion smoothness | Free action changes | Physical deceleration | `xy_flip_rate = 0.40` RED | [action_inspect.py](../lightweight_train/action_inspect.py) |
| 2. Altitude feature density | Uniform across layers | Ceiling sparse | `trk_lost` after 50 s at alt=4 m | Tier 3 dry-run console |
| 3. Z-filter transition cost | Zero | 0.5–2 s stall | `cov_trace` spikes per alt change | 300-step dry-run CSV |
| 4. Frontier definition | `free ∧ adj-unknown` | All-occupied grid | `frontier_count = 0` initially | Step-100 debug print |

## Deliverables

- **Code:** three-tier diagnostic framework (Tier 1 / 1b / 2), reusable for other sim-to-sim active-SLAM transfer studies. See [`lightweight_train/TRANSFER_DIAGNOSTICS.md`](../lightweight_train/TRANSFER_DIAGNOSTICS.md).
- **Empirical:** 91.9 ± 1.7% lightweight coverage (MS600, 3 seeds × 50 eps), dominating every classical baseline (p < 1e-25, Cliff's δ > 0.9). Full table and significance tests at [`lightweight_train/results_3d_seed42/summary/full_table.md`](../lightweight_train/results_3d_seed42/summary/full_table.md) and [`statistical_tests.md`](../lightweight_train/results_3d_seed42/summary/statistical_tests.md).
- **Diagnostic figures:** EMA α Pareto curve showing the motion-smoothness / exploration trade-off discovered by the framework.
- **Future work:** four concrete lightweight-env modifications with expected impact on sim-to-sim transfer success.
