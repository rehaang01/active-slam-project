# Wilcoxon signed-rank tests (RL vs each baseline)

Hypothesis: RL dominates on coverage (higher) and on cov_trace (lower).
Cliff's delta ∈ [-1,1]: positive means RL favored; |δ|>0.47 is 'large' effect.

## POOLED across 3 seeds

| Baseline | n | Cov RL | Cov BL | Cov p | Cov δ | Trace RL | Trace BL | Trace p | Trace δ |
|---|---|---|---|---|---|---|---|---|---|
| random_walk | 150 | 87.56 | 54.92 | 5.68e-26*** | +0.958 | 3.708 | 3.690 | 6.50e-01 | -0.013 |
| nearest_frontier | 150 | 87.56 | 52.89 | 1.03e-25*** | +0.922 | 3.708 | 3.628 | 7.96e-01 | +0.038 |
| spiral | 150 | 87.56 | 28.72 | 1.22e-26*** | +0.977 | 3.708 | 2.410 | 1.00e+00 | -0.838 |
| potential_field | 150 | 87.56 | 33.51 | 6.15e-26*** | +0.919 | 3.708 | 2.081 | 1.00e+00 | -0.646 |
| greedy_info_gain | 150 | 87.56 | 83.38 | 1.56e-03** | +0.197 | 3.708 | 3.795 | 2.93e-02* | +0.165 |
| rrt_explorer | 150 | 87.56 | 39.58 | 2.02e-26*** | +0.971 | 3.708 | 3.501 | 9.95e-01 | -0.153 |
| vanilla_ppo | 150 | 87.56 | 86.87 | 2.94e-01 | +0.079 | 3.708 | 3.780 | 1.12e-01 | +0.106 |

## Shape-Delta (ours, proposed) vs Vanilla PPO, pooled

Hypothesis: the new W_COV_DELTA step-wise trace-reduction reward beats vanilla PPO.

| Comparison | n | Cov Shape-Delta | Cov Vanilla | Cov p | Cov δ | Trace Shape-Delta | Trace Vanilla | Trace p | Trace δ |
|---|---|---|---|---|---|---|---|---|---|
| Shape-Delta vs Vanilla | 150 | 88.69 | 86.87 | 1.84e-02* | +0.170 | 3.738 | 3.780 | 1.84e-01 | +0.055 |

## Learned baseline (Vanilla PPO) vs hand-crafted best (Greedy Info Gain), pooled

Hypothesis: PPO on raw coverage + collision beats hand-crafted greedy.

| Comparison | n | Cov Vanilla | Cov Greedy | Cov p | Cov δ | Trace Vanilla | Trace Greedy | Trace p | Trace δ |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla PPO vs Greedy | 150 | 86.87 | 83.38 | 6.65e-03** | +0.136 | 3.780 | 3.795 | 4.13e-01 | +0.059 |

## MS600 (Δcov-trace + MAX_STEPS=600, final recipe) vs Vanilla PPO, pooled

Hypothesis: the final recipe (reward shaping + longer episode horizon) beats vanilla PPO.

| Comparison | n | Cov MS600 | Cov Vanilla | Cov p | Cov δ | Trace MS600 | Trace Vanilla | Trace p | Trace δ |
|---|---|---|---|---|---|---|---|---|---|
| MS600 vs Vanilla | 150 | 91.90 | 86.87 | 7.36e-11*** | +0.486 | 3.729 | 3.780 | 1.36e-01 | +0.040 |

## MS600 vs Shape-Delta (ablate horizon), pooled

Hypothesis: extending episode horizon from 400 to 600 steps strictly improves the shape-delta policy.

| Comparison | n | Cov MS600 | Cov 400-step | Cov p | Cov δ | Trace MS600 | Trace 400-step | Trace p | Trace δ |
|---|---|---|---|---|---|---|---|---|---|
| MS600 vs 400-step | 150 | 91.90 | 88.69 | 7.73e-06*** | +0.306 | 3.729 | 3.738 | 5.23e-01 | -0.012 |

## MS600 vs Greedy Info Gain (hand-crafted best), pooled

Hypothesis: our learned policy beats the strongest classical active-SLAM planner.

| Comparison | n | Cov MS600 | Cov Greedy | Cov p | Cov δ | Trace MS600 | Trace Greedy | Trace p | Trace δ |
|---|---|---|---|---|---|---|---|---|---|
| MS600 vs Greedy | 150 | 91.90 | 83.38 | 8.12e-14*** | +0.534 | 3.729 | 3.795 | 1.38e-01 | +0.098 |

## Per-seed breakdown

### seed42

| Baseline | n | Cov RL | Cov BL | Cov p | Cov δ | Trace RL | Trace BL | Trace p | Trace δ |
|---|---|---|---|---|---|---|---|---|---|
| random_walk | 50 | 88.83 | 55.16 | 8.88e-16*** | +0.986 | 3.787 | 3.716 | 7.02e-01 | -0.085 |
| nearest_frontier | 50 | 88.83 | 52.53 | 8.88e-16*** | +0.938 | 3.787 | 3.627 | 8.73e-01 | -0.029 |
| spiral | 50 | 88.83 | 27.94 | 8.88e-16*** | +1.000 | 3.787 | 2.405 | 1.00e+00 | -0.898 |
| potential_field | 50 | 88.83 | 36.26 | 1.24e-14*** | +0.947 | 3.787 | 2.227 | 1.00e+00 | -0.650 |
| greedy_info_gain | 50 | 88.83 | 82.73 | 7.04e-03** | +0.238 | 3.787 | 3.760 | 6.58e-01 | +0.050 |
| rrt_explorer | 50 | 88.83 | 39.17 | 8.88e-16*** | +0.996 | 3.787 | 3.501 | 9.87e-01 | -0.212 |

### seed100

| Baseline | n | Cov RL | Cov BL | Cov p | Cov δ | Trace RL | Trace BL | Trace p | Trace δ |
|---|---|---|---|---|---|---|---|---|---|
| random_walk | 50 | 89.11 | 56.62 | 4.44e-15*** | +0.979 | 3.691 | 3.668 | 5.43e-01 | +0.018 |
| nearest_frontier | 50 | 89.11 | 53.63 | 4.44e-15*** | +0.957 | 3.691 | 3.616 | 7.01e-01 | +0.021 |
| spiral | 50 | 89.11 | 28.34 | 8.88e-16*** | +1.000 | 3.691 | 2.330 | 1.00e+00 | -0.857 |
| potential_field | 50 | 89.11 | 28.46 | 6.22e-15*** | +0.951 | 3.691 | 1.765 | 1.00e+00 | -0.684 |
| greedy_info_gain | 50 | 89.11 | 85.76 | 6.04e-02 | +0.176 | 3.691 | 3.865 | 1.37e-02* | +0.216 |
| rrt_explorer | 50 | 89.11 | 39.31 | 8.88e-16*** | +0.995 | 3.691 | 3.532 | 8.61e-01 | -0.145 |

### seed200

| Baseline | n | Cov RL | Cov BL | Cov p | Cov δ | Trace RL | Trace BL | Trace p | Trace δ |
|---|---|---|---|---|---|---|---|---|---|
| random_walk | 50 | 84.73 | 52.98 | 1.32e-12*** | +0.907 | 3.646 | 3.686 | 5.07e-01 | +0.026 |
| nearest_frontier | 50 | 84.73 | 52.52 | 7.72e-12*** | +0.877 | 3.646 | 3.640 | 4.15e-01 | +0.121 |
| spiral | 50 | 84.73 | 29.89 | 4.44e-15*** | +0.928 | 3.646 | 2.494 | 1.00e+00 | -0.766 |
| potential_field | 50 | 84.73 | 35.82 | 5.68e-13*** | +0.847 | 3.646 | 2.250 | 1.00e+00 | -0.598 |
| greedy_info_gain | 50 | 84.73 | 81.65 | 1.29e-01 | +0.164 | 3.646 | 3.761 | 9.33e-02 | +0.222 |
| rrt_explorer | 50 | 84.73 | 40.25 | 1.22e-13*** | +0.926 | 3.646 | 3.469 | 8.81e-01 | -0.094 |

