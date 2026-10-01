# Preregistered Experimental Protocol

## 1. Research question

Which of PPO, CrossQ, DSAC-T and DreamerV3 gives the most reliable balance between maize yield protection and irrigation efficiency under normal, dry, extreme-dry, limited-water and wet conditions?

The benchmark is deliberately **multi-objective**. It does not declare a winner from episode reward alone and does not assume that one algorithm must dominate every agronomic, economic and computational measure.

## 2. Experimental unit and blocking

Each trained policy is evaluated once on every held-out climate year, producing detailed **algorithm × training seed × year** rows. For inferential statistics, those years are averaged within each independently trained seed. The primary inferential unit is therefore one **algorithm × training seed**, paired across algorithms by `(scenario, seed)`. This avoids counting several evaluations of the same trained policy as independent replicates.

A deterministic crop simulator and deterministic policy evaluation produce the same result when the same seed-year is repeated. Therefore, the protocol uses one deterministic rollout per seed-year. Repeating an identical rollout ten times would be pseudo-replication and would artificially narrow uncertainty intervals. Seed-year rows remain available for climate-specific plots and Pareto analysis; extra stochastic rollouts are allowed only as a clearly labelled secondary robustness analysis.

## 3. Data split

Each rainfall stratum is divided deterministically into:

- training years: approximately 60%;
- validation years: approximately 20%;
- final test years: approximately 20%.

The hash-based split is fixed by seed `4279123`. Hyperparameters, reward constants and scenario thresholds must be frozen before test evaluation. Test-year results are opened only for the final benchmark.

## 4. Scenarios

Seasonal precipitation is summed from planting (`05/01`) to season end (`12/31`) for every climate year. The benchmark uses:

| Scenario | Operational definition |
|---|---|
| `extreme_dry` | bottom 15% of historical seasonal precipitation |
| `dry` | 15th–35th percentiles |
| `normal` | 35th–65th percentiles |
| `limited_water` | the same climate stratum as normal, with a seasonal allocation cap |
| `wet` | top 20% of historical seasonal precipitation |

The percentile cut-offs are transparent benchmark design decisions. The literature supports the need to test drought, water restriction, ordinary variability and wet/waterlogging risk; it does not prescribe these exact numerical cut-offs.

The limited-water cap is calculated separately for each year as 70% of the seasonal irrigation used by an AquaCrop soil-moisture-target reference policy (`SMT=80`). The cap is independent of DRL outcomes, preventing algorithm-specific leakage.

## 5. Common MDP

All algorithms receive the same 26-dimensional observation:

1. crop age;
2. canopy cover;
3. biomass;
4. root-zone depletion;
5. total available water;
6. precipitation for the previous seven days;
7. minimum temperature for the previous seven days;
8. maximum temperature for the previous seven days.

All algorithms receive one normalized continuous action in `[-1,1]`, mapped linearly to irrigation depth in `[0,25]` mm before AquaCrop is stepped. The transform is common, fixed and non-learning. This is required for a fair four-way comparison because CrossQ and DSAC-T are continuous-control algorithms and RLlib DreamerV3 emits tanh-centred continuous actions. A binary `0/25 mm` ablation reproduces the source environment's original decision structure.

The primary reward is:

`terminal_yield_scale × dry_yield^4 − water_penalty × incremental_irrigation − optional_stress_penalty`.

Incremental irrigation is penalised once. The source implementation's cumulative-water subtraction is retained only as a `legacy_source` ablation because it repeatedly penalises earlier irrigation.

## 6. Training budget

Primary target budget: 512,000 **real AquaCrop environment steps** for every algorithm, scenario and seed. This is exactly 250 PPO rollouts at `n_steps=2048`, so PPO, CrossQ and DSAC-T can stop at the same interaction count. DreamerV3 stops at the first RLlib training-iteration boundary at or above the target; its actual sampled-step count is logged and any overshoot must be reported.

Equal environment interaction is the main fairness constraint. Gradient updates are not forced to be equal because replay-based and world-model methods use data differently. Wall-clock time, peak memory, actual sampled steps and training ratio are reported so compute trade-offs remain visible.

No algorithm receives extra test interaction, extra training years or a different reward. Failed runs are retained in a failure log and are not silently replaced with a favourable seed.

## 7. Hyperparameter policy

- Start from published/default settings.
- One small pilot grid may be run on validation years only.
- Use the same pilot budget per algorithm.
- Select one configuration per algorithm before final runs.
- Never tune against final test years.
- Report all tried configurations and failed runs.

The supplied YAML files are the frozen starting configurations. Changes must create a new experiment ID rather than overwriting old runs.

## 8. Evaluation metrics

### Agronomic and economic

- dry yield (tonne/ha), maximise;
- total irrigation (mm), minimise;
- irrigation water productivity (kg/ha/mm), maximise;
- benchmark profit, maximise;
- cumulative stress exposure, minimise;
- stress days, minimise;
- maximum daily stress, minimise.

### Learning and computational

- episode reward, maximise, but never use alone;
- learning-curve area under the curve against real environment steps;
- final rolling training reward;
- wall-clock time;
- peak main-process and process-tree resident memory;
- hardware and package manifest.

## 9. Statistical analysis

For each scenario and metric:

1. average held-out years within each training seed;
2. summarise seed-level means, standard deviations, medians and 95% bootstrap confidence intervals;
3. run a Friedman test across the four algorithms paired by seed;
4. run paired Wilcoxon signed-rank comparisons;
5. adjust pairwise p-values with Holm's procedure;
6. report median paired differences and win rates, not p-values alone;
7. report yield–irrigation Pareto-efficient frequency on the detailed seed-year rows.

The analysis is exploratory if there are too few complete paired blocks. A non-significant test is not proof of equivalence. A statistically significant reward difference is not automatically agronomically important.

## 10. Required ablations

- original binary `0/25 mm` action;
- exact source reward versus incremental-water reward;
- limited-water cap at 60%, 70% and 80% of reference irrigation;
- with and without fixed feature scaling;
- DreamerV3 RLlib Windows track versus upstream Danijar WSL track when compute permits.

## 11. Completion criteria

The final report is complete only when:

- all four algorithms have five successful seeds in every primary scenario;
- every algorithm has complete test rows for the same held-out years, and seed-aggregated inference is paired by seed;
- package, hardware and upstream commit manifests are present;
- no prototype DSAC surrogate is labelled as full DSAC-T;
- no rankings are based on a single seed;
- tables include uncertainty and compute cost;
- limitations and failures are reported.
