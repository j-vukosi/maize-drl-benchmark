# Fair-Comparison Checklist

Use this before every final run.

## Environment

- [ ] Same climate archive and climate-file hash.
- [ ] Same crop, soil, planting date and season end.
- [ ] Same 26 observations and fixed scaling.
- [ ] Same normalized `[-1,1]` policy action and identical linear mapping to `[0,25]` mm.
- [ ] Same reward constants and profit constants.
- [ ] Same cap enforcement in `limited_water`.
- [ ] Same termination rule.

## Data and randomisation

- [ ] Same train, validation and test years.
- [ ] Same seed list: `11, 23, 37, 53, 71`.
- [ ] Test years never used for tuning.
- [ ] Seed/year pairing preserved in statistical tables.
- [ ] Failed runs logged; no cherry-picked replacement seeds.

## Budget

- [ ] Same real environment-step budget.
- [ ] Checkpoints use environment steps, not algorithm iterations.
- [ ] Evaluation interaction is excluded from the training budget.
- [ ] Wall-clock time and peak memory recorded.
- [ ] Dreamer training ratio and replay update ratios disclosed.

## Implementation identity

- [ ] PPO version and SB3 commit/version recorded.
- [ ] CrossQ identified as the SB3-Contrib implementation of the ICLR 2024 algorithm.
- [ ] DSAC-T implementation notes match the official DSAC-v2 equations.
- [ ] DreamerV3 backend identified as RLlib or upstream Danijar, never ambiguously mixed.
- [ ] Upstream repository commits stored in `outputs/manifests/upstream_commits.json`.

## Reporting

- [ ] No ranking from the prototype's single seed or 3000 steps.
- [ ] All main metrics reported, including water and stress.
- [ ] Confidence intervals and paired effects shown.
- [ ] Statistical and agronomic significance discussed separately.
- [ ] Pareto trade-offs reported.
- [ ] Limitations, missing runs and dependency problems disclosed.
