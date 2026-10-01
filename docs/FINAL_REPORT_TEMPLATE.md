# Final Honours Report Template

Use this structure after the honours run. Replace bracketed prompts with genuine output only.

## 1. Research question and contribution

State the four-way fair-comparison question and explain that the contribution is a reproducible benchmark rather than a claim that DRL is universally superior.

## 2. Environment and common MDP

Report the AquaCrop climate file hash, crop, soil, planting date, 26 observations, shared normalized action interface and native depth mapping, reward equation, feature transform and limited-water cap method. Explain why a common continuous action was necessary, and report the source binary-action experiment separately as an ablation.

## 3. Algorithms

For PPO, CrossQ, DSAC-T and DreamerV3, give the paradigm, implementation source, exact upstream commit, frozen hyperparameters, parameter count if available and training budget. Distinguish the primary RLlib DreamerV3 Windows backend from the optional canonical Danijar/JAX WSL2 replication.

## 4. Scenario design

Include the exact train, validation and test years from `scenarios/*/manifest.json`, the precipitation quantile definitions and the independently calibrated limited-water allocation. Cite the recent articles in `docs/SCENARIO_EVIDENCE.md`. Make clear that the numerical percentile boundaries are benchmark choices.

## 5. Reproducibility and fairness

Present the fairness checklist, five seeds, equal real environment steps, deterministic final evaluation, held-out years, failure handling, hardware/package manifests and no-test-tuning rule.

## 6. Results

### 6.1 Agronomic and economic outcomes

First verify that `outputs/manifests/completion_status.json` reports `valid_for_final_ranking: true` and inspect `outputs/summaries/run_completeness.csv`. Then use `outputs/summaries/algorithm_scenario_summary.csv` and the high-resolution figures. Report mean, standard deviation and 95% bootstrap interval for yield, irrigation, water productivity, profit and stress metrics. Discuss trade-offs rather than selecting a winner from one chart.

### 6.2 Statistical comparison

Use the Friedman omnibus table first. The inferential rows are year-averaged values from independently trained seeds, paired by seed within each scenario; the held-out seed-year table remains descriptive. Discuss pairwise Wilcoxon-Holm comparisons only where the omnibus result and corrected pairwise evidence support them. Report effect direction, median paired difference and win rate, not p-values alone.

### 6.3 Pareto analysis

Use `yield_water_pareto_frequency.csv` to show how often each agent lies on the yield–water frontier across matched seed-year blocks.

### 6.4 Learning and computational practicality

Compare environment-step reward AUC, final rolling reward, wall-clock time and peak main-process and process-tree resident memory. Use `interaction_budget_summary.csv` to disclose any DreamerV3 iteration-boundary overshoot. Explain that equal environment interaction does not imply equal gradient updates or equal compute.

### 6.5 Baselines

Compare the learned agents descriptively with the single deterministic reference evaluation for rainfed, soil-moisture-threshold and fixed-interval policies in each scenario. These references are contextual comparators, not stochastic replicates and are excluded from the four-agent inferential tests.

## 7. Robustness and ablations

Recommended secondary analyses:

- binary `0/25 mm` action;
- legacy source reward versus corrected incremental-water reward;
- limited-water cap at 60%, 70% and 80% of reference requirement;
- stochastic policy evaluation where meaningful;
- canonical Danijar DreamerV3 replication in WSL2/Linux.

Label every changed protocol as secondary and do not merge it into the preregistered primary table.

## 8. Limitations

Discuss AquaCrop calibration, the small number of independent historical test years, simulator-to-field transfer, imperfect waterlogging representation, economic assumptions, hardware dependence, and the fact that algorithm defaults may favour some methods. State that field deployment and sensor integration are outside scope.

## 9. Conclusion

Answer the research question in terms of scenario-dependent trade-offs and reliability. A defensible conclusion may identify different leaders for yield protection, water conservation, return stability and compute efficiency.
