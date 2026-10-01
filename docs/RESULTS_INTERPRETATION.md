# Results Interpretation Guide

## Do not search for one universal winner

Irrigation scheduling is a trade-off problem. A policy can raise yield by using much more water, or save water by accepting an unacceptable yield loss. The final discussion should identify profiles such as:

- yield-protecting;
- water-conserving;
- economically balanced;
- robust under scarcity;
- computationally practical.

## Primary tables

Report each algorithm × scenario with mean, standard deviation, median and 95% confidence interval for yield, irrigation, water productivity, profit, stress and reward. Add wall-clock time and memory in a separate table.

## Paired inference

The Friedman and Wilcoxon analyses use matched training seeds within each scenario after averaging that seed's shared held-out climate years. This keeps the independently trained policy—not each repeated climate evaluation—as the inferential unit. Interpret adjusted p-values together with median paired differences and win rates. With five seeds, power remains limited, so uncertainty and effect size must remain central.

## Pareto analysis

An algorithm is yield–water Pareto-efficient in a block when no competitor has both at least as much yield and no more irrigation, with one strict improvement. Pareto frequency is more defensible than inventing one weighted score.

## Learning efficiency

Use real environment steps on the horizontal axis. Do not compare PPO iterations with Dreamer iterations. The supplied report computes reward-curve AUC against environment steps and reports compute cost separately.

## Claims to avoid

Do not write:

- “Algorithm X is best” from one seed;
- “significantly better” without a statistical test and effect size;
- “water efficient” based only on low irrigation when yield collapsed;
- “robust to flooding” when the simulator only represents high rainfall imperfectly;
- “full DSAC-T” for the old simplified prototype;
- “official DreamerV3” when using the RLlib backend without naming it.

## Suitable conclusion pattern

“Under the preregistered held-out years, Algorithm A most frequently occupied the yield–irrigation Pareto set, while Algorithm B achieved the lowest irrigation and Algorithm C learned fastest per environment step. Differences were scenario-dependent, and compute cost favoured …”
