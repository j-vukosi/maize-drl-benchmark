# Evidence Base for the Five Scenarios

## Principle

Recent studies justify testing distinct moisture regimes, but the exact percentile thresholds are an explicit benchmark operationalisation. This distinction prevents the report from falsely claiming that a paper prescribed a particular folder name or quantile.

## `normal`

Normal represents central historical rainfall, not an idealised constant season. Marcos-Garcia, Carmona-Moreno and Pastori (2024) found that within-season dry–wet spell patterns strongly influence maize yield variability in sub-Saharan Africa. This supports testing ordinary interannual and intra-season variability rather than only a single average weather year.

Primary source: *Nature Food*, 2024, “Intra-growing season dry–wet spell pattern is a pivotal driver of maize yield variability in sub-Saharan Africa.” DOI: https://doi.org/10.1038/s43016-024-01040-8

## `dry`

Dry covers sustained below-normal seasonal rainfall. Du et al. (2024) identify drought among the major weather extremes threatening African maize production. The scenario tests whether an agent protects yield while avoiding wasteful irrigation during a moderate rainfall deficit.

Primary source: *Weather Extremes Shock Maize Production*, 2024: https://pmc.ncbi.nlm.nih.gov/articles/PMC11207875/

## `extreme_dry`

Extreme dry isolates the lowest historical rainfall tail. It is separated from `dry` because average performance can hide failure under severe scarcity. The 2024 southern African drought also demonstrated the practical importance of extreme maize-season water deficits, while the scientific benchmark uses the historical climate archive rather than a news event as data.

Scientific support: Du et al. (2024), above; Marcos-Garcia et al. (2024), above.

Regional context: SADC Agromet Special Update, 2024: https://fews.net/sites/default/files/2024-04/SADC%20Agromet%20Special%20Update%20Issue-05%20-%202023-2024%20Season_Final.pdf

## `limited_water`

Limited water is not simply another dry-weather folder. It holds climate near the normal stratum while imposing an explicit seasonal irrigation allocation. This separates **resource restriction** from **meteorological drought**.

Jiao et al. (2024) report that regulated deficit irrigation can stabilise maize yield while improving water productivity, with their 65/80ET treatment providing a strong water-saving balance. Melkie et al. (2024) also report improved water-use efficiency from deficit irrigation under water scarcity. These studies support a controlled allocation scenario and sensitivity analysis, not one universal cap.

Primary sources:

- Jiao et al., *Agricultural Water Management* 297 (2024), 108827. DOI: https://doi.org/10.1016/j.agwat.2024.108827
- Melkie et al., *Frontiers in Agronomy* (2024), “Optimizing water use efficiency in maize…” DOI: https://doi.org/10.3389/fagro.2024.1490423

The primary cap is 70% of an independently simulated SMT-80 reference requirement, with 60% and 80% preregistered as sensitivity analyses.

## `wet`

Wet tests whether an irrigation policy recognises abundant rainfall and avoids unnecessary water application. It also probes robustness to excessive soil moisture. A 2024 meta-analysis found that waterlogging reduces crop yield substantially, while Marcos-Garcia et al. (2024) show that dry–wet spell sequencing matters for African maize yield.

Primary sources:

- Yang et al., *Crop and Environment* (2024), “Implications of soil waterlogging for crop quality: A meta-analysis.” https://www.sciencedirect.com/science/article/pii/S1161030124003162
- Marcos-Garcia et al. (2024), DOI above.

AquaCrop's representation of waterlogging must be discussed as a model limitation; this scenario is best interpreted as a high-rainfall/over-irrigation avoidance test rather than a perfect flood experiment.

## Scenario audit outputs

`python scripts/prepare_scenarios.py` writes:

- the rainfall and ET0 summary for every year;
- exact train/validation/test year lists;
- the climate-file SHA-256 hash;
- each scenario's quantile bounds and split manifest.

`python scripts/calibrate_water_caps.py` writes the reference irrigation, 70% cap and reference yield for each limited-water year.
