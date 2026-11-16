#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.cwd()
OUTPUT_ROOT = ROOT / "outputs"
RUNS_ROOT = OUTPUT_ROOT / "runs"
FIG_DIR = OUTPUT_ROOT / "figures" / "baseline_full_climate"
SUM_DIR = OUTPUT_ROOT / "summaries"

FIG_DIR.mkdir(parents=True, exist_ok=True)
SUM_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# User-requested colours
# -----------------------------
BAR_COLORS = {
    "ppo": "#1f77b4",        # blue
    "crossq": "#ff7f0e",     # orange
    "dsac_t": "#2ca02c",     # green
    "dreamerv3": "#d62728",  # red
}

DISPLAY_NAMES = {
    "ppo": "PPO",
    "crossq": "CrossQ",
    "dsac_t": "DSAC-T",
    "dreamerv3": "DreamerV3",
}

SCENARIO_ORDER = ["extreme_dry", "dry", "normal", "wet"]
SCENARIO_LABELS = {
    "extreme_dry": "Extreme Dry",
    "dry": "Dry",
    "normal": "Normal",
    "wet": "Wet",
}

METRICS = {
    "dry_yield_tonne_per_ha": {
        "ylabel": "Yield (t/ha)",
        "filename": "mean_dry_yield_by_climate_scenario.png",
        "show_errorbars": True,
    },
    "total_irrigation_mm": {
        "ylabel": "Seasonal Irrigation (mm)",
        "filename": "total_irrigation_by_climate_scenario.png",
        "show_errorbars": True,
    },
    "water_productivity_kg_per_ha_per_mm": {
        "ylabel": "Water Efficiency (kg/ha/mm)",
        "filename": "water_efficiency_by_climate_scenario.png",
        # keep baseline-like presentation: no error bars here
        "show_errorbars": False,
    },
    "profit": {
        "ylabel": "Profit (USD/ha)",
        "filename": "profit_usd_per_ha_by_climate_scenario.png",
        # keep baseline-like presentation: no error bars here
        "show_errorbars": False,
    },
}

ALGORITHMS = ["ppo", "crossq", "dsac_t", "dreamerv3"]

def bootstrap_mean_ci(values, resamples=10000, alpha=0.05, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, n), replace=True)
    means = draws.mean(axis=1)
    lower = float(np.quantile(means, alpha / 2))
    upper = float(np.quantile(means, 1 - alpha / 2))
    return lower, upper

# ----------------------------------------------------
# 1) Build / refresh year classification from baseline
# ----------------------------------------------------
manifest_path = ROOT / "scenarios" / "baseline_full" / "climate_manifest.csv"
manifest = pd.read_csv(manifest_path)

train = manifest[manifest["split"] == "train"].copy()
test = manifest[manifest["split"] == "test"].copy()

q = train["seasonal_precip_mm"].quantile([0.10, 0.33, 0.67])
q10, q33, q67 = q.iloc[0], q.iloc[1], q.iloc[2]

def classify_year(x: float) -> str:
    if x < q10:
        return "extreme_dry"
    elif x < q33:
        return "dry"
    elif x < q67:
        return "normal"
    else:
        return "wet"

test["climate_scenario"] = test["seasonal_precip_mm"].map(classify_year)

classification = test[["year", "climate_scenario", "seasonal_precip_mm", "seasonal_et0_mm"]].copy()
classification.to_csv(SUM_DIR / "climate_year_classification.csv", index=False)

# ----------------------------------------------------
# 2) Load all baseline_full evaluation files
# ----------------------------------------------------
files = sorted(RUNS_ROOT.glob("*/baseline_full/seed_*/evaluation.csv"))
if not files:
    raise FileNotFoundError("No evaluation.csv files found under outputs/runs/*/baseline_full/seed_*/evaluation.csv")

frames = []
for path in files:
    algorithm = path.parts[-4]
    if algorithm not in ALGORITHMS:
        continue
    seed = int(path.parent.name.split("_")[-1])
    frame = pd.read_csv(path)
    frame["algorithm"] = algorithm
    frame["seed"] = seed
    frames.append(frame)

if not frames:
    raise RuntimeError("No recognised baseline_full evaluation files found for PPO, CrossQ, DSAC-T, or DreamerV3.")

all_eval = pd.concat(frames, ignore_index=True)
all_eval = all_eval.merge(classification[["year", "climate_scenario"]], on="year", how="inner", validate="many_to_one")

# ----------------------------------------------------
# 3) Per-seed climate-scenario means
# ----------------------------------------------------
metric_names = list(METRICS.keys())

seed_summary = (
    all_eval
    .groupby(["climate_scenario", "algorithm", "seed"], as_index=False)[metric_names]
    .mean()
)

seed_summary.to_csv(SUM_DIR / "climate_scenario_seed_summary.csv", index=False)

# ----------------------------------------------------
# 4) Across-seed summary + bootstrap CIs
# ----------------------------------------------------
summary_rows = []
years_per_group = classification.groupby("climate_scenario")["year"].nunique().to_dict()

for climate_scenario in SCENARIO_ORDER:
    subset_cs = seed_summary[seed_summary["climate_scenario"] == climate_scenario]
    for algorithm in ALGORITHMS:
        subset_alg = subset_cs[subset_cs["algorithm"] == algorithm]
        for metric in metric_names:
            vals = pd.to_numeric(subset_alg[metric], errors="coerce").dropna().to_numpy(dtype=float)
            mean = float(np.mean(vals)) if len(vals) else np.nan
            sd = float(np.std(vals, ddof=1)) if len(vals) >= 2 else np.nan
            ci_lower, ci_upper = bootstrap_mean_ci(vals, resamples=10000, alpha=0.05, seed=42)
            summary_rows.append(
                {
                    "climate_scenario": climate_scenario,
                    "algorithm": algorithm,
                    "metric": metric,
                    "mean": mean,
                    "sd": sd,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "n_valid_seeds": int(len(vals)),
                    "n_years": int(years_per_group.get(climate_scenario, 0)),
                }
            )

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUM_DIR / "climate_scenario_summary.csv", index=False)

# ----------------------------------------------------
# 5) Plot helper
# ----------------------------------------------------
def plot_metric(metric: str) -> None:
    info = METRICS[metric]
    data = summary[summary["metric"] == metric].copy()

    x = np.arange(len(SCENARIO_ORDER), dtype=float)
    width = 0.84 / len(ALGORITHMS)

    fig, ax = plt.subplots(figsize=(14, 8))
    max_seen = []

    for i, algorithm in enumerate(ALGORITHMS):
        subset = (
            data[data["algorithm"] == algorithm]
            .set_index("climate_scenario")
            .reindex(SCENARIO_ORDER)
        )

        means = subset["mean"].to_numpy(dtype=float)
        ci_lower = subset["ci_lower"].to_numpy(dtype=float)
        ci_upper = subset["ci_upper"].to_numpy(dtype=float)
        n_valid = subset["n_valid_seeds"].to_numpy(dtype=float)

        positions = x - 0.42 + width / 2 + i * width
        valid = np.isfinite(means)

        ax.bar(
            positions[valid],
            means[valid],
            width=width * 0.94,
            color=BAR_COLORS[algorithm],
            edgecolor="black",
            linewidth=0.8,
            label=DISPLAY_NAMES[algorithm],
            zorder=2,
        )

        if info["show_errorbars"]:
            ci_valid = valid & (n_valid >= 2) & np.isfinite(ci_lower) & np.isfinite(ci_upper)
            if ci_valid.any():
                lo = np.maximum(0.0, means[ci_valid] - ci_lower[ci_valid])
                hi = np.maximum(0.0, ci_upper[ci_valid] - means[ci_valid])
                ax.errorbar(
                    positions[ci_valid],
                    means[ci_valid],
                    yerr=np.vstack([lo, hi]),
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.5,
                    capsize=4,
                    capthick=1.5,
                    zorder=5,
                )

        for j, (xp, mean) in enumerate(zip(positions, means)):
            if not np.isfinite(mean):
                continue

            top = mean
            if info["show_errorbars"] and np.isfinite(ci_upper[j]) and int(n_valid[j]) >= 2:
                top = ci_upper[j]

            max_seen.append(float(top))
            ax.annotate(
                f"{mean:,.2f}",
                xy=(xp, top),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.25),
                clip_on=False,
                zorder=6,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
    ax.set_xlabel("Climate scenario")
    ax.set_ylabel(info["ylabel"])

    # no chart title, in line with your preference
    ax.grid(axis="y", color="#d0d0d0", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    # keep full border around plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    ax.legend(ncol=4, loc="upper left", frameon=False)

    if max_seen:
        ymin, ymax = ax.get_ylim()
        top = max(max_seen)
        span = max(top - ymin, 1.0)
        ax.set_ylim(ymin, max(ymax, top + 0.16 * span))

    fig.tight_layout()
    out_path = FIG_DIR / info["filename"]
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED: {out_path}")

# ----------------------------------------------------
# 6) Generate the 4 PNG graphs
# ----------------------------------------------------
for metric in METRICS:
    plot_metric(metric)

print("\nDONE")
print(f"Figures: {FIG_DIR}")
print(f"Summaries: {SUM_DIR}")
