"""Scientific protocol constants and pure calculations for the maize benchmark.

These definitions are intentionally kept independent of AquaCrop and Gymnasium
so the fixed irrigation and reward methodology can be unit-tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

DEPLETION_TRIGGER_FRACTION = 0.40
MAX_DAILY_IRRIGATION_MM = 25.0
MIN_IRRIGATION_EVENT_MM = 1.0
ACTION_MODE = "deficit_fraction"
REWARD_VERSION = "legacy_source"
YIELD_POWER = 4.0
MAIZE_PRICE_PER_TONNE_USD = 180.0
IRRIGATION_COST_PER_MM_HA_USD = 1.0
FIXED_COST_PER_HA_USD = 1728.0


@dataclass(frozen=True)
class IrrigationDecision:
    """Auditable result of mapping one policy action to applied irrigation."""

    action_fraction: float
    depletion_fraction: float
    requested_mm: float
    applied_mm: float


def deficit_fraction_irrigation(
    *,
    native_action_mm: float,
    depletion_mm: float,
    taw_mm: float,
    seasonal_irrigation_mm: float,
    seasonal_cap_mm: float | None,
) -> IrrigationDecision:
    """Apply the frozen deficit-fraction irrigation rule.

    The policy interface is normalized to [-1, 1]. The action wrapper maps it
    to the native [0, 25] mm scale; that native value is used only to recover a
    [0, 1] fraction. The fraction is applied to current root-zone depletion.
    """

    values = {
        "native_action_mm": native_action_mm,
        "depletion_mm": depletion_mm,
        "taw_mm": taw_mm,
        "seasonal_irrigation_mm": seasonal_irrigation_mm,
    }
    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, received {value!r}.")

    native_action_mm = float(native_action_mm)
    depletion_mm = float(depletion_mm)
    taw_mm = float(taw_mm)
    seasonal_irrigation_mm = float(seasonal_irrigation_mm)

    if depletion_mm < 0 or taw_mm < 0 or seasonal_irrigation_mm < 0:
        raise ValueError("Water-state values cannot be negative.")

    if seasonal_cap_mm is not None:
        seasonal_cap_mm = float(seasonal_cap_mm)
        if not math.isfinite(seasonal_cap_mm) or seasonal_cap_mm < 0:
            raise ValueError("seasonal_cap_mm must be finite and non-negative.")

    clipped_native = min(max(native_action_mm, 0.0), MAX_DAILY_IRRIGATION_MM)
    action_fraction = clipped_native / MAX_DAILY_IRRIGATION_MM
    depletion_fraction = depletion_mm / taw_mm if taw_mm > 0 else 0.0

    if depletion_fraction < DEPLETION_TRIGGER_FRACTION:
        requested_mm = 0.0
    else:
        requested_mm = action_fraction * depletion_mm

    limits = [requested_mm, MAX_DAILY_IRRIGATION_MM, depletion_mm]
    if seasonal_cap_mm is not None:
        remaining_allocation = max(
            0.0, seasonal_cap_mm - seasonal_irrigation_mm
        )
        limits.append(remaining_allocation)

    applied_mm = max(0.0, min(limits))

    # Events smaller than one millimetre are treated as zero. Do not round
    # small requests upward, as that could exceed the policy's requested depth.
    if 0.0 < applied_mm < MIN_IRRIGATION_EVENT_MM:
        applied_mm = 0.0

    return IrrigationDecision(
        action_fraction=float(action_fraction),
        depletion_fraction=float(depletion_fraction),
        requested_mm=float(requested_mm),
        applied_mm=float(applied_mm),
    )


def legacy_source_step_reward(
    *,
    seasonal_irrigation_after_step_mm: float,
    applied_irrigation_mm: float,
    terminated: bool,
    final_dry_yield_tonne_per_ha: float,
) -> float:
    """Return one step of the Implementation Document's legacy_source reward.

    At every irrigation event the penalty is the cumulative seasonal depth
    applied up to and including that timestep. The final step adds Y**4.
    """

    cumulative = float(seasonal_irrigation_after_step_mm)
    applied = float(applied_irrigation_mm)
    dry_yield = float(final_dry_yield_tonne_per_ha)
    if not all(math.isfinite(v) for v in (cumulative, applied, dry_yield)):
        raise ValueError("Reward inputs must be finite.")
    if cumulative < 0 or applied < 0 or dry_yield < 0:
        raise ValueError("Reward inputs cannot be negative.")

    reward = -cumulative if applied > 0.0 else 0.0
    if terminated:
        reward += dry_yield ** YIELD_POWER
    return float(reward)


def evaluation_metrics(
    *, dry_yield_tonne_per_ha: float, seasonal_irrigation_mm: float
) -> dict[str, float]:
    """Calculate the Implementation Document's separate evaluation metrics."""

    dry_yield = float(dry_yield_tonne_per_ha)
    irrigation = float(seasonal_irrigation_mm)
    if not math.isfinite(dry_yield) or not math.isfinite(irrigation):
        raise ValueError("Evaluation inputs must be finite.")
    if dry_yield < 0 or irrigation < 0:
        raise ValueError("Evaluation inputs cannot be negative.")

    water_efficiency = (
        1000.0 * dry_yield / irrigation if irrigation > 0.0 else math.nan
    )
    profit = (
        MAIZE_PRICE_PER_TONNE_USD * dry_yield
        - IRRIGATION_COST_PER_MM_HA_USD * irrigation
        - FIXED_COST_PER_HA_USD
    )
    return {
        "water_productivity_kg_per_ha_per_mm": float(water_efficiency),
        "profit": float(profit),
    }
