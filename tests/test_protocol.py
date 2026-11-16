import math

import numpy as np
import pytest

from maize_benchmark.protocol import (
    ACTION_MODE,
    DEPLETION_TRIGGER_FRACTION,
    MAX_DAILY_IRRIGATION_MM,
    MIN_IRRIGATION_EVENT_MM,
    REWARD_VERSION,
    deficit_fraction_irrigation,
    evaluation_metrics,
    legacy_source_step_reward,
)


def test_frozen_protocol_constants_match_implementation_document():
    assert ACTION_MODE == "deficit_fraction"
    assert REWARD_VERSION == "legacy_source"
    assert DEPLETION_TRIGGER_FRACTION == 0.40
    assert MAX_DAILY_IRRIGATION_MM == 25.0
    assert MIN_IRRIGATION_EVENT_MM == 1.0


def test_deficit_fraction_action_applies_fraction_of_current_deficit():
    # Native 12.5 mm is only a carrier for a 0.5 action fraction.
    decision = deficit_fraction_irrigation(
        native_action_mm=12.5,
        depletion_mm=80.0,
        taw_mm=200.0,
        seasonal_irrigation_mm=0.0,
        seasonal_cap_mm=None,
    )
    assert decision.action_fraction == pytest.approx(0.5)
    assert decision.depletion_fraction == pytest.approx(0.4)
    assert decision.requested_mm == pytest.approx(40.0)
    assert decision.applied_mm == pytest.approx(25.0)


def test_irrigation_is_blocked_below_forty_percent_taw():
    decision = deficit_fraction_irrigation(
        native_action_mm=25.0,
        depletion_mm=79.0,
        taw_mm=200.0,
        seasonal_irrigation_mm=0.0,
        seasonal_cap_mm=None,
    )
    assert decision.requested_mm == 0.0
    assert decision.applied_mm == 0.0


def test_irrigation_respects_current_deficit_daily_maximum_and_seasonal_cap():
    decision = deficit_fraction_irrigation(
        native_action_mm=25.0,
        depletion_mm=80.0,
        taw_mm=200.0,
        seasonal_irrigation_mm=90.0,
        seasonal_cap_mm=100.0,
    )
    assert decision.requested_mm == 80.0
    assert decision.applied_mm == 10.0


def test_irrigation_requests_below_one_mm_are_zeroed():
    decision = deficit_fraction_irrigation(
        native_action_mm=0.5,
        depletion_mm=80.0,
        taw_mm=200.0,
        seasonal_irrigation_mm=0.0,
        seasonal_cap_mm=None,
    )
    assert decision.requested_mm == pytest.approx(1.6)
    assert decision.applied_mm == pytest.approx(1.6)

    tiny = deficit_fraction_irrigation(
        native_action_mm=0.25,
        depletion_mm=80.0,
        taw_mm=200.0,
        seasonal_irrigation_mm=0.0,
        seasonal_cap_mm=None,
    )
    assert tiny.requested_mm == pytest.approx(0.8)
    assert tiny.applied_mm == 0.0


def test_legacy_source_reward_is_terminal_yield_power_four_minus_cumulative_penalties():
    first_event = legacy_source_step_reward(
        seasonal_irrigation_after_step_mm=5.0,
        applied_irrigation_mm=5.0,
        terminated=False,
        final_dry_yield_tonne_per_ha=0.0,
    )
    no_event = legacy_source_step_reward(
        seasonal_irrigation_after_step_mm=5.0,
        applied_irrigation_mm=0.0,
        terminated=False,
        final_dry_yield_tonne_per_ha=0.0,
    )
    second_event_and_terminal = legacy_source_step_reward(
        seasonal_irrigation_after_step_mm=8.0,
        applied_irrigation_mm=3.0,
        terminated=True,
        final_dry_yield_tonne_per_ha=10.0,
    )
    assert first_event == -5.0
    assert no_event == 0.0
    assert second_event_and_terminal == 10.0**4 - 8.0
    assert first_event + no_event + second_event_and_terminal == 10.0**4 - (5.0 + 8.0)


def test_evaluation_metrics_are_separate_from_training_reward():
    metrics = evaluation_metrics(
        dry_yield_tonne_per_ha=13.0,
        seasonal_irrigation_mm=100.0,
    )
    assert metrics["water_productivity_kg_per_ha_per_mm"] == pytest.approx(130.0)
    assert metrics["profit"] == pytest.approx(512.0)

    rainfed_metrics = evaluation_metrics(
        dry_yield_tonne_per_ha=13.0,
        seasonal_irrigation_mm=0.0,
    )
    assert math.isnan(rainfed_metrics["water_productivity_kg_per_ha_per_mm"])
    assert rainfed_metrics["profit"] == pytest.approx(612.0)


def test_protocol_rejects_invalid_values():
    with pytest.raises(ValueError, match="finite"):
        deficit_fraction_irrigation(
            native_action_mm=np.nan,
            depletion_mm=80.0,
            taw_mm=200.0,
            seasonal_irrigation_mm=0.0,
            seasonal_cap_mm=None,
        )
    with pytest.raises(ValueError, match="negative"):
        evaluation_metrics(
            dry_yield_tonne_per_ha=13.0,
            seasonal_irrigation_mm=-1.0,
        )
