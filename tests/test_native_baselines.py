from maize_benchmark.baselines.runner import native_irrigation_settings
import pytest


def test_native_reference_strategies_match_paper():
    assert native_irrigation_settings('rainfed', None) == {
        'irrigation_method': 0, 'MaxIrr': 25.0,
    }
    assert native_irrigation_settings('soil_moisture_threshold', 300) == {
        'irrigation_method': 1, 'SMT': [23.72, 26.46, 38.19, 50.11],
        'MaxIrr': 25.0, 'MaxIrrSeason': 300.0,
    }
    assert native_irrigation_settings('fixed_interval', 0) == {
        'irrigation_method': 2, 'IrrInterval': 7,
        'MaxIrr': 25.0, 'MaxIrrSeason': 0.0,
    }
    with pytest.raises(ValueError):
        native_irrigation_settings('unknown', None)
    with pytest.raises(ValueError):
        native_irrigation_settings('rainfed', -1)


def test_native_baselines_do_not_use_40pct_gate():
    for strategy in ['rainfed', 'soil_moisture_threshold', 'fixed_interval']:
        assert 'depletion_trigger_fraction' not in native_irrigation_settings(strategy, None)
