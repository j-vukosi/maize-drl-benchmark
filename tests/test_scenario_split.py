from pathlib import Path

from maize_benchmark.envs.scenarios import load_scenario


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_scenario_splits_are_disjoint_and_loaded():
    expected_counts = {
        "normal": (7, 2, 2),
        "dry": (5, 1, 1),
        "extreme_dry": (4, 1, 1),
        "limited_water": (7, 2, 2),
        "wet": (4, 2, 2),
    }
    for name, counts in expected_counts.items():
        scenario = load_scenario(PROJECT_ROOT, name)
        assert (
            len(scenario.train_years),
            len(scenario.validation_years),
            len(scenario.test_years),
        ) == counts
        train, validation, test = map(
            set,
            (
                scenario.train_years,
                scenario.validation_years,
                scenario.test_years,
            ),
        )
        assert not train & validation
        assert not train & test
        assert not validation & test


def test_limited_water_uses_the_checked_in_smt80_cap_table():
    scenario = load_scenario(PROJECT_ROOT, "limited_water")
    assert len(scenario.cap_lookup) == 11
    assert scenario.cap_for_year(1988) == 363.303
    assert scenario.cap_for_year(1990) == 349.097


def test_scenario_years_match_the_frozen_research_partitions():
    expected = {
        "normal": {
            "train": (1990, 1994, 1995, 2005, 2006, 2010, 2018),
            "validation": (1999, 2007),
            "test": (1988, 1998),
        },
        "dry": {
            "train": (1985, 1997, 2013, 2016, 2017),
            "validation": (2001,),
            "test": (1989,),
        },
        "extreme_dry": {
            "train": (1984, 2002, 2003, 2012),
            "validation": (1983,),
            "test": (2000,),
        },
        "limited_water": {
            "train": (1990, 1994, 1995, 2005, 2006, 2010, 2018),
            "validation": (1999, 2007),
            "test": (1988, 1998),
        },
        "wet": {
            "train": (1993, 1996, 2009, 2015),
            "validation": (1991, 2004),
            "test": (1992, 2014),
        },
    }
    for name, splits in expected.items():
        scenario = load_scenario(PROJECT_ROOT, name)
        assert scenario.train_years == splits["train"]
        assert scenario.validation_years == splits["validation"]
        assert scenario.test_years == splits["test"]
