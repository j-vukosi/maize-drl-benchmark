import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_scenarios.py"
spec = importlib.util.spec_from_file_location("prepare_scenarios", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_deterministic_split_has_no_leakage():
    years = list(range(1982, 1992))
    split1 = module.deterministic_split(
        years, validation_fraction=0.2, test_fraction=0.2, seed=4279123
    )
    split2 = module.deterministic_split(
        years, validation_fraction=0.2, test_fraction=0.2, seed=4279123
    )
    assert split1 == split2
    train, validation, test = map(set, split1)
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert train | validation | test == set(years)
