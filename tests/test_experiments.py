"""Checks of the experiment scripts (no training, a few seconds): every module imports, and the LaTeX tables
regenerated from the stored evaluation files are those of the paper."""
import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import experiments  # noqa: E402

MODULES = sorted(m.name for m in pkgutil.walk_packages(experiments.__path__, "experiments."))


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_tables_from_stored_results():
    from experiments.appC_trajectories import plot as appC
    from experiments.exp1_dataset_size import plot as ds

    stored = lambda exp, name: (ROOT / "experiments" / exp / "figures" / name).read_text()
    results, _ = ds.load()
    assert ds.table(results) + "\n" == stored("exp1_dataset_size", "table.tex")
    results_c, _ = appC.load()
    assert appC.table(results_c) + "\n" == stored("appC_trajectories", "table.tex")
    assert appC.table_equilibria(results_c) + "\n" == stored("appC_trajectories", "table_equilibria.tex")
