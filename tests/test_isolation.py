"""The agent must not be able to reach hidden data (README 9.1). These tests fail if that ever changes."""
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = sorted((ROOT / "src" / "agent4").rglob("*.py"))


def test_agent_package_never_imports_the_evaluator():
    pattern = re.compile(r"^\s*(from|import)\s+bench(\.|\s|$)", flags=re.M)
    offenders = [str(p.relative_to(ROOT)) for p in AGENT_SRC if pattern.search(p.read_text())]
    assert not offenders, f"agent code imports the evaluator: {offenders}"


def test_agent_package_never_names_private_locations():
    offenders = [str(p.relative_to(ROOT)) for p in AGENT_SRC
                 if re.search(r"bench_private|hidden_rna|true_scores", p.read_text())]
    assert not offenders, f"agent code refers to evaluator-only data: {offenders}"


def test_task_m_view_holds_no_rna_of_held_out_cells():
    view = ROOT / "outputs" / "task_m" / "agent_view"
    if not view.exists():
        pytest.skip("view not built")
    names = {p.name for p in view.iterdir()}
    assert not {n for n in names if "hidden" in n and "rna" in n}, names
    dev, hidden = pd.read_parquet(view / "dev_obs.parquet"), pd.read_parquet(view / "hidden_obs.parquet")
    assert not set(dev.index) & set(hidden.index), "development and held-out cells overlap"
    assert list(hidden.columns) == ["n_atac"], f"held-out table exposes more than ATAC depth: {list(hidden.columns)}"


def test_task_m_loader_exposes_no_hidden_rna():
    view_dir = ROOT / "outputs" / "task_m" / "agent_view"
    if not (view_dir / "rules.json").exists():
        pytest.skip("view not built")
    from agent4.data import load_cell_view
    view = load_cell_view(view_dir)
    assert not [a for a in vars(view) if "hidden" in a and "rna" in a]
    assert view.dev_rna.shape[0] == len(view.dev_obs) and view.hidden_atac.shape[0] == len(view.hidden_obs)
