"""Locations known only to the evaluator. Nothing under src/agent4 may import this module (tests enforce it)."""
from pathlib import Path

from agent4.io import ROOT

DATA = ROOT.parent / "data" / "pbmc_granulocyte_sorted_10k"
AGENT3 = ROOT.parent / "Agent3_RepresentationLearning" / "outputs"
TCGA_PREPROCESS = AGENT3 / "tcga_brca" / "preprocess"
FEATURE_BANK = AGENT3 / "tcga_brca_hw4_feature_bank"

PRIVATE = ROOT / "bench_private"            # hidden measurements, evaluator folds, true scores


def view_dir(task_id: str) -> Path:
    return ROOT / "outputs" / task_id / "agent_view"


def private_dir(task_id: str) -> Path:
    return PRIVATE / task_id
