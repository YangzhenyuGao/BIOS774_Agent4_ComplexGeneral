"""Decision rules shared by the agent and the evaluator (README 4.3)."""
import numpy as np
from scipy import sparse

from agent4 import stats
from agent4.tournament import bradley_terry


def test_read_split_conserves_counts_and_is_seeded():
    rng = np.random.default_rng(0)
    counts = sparse.csr_matrix(rng.poisson(1.0, size=(50, 40)))
    a, b = stats.thin_counts(counts, seed=1)
    assert ((a + b) != counts).nnz == 0 and a.min() >= 0 and b.min() >= 0
    a2, _ = stats.thin_counts(counts, seed=1)
    assert (a != a2).nnz == 0


def test_benjamini_hochberg_matches_manual_case():
    reject, adj = stats.bh_reject([0.001, 0.008, 0.039, 0.041, 0.9, np.nan], alpha=0.05)
    assert reject.tolist() == [True, True, False, False, False, False]
    assert np.isnan(adj[-1])


def test_paired_bootstrap_detects_a_better_predictor():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(400, 30)).astype(np.float32)
    good = truth + rng.normal(scale=0.5, size=truth.shape).astype(np.float32)
    weak = truth + rng.normal(scale=3.0, size=truth.shape).astype(np.float32)
    out = stats.paired_bootstrap_delta_r(good, weak, truth, n_boot=100, seed=0)
    assert np.all(out["delta"] > 0) and np.mean(out["p"] < 0.05) > 0.9
    same = stats.paired_bootstrap_delta_r(weak, weak, truth, n_boot=100, seed=0)
    assert np.allclose(same["delta"], 0)


def test_three_way_conclusion_uses_only_the_interval():
    assert stats.three_way_conclusion(0.01, 0.05) == "improves"
    assert stats.three_way_conclusion(-0.01, 0.05) == "insufficient_evidence"
    assert stats.three_way_conclusion(-0.05, -0.01) == "worse"


def test_bradley_terry_orders_a_clear_hierarchy():
    wins = np.array([[0, 2, 2], [0, 0, 2], [0, 0, 0]], dtype=float)
    s = bradley_terry(["a", "b", "c"], wins)
    assert s["a"] > s["b"] > s["c"]


def test_constant_prediction_counts_as_zero_correlation():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=(100, 3))
    pred = truth.copy()
    pred[:, 1] = 5.0                      # a method that predicts a constant for this gene
    r = stats.colcorr(pred, truth)
    assert r[1] == 0.0 and np.isclose(r[0], 1.0)
    assert np.isclose(np.nanmean(r), 2 / 3)
