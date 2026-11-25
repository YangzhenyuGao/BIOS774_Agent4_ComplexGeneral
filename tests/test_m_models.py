"""Fast checks of the task M models on tiny synthetic data (no scvi components; those are smoke-tested on real data)."""
import os

os.environ.setdefault("A4_THREADS", "2")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from agent4.data import CellView
from agent4.tools import m_models as mm

N_CELLS, N_PEAKS, N_GENES = 260, 300, 40


@pytest.fixture(scope="module")
def view():
    rng = np.random.default_rng(0)
    cluster = rng.integers(0, 3, N_CELLS)
    peak_rate = rng.gamma(1.0, 0.25, (3, N_PEAKS))
    atac = rng.poisson(peak_rate[cluster] * rng.uniform(0.6, 1.6, (N_CELLS, 1)))
    gene_rate = rng.gamma(1.5, 1.0, (3, N_GENES))
    cis_peak = np.arange(N_GENES) * (N_PEAKS // N_GENES)           # one driving peak per gene
    rna = rng.poisson(gene_rate[cluster] * (1.0 + 0.8 * atac[:, cis_peak]))
    genes = pd.DataFrame(dict(name=[f"g{i}" for i in range(N_GENES)], gene_id=[f"E{i}" for i in range(N_GENES)],
                              chrom="chr1", tss=(np.arange(N_GENES) * (N_PEAKS // N_GENES) * 1000.0 + 500.0), n_cells_dev=N_CELLS))
    peaks = pd.DataFrame(dict(name=[f"chr1:{i * 1000}-{i * 1000 + 400}" for i in range(N_PEAKS)], chrom="chr1",
                              start=np.arange(N_PEAKS) * 1000, end=np.arange(N_PEAKS) * 1000 + 400))
    peak_gene = pd.DataFrame(dict(peak_idx=cis_peak, gene_idx=np.arange(N_GENES), distance=0, peak_type="promoter"))
    obs = pd.DataFrame(dict(celltype=cluster.astype(str), n_rna=rna.sum(1), n_atac=atac.sum(1)))
    rules = dict(gene_scope=list(range(N_GENES)), cluster_k=3, baseline_lsi=dict(n_components=8, max_peaks=N_PEAKS))
    return CellView(root=Path("."), meta={}, dev_rna=sparse.csr_matrix(rna), dev_atac=sparse.csr_matrix(atac),
                    hidden_atac=sparse.csr_matrix(atac[:20]), dev_obs=obs, hidden_obs=obs.iloc[:20], genes=genes, peaks=peaks,
                    peak_gene=peak_gene, rules=rules)


def spec(component, params=None, inputs=None, training=None):
    return dict(model=dict(component=component, params=params or {}), inputs=inputs or {},
                training=dict(dict(max_epochs=6, early_stopping_patience=3), **(training or {})))


LSI8 = {"atac_representation": [{"type": "lsi", "n_components": 8, "max_peaks": N_PEAKS}]}
CASES = [
    ("cluster_mean", {"k": "frozen"}, {}, None),
    ("latent_ridge", {"alpha": 5.0}, LSI8, None),
    ("latent_knn", {"k": 10}, LSI8, None),
    ("latent_pls", {"n_components": 4}, LSI8, None),
    ("gene_activity_score", {"mode": "promoter"}, {}, None),
    ("translation_network", {"likelihood": "poisson", "encoder_hidden": [32], "latent_dim": 8}, {"tfidf_peaks": {"max_peaks": 200}}, None),
    ("translation_network", {"likelihood": "nb", "encoder_hidden": [32], "latent_dim": 8, "decoder": "linear"}, LSI8, None),
    ("translation_network", {"likelihood": "mse", "encoder_hidden": 32, "latent_dim": 8}, [{"type": "lsi", "n_components": 8}, "depth", {"type": "gene_activity", "mode": "promoter"}], None),
    ("residual_over_cluster_mean", {"learner": "mlp", "hidden": [32]}, LSI8, None),
    ("residual_over_cluster_mean", {"learner": "gbm"}, LSI8, 6),
    ("cis_regulatory", {"learner": "masked_nn", "window_kb": 3, "max_peaks_per_gene": 4, "global_state_dim": 6}, {}, None),
    ("cis_regulatory", {"learner": "elastic_net", "window_kb": 3, "max_peaks_per_gene": 4, "global_state_dim": 6}, {}, 6),
    ("cis_regulatory", {"learner": "gbm", "window_kb": 3, "max_peaks_per_gene": 4, "global_state_dim": 6}, {"gene_activity": {"mode": "promoter"}}, 6),
    ("multitask_lowrank", {"rank": 4, "trunk_hidden": [32], "gene_embedding_dim": 8}, LSI8, None),
    ("contrastive_pretrain_head", {"embed_dim": 8, "pretrain_epochs": 4, "head": "mlp", "encoder_hidden": [32]}, LSI8, None),
    ("contrastive_pretrain_head", {"embed_dim": 8, "pretrain_epochs": 4, "head": "knn_transfer", "encoder_hidden": [32]}, LSI8, None),
    ("stacking_ensemble", {"members": [{"component": "latent_ridge", "params": {}, "inputs": LSI8},
                                       {"component": "residual_over_cluster_mean", "params": {"learner": "mlp", "hidden": [16]}, "inputs": LSI8}],
                           "meta": "per_gene_ridge", "oof_folds": 2}, {}, None),
    ("stacking_ensemble", {"members": ["latent_ridge", {"component": "latent_pls", "params": {"n_components": 3}}],
                           "meta": "global_weights", "oof_folds": 2}, LSI8, None),
]


@pytest.mark.parametrize("component,params,inputs,n_genes", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_fit_predict(view, component, params, inputs, n_genes):
    genes = None if n_genes is None else np.arange(n_genes)
    model = mm.build(spec(component, params, inputs), view, scale="pilot", seed=1, genes=genes)
    train, test = np.arange(200), np.arange(200, N_CELLS)
    report = model.fit(view.dev_atac[train], view.dev_rna[train])
    pred = model.predict(view.dev_atac[test])
    assert pred.shape == (len(test), n_genes or N_GENES) and pred.dtype == np.float32
    assert np.isfinite(pred).all()
    for key in ("seconds", "device", "epochs", "stopped_by", "converged", "convergence_note", "curves", "n_parameters"):
        assert key in report, key
    assert report["stopped_by"] in ("early_stopping", "max_epochs", "time_budget", "closed_form")
    assert isinstance(model.resolved, dict) and isinstance(model.ignored, list)
    if report["stopped_by"] not in ("closed_form",) and component != "stacking_ensemble":
        assert any(len(v) > 0 for v in report["curves"].values()), "iterative models must record a training curve"


def test_signal_is_learned(view):
    """A model with access to the driving peaks should beat the cluster-mean baseline on this synthetic data."""
    from agent4.stats import colcorr
    train, test = np.arange(200), np.arange(200, N_CELLS)
    truth = view.target(view.dev_rna[test])
    scores = {}
    for name, sp in (("cluster_mean", spec("cluster_mean", {"k": "frozen"})),
                     ("cis", spec("cis_regulatory", {"learner": "elastic_net", "window_kb": 3, "max_peaks_per_gene": 4, "global_state_dim": 6}))):
        model = mm.build(sp, view, seed=0)
        model.fit(view.dev_atac[train], view.dev_rna[train])
        scores[name] = float(np.nanmean(colcorr(model.predict(view.dev_atac[test]), truth)))
    assert scores["cis"] > scores["cluster_mean"], scores


def test_deterministic_given_seed(view):
    preds = []
    for _ in range(2):
        model = mm.build(spec("translation_network", {"likelihood": "mse", "encoder_hidden": [32], "latent_dim": 8}, LSI8), view, seed=3)
        model.fit(view.dev_atac[:200], view.dev_rna[:200])
        preds.append(model.predict(view.dev_atac[200:]))
    assert np.allclose(preds[0], preds[1], atol=1e-4)


def test_unknown_parameters_are_reported_not_fatal(view):
    sp = spec("residual_over_cluster_mean", {"learner": "mlp", "hidden": [16], "magic_number": 7}, LSI8, {"warmup_schedule": "cosine"})
    model = mm.build(sp, view, seed=0)
    model.fit(view.dev_atac[:200], view.dev_rna[:200])
    assert "model.params.magic_number" in model.ignored and "training.warmup_schedule" in model.ignored


def test_time_budget_never_claims_convergence(view):
    sp = spec("translation_network", {"likelihood": "mse", "encoder_hidden": [32], "latent_dim": 8}, LSI8, {"max_epochs": 50, "early_stopping_patience": 50})
    model = mm.build(sp, view, seed=0, time_budget=1e-6)
    report = model.fit(view.dev_atac[:200], view.dev_rna[:200])
    assert report["stopped_by"] == "time_budget" and report["converged"] is False


def test_target_denoising_and_explicit_validation(view):
    sp = spec("translation_network", {"likelihood": "poisson", "encoder_hidden": [32], "latent_dim": 8}, LSI8, {"target_denoising": {"type": "knn_pool", "k": 5}})
    model = mm.build(sp, view, seed=0)
    report = model.fit(view.dev_atac[:180], view.dev_rna[:180], val=(view.dev_atac[180:220], view.dev_rna[180:220]))
    assert report["n_val_cells"] == 40 and model.resolved["network"]["denoise_k"] == 5
    assert np.isfinite(model.predict(view.dev_atac[220:])).all()


def test_unknown_component_is_an_error(view):
    with pytest.raises(KeyError):
        mm.build(spec("no_such_component"), view)
