"""Smoke test of every task M component on real development cells (never touches held-out RNA).

    python scripts/smoke_m_models.py --group basic|nets|pergene|scvi|stack [--n-train 2500 --n-test 800 --n-genes 300]
    python scripts/smoke_m_models.py --group full --components joint_latent_multivi      full-size timing of one component

Each component is fitted on a subsample of development cells and scored on other development cells against a
thinned half of their RNA reads, exactly as the evaluator scores held-out cells. One line per component:
mean per-gene r, the frozen cluster-mean baseline's mean r on the same cells and genes, seconds, convergence.
These numbers show that the code paths work and how long they take; they are not results.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from agent4 import stats
from agent4.data import load_cell_view
from agent4.io import ROOT, write_json
from agent4.tools import m_models as mm

OUT = ROOT / "outputs" / "_smoke" / "m_models"
LSI50 = {"atac_representation": [{"type": "lsi", "n_components": 50, "max_peaks": 60000}]}
TFIDF = {"atac_representation": [{"type": "tfidf_peaks", "max_peaks": 30000}]}

SPECS = {
    "basic": [
        ("cluster_mean", dict(component="cluster_mean", params={"k": "frozen"}), {}, {}),
        ("latent_ridge", dict(component="latent_ridge", params={"alpha": 10.0}), LSI50, {}),
        ("latent_knn", dict(component="latent_knn", params={"k": 20}), LSI50, {}),
        ("latent_pls", dict(component="latent_pls", params={"n_components": 20}), LSI50, {}),
        ("gene_activity_score", dict(component="gene_activity_score", params={"mode": "promoter_and_distal"}), {}, {}),
    ],
    "nets": [
        ("translation_network[poisson,tfidf]", dict(component="translation_network", params={"likelihood": "poisson"}), TFIDF, {}),
        ("translation_network[nb,lsi]", dict(component="translation_network", params={"likelihood": "nb", "encoder_hidden": [512, 256]}),
         {"atac_representation": [{"type": "lsi", "n_components": 100}, {"type": "depth"}]}, {}),
        ("translation_network[mse,tfidf,knn_pool]", dict(component="translation_network", params={"likelihood": "mse"}), TFIDF,
         {"target_denoising": "knn_pool", "k": 10}),
        ("residual_over_cluster_mean[mlp]", dict(component="residual_over_cluster_mean", params={"learner": "mlp"}),
         {"atac_representation": [{"type": "lsi", "n_components": 100}, {"type": "gene_activity", "mode": "promoter_and_distal"}]}, {}),
        ("multitask_lowrank", dict(component="multitask_lowrank", params={"rank": 32, "gene_embedding_dim": 64}),
         {"atac_representation": [{"type": "lsi", "n_components": 100}]}, {}),
        ("cis_regulatory[masked_nn]", dict(component="cis_regulatory", params={"learner": "masked_nn", "window_kb": 100, "max_peaks_per_gene": 20,
                                                                                "global_state_dim": 50}), {}, {}),
        ("contrastive_pretrain_head[mlp]", dict(component="contrastive_pretrain_head", params={"head": "mlp", "embed_dim": 64}), TFIDF, {}),
        ("contrastive_pretrain_head[knn_transfer]", dict(component="contrastive_pretrain_head", params={"head": "knn_transfer", "embed_dim": 64}),
         {"atac_representation": [{"type": "lsi", "n_components": 100}]}, {}),
    ],
    "pergene": [
        ("residual_over_cluster_mean[gbm]", dict(component="residual_over_cluster_mean", params={"learner": "gbm"}), LSI50, {}),
        ("cis_regulatory[gbm]", dict(component="cis_regulatory", params={"learner": "gbm", "window_kb": 100, "max_peaks_per_gene": 20,
                                                                          "global_state_dim": 30}), {}, {}),
        ("cis_regulatory[elastic_net]", dict(component="cis_regulatory", params={"learner": "elastic_net", "window_kb": 250,
                                                                                  "max_peaks_per_gene": 30, "global_state_dim": 30}),
         {"atac_representation": [{"type": "gene_activity", "mode": "promoter"}]}, {}),
    ],
    "scvi": [
        ("joint_latent_multivi", dict(component="joint_latent_multivi", params={"n_latent": 20, "n_hidden": 256, "n_peaks": 30000,
                                                                                "max_epochs": 300, "early_stopping_patience": 20}), {}, {}),
        ("peakvi_latent_head[ridge]", dict(component="peakvi_latent_head", params={"n_latent": 30, "head": "ridge"}), {}, {}),
        ("peakvi_latent_head[mlp]", dict(component="peakvi_latent_head", params={"n_latent": 30, "head": "mlp"}), {}, {}),
    ],
    "stack": [
        ("stacking_ensemble[ridge+residual_mlp+translation]", dict(component="stacking_ensemble", params={
            "members": [{"component": "latent_ridge", "params": {"alpha": 10.0}, "inputs": LSI50},
                        {"component": "residual_over_cluster_mean", "params": {"learner": "mlp"}, "inputs": LSI50},
                        {"component": "translation_network", "params": {"likelihood": "poisson"}, "inputs": TFIDF}],
            "meta": "per_gene_ridge", "oof_folds": 3}), {}, {}),
    ],
}
PER_GENE_GROUPS = {"pergene"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=list(SPECS) + ["full"])
    ap.add_argument("--components", default="", help="comma-separated name prefixes to keep")
    ap.add_argument("--n-train", type=int, default=2500)
    ap.add_argument("--n-test", type=int, default=800)
    ap.add_argument("--n-genes", type=int, default=300, help="genes for per-gene learners (0 = all in-scope genes)")
    ap.add_argument("--time-budget", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None, help="override training lr (scvi sensitivity)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    view = load_cell_view(ROOT / "outputs" / "task_m" / "agent_view")
    rng = np.random.default_rng(2026)
    order = rng.permutation(view.dev_rna.shape[0])
    test = np.sort(order[: a.n_test])
    full = a.group == "full"
    train = np.sort(order[a.n_test:] if full else order[a.n_test: a.n_test + a.n_train])
    _, half_b = stats.thin_counts(view.dev_rna[test], seed=7)
    keep = [k.strip() for k in a.components.split(",") if k.strip()]
    groups = [g for g in SPECS] if full else [a.group]
    todo = [(g, item) for g in groups for item in SPECS[g] if not keep or any(item[0].startswith(k) for k in keep)]
    if full and not keep:
        raise SystemExit("--group full needs --components")
    print(f"host={os.uname().nodename} threads={mm.THREADS} train={len(train)} test={len(test)} scope={len(view.scope)} group={a.group}", flush=True)

    baselines = {}

    def baseline_r(genes):
        key = None if genes is None else len(genes)
        if key not in baselines:
            m = mm.build(dict(model=dict(component="cluster_mean", params={"k": "frozen"}), inputs={}, training={}), view, seed=0, genes=genes)
            m.fit(view.dev_atac[train], view.dev_rna[train])
            truth = view.target(half_b, m.gene_idx)
            baselines[key] = (float(np.nan_to_num(stats.colcorr(m.predict(view.dev_atac[test]), truth), nan=0.0).mean()), truth)
        return baselines[key]

    for group, (name, model, inputs, training) in todo:
        genes = None
        if group in PER_GENE_GROUPS and a.n_genes > 0:
            genes = np.sort(np.random.default_rng(5).choice(len(view.scope), a.n_genes, replace=False))
        if a.lr is not None:
            training = dict(training, lr=a.lr)
        spec = dict(model=model, inputs=inputs, training=training)
        t0 = time.time()
        try:
            base_r, truth = baseline_r(genes)
            m = mm.build(spec, view, scale="full" if full else "pilot", seed=0, genes=genes, time_budget=a.time_budget)
            report = m.fit(view.dev_atac[train], view.dev_rna[train])
            t_pred = time.time()
            pred = m.predict(view.dev_atac[test])
            per_gene = stats.colcorr(pred, truth)
            # A constant prediction has no correlation (NaN). Count it as 0 so that a model cannot raise its mean
            # by predicting a constant for the genes it cannot predict; report how many there were.
            n_constant = int(np.isnan(per_gene).sum())
            r = float(np.nan_to_num(per_gene, nan=0.0).mean())
            row = dict(name=name, group=a.group, n_train=len(train), n_test=len(test), n_genes=int(pred.shape[1]), mean_r=r,
                       mean_r_dropping_constant_genes=float(np.nanmean(per_gene)), n_constant_predictions=n_constant,
                       cluster_mean_r=base_r, delta=r - base_r, fit_seconds=report["seconds"], predict_seconds=round(time.time() - t_pred, 2),
                       converged=report["converged"], stopped_by=report["stopped_by"], epochs=report["epochs"], device=report["device"],
                       n_parameters=report["n_parameters"], note=report["convergence_note"], ignored=m.ignored, resolved=m.resolved,
                       curves={k: v for k, v in report["curves"].items()}, threads=mm.THREADS, finite=bool(np.isfinite(pred).all()))
            print(f"SMOKE {name:52s} mean_r={r:.4f} cluster_mean={base_r:.4f} delta={r - base_r:+.4f} genes={pred.shape[1]:5d} "
                  f"fit={report['seconds']:8.1f}s predict={row['predict_seconds']:6.1f}s epochs={report['epochs']:4d} "
                  f"stopped_by={report['stopped_by']:14s} converged={report['converged']} constant_genes={n_constant} "
                  f"ignored={m.ignored}", flush=True)
        except Exception as exc:
            row = dict(name=name, group=a.group, error=repr(exc)[:500], trace=traceback.format_exc()[-2500:], seconds=round(time.time() - t0, 1))
            print(f"SMOKE {name:52s} FAILED after {row['seconds']}s: {row['error']}\n{row['trace']}", flush=True)
        safe = "".join(c if c.isalnum() else "_" for c in name)
        write_json(OUT / f"{a.group}{'_' + a.tag if a.tag else ''}__{safe}.json", row)
    print("SMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()
