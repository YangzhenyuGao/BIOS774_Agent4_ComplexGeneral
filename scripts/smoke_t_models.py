"""Smoke test of the task T components on the real cohort (run through Slurm, never on the login node).

For each component with default parameters: one stratified 5-fold cross-validation on the labelled patients,
no tuning (first configuration only). Prints macro-F1, the untuned clinical-only macro-F1 on the same folds,
the gain, seconds per fit and the share of folds that converged. These numbers show that the code paths work
and how long a fit takes; they are not results of the project.

    python scripts/smoke_t_models.py --components attention_mil_gated,incremental_offset [--encoder H-Optimus-1]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from agent4 import stats
from agent4.data import load_cohort_view
from agent4.io import ROOT, write_json
from agent4.tools import t_basic as tb
from agent4.tools.registry import build_runner


def specs(encoder, all_encoders):
    one = dict(image=dict(encoders=[encoder], level="tile"), clinical=dict(use=True))
    many_tile = dict(image=dict(encoders=all_encoders, level="tile"), clinical=dict(use=True))
    many_patient = dict(image=dict(encoders=all_encoders, level="patient"), clinical=dict(use=True))
    mk = lambda component, inputs=one, params=None, fusion=None, training=None: dict(
        inputs=inputs, model=dict(component=component, params=params or {}), fusion=fusion, training=training or {})
    return {
        "clinical_only": mk("clinical_only", params=dict(tune=False)),
        "patient_mean_logreg": mk("patient_mean_logreg", params=dict(grid=[dict(encoders=[encoder], n_pca=64, C=0.3)])),
        "attention_mil_gated": mk("attention_mil", params=dict(aggregator="gated_attention")),
        "attention_mil_multihead": mk("attention_mil", params=dict(aggregator="multihead_attention", heads=4)),
        "attention_mil_transformer": mk("attention_mil", params=dict(aggregator="transformer", layers=1, heads=4)),
        "attention_mil_late": mk("attention_mil", params=dict(aggregator="gated_attention"), fusion="late_average"),
        "attention_mil_transformer_lr1e-3": mk("attention_mil", params=dict(aggregator="transformer", layers=1, heads=4), training=dict(lr=1e-3)),
        "attention_mil_transformer_2layer": mk("attention_mil", params=dict(aggregator="transformer", layers=2, heads=4, dropout=0.3),
                                               training=dict(augmentation=dict(tile_resample=64), max_epochs=100)),
        "multi_encoder_fusion_patient": mk("multi_encoder_fusion", many_patient, dict(level="patient", encoder_fusion="attention")),
        "multi_encoder_fusion_tile": mk("multi_encoder_fusion", many_tile, dict(level="tile", encoder_fusion="gated")),
        "morphology_prototypes": mk("morphology_prototypes", params=dict(n_prototypes=32, classifier="logreg")),
        "conditional_fusion_film": mk("conditional_fusion", params=dict(mode="film")),
        "conditional_fusion_cross": mk("conditional_fusion", params=dict(mode="cross_attention")),
        "incremental_offset": mk("incremental_offset", params=dict(init_zero=True)),
        "hierarchical": mk("hierarchical", params=dict(groups=[["LumA", "LumB"], ["Basal"], ["Her2"]], fine_inputs="image+clinical")),
        "ssl_preadapted_mil": mk("ssl_preadapted_mil", params=dict(pretrain_epochs=20, uses_unlabelled=True)),
        "gradient_boosting": mk("gradient_boosting", many_patient, dict(n_pca=16, max_iter=100)),
        "stacking_ensemble": mk("stacking_ensemble", params=dict(members=[
            dict(component="patient_mean_logreg", params=dict(grid=[dict(encoders=[encoder], n_pca=64, C=0.3)])),
            dict(component="incremental_offset"), dict(component="morphology_prototypes")], oof_folds=3)),
        "aug_mixup_ensemble": mk("attention_mil", params=dict(aggregator="gated_attention"),
                                 training=dict(augmentation=dict(mixup_alpha=0.2, tile_dropout=0.2, tile_resample=64), n_seeds_ensemble=2)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", default="all")
    ap.add_argument("--encoder", default="H-Optimus-1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    view = load_cohort_view(ROOT / "outputs" / "task_t" / "agent_view")
    ids = view.labelled.index.values
    y = view.y(ids)
    k = len(view.classes)
    folds = tb.make_folds(y, a.folds, 1, seed=774)
    table = specs(a.encoder, view.encoders)
    names = list(table) if a.components == "all" else [n for n in a.components.split(",") if n]
    clin = tb.cross_validate(build_runner(table["clinical_only"], view), ids, y, folds, selection="first", seed=774, log=lambda *_: None)
    clin_f1 = stats.macro_f1(y, clin["proba"][0], k)
    rows = []
    for name in names:
        if name == "clinical_only":
            continue
        t0 = time.time()
        try:
            runner = build_runner(table[name], view, scale="full", seed=774)
            out = tb.cross_validate(runner, ids, y, folds, selection="first", seed=774, log=lambda *_: None)
            proba, reports = out["proba"][0], out["reports"][0]
            assert np.isfinite(proba).all() and np.allclose(proba.sum(1), 1.0, atol=1e-4)
            gain = stats.paired_gain_bootstrap(y, [proba], [clin["proba"][0]], n_boot=500, seed=0)
            row = dict(component=name, macro_f1=stats.macro_f1(y, proba, k), clinical_f1=clin_f1, gain=gain["gain"],
                       gain_ci=[gain["ci_low"], gain["ci_high"]], seconds_per_fit=float(np.mean([r["seconds"] for r in reports])),
                       converged_share=float(np.mean([bool(r.get("converged", True)) for r in reports])),
                       epochs=[int(r.get("epochs", 0)) for r in reports], stopped_by=[r.get("stopped_by") for r in reports],
                       n_parameters=int(reports[0].get("n_parameters", 0)), device=reports[0].get("device", "cpu"),
                       recall=dict(zip(view.classes, stats.per_class_recall(y, proba, k))),
                       ignored=getattr(runner, "ignored", []), total_seconds=round(time.time() - t0, 1))
            print(f"SMOKE {name:30s} macroF1={row['macro_f1']:.4f} clinical={clin_f1:.4f} gain={row['gain']:+.4f} "
                  f"[{gain['ci_low']:+.3f},{gain['ci_high']:+.3f}] sec/fit={row['seconds_per_fit']:.1f} "
                  f"converged={row['converged_share']:.2f} epochs={row['epochs']} params={row['n_parameters']} ignored={row['ignored']}",
                  flush=True)
        except Exception as exc:  # report and continue: one broken component must not hide the others
            import traceback
            row = dict(component=name, error=repr(exc)[:500], trace=traceback.format_exc()[-1500:])
            print(f"SMOKE {name:30s} FAILED {row['error']}\n{row['trace']}", flush=True)
        rows.append(row)
    out_dir = ROOT / "outputs" / "_smoke" / "t_models"
    write_json(out_dir / f"{a.name or 'smoke'}.json", dict(encoder=a.encoder, folds=a.folds, clinical_f1=clin_f1, rows=rows,
                                                           note="setup check with default parameters; not a project result"))


if __name__ == "__main__":
    main()
