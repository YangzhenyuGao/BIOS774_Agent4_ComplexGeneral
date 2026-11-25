"""Evaluator for task T (README 4.2): cohort view, evaluator-owned folds, unified nested cross-validation.

    python -m bench.task_t prepare
    python -m bench.task_t fold --plan <plan.json> --set main --repeat 0 --fold 0     one outer fold
    python -m bench.task_t run --plan <plan.json> [--set main]                        all outer folds in one process
    python -m bench.task_t collect --plan-id <id>                                     true score versus the clinical floor
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from agent4 import stats
from agent4.data import load_cohort_view
from agent4.io import Timer, now, read_json, write_index, write_json
from agent4.tools import t_basic as tb
from bench import paths

TASK = "task_t"
SEED = 774
CLASSES = ["LumA", "LumB", "Basal", "Her2"]
ENCODERS = ["UNI2-h", "CONCH-Z", "Virchow2", "MUSK", "H-Optimus-1"]
OUTER_K, REPEATS, INNER_K = 5, 3, 3
FLOOR_ID, REFERENCE_ID = "ruler_clinical_floor", "ruler_simple_reference"
FLOOR_SPEC = dict(plan_id=FLOOR_ID, model=dict(component="clinical_only", params=dict(tune=True)))
REFERENCE_SPEC = dict(plan_id=REFERENCE_ID, model=dict(component="patient_mean_logreg", params={}),
                      inputs=dict(image=dict(encoders=ENCODERS, level="patient"), clinical=dict(use=True)))


def prepare():
    view, private = paths.view_dir(TASK), paths.private_dir(TASK)
    view.mkdir(parents=True, exist_ok=True), private.mkdir(parents=True, exist_ok=True)
    clin = pd.read_parquet(paths.TCGA_PREPROCESS / "clinical_cohort.parquet")
    clin = clin[clin.has_primary_slide]
    out = clin[["site", "age", "stage", "histological_type", "er", "pr", "her2"]].copy()
    out["label"] = clin["label"].where(clin["label"].isin(CLASSES), None)
    for c in ("er", "pr", "her2", "stage", "histological_type"):
        out[c] = out[c].where(~out[c].isin(["missing", "", "nan"]), None)
    out.index.name = "patient_id"
    out.to_parquet(view / "clinical.parquet")

    for enc in ENCODERS:
        with np.load(paths.FEATURE_BANK / f"representations/{enc}/patient_embeddings.npz", allow_pickle=False) as z:
            frame = pd.DataFrame(z["embedding"], index=z["ids"].astype(str))
        np.save(view / f"patient_{enc}.npy", frame.loc[out.index].values.astype(np.float32))

    index = pd.read_parquet(paths.FEATURE_BANK / "feature_index.parquet")
    slides = pd.read_parquet(paths.FEATURE_BANK / "slide_index.parquet")
    slides = slides[slides.patient_id.isin(out.index)].sort_values(["patient_id", "slide_id"]).reset_index(drop=True)
    rows, start = [], 0
    for s in slides.itertuples():
        rows.append(dict(patient_id=s.patient_id, slide_id=s.slide_id, start=start, end=start + int(s.tiles)))
        start += int(s.tiles)
    tile_index = pd.DataFrame(rows)
    tile_index.to_parquet(view / "tile_index.parquet")
    for enc in ENCODERS:
        path_of = index[index.model == enc].set_index("slide_id").path
        packed = None
        for s in tile_index.itertuples():
            with h5py.File(path_of[s.slide_id], "r") as f:
                a = f["features"][:]
            assert len(a) == s.end - s.start, (enc, s.slide_id, len(a))
            if packed is None:
                packed = np.lib.format.open_memmap(view / f"tiles_{enc}.npy", mode="w+", dtype=np.float16, shape=(start, a.shape[1]))
            packed[s.start:s.end] = a
        packed.flush()
        del packed
        print(f"packed tiles for {enc}", flush=True)

    labelled = out[out.label.notna()]
    write_json(view / "view.json", dict(
        adapter="cohort_features", task_id=TASK, created=now(), encoders=ENCODERS, classes=CLASSES,
        n_patients=len(out), n_labelled=len(labelled), n_slides=len(tile_index), n_tiles=int(start),
        class_counts=labelled.label.value_counts().to_dict(), group_column="site",
        note="Tile arrays are float16 copies of frozen-encoder features in tile_index order; all encoders share tile order."))
    write_index(view, note="agent view for task T")

    ids = labelled.index.values
    y = labelled.label.map({c: i for i, c in enumerate(CLASSES)}).values
    main = [[dict(train=ids[tr].tolist(), test=ids[te].tolist()) for tr, te in rep]
            for rep in tb.make_folds(y, OUTER_K, REPEATS, SEED)]
    site = [[dict(train=ids[tr].tolist(), test=ids[te].tolist())
             for tr, te in GroupKFold(OUTER_K).split(ids, y, groups=labelled.site.values)]]
    write_json(private / "folds.json", dict(seed=SEED, outer_k=OUTER_K, repeats=REPEATS, inner_k=INNER_K, main=main, site=site))
    print(f"prepared: patients={len(out)} labelled={len(labelled)} slides={len(tile_index)} tiles={start}")


def _load(plan_path):
    if plan_path in (FLOOR_ID, REFERENCE_ID):
        return FLOOR_SPEC if plan_path == FLOOR_ID else REFERENCE_SPEC
    return read_json(plan_path)


def _runner(spec, view, scale="full"):
    from agent4.tools.registry import build_runner
    return build_runner(spec, view, scale=scale, seed=SEED)


def fold(plan_path, fold_set="main", repeat=0, fold_idx=0):
    """Run one outer fold with nested configuration selection and store the test probabilities."""
    spec = _load(plan_path)
    view, private = load_cohort_view(paths.view_dir(TASK)), paths.private_dir(TASK)
    f = read_json(private / "folds.json")[fold_set][repeat][fold_idx]
    train_ids, test_ids = np.array(f["train"]), np.array(f["test"])
    ids = np.concatenate([train_ids, test_ids])
    y = view.y(ids)
    runner = _runner(spec, view)
    res = tb.run_outer_fold(runner, ids, y, np.arange(len(train_ids)), np.arange(len(train_ids), len(ids)),
                            "nested", INNER_K, SEED + 100 * repeat + fold_idx)
    out = private / "true_scores" / spec["plan_id"] / fold_set
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"r{repeat}_f{fold_idx}.npz", ids=test_ids, proba=res["proba"])
    write_json(out / f"r{repeat}_f{fold_idx}.json", dict(
        chosen=res["chosen"], config=res["config"], inner_scores=res["inner_scores"], n_configs=res["n_configs"],
        report=res["report"], seconds=res["seconds"], cpus=int(os.environ.get("A4_THREADS", 1)), finished=now()))
    print(f"{spec['plan_id']} {fold_set} r{repeat} f{fold_idx}: chosen {res['chosen']}/{res['n_configs']} in {res['seconds']}s", flush=True)


def run(plan_path, fold_set="main"):
    folds = read_json(paths.private_dir(TASK) / "folds.json")[fold_set]
    for r, rep in enumerate(folds):
        for f in range(len(rep)):
            fold(plan_path, fold_set, r, f)


def _oof(plan_id, fold_set, ids):
    private = paths.private_dir(TASK)
    folds = read_json(private / "folds.json")[fold_set]
    pos = pd.Series(np.arange(len(ids)), index=ids)
    out, meta = [], []
    for r, rep in enumerate(folds):
        oof = np.full((len(ids), len(CLASSES)), np.nan)
        for f in range(len(rep)):
            z = np.load(private / "true_scores" / plan_id / fold_set / f"r{r}_f{f}.npz", allow_pickle=False)
            oof[pos[z["ids"]].values] = z["proba"]
            meta.append(read_json(private / "true_scores" / plan_id / fold_set / f"r{r}_f{f}.json"))
        assert np.isfinite(oof).all(), f"missing folds for {plan_id} {fold_set} repeat {r}"
        out.append(oof)
    return out, meta


def collect(plan_id, fold_set="main"):
    """True score: paired gain over the evaluator's clinical floor with a patient-level bootstrap interval."""
    view, private = load_cohort_view(paths.view_dir(TASK)), paths.private_dir(TASK)
    ids = view.labelled.index.values
    y = view.y(ids)
    k = len(CLASSES)
    plan, meta = _oof(plan_id, fold_set, ids)
    floor, _ = _oof(FLOOR_ID, fold_set, ids)
    with Timer() as t:
        f1 = stats.paired_gain_bootstrap(y, plan, floor, n_boot=2000, seed=SEED)
        ll = stats.paired_gain_bootstrap(y, plan, floor, n_boot=500, seed=SEED, metric=lambda yy, p, kk: -stats.log_loss(yy, p))
        auc = stats.paired_gain_bootstrap(y, plan, floor, n_boot=300, seed=SEED, metric=lambda yy, p, kk: stats.ovr_auroc(yy, p))
    recall_plan = np.mean([stats.per_class_recall(y, p, k) for p in plan], axis=0)
    recall_floor = np.mean([stats.per_class_recall(y, p, k) for p in floor], axis=0)
    result = dict(
        plan=plan_id, fold_set=fold_set, scored=now(), n_patients=len(ids), repeats=len(plan),
        macro_f1_plan=float(np.mean([stats.macro_f1(y, p, k) for p in plan])),
        macro_f1_floor=float(np.mean([stats.macro_f1(y, p, k) for p in floor])),
        macro_f1_plan_by_repeat=[stats.macro_f1(y, p, k) for p in plan],
        true_gain=f1["gain"], ci_low=f1["ci_low"], ci_high=f1["ci_high"], se=f1["se"],
        conclusion=stats.three_way_conclusion(f1["ci_low"], f1["ci_high"]),
        neg_log_loss_gain=dict(gain=ll["gain"], ci_low=ll["ci_low"], ci_high=ll["ci_high"]),
        auroc_gain=dict(gain=auc["gain"], ci_low=auc["ci_low"], ci_high=auc["ci_high"]),
        recall_plan=dict(zip(CLASSES, recall_plan.tolist())), recall_floor=dict(zip(CLASSES, recall_floor.tolist())),
        recall_change=dict(zip(CLASSES, (recall_plan - recall_floor).tolist())),
        chosen_configs=[m["chosen"] for m in meta], n_configs=meta[0]["n_configs"],
        converged_folds=int(sum(bool(m["report"].get("converged", True)) for m in meta)), n_folds=len(meta),
        compute_seconds=float(sum(m["seconds"] for m in meta)), cpu_core_seconds=float(sum(m["seconds"] * m.get("cpus", 1) for m in meta)),
        bootstrap_seconds=t.seconds)
    out = private / "true_scores" / plan_id
    np.savez(out / f"oof_{fold_set}.npz", ids=ids, y=y, **{f"repeat{r}": p for r, p in enumerate(plan)})
    write_json(out / f"true_score_{fold_set}.json", result)
    print(f"{plan_id} [{fold_set}]: F1 {result['macro_f1_plan']:.4f} vs floor {result['macro_f1_floor']:.4f}; gain "
          f"{result['true_gain']:+.4f} [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}] -> {result['conclusion']}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "fold", "run", "collect"])
    ap.add_argument("--plan"), ap.add_argument("--plan-id"), ap.add_argument("--set", default="main")
    ap.add_argument("--repeat", type=int, default=0), ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--task-index", type=int, default=None, help="array index: repeat * outer_k + fold")
    a = ap.parse_args(argv)
    if a.command == "prepare":
        prepare()
    elif a.command == "fold":
        if a.task_index is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
            a.task_index = int(os.environ["SLURM_ARRAY_TASK_ID"])      # one array task per outer fold
        if a.task_index is not None:
            a.repeat, a.fold = divmod(a.task_index, OUTER_K)
        fold(a.plan, a.set, a.repeat, a.fold)
    elif a.command == "run":
        run(a.plan, a.set)
    else:
        collect(a.plan_id, a.set)


if __name__ == "__main__":
    sys.exit(main())
