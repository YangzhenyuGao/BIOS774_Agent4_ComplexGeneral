"""Incremental-value classification: a plan's own validation (pilot and full scale), its evidence card, the full run.

The plan is validated the way the plan itself declared: its folds, its way of choosing a configuration, its
clinical comparison arm. The evaluator's unified nested cross-validation is separate and never read here.
"""
from __future__ import annotations

import os
import time

import numpy as np

from . import stats
from .io import now, write_json
from .schemas import EvidenceCard, EvidenceItem
from .tools import registry
from .tools import t_basic as tb

FAMILY = "incremental_value_classification"
OWN_FOLD_SEED = 2026          # the plan's own folds; unrelated to the evaluator's
# Wall-clock budgets in seconds for the image arm. A full run (own validation, then the site-held-out analysis)
# has to fit one two-hour job (README 6.2); outer folds are run in the declared order until the next would not fit.
PILOT_BUDGET, FULL_BUDGET, SITE_BUDGET = 30 * 60, 75 * 60, 35 * 60


def budgeted_cv(runner, ids, y, folds, selection, inner_k, seed, budget, log=print):
    """Out-of-fold probabilities with a time budget. Rows of patients whose fold was not reached stay NaN."""
    k = len(runner.view.classes)
    if selection == "same_folds":                    # needs every fold for every configuration; cannot be cut short
        out = tb.cross_validate(runner, ids, y, folds, selection=selection, inner_k=inner_k, seed=seed, log=log)
        out.update(n_folds_declared=sum(len(r) for r in folds), n_folds_run=sum(len(r) for r in folds), stopped_for_budget=False)
        return out
    out = dict(proba=[], chosen=[], reports=[], selection=selection, n_configs=len(runner.configs()),
               n_folds_declared=sum(len(r) for r in folds), n_folds_run=0, stopped_for_budget=False)
    t0 = time.time()
    for r, repeat in enumerate(folds):
        oof, chosen, reports = np.full((len(ids), k), np.nan), [], []
        for f, (tr, te) in enumerate(repeat):
            elapsed, done = time.time() - t0, out["n_folds_run"]
            if done >= 1 and elapsed + elapsed / done > budget:
                out["stopped_for_budget"] = True
                log(f"  stopping after {done} of {out['n_folds_declared']} outer folds: the next one would exceed the {budget / 60:.0f}-minute budget")
                break
            res = tb.run_outer_fold(runner, ids, y, tr, te, selection, inner_k, seed + r)
            oof[te] = res["proba"]
            chosen.append(res["chosen"]), reports.append(res["report"])
            out["n_folds_run"] += 1
            log(f"  repeat {r} fold {f}: chosen config {res['chosen']} of {res['n_configs']} ({res['seconds']}s)")
        if chosen:
            out["proba"].append(oof), out["chosen"].append(chosen), out["reports"].append(reports)
        if out["stopped_for_budget"]:
            break
    return out


def _scored(cv):
    """Repeats to score and the patients covered: complete repeats if there are any, else the partial first one."""
    complete = [p for p in cv["proba"] if np.isfinite(p).all()]
    if complete:
        return complete, np.ones(len(complete[0]), dtype=bool), len(complete)
    first = cv["proba"][0]
    return [first], np.isfinite(first).all(1), 0


def _validation(plan):
    v = plan.get("validation", {}) or {}
    scheme = str(v.get("scheme", "kfold")).lower()
    selection = str(v.get("selection", "nested")).lower()
    comparator = v.get("clinical_comparator", {})
    tuned = comparator.get("tuned", True) if isinstance(comparator, dict) else bool(comparator)
    return dict(k=int(min(max(int(v.get("k", 5) or 5), 3), 5)),
                repeats=int(min(max(int(v.get("repeats", 1) or 1), 1), 3)) if "repeat" in scheme else 1,
                selection=selection if selection in ("nested", "same_folds", "none") else "nested",
                inner_k=int(min(max(int(v.get("inner_k", 3) or 3), 2), 3)), clinical_tuned=bool(tuned),
                split_unit=str(v.get("split_unit", "patient")).lower())


def _curve_stats(reports):
    flat = [r for rep in reports for r in rep] if reports and isinstance(reports[0], list) else reports
    conv = [bool(r.get("converged", True)) for r in flat]
    epochs = [r.get("epochs") for r in flat if r.get("epochs") is not None]
    secs = [r.get("seconds", 0.0) for r in flat]
    return dict(n_fits_reported=len(flat), share_converged=float(np.mean(conv)) if conv else None,
                median_epochs=float(np.median(epochs)) if epochs else None, median_seconds_per_fit=float(np.median(secs)) if secs else None,
                stopped_by=sorted({str(r.get("stopped_by")) for r in flat}), example_curve=next((r.get("curves") for r in flat if r.get("curves")), None))


def validate(plan, view, scale, out_dir, seed=0, log=print):
    v = _validation(plan)
    ids = view.labelled.index.values
    y = view.y(ids)
    k = len(view.classes)
    repeats = 1 if scale == "pilot" else v["repeats"]
    folds = tb.make_folds(y, v["k"], repeats, OWN_FOLD_SEED + seed)
    t0 = time.time()
    runner = registry.build_runner(plan, view, scale=scale, seed=seed)
    sel = {"nested": "nested", "same_folds": "same_folds", "none": "first"}[v["selection"]]
    cv = budgeted_cv(runner, ids, y, folds, sel, v["inner_k"], seed, PILOT_BUDGET if scale == "pilot" else FULL_BUDGET, log)
    t_plan = time.time() - t0
    plan_proba, covered, n_complete = _scored(cv)
    use = len(plan_proba)                              # number of repeats that are scored
    clin = registry.build_runner(dict(model=dict(component="clinical_only", params=dict(tune=v["clinical_tuned"])),
                                      inputs=dict(clinical=plan.get("inputs", {}).get("clinical", {}))), view, seed=seed)
    cv_clin = tb.cross_validate(clin, ids, y, folds, selection="nested" if v["clinical_tuned"] else "first", inner_k=v["inner_k"], seed=seed, log=lambda *_: None)
    ref = registry.build_runner(dict(model=dict(component="patient_mean_logreg", params={}), inputs=dict(image=dict(encoders=view.encoders))),
                                view, scale="pilot", seed=seed)
    cv_ref = tb.cross_validate(ref, ids, y, folds, selection="nested", inner_k=v["inner_k"], seed=seed, log=lambda *_: None)
    # Every arm is scored on exactly the patients the image arm covered, in the same repeats.
    ys = y[covered]
    p_plan = [p[covered] for p in plan_proba]
    p_clin = [p[covered] for p in cv_clin["proba"][:use]]
    p_ref = [p[covered] for p in cv_ref["proba"][:use]]
    gain = stats.paired_gain_bootstrap(ys, p_plan, p_clin, n_boot=1000, seed=seed)
    gain_ref = stats.paired_gain_bootstrap(ys, p_plan, p_ref, n_boot=500, seed=seed)
    ref_gain = stats.paired_gain_bootstrap(ys, p_ref, p_clin, n_boot=500, seed=seed)
    recall_plan = np.mean([stats.per_class_recall(ys, p, k) for p in p_plan], axis=0)
    recall_clin = np.mean([stats.per_class_recall(ys, p, k) for p in p_clin], axis=0)
    curve = _curve_stats(cv["reports"])
    n_fits = len(folds) * v["k"] * (1 + (cv["n_configs"] * v["inner_k"] if sel == "nested" and cv["n_configs"] > 1 else 0)) \
        if sel != "same_folds" else len(folds) * v["k"] * cv["n_configs"]
    summary = dict(
        plan_id=plan["plan_id"], scale=scale, finished=now(), validation=dict(v, repeats_run=repeats), n_patients=int(covered.sum()),
        n_patients_cohort=len(ids), n_folds_declared=cv["n_folds_declared"] * (v["repeats"] if scale == "pilot" else 1),
        n_folds_run=cv["n_folds_run"], n_complete_repeats=n_complete, stopped_for_budget=cv["stopped_for_budget"],
        macro_f1_plan=float(np.mean([stats.macro_f1(ys, p, k) for p in p_plan])),
        macro_f1_clinical=float(np.mean([stats.macro_f1(ys, p, k) for p in p_clin])),
        macro_f1_simple_reference=float(np.mean([stats.macro_f1(ys, p, k) for p in p_ref])),
        self_gain=gain["gain"], self_ci=[gain["ci_low"], gain["ci_high"]], self_se=gain["se"],
        self_conclusion=stats.three_way_conclusion(gain["ci_low"], gain["ci_high"]),
        gain_over_simple_reference=gain_ref["gain"], gain_over_simple_reference_ci=[gain_ref["ci_low"], gain_ref["ci_high"]],
        simple_reference_gain=ref_gain["gain"], simple_reference_gain_ci=[ref_gain["ci_low"], ref_gain["ci_high"]],
        recall_plan=dict(zip(view.classes, recall_plan.tolist())), recall_clinical=dict(zip(view.classes, recall_clin.tolist())),
        recall_change=dict(zip(view.classes, (recall_plan - recall_clin).tolist())),
        n_configs_image_arm=cv["n_configs"], n_configs_clinical_arm=cv_clin["n_configs"], chosen_configs=cv["chosen"],
        n_fits=int(n_fits), plan_seconds=round(t_plan, 1), cpus=int(os.environ.get("A4_THREADS", 1)), training=curve,
        resolved_config=getattr(runner, "resolved", None), ignored_params=getattr(runner, "ignored", []))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "self_validation.json", summary)
    np.savez_compressed(out_dir / "self_validation_oof.npz", ids=ids, y=y, covered=covered, **{f"plan_repeat{r}": p for r, p in enumerate(plan_proba)},
                        **{f"clinical_repeat{r}": p for r, p in enumerate(cv_clin["proba"][:use])},
                        **{f"reference_repeat{r}": p for r, p in enumerate(cv_ref["proba"][:use])})
    return summary


def evidence_card(plan, summary, compliance, limits) -> EvidenceCard:
    v, items = summary["validation"], []

    def add(kind, text, value=None):
        items.append(EvidenceItem(id=f"{plan['plan_id']}-E{len(items) + 1}", kind=kind, text=text, value=value))

    add("self_score", f"Self-reported gain in macro-F1 over the plan's own clinical-only arm: {summary['self_gain']:+.4f} "
                      f"(95% bootstrap interval {summary['self_ci'][0]:+.4f} to {summary['self_ci'][1]:+.4f}); macro-F1 {summary['macro_f1_plan']:.4f} "
                      f"with images versus {summary['macro_f1_clinical']:.4f} clinical only, {summary['n_patients']} patients in both arms.", summary["self_gain"])
    add("ruler", f"Against the simple reference (clinical variables plus mean-pooled image features, logistic regression) on the same folds: "
                 f"plan minus reference = {summary['gain_over_simple_reference']:+.4f} (interval {summary['gain_over_simple_reference_ci'][0]:+.4f} to "
                 f"{summary['gain_over_simple_reference_ci'][1]:+.4f}); the reference itself gains {summary['simple_reference_gain']:+.4f} over clinical only.",
        summary["gain_over_simple_reference"])
    add("self_score", "Change in recall by class (with images minus clinical only): " +
        ", ".join(f"{c} {d:+.3f}" for c, d in summary["recall_change"].items()) + ".", summary["recall_change"])
    tr = summary["training"]
    share = "not reported" if tr["share_converged"] is None else f"{tr['share_converged']:.0%}"
    add("convergence", f"Training: {tr['n_fits_reported']} reported fits, {share} converged, stopped by {tr['stopped_by']}, "
                       f"median {tr['median_epochs']} epochs, median {tr['median_seconds_per_fit']} s per fit.", tr["share_converged"])
    sel_text = {"nested": "chosen by inner cross-validation inside each training fold",
                "same_folds": "chosen on the same folds whose score is reported (the reported score is the best of the candidates)",
                "none": "not tuned (single configuration)"}[v["selection"]]
    add("validation_design", f"Validation design: split unit {v['split_unit']}; {v['k']}-fold cross-validation, {v['repeats_run']} repeat(s) run "
                             f"(the plan declares {v['repeats']}); image-arm configuration {sel_text}; {summary['n_configs_image_arm']} candidate "
                             f"configuration(s) for the image arm versus {summary['n_configs_clinical_arm']} for the clinical arm "
                             f"({'tuned' if v['clinical_tuned'] else 'not tuned'}); both arms use the same patients, folds and in-fold missing-value handling.", v)
    cut = (" The run stopped there because the next outer fold would not have fitted the time budget; all arms are scored on the patients covered."
           if summary.get("stopped_for_budget") else "")
    add("validation_design", f"Scale of this run: {summary['scale']}; {summary['n_folds_run']} of the {summary['n_folds_declared']} declared outer folds were run, "
                             f"{summary['n_patients']} of {summary['n_patients_cohort']} patients scored.{cut}")
    add("cost", f"Cost: {summary['plan_seconds']:.0f} s for the image arm on {summary['cpus']} CPU cores; declared budget for a full run "
                f"{plan.get('budget', {}).get('full_minutes')} min, needs GPU: {plan.get('budget', {}).get('needs_gpu')}.", summary["plan_seconds"])
    add("compliance", "Compliance checks: " + "; ".join(f"{r['rule']} {'passed' if r['passed'] else 'FAILED'} ({r['detail']})" for r in compliance["rules"]) + ".",
        compliance["passed"])
    rc = summary.get("resolved_config") or {}
    if rc:
        shown = {k: rc[k] for k in ("component", "encoders", "level", "reduce_type", "reduce_dim", "aggregator", "encoder_fusion", "fusion", "head", "embed_dim",
                                    "dropout", "max_epochs", "patience", "lr", "weight_decay", "class_weighting", "tile_resample", "mixup_alpha", "tile_dropout",
                                    "n_seeds_ensemble") if rc.get(k) not in (None, [], "")}
        if rc.get("component") == "ssl_preadapted_mil":
            shown.update(self_supervised_pretrain_epochs=rc.get("pretrain_epochs"), pretraining_uses_unlabelled_patients=rc.get("uses_unlabelled"))
        if rc.get("members"):
            shown["stacking_members"] = [m.get("component") if isinstance(m, dict) else str(m) for m in rc["members"]]
        add("note", f"Configuration the executor actually ran (first configuration; alternatives come from the plan's tuning grid): {shown}.", shown)
    if summary.get("ignored_params"):
        training = plan.get("training", {}) or {}
        value = lambda key: training.get(key.split(".")[-1])
        text = [k for k in summary["ignored_params"] if isinstance(value(k), str) and len(value(k)) > 40]
        settings = [k for k in summary["ignored_params"] if k not in text]
        parts = []
        if settings:
            parts.append(f"settings the executor has no implementation for, so its defaults applied: {settings}")
        if text:
            parts.append(f"free-text fields that describe intent and contain nothing to execute: {text} (this does not mean the corresponding step was skipped; see the configuration above)")
        add("note", "Plan fields not used by the executor - " + "; ".join(parts) + ".")
    return EvidenceCard(plan_id=plan["plan_id"], scale=summary["scale"], items=items, raw_files=["self_validation.json", "self_validation_oof.npz"])


def site_sensitivity(plan, view, v, seed=0, log=print, budget=SITE_BUDGET):
    """Sensitivity analysis the objective asks for: the same comparison with whole collection sites held out."""
    from sklearn.model_selection import GroupKFold
    frame = view.labelled
    ids, y, k = frame.index.values, view.y(frame.index.values), len(view.classes)
    folds = [[(tr, te) for tr, te in GroupKFold(5).split(ids, y, groups=frame["site"].values)]]
    sel = {"nested": "nested", "same_folds": "same_folds", "none": "first"}[v["selection"]]
    runner = registry.build_runner(plan, view, scale="full", seed=seed)
    cv = budgeted_cv(runner, ids, y, folds, sel, v["inner_k"], seed, budget, log)
    clin = registry.build_runner(dict(model=dict(component="clinical_only", params=dict(tune=v["clinical_tuned"])),
                                      inputs=dict(clinical=plan.get("inputs", {}).get("clinical", {}))), view, seed=seed)
    cv_clin = tb.cross_validate(clin, ids, y, folds, selection="nested" if v["clinical_tuned"] else "first", inner_k=v["inner_k"], seed=seed, log=lambda *_: None)
    plan_proba, covered, _ = _scored(cv)
    ys, p_plan, p_clin = y[covered], [plan_proba[0][covered]], [cv_clin["proba"][0][covered]]
    gain = stats.paired_gain_bootstrap(ys, p_plan, p_clin, n_boot=1000, seed=seed)
    return dict(design="5 folds, each holding out whole collection sites", n_sites=int(frame["site"].nunique()), n_folds_run=cv["n_folds_run"],
                n_patients_scored=int(covered.sum()), stopped_for_budget=cv["stopped_for_budget"], gain=gain["gain"],
                ci=[gain["ci_low"], gain["ci_high"]], macro_f1_plan=stats.macro_f1(ys, p_plan[0], k), macro_f1_clinical=stats.macro_f1(ys, p_clin[0], k),
                conclusion=stats.three_way_conclusion(gain["ci_low"], gain["ci_high"]))


def execute(plan, view, out_dir, seed=0, log=print):
    """Full run: the plan's own validation at the scale it declared, plus the site-held-out sensitivity analysis.
    The deliverable is the gain with its interval, the three-way conclusion and the out-of-fold predictions."""
    t0 = time.time()
    summary = validate(plan, view, "full", out_dir, seed=seed, log=log)
    site = site_sensitivity(plan, view, summary["validation"], seed=seed, log=log)
    result = dict(plan_id=plan["plan_id"], finished=now(), self_gain=summary["self_gain"], self_ci=summary["self_ci"],
                  conclusion=summary["self_conclusion"], macro_f1_plan=summary["macro_f1_plan"], macro_f1_clinical=summary["macro_f1_clinical"],
                  n_patients_both_arms=summary["n_patients"], n_patients_cohort=summary["n_patients_cohort"],
                  n_folds_declared=summary["n_folds_declared"], n_folds_run=summary["n_folds_run"], stopped_for_budget=summary["stopped_for_budget"],
                  comparators=dict(clinical_only=dict(macro_f1=summary["macro_f1_clinical"], tuned=summary["validation"]["clinical_tuned"],
                                                      n_configurations=summary["n_configs_clinical_arm"], same_patients_and_folds=True),
                                   simple_reference=dict(macro_f1=summary["macro_f1_simple_reference"], gain_over_clinical=summary["simple_reference_gain"],
                                                         gain_over_clinical_ci=summary["simple_reference_gain_ci"], same_patients_and_folds=True)),
                  gain_over_simple_reference=summary["gain_over_simple_reference"], gain_over_simple_reference_ci=summary["gain_over_simple_reference_ci"],
                  site_held_out=site, recall_plan=summary["recall_plan"], recall_clinical=summary["recall_clinical"], recall_change=summary["recall_change"],
                  validation=summary["validation"], n_configs_image_arm=summary["n_configs_image_arm"], training=summary["training"],
                  deliverable=dict(file="self_validation_oof.npz", complete=True), seconds_total=round(time.time() - t0, 1), cpus=summary["cpus"],
                  scope="incremental predictive value in this labelled cohort; not an independent biological mechanism")
    write_json(out_dir / "full_run.json", result)
    return result
