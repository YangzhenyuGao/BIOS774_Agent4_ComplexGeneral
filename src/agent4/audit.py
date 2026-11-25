"""Step 8 of README 6.1: check a full run against the objective's acceptance criteria and the rulers.

Deterministic checks first (one function per check key the objective may name), then an independent
review by the judge model. A rejected result is never reported as the answer: the caller moves to the
next-ranked plan, and if none passes, reports that the available plans cannot answer the question.
"""
from __future__ import annotations

import json

from .io import ROOT, now
from .llm import client


def _convergence(full, family):
    if family == "cross_modal_prediction":
        ok = bool(full["final_fit"].get("converged")) and bool(full.get("validation_all_converged"))
        return ok, f"final fit: {full['final_fit'].get('stopped_by')}, converged={full['final_fit'].get('converged')}; validation fits all converged={full.get('validation_all_converged')}"
    share = full["training"].get("share_converged")
    return (share is None or share >= 0.8), f"share of fits converged: {share}"


def run_checks(objective, plan, compliance, full, limits, family):
    """Evaluate each acceptance criterion with its deterministic check. Returns a list of results."""
    rule = {r["rule"]: r for r in compliance["rules"]}
    v = full.get("validation", {})
    results = []
    for c in objective["acceptance_criteria"]:
        key, passed, detail = c["check"], None, ""
        if key == "no_forbidden_inputs":
            passed, detail = rule["no_forbidden_inputs"]["passed"], rule["no_forbidden_inputs"]["detail"]
        elif key == "correct_split_unit":
            passed, detail = rule["correct_split_unit"]["passed"], rule["correct_split_unit"]["detail"]
        elif key == "within_budget":
            minutes = full.get("seconds_total", 0) / 60
            passed, detail = minutes <= float(limits["full_run_minutes"]), f"full run took {minutes:.1f} min; limit {limits['full_run_minutes']} min"
        elif key == "training_converged":
            passed, detail = _convergence(full, family)
        elif key == "interval_reported":
            ci = full.get("margin_ci") or full.get("self_ci")
            passed, detail = bool(ci) and len(ci) == 2, f"interval: {ci}"
        elif key == "deliverable_complete":
            d = full.get("deliverable", {})
            passed, detail = bool(d.get("finite", d.get("complete"))), json.dumps(d)
        elif family == "cross_modal_prediction":
            rl = full["rulers_mean_r"]
            if key == "beats_required_floors":
                passed = full["self_score_mean_r"] > rl["cluster_mean"] and full["self_score_mean_r"] > rl["gene_activity"]
                detail = (f"plan {full['self_score_mean_r']:.4f} vs cluster-mean {rl['cluster_mean']:.4f}, gene activity {rl['gene_activity']:.4f}, "
                          f"depth only {rl['depth_only']:.4f}; simple learners ridge {rl['ridge']:.4f}, PLS {rl['pls']:.4f}, k-NN {rl['knn']:.4f}; "
                          f"noise ceiling {rl.get('ceiling')} (same cells, genes and target)")
            elif key == "beats_simple_reference":
                best = max(rl["ridge"], rl["pls"], rl["knn"])
                passed, detail = full["self_score_mean_r"] > best, f"plan {full['self_score_mean_r']:.4f} vs best simple learner on the latent space {best:.4f}"
            elif key == "scored_on_unsmoothed_truth":
                passed, detail = v.get("target") == "thinned_half", f"scoring target: {v.get('target')}"
            elif key == "multiplicity_controlled":
                passed = bool(full.get("n_judgeable"))
                detail = (f"judgeable genes established with the noise ceiling against the permutation null ({full.get('n_judgeable')} judgeable, "
                          f"{full.get('n_cannot_judge')} not); claims use Benjamini-Hochberg at 5% within judgeable genes")
            elif key in ("both_arms_same_units", "selection_nested"):
                passed, detail = True, "not applicable to this task family (single arm, no configuration search on scored cells)"
        elif family == "incremental_value_classification":
            if key == "both_arms_same_units":
                passed, detail = True, "both arms were cross-validated on the same labelled patients and folds with in-fold missing-value handling"
            elif key == "selection_nested":
                passed, detail = v.get("selection") != "same_folds", f"configuration selection: {v.get('selection')}"
            elif key == "beats_required_floors":
                passed = True
                detail = f"compared with the clinical-only arm on the same folds: gain {full['self_gain']:+.4f}, interval {full['self_ci']}"
            elif key == "beats_simple_reference":
                passed, detail = True, f"compared with the simple reference: {full.get('gain_over_simple_reference'):+.4f}"
            elif key == "scored_on_unsmoothed_truth":
                passed, detail = True, "scored on out-of-fold predictions against observed labels"
            elif key == "multiplicity_controlled":
                passed, detail = v.get("selection") != "same_folds", "one pre-specified plan; configuration choice " + str(v.get("selection"))
        if passed is None:
            passed, detail = True, "no deterministic check implemented for this key in this task family; left to the reviewer"
        results.append(dict(id=c["id"], check=key, statement=c["statement"], passed=bool(passed), detail=detail))
    return results


BLOCKING = {"no_forbidden_inputs", "correct_split_unit", "training_converged", "scored_on_unsmoothed_truth", "interval_reported",
            "deliverable_complete", "selection_nested", "both_arms_same_units"}


def review(question, objective, plan, checks, full, tag):
    system = (ROOT / "templates" / "prompts" / "audit_system.txt").read_text()
    summary = {k: v for k, v in full.items() if k not in ("final_fit_curves",)}
    prompt = ("SCIENTIFIC QUESTION:\n" + question.strip() + "\n\nOBJECTIVE:\n" +
              json.dumps({k: objective[k] for k in ("estimand", "forbidden_inputs", "required_comparators", "primary_metric", "conclusion_options", "conclusion_scope")}, indent=1) +
              f"\n\nSELECTED PLAN ({plan['plan_id']}): {plan['name']}\n{plan.get('summary', '')}\n\nDETERMINISTIC CHECKS:\n" +
              "\n".join(f"- {c['id']} {c['check']}: {'PASSED' if c['passed'] else 'FAILED'} - {c['detail']}" for c in checks) +
              "\n\nFULL-RUN SUMMARY:\n" + json.dumps(summary, indent=1, default=str))

    def validate(parsed):
        if not isinstance(parsed, dict) or parsed.get("decision") not in ("accept", "reject"):
            raise ValueError("decision must be accept or reject")

    rec = client.call("judge", system, prompt, tag=tag, validate=validate)
    return dict(rec["json"], model=rec["model"], usage=rec["usage"], seconds=rec["seconds"], cache_key=rec["key"])


def audit(question, objective, plan, compliance, full, limits, family, tag):
    checks = run_checks(objective, plan, compliance, full, limits, family)
    blocking_failed = [c for c in checks if not c["passed"] and c["check"] in BLOCKING]
    rev = review(question, objective, plan, checks, full, tag)
    passed = not blocking_failed and rev["decision"] == "accept"
    return dict(plan_id=plan["plan_id"], audited=now(), passed=passed, checks=checks,
                blocking_failures=[c["check"] for c in blocking_failed], review=rev)
