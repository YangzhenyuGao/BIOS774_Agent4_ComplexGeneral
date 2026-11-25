"""Steps 2-4 of README 6.1: compile the objective, generate and de-duplicate plans, check compliance."""
from __future__ import annotations

import json

import yaml
from pydantic import ValidationError

from .io import ROOT, now, sha256_text, write_json
from .llm import client
from .schemas import TASK_FAMILIES, DataProfile, Objective, Plan
from .tools import registry

PROMPTS = ROOT / "templates" / "prompts"

# Deterministic checks the audit knows how to run (README 6.1 step 8). The objective picks from these.
CHECKS = {
    "no_forbidden_inputs": "the plan uses no input the objective forbids",
    "correct_split_unit": "the plan's own validation keeps each analysis unit (and everything nested in it) on one side of every split",
    "beats_required_floors": "the method's primary metric is above each required floor / comparator on the same units",
    "beats_simple_reference": "the method is compared directly with the simplest valid plan on the same units",
    "training_converged": "training curves exist and the last part of the curve is no longer improving",
    "scored_on_unsmoothed_truth": "scores are computed against measured, unsmoothed values of units not used for fitting",
    "both_arms_same_units": "the comparison arms use the same units and the same missing-value handling",
    "selection_nested": "model or configuration choices were made without the units the reported score is computed on",
    "interval_reported": "the primary estimate comes with a confidence interval at the 95% level",
    "multiplicity_controlled": "claims about many features control the false discovery rate at 5%",
    "within_budget": "the full run fits the declared budget",
    "deliverable_complete": "the required deliverable exists, has the right shape and contains only finite values",
}


def _prompt(name):
    return (PROMPTS / name).read_text()


def _profile_block(profile: DataProfile) -> str:
    lines = ["DATASET DESCRIPTION (from the data provider):", yaml.safe_dump(profile.dataset_card, sort_keys=False, width=110).strip(),
             "", "MEASURED DATA PROFILE:", json.dumps(profile.summary, indent=1), "", "MEASURED DIFFICULTIES (most severe first):"]
    for d in profile.difficulties:
        lines.append(f"- {d.id} [{d.tag}, severity {d.severity}] {d.evidence} Implication: {d.implication}")
    return "\n".join(lines)


def rule_based_objective(question, profile: DataProfile) -> Objective:
    """Fallback when the generator's answer cannot be validated (README 6.1 step 2): fill the objective from the dataset card."""
    card = profile.dataset_card
    family = "cross_modal_prediction" if profile.adapter == "paired_single_cell" else "incremental_value_classification"
    checks = ["no_forbidden_inputs", "correct_split_unit", "beats_required_floors", "training_converged", "interval_reported", "deliverable_complete"]
    return Objective(
        question=question, task_family=family, analysis_unit=card["unit"], estimand=card.get("decision_rule", ""),
        target="see dataset description", allowed_inputs=list(profile.summary.get("modalities", {}).keys()), forbidden_inputs=[],
        required_comparators=list(card.get("rulers", {}).keys()), primary_metric="see decision rule",
        acceptance_criteria=[dict(id=f"A{i + 1}", statement=CHECKS[c], check=c, rule="as in the dataset decision rule") for i, c in enumerate(checks)],
        conclusion_options=["see decision rule"], conclusion_scope=card.get("scope_of_inference", ""),
        notes="rule-based fallback: the generator's answer did not validate")


def compile_objective(question, profile: DataProfile, tag) -> tuple[Objective, dict]:
    system = _prompt("objective_system.txt")
    families = "\n".join(f"- {k}: {v['what']}" for k, v in registry.FAMILIES.items())
    checks = "\n".join(f"- {k}: {v}" for k, v in CHECKS.items())
    prompt = (f"SCIENTIFIC QUESTION:\n{question}\n\n{_profile_block(profile)}\n\nTASK FAMILIES (choose one key):\n{families}\n\n"
              f"DETERMINISTIC CHECKS AVAILABLE TO THE AUDIT (use these keys in acceptance_criteria.check):\n{checks}\n")
    record, error = client.call("generator", system, prompt, tag=tag), None
    for attempt in range(2):
        try:
            obj = Objective(**record["json"])
            bad = [c.check for c in obj.acceptance_criteria if c.check not in CHECKS]
            if obj.task_family not in TASK_FAMILIES or bad:
                raise ValueError(f"unknown task family or check keys: {obj.task_family} {bad}")
            return obj, dict(source="generator", model=record["model"], cache_key=record["key"], repair_attempts=attempt, error=None)
        except (ValidationError, ValueError, TypeError) as exc:
            error = str(exc)[:1500]
            repair = prompt + f"\n\nYOUR PREVIOUS ANSWER DID NOT VALIDATE:\n{error}\n\nReturn the corrected JSON object only."
            record = client.call("generator", system, repair, tag=f"{tag}-repair{attempt + 1}")
    return rule_based_objective(question, profile), dict(source="rule_fallback", error=error)


def retrieve_lessons(profile: DataProfile, max_items=8):
    lessons = yaml.safe_load((ROOT / "knowhow" / "lessons.yaml").read_text())
    tags = {d.tag: d.severity for d in profile.difficulties}
    weight = {"high": 3, "medium": 2, "low": 1}
    scored = [(sum(weight[tags[t]] for t in e["tags"] if t in tags), e) for e in lessons]
    return [e for s, e in sorted(scored, key=lambda x: -x[0]) if s > 0][:max_items]


def _brief(p: Plan) -> str:
    inp = json.dumps(p.inputs)[:220]
    return (f"- {p.name} [{p.complexity}] component={p.model.get('component')} fusion={p.fusion} inputs={inp} "
            f"validation={json.dumps(p.validation)[:160]} diversity={json.dumps(p.diversity)}")


def generate_plans(objective: Objective, profile: DataProfile, limits, tag, round_idx=0, exclude=None):
    """One round of generation, made of several small generator calls (README 6.7: a call may take at most
    six minutes, so each call designs only a few plans and sees what earlier calls proposed).
    Returns (valid plans, invalid entries with reasons, list of call records)."""
    n_total, per_call = int(limits["n_candidates_requested"]), int(limits.get("plans_per_call", 3))
    n_calls = -(-n_total // per_call)
    lessons = retrieve_lessons(profile)
    base = ("COMPILED OBJECTIVE:\n" + json.dumps(objective.model_dump(), indent=1) + "\n\n" + _profile_block(profile) +
            "\n\nCOMPONENT CATALOGUE:\n" + registry.describe(objective.task_family) +
            "\n\nLESSONS FROM EARLIER PROJECTS (retrieved by difficulty):\n" +
            "\n".join(f"- {e['id']} ({e['source']}): {e['lesson']} Practice: {e['practice']}" for e in lessons) + "\n")
    valid, invalid, calls = [], [], []
    for b in range(n_calls):
        seen = list(exclude or []) + valid
        need_simple = not any(p.complexity == "simple_reference" for p in seen)
        simple_rule = ("Exactly ONE plan in this call must be the simplest valid plan (complexity \"simple_reference\"); it exists to show what the complex plans buy. Every other plan must be ambitious."
                       if need_simple else
                       "A simple reference plan has already been proposed. Every plan in this call must be ambitious.")
        batch_note = (f"This is call {b + 1} of {n_calls} in this round; {len(seen)} plan(s) proposed earlier are listed at the end of the user message. "
                      "Your plans must differ in design from all of them and widen the coverage of the three diversity directions.")
        system = (_prompt("plans_system.txt").replace("{n_candidates}", str(per_call)).replace("{full_minutes}", str(limits["full_run_minutes"]))
                  .replace("{cpus}", str(limits["cpus_per_job"])).replace("{max_new}", "1").replace("{batch_note}", batch_note)
                  .replace("{simple_rule}", simple_rule))
        prompt = base + ("\nPLANS ALREADY PROPOSED (do not repeat these designs):\n" + "\n".join(_brief(p) for p in seen) + "\n" if seen else
                         "\nPLANS ALREADY PROPOSED: none yet.\n")
        record = client.call("generator", system, prompt, tag=f"{tag}-round{round_idx}-call{b + 1}")
        raw = record["json"]["plans"] if isinstance(record["json"], dict) else record["json"]
        for i, item in enumerate(raw):
            try:
                item = dict(item)
                item["plan_id"] = f"R{round_idx}B{b + 1}C{i + 1}"
                valid.append(Plan(**item))
            except (ValidationError, TypeError) as exc:
                invalid.append(dict(call=b + 1, index=i, name=item.get("name") if isinstance(item, dict) else None, error=str(exc)[:600]))
        calls.append(dict(model=record["model"], cache_key=record["key"], system_sha256=sha256_text(system), prompt_sha256=sha256_text(prompt),
                          lessons=[e["id"] for e in lessons], seconds=record["seconds"], attempts=record["attempts"], usage=record["usage"]))
    return valid, invalid, calls


# ----------------------------------------------------------------------------- de-duplication and pool selection
def _signature(plan: Plan) -> dict:
    inp = plan.inputs if isinstance(plan.inputs, dict) else {}
    reps = inp.get("atac_representation") or inp.get("representations") or []
    rep_types = sorted({r.get("type") for r in reps if isinstance(r, dict)}) if isinstance(reps, list) else []
    image = inp.get("image", {}) if isinstance(inp.get("image"), dict) else {}
    members = plan.model.get("params", {}).get("members") or []
    return dict(
        component=plan.model.get("component"), input_types=rep_types, image_level=image.get("level"),
        encoders=sorted(image.get("encoders") or []), fusion=getattr(plan, "fusion", None),
        members=sorted(m.get("component", "") for m in members if isinstance(m, dict)),
        learner=plan.model.get("params", {}).get("learner"), aggregator=plan.model.get("params", {}).get("aggregator"),
        validation=(plan.validation.get("scheme"), plan.validation.get("scoring_target"), plan.validation.get("selection")))


def similarity(a: Plan, b: Plan) -> float:
    """Share of structured design fields on which two plans agree (component counts double)."""
    sa, sb = _signature(a), _signature(b)
    weights = dict(component=2.0, input_types=1.0, image_level=1.0, encoders=1.0, fusion=1.0, members=1.0, learner=1.0,
                   aggregator=1.0, validation=1.0)
    used = [k for k in weights if sa[k] not in (None, [], ()) or sb[k] not in (None, [], ())]
    return sum(weights[k] for k in used if sa[k] == sb[k]) / max(sum(weights[k] for k in used), 1e-9)


def select_pool(candidates: list[Plan], family, limits, threshold=0.75):
    """Drop near-duplicates, then keep one simple reference plus ambitious plans up to the pool size."""
    kept, dropped = [], []
    for p in candidates:
        twin = next((q for q in kept if similarity(p, q) >= threshold), None)
        if twin is not None:
            dropped.append(dict(plan_id=p.plan_id, name=p.name, reason=f"too similar to {twin.plan_id} (similarity {similarity(p, twin):.2f})",
                                plan=p.model_dump()))
        else:
            kept.append(p)
    n, simple = int(limits["n_plans"]), [p for p in kept if p.complexity == "simple_reference"]
    ambitious = [p for p in kept if p.complexity == "ambitious"]
    pool = simple[:1] + ambitious[: n - min(len(simple), 1)]
    for p in kept:
        if p not in pool:
            why = "a simple reference is already in the pool" if p.complexity == "simple_reference" else "pool is full"
            dropped.append(dict(plan_id=p.plan_id, name=p.name, reason=why, plan=p.model_dump()))
    coverage = {axis: sorted({str(p.diversity.get(axis)) for p in pool}) for axis in ("data", "model", "method")}
    report = dict(n_candidates=len(candidates), n_after_dedup=len(kept), n_pool=len(pool), n_ambitious=sum(p.complexity == "ambitious" for p in pool),
                  n_simple=sum(p.complexity == "simple_reference" for p in pool), diversity_coverage=coverage,
                  diversity_ok=all(len(v) >= 2 for v in coverage.values()), threshold=threshold,
                  pairwise_similarity={f"{a.plan_id}|{b.plan_id}": round(similarity(a, b), 3) for i, a in enumerate(pool) for b in pool[i + 1:]})
    return pool, dropped, report


def unknown_components(pool: list[Plan], family):
    """Components a plan needs that the registry does not have (README 6.5: extendable before the freeze)."""
    known = registry.component_keys(family)
    out = []
    for p in pool:
        keys = [p.model.get("component")] + [m.get("component") for m in p.model.get("params", {}).get("members", []) if isinstance(m, dict)]
        for k in keys:
            if k not in known:
                desc = next((c.get("description") for c in p.new_components if c.get("name") == k), None)
                out.append(dict(plan_id=p.plan_id, component=k, description=desc))
    return out


# ----------------------------------------------------------------------------- compliance (step 4)
CLINICAL_COVARIATES = {"age", "stage", "histological_type", "er", "pr", "her2"}
ATAC_REPRESENTATIONS = {"lsi", "tfidf_peaks", "gene_activity", "cis_peaks", "depth"}


def _coverage(statement, family):
    """How one forbidden-input statement of the objective is guaranteed. Only explains; pass/fail comes from the structured rules."""
    s = statement.lower()
    leak_scope = any(w in s for w in ("fold", "split", "held-out", "validation", "evaluation"))
    leak_step = any(w in s for w in ("during", "training", "preprocessing", "selection", "tuning", "prediction"))
    if "smooth" in s or "full-depth" in s or "scoring target" in s:
        return "rule scoring_target_allowed"
    if leak_scope and leak_step:
        return "enforced by the executor: every transform and model is fitted inside the training units of a split; other units are only predicted"
    if "baseline arm" in s:
        return "enforced by the executor: the clinical-only arm is built from clinical variables alone"
    if "external" in s or "outside the provided" in s or "pretrained" in s:
        return "rule inputs_from_view: the executor can read nothing but the task view"
    if "site" in s:
        return "rule clinical_variables_allowed: site is a grouping column, not an allowed covariate"
    if "annotation" in s or "cell-type" in s or "cell type" in s:
        return "rule no_label_inputs"
    if "rna" in s:
        return "rule inputs_from_view: the view offers no such variable for the units to be predicted"
    if "clinical" in s or "receptor" in s:
        return "rule clinical_block_matches_question"
    return "no deterministic rule; left to the audit reviewer"


def check_compliance(plan: Plan, objective: Objective, family, limits, view_meta=None) -> dict:
    """Deterministic checks on the plan's structured fields that can eliminate it before any computation."""
    rules, inp = [], (plan.inputs if isinstance(plan.inputs, dict) else {})
    input_rules = []

    def rule(name, passed, detail, is_input=False):
        rules.append(dict(rule=name, passed=bool(passed), detail=detail))
        if is_input:
            input_rules.append(bool(passed))

    if family in ("incremental_value_classification", "single_modality_classification"):
        clin = inp.get("clinical") if isinstance(inp.get("clinical"), dict) else {}
        image = inp.get("image") if isinstance(inp.get("image"), dict) else {}
        use_clin = bool(clin.get("use", True))
        variables = set(clin.get("variables") or (CLINICAL_COVARIATES if use_clin else set()))
        extra_blocks = sorted(set(inp) - {"clinical", "image"})
        encoders, known = set(image.get("encoders") or []), set((view_meta or {}).get("encoders") or [])
        unknown_enc = sorted(encoders - known) if known else []
        rule("inputs_from_view", not extra_blocks and not unknown_enc,
             f"input blocks {sorted(inp)}; unknown blocks {extra_blocks}; unknown encoders {unknown_enc}", True)
        rule("clinical_variables_allowed", variables <= CLINICAL_COVARIATES,
             f"declared clinical variables {sorted(variables)}; outside the six covariates: {sorted(variables - CLINICAL_COVARIATES)}", True)
        clinical_forbidden = any(("clinical" in f.lower() or "receptor" in f.lower()) and "baseline arm" not in f.lower()
                                 and not any(w in f.lower() for w in ("external", "outside", "fold", "split")) for f in objective.forbidden_inputs)
        want = not clinical_forbidden
        rule("clinical_block_matches_question", use_clin == want,
             f"the question {'conditions on' if want else 'excludes'} clinical variables; the plan {'uses' if use_clin else 'does not use'} them", True)
        unit = str(plan.validation.get("split_unit", "patient")).lower()
        rule("correct_split_unit", unit == "patient", f"declared split unit: {unit}; the analysis unit is the patient")
    else:
        reps = inp.get("atac_representation") or inp.get("representations")
        types = {str(r.get("type")) for r in reps if isinstance(r, dict)} if isinstance(reps, list) else {k for k in inp if k not in ("atac_representation", "representations")}
        extra = sorted(types - ATAC_REPRESENTATIONS)
        rule("inputs_from_view", not extra, f"input representations {sorted(types)}; not ATAC-derived or not in the catalogue: {extra}", True)
        text = json.dumps(plan.inputs).lower()
        rule("no_label_inputs", not any(w in text for w in ("celltype", "cell_type", "annotation")),
             "the supplied cell-type annotation is not among the model inputs" if not any(w in text for w in ("celltype", "cell_type", "annotation"))
             else "the supplied cell-type annotation appears among the model inputs", True)
        if any(c.check == "scored_on_unsmoothed_truth" for c in objective.acceptance_criteria):
            target = str(plan.validation.get("scoring_target", "thinned_half")).lower()
            rule("scoring_target_allowed", target == "thinned_half",
                 f"declared scoring target: {target}; the objective requires measured, unsmoothed, independently thinned values", True)
        unit = str(plan.validation.get("split_unit", "cell")).lower()
        rule("correct_split_unit", unit == "cell", f"declared split unit: {unit}; the analysis unit is the cell")
    coverage = {f: _coverage(f, family) for f in objective.forbidden_inputs}
    rule("no_forbidden_inputs", all(input_rules),
         "all input rules passed; each forbidden input of the objective is covered as follows: " + " | ".join(f"[{k}] -> {v}" for k, v in coverage.items()))
    known = registry.component_keys(family)
    keys = [plan.model.get("component")] + [m.get("component") for m in plan.model.get("params", {}).get("members", []) if isinstance(m, dict)]
    missing = [k for k in keys if k not in known]
    rule("components_executable", not missing, f"not in registry: {missing}" if missing else "all components registered")
    over = plan.budget.full_minutes > float(limits["full_run_minutes"])
    rule("within_budget", not over, f"declared full run {plan.budget.full_minutes} min; limit {limits['full_run_minutes']} min")
    return dict(plan_id=plan.plan_id, passed=all(r["passed"] for r in rules), rules=rules, forbidden_input_coverage=coverage, checked=now())


def save_pool(out_dir, objective, pool, dropped, report, calls, compliance=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in pool:
        write_json(out_dir / f"{p.plan_id}.json", p.model_dump())
        (out_dir / f"{p.plan_id}.txt").write_text(f"{p.name}\n\n{p.summary}\n\nRationale: {p.rationale}\n")
    write_json(out_dir / "pool.json", dict(frozen=now(), plan_ids=[p.plan_id for p in pool], selection_report=report, generation_calls=calls))
    write_json(out_dir / "dropped.json", dropped)
    if compliance is not None:
        write_json(out_dir / "compliance.json", compliance)
