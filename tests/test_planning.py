"""Plan handling that needs no LLM and no data: de-duplication, pool selection, compliance rules."""
import copy

from agent4 import planning
from agent4.schemas import Objective, Plan

LIMITS = dict(n_plans=3, min_ambitious=1, n_candidates_requested=4, full_run_minutes=120, cpus_per_job=16)


def t_plan(pid, component="attention_mil", complexity="ambitious", **over):
    base = dict(plan_id=pid, name=f"plan {pid}", summary="s", complexity=complexity, analysis_unit="patient",
                inputs=dict(image=dict(encoders=["UNI2-h"], level="tile"), clinical=dict(use=True, variables=["age", "er", "pr", "her2"])),
                model=dict(component=component, params={}), fusion="early_concat",
                validation=dict(scheme="kfold", k=5, split_unit="patient", selection="nested"),
                budget=dict(full_minutes=60), diversity=dict(data="tile", model=component, method="early"))
    base.update(over)
    return Plan(**base)


def objective(family="incremental_value_classification", forbidden=("RNA expression or RNA-derived variables", "Collection site as a predictive model input"), checks=("no_forbidden_inputs",)):
    return Objective(question="q", task_family=family, analysis_unit="patient", estimand="e", target="t", allowed_inputs=["clinical", "image"],
                     forbidden_inputs=list(forbidden), required_comparators=["clinical only"], primary_metric="m",
                     acceptance_criteria=[dict(id=f"A{i}", statement="s", check=c) for i, c in enumerate(checks, 1)],
                     conclusion_options=["improves"], conclusion_scope="scope")


META = dict(encoders=["UNI2-h", "CONCH-Z"])


def rules(result):
    return {r["rule"]: r["passed"] for r in result["rules"]}


def test_near_duplicates_are_dropped_and_one_simple_reference_is_kept():
    a, b = t_plan("a"), t_plan("b")                                   # identical design
    c = t_plan("c", component="incremental_offset", fusion="offset")
    s1, s2 = t_plan("s1", "patient_mean_logreg", "simple_reference"), t_plan("s2", "gradient_boosting", "simple_reference")
    pool, dropped, report = planning.select_pool([s1, a, b, c, s2], "incremental_value_classification", LIMITS)
    assert [p.plan_id for p in pool] == ["s1", "a", "c"]
    reasons = {d["plan_id"]: d["reason"] for d in dropped}
    assert "too similar" in reasons["b"] and "simple reference" in reasons["s2"]
    assert report["n_simple"] == 1 and report["n_ambitious"] == 2


def test_compliant_plan_passes_every_rule():
    res = planning.check_compliance(t_plan("p"), objective(), "incremental_value_classification", LIMITS, META)
    assert res["passed"], res


def test_slide_level_split_is_rejected():
    p = t_plan("p", validation=dict(scheme="kfold", k=5, split_unit="slide", selection="nested"))
    assert not rules(planning.check_compliance(p, objective(), "incremental_value_classification", LIMITS, META))["correct_split_unit"]


def test_site_or_unknown_covariates_and_unknown_encoders_are_rejected():
    p = t_plan("p")
    q = copy.deepcopy(p)
    q.inputs["clinical"]["variables"] = ["age", "site"]
    r = planning.check_compliance(q, objective(), "incremental_value_classification", LIMITS, META)
    assert not rules(r)["clinical_variables_allowed"] and not rules(r)["no_forbidden_inputs"] and not r["passed"]
    q = copy.deepcopy(p)
    q.inputs["image"]["encoders"] = ["SomeOtherEncoder"]
    assert not rules(planning.check_compliance(q, objective(), "incremental_value_classification", LIMITS, META))["inputs_from_view"]
    q = copy.deepcopy(p)
    q.inputs["rna"] = dict(use=True)
    assert not rules(planning.check_compliance(q, objective(), "incremental_value_classification", LIMITS, META))["inputs_from_view"]


def test_question_about_images_alone_forbids_the_clinical_block():
    obj = objective("single_modality_classification", forbidden=("er, pr, her2 (immunohistochemistry receptor statuses)", "age and stage"))
    with_clin = planning.check_compliance(t_plan("p"), obj, "single_modality_classification", LIMITS, META)
    assert not rules(with_clin)["clinical_block_matches_question"]
    p = t_plan("p", inputs=dict(image=dict(encoders=["UNI2-h"], level="tile"), clinical=dict(use=False)))
    assert planning.check_compliance(p, obj, "single_modality_classification", LIMITS, META)["passed"]


def test_cross_modal_plan_rules():
    obj = objective("cross_modal_prediction", forbidden=("Smoothed or full-depth RNA as the validation scoring target",),
                    checks=("no_forbidden_inputs", "scored_on_unsmoothed_truth"))
    base = dict(plan_id="m", name="m", summary="s", complexity="ambitious", analysis_unit="cell", inputs=dict(lsi=dict(n_components=50), depth={}),
                model=dict(component="translation_network", params={}), validation=dict(scheme="kfold", k=3, split_unit="cell", scoring_target="thinned_half"),
                budget=dict(full_minutes=60), diversity=dict(data="lsi", model="net", method="joint"))
    assert planning.check_compliance(Plan(**base), obj, "cross_modal_prediction", LIMITS)["passed"]
    smooth = copy.deepcopy(base)
    smooth["validation"]["scoring_target"] = "knn_smoothed"
    assert not rules(planning.check_compliance(Plan(**smooth), obj, "cross_modal_prediction", LIMITS))["scoring_target_allowed"]
    labels = copy.deepcopy(base)
    labels["inputs"]["celltype_annotation"] = {}
    r = rules(planning.check_compliance(Plan(**labels), obj, "cross_modal_prediction", LIMITS))
    assert not r["inputs_from_view"] and not r["no_label_inputs"]
    over = copy.deepcopy(base)
    over["budget"]["full_minutes"] = 300
    assert not rules(planning.check_compliance(Plan(**over), obj, "cross_modal_prediction", LIMITS))["within_budget"]
    unknown = copy.deepcopy(base)
    unknown["model"]["component"] = "not_in_registry"
    assert not rules(planning.check_compliance(Plan(**unknown), obj, "cross_modal_prediction", LIMITS))["components_executable"]
