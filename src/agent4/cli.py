"""Stage entry points of the agent (README 6.1). Each stage reads and writes files under outputs/<task>/ and
can be re-run; stages that need data run inside Slurm jobs, stages that only talk to an LLM run in the controller.

    python -m agent4.cli profile   --task task_m          step 1 (data job)
    python -m agent4.cli objective --task task_m [--alt]  step 2 (generator)
    python -m agent4.cli plans     --task task_m [--alt]  step 3 (generator, de-duplication) and step 4 (compliance)
"""
from __future__ import annotations

import argparse
import sys

import yaml

from . import planning
from .io import ROOT, now, read_json, sha256_file, write_index, write_json
from .schemas import DataProfile, Objective, Plan


def task_config(task_id):
    return yaml.safe_load((ROOT / "config" / f"{task_id}.yaml").read_text())


def task_dir(task_id, alt=False):
    return ROOT / "outputs" / task_id / ("alt_question" if alt else "")


def stage_profile(task_id):
    from .data import load_view
    from .diagnose import profile_and_diagnose
    cfg = task_config(task_id)
    view = load_view(ROOT / cfg["view_dir"])
    profile = profile_and_diagnose(view, cfg)
    out = task_dir(task_id) / "shared_inputs"
    write_json(out / "profile.json", profile.model_dump())
    write_json(out / "difficulties.json", [d.model_dump() for d in profile.difficulties])
    for d in profile.difficulties:
        print(f"{d.id} [{d.tag}/{d.severity}] {d.evidence}")


def stage_objective(task_id, alt=False):
    cfg = task_config(task_id)
    profile = DataProfile(**read_json(task_dir(task_id) / "shared_inputs" / "profile.json"))
    question = cfg["alternative_question"] if alt else cfg["question"]
    objective, meta = planning.compile_objective(question, profile, tag=f"{task_id}{'-alt' if alt else ''}-objective")
    out = task_dir(task_id, alt) / "shared_inputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "question.txt").write_text(question + "\n")
    write_json(out / "objective.json", objective.model_dump())
    write_json(out / "objective_provenance.json", dict(compiled=now(), **meta))
    write_json(out / "constraints.json", dict(
        limits=cfg["limits"], decision_rule=cfg["dataset_card"].get("decision_rule"), rulers=cfg["dataset_card"].get("rulers"),
        scope_of_inference=cfg["dataset_card"].get("scope_of_inference"),
        how_predictions_are_scored=cfg["dataset_card"].get("how_predictions_are_scored")))
    print(f"objective [{meta['source']}]: family={objective.task_family}; forbidden={objective.forbidden_inputs}; "
          f"criteria={[c.check for c in objective.acceptance_criteria]}")


def stage_plans(task_id, alt=False):
    cfg = task_config(task_id)
    base, out = task_dir(task_id), task_dir(task_id, alt)
    profile = DataProfile(**read_json(base / "shared_inputs" / "profile.json"))
    objective = Objective(**read_json(out / "shared_inputs" / "objective.json"))
    limits, family = cfg["limits"], objective.task_family
    tag = f"{task_id}{'-alt' if alt else ''}-plans"
    candidates, invalid, calls = [], [], []
    for round_idx in range(2):                      # README 6.2: one more round if fewer than the pool size survive
        got, bad, round_calls = planning.generate_plans(objective, profile, limits, tag, round_idx, exclude=candidates or None)
        candidates += got
        invalid += [dict(round=round_idx, **b) for b in bad]
        calls += [dict(round=round_idx, **rc) for rc in round_calls]
        pool, dropped, report = planning.select_pool(candidates, family, limits)
        if len(pool) >= int(limits["n_plans"]):
            break
    renamed = {}
    for i, p in enumerate(pool, 1):
        renamed[p.plan_id] = f"P{i}"
        p.plan_id = f"P{i}"
    report["renamed_from_candidate_ids"] = renamed
    report["invalid_candidates"] = invalid
    view_meta = read_json(ROOT / cfg["view_dir"] / "view.json")
    compliance = [planning.check_compliance(p, objective, family, limits, view_meta) for p in pool]
    planning.save_pool(out / "plans", objective, pool, dropped, report, calls, compliance)
    write_json(out / "plans" / "registry_gaps.json", planning.unknown_components(pool, family))
    provenance = out / "plans" / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    for i, call in enumerate(calls):
        rec = read_json(ROOT / "llm_cache" / "generator" / f"{call['cache_key']}.json")
        write_json(provenance / f"generation_call{i + 1}.json", dict(model=rec["model"], created=rec["created"], system=rec["system"],
                                                                   prompt=rec["prompt"], raw_reply=rec["text"], usage=rec["usage"],
                                                                   attempts=rec["attempts"]))
    print(f"pool: {len(pool)} plans ({report['n_ambitious']} ambitious, {report['n_simple']} simple); dropped {len(dropped)}; "
          f"invalid {len(invalid)}; diversity ok: {report['diversity_ok']}")
    for p, c in zip(pool, compliance):
        print(f"  {p.plan_id} [{p.complexity}] {p.name} :: {p.model.get('component')} :: compliant={c['passed']}")
    gaps = read_json(out / "plans" / "registry_gaps.json")
    if gaps:
        print("components not in the registry:", [(g["plan_id"], g["component"]) for g in gaps])


def stage_recheck(task_id, alt=False):
    """Re-run the deterministic compliance check on the frozen pool (no generator call; the pool itself is untouched)."""
    cfg, out = task_config(task_id), task_dir(task_id, alt)
    objective = Objective(**read_json(out / "shared_inputs" / "objective.json"))
    pool = [Plan(**read_json(out / "plans" / f"{p}.json")) for p in read_json(out / "plans" / "pool.json")["plan_ids"]]
    view_meta = read_json(ROOT / cfg["view_dir"] / "view.json")
    compliance = [planning.check_compliance(p, objective, objective.task_family, cfg["limits"], view_meta) for p in pool]
    write_json(out / "plans" / "compliance.json", compliance)
    write_json(out / "plans" / "registry_gaps.json", planning.unknown_components(pool, objective.task_family))
    for p, c in zip(pool, compliance):
        failed = [r["rule"] for r in c["rules"] if not r["passed"]]
        print(f"  {p.plan_id} [{p.complexity}] {p.name} :: compliant={c['passed']} {failed or ''}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--task", required=True)
    ap.add_argument("--alt", action="store_true", help="the alternative question (plan generation only, README 4.4)")
    a, rest = ap.parse_known_args(argv)
    if a.stage == "profile":
        stage_profile(a.task)
    elif a.stage == "objective":
        stage_objective(a.task, a.alt)
    elif a.stage == "plans":
        stage_plans(a.task, a.alt)
    elif a.stage == "recheck":
        stage_recheck(a.task, a.alt)
    else:
        from . import run_stages
        run_stages.dispatch(a.stage, a.task, rest)


if __name__ == "__main__":
    sys.exit(main())
