"""Stages 5-8 of README 6.1 (pilot, tournament, full execution, audit) plus bookkeeping.

    python -m agent4.cli pilot      --task T --plan P3     data job
    python -m agent4.cli tournament --task T               controller (judge calls)
    python -m agent4.cli execute    --task T --plan P3     data job
    python -m agent4.cli audit      --task T               controller; exit code 3 means "run `execute` for the plan printed"
    python -m agent4.cli costs      --task T
    python -m agent4.cli index      --task T

The task family named in the compiled objective selects the flow; nothing here depends on a dataset name.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback

from . import audit as audit_mod
from . import flow_m, flow_t, tournament
from .io import ROOT, now, read_json, write_index, write_json

FLOWS = {flow_m.FAMILY: flow_m, flow_t.FAMILY: flow_t}


class Context:
    def __init__(self, task_id):
        from .cli import task_config, task_dir
        self.task_id, self.cfg, self.dir = task_id, task_config(task_id), task_dir(task_id)
        shared = self.dir / "shared_inputs"
        self.question = (shared / "question.txt").read_text()
        self.objective = read_json(shared / "objective.json")
        self.difficulties = read_json(shared / "difficulties.json")
        self.constraints = read_json(shared / "constraints.json")
        self.family = self.objective["task_family"]
        self.flow = FLOWS[self.family]
        self.pool = read_json(self.dir / "plans" / "pool.json")["plan_ids"]
        self.compliance = {c["plan_id"]: c for c in read_json(self.dir / "plans" / "compliance.json")}
        self.limits = dict(self.cfg["limits"])

    def plan(self, plan_id):
        return read_json(self.dir / "plans" / f"{plan_id}.json")

    def view(self):
        from .data import load_view
        v = load_view(ROOT / self.cfg["view_dir"])
        if v.adapter == "paired_single_cell":
            self.limits.update(n_dev=int(v.dev_rna.shape[0]), n_scope=int(len(v.scope)))
        return v

    @property
    def compliant(self):
        return [p for p in self.pool if self.compliance[p]["passed"]]


def stage_pilot(ctx, plan_id):
    plan, out = ctx.plan(plan_id), ctx.dir / "pilot" / plan_id
    out.mkdir(parents=True, exist_ok=True)
    try:
        summary = ctx.flow.validate(plan, ctx.view(), "pilot", out)
        card = ctx.flow.evidence_card(plan, summary, ctx.compliance[plan_id], ctx.limits)
        write_json(out / "evidence_card.json", card.model_dump())
        write_json(out / "status.json", dict(plan_id=plan_id, status="ok", finished=now()))
        for it in card.items:
            print(f"[{it.id}] ({it.kind}) {it.text}")
    except Exception as exc:
        write_json(out / "status.json", dict(plan_id=plan_id, status="failed", finished=now(), error=repr(exc)[:500],
                                             traceback=traceback.format_exc()[-3000:]))
        raise


def stage_cards(ctx):
    """Rebuild every evidence card from the saved pilot results (no computation; used when only the card wording changes)."""
    for plan_id in ctx.compliant:
        out = ctx.dir / "pilot" / plan_id
        summary = read_json(out / "self_validation.json")
        ctx.view()                                           # fills size limits used in the card text
        card = ctx.flow.evidence_card(ctx.plan(plan_id), summary, ctx.compliance[plan_id], ctx.limits)
        write_json(out / "evidence_card.json", card.model_dump())
        print(f"{plan_id}: {len(card.items)} evidence items")


def stage_tournament(ctx):
    plans, cards = [], {}
    for p in ctx.compliant:
        status = read_json(ctx.dir / "pilot" / p / "status.json")
        if status["status"] != "ok":
            raise RuntimeError(f"pilot of {p} did not finish: {status.get('error')}")
        plans.append(ctx.plan(p))
        cards[p] = read_json(ctx.dir / "pilot" / p / "evidence_card.json")
    result = tournament.run(ctx.question, ctx.objective, ctx.difficulties, ctx.constraints, plans, cards,
                            ctx.dir / "tournament", tag=f"{ctx.task_id}-match")
    print("ranking:", " > ".join(result["ranking"]), "| order consistency", round(result["order_consistency"], 3),
          "| strengths", {k: round(v, 3) for k, v in result["bradley_terry_strength"].items()})
    return result


def stage_execute(ctx, plan_id):
    out = ctx.dir / "full" / plan_id
    out.mkdir(parents=True, exist_ok=True)
    result = ctx.flow.execute(ctx.plan(plan_id), ctx.view(), out)
    print(json.dumps({k: v for k, v in result.items() if k not in ("final_fit_curves", "training")}, indent=1, default=str)[:3000])


def stage_audit(ctx, max_plans=None):
    """Audit plans in tournament order until one passes. Exit code 3 asks the caller to execute a plan first."""
    ranking = read_json(ctx.dir / "tournament" / "ranking.json")["ranking"]
    out = ctx.dir / "audit"
    out.mkdir(parents=True, exist_ok=True)
    trail = []
    for rank, plan_id in enumerate(ranking[: max_plans or len(ranking)], 1):
        full_path = ctx.dir / "full" / plan_id / "full_run.json"
        if not full_path.exists():
            print(f"NEED_EXECUTE {plan_id}")
            return 3
        res = audit_mod.audit(ctx.question, ctx.objective, ctx.plan(plan_id), ctx.compliance[plan_id], read_json(full_path),
                              ctx.limits, ctx.family, tag=f"{ctx.task_id}-audit-{plan_id}")
        res["tournament_rank"] = rank
        write_json(out / f"{plan_id}.json", res)
        trail.append(dict(plan_id=plan_id, rank=rank, passed=res["passed"], blocking_failures=res["blocking_failures"],
                          reviewer_decision=res["review"]["decision"]))
        print(f"audit {plan_id} (rank {rank}): passed={res['passed']} blocking={res['blocking_failures']} reviewer={res['review']['decision']}")
        if res["passed"]:
            write_json(out / "final.json", dict(decided=now(), accepted_plan=plan_id, tournament_rank=rank, trail=trail,
                                                outcome="answer_reported"))
            return 0
    write_json(out / "final.json", dict(decided=now(), accepted_plan=None, trail=trail, outcome="no_available_plan_can_answer"))
    return 0


def stage_costs(ctx):
    """Collect compute and LLM costs recorded by the stages (README 5.5, last row)."""
    plans = {}
    for p in ctx.pool:
        row = {}
        for scale, path in (("pilot", ctx.dir / "pilot" / p / "self_validation.json"), ("full", ctx.dir / "full" / p / "full_run.json")):
            if path.exists():
                d = read_json(path)
                seconds = d.get("fit_seconds") or d.get("plan_seconds") or d.get("seconds_total") or 0
                device = ((d.get("fit_reports") or [{}])[0].get("device") or (d.get("final_fit") or {}).get("device") or "cpu")
                row[scale] = dict(wall_seconds=seconds, cpus=d.get("cpus"), cpu_core_hours=round(seconds * (d.get("cpus") or 1) / 3600, 3),
                                  gpu_hours=round(seconds / 3600, 3) if "cuda" in str(device) else 0.0, device=device)
        plans[p] = row
    calls = [json.loads(line) for line in open(ROOT / "logs" / "llm_calls.jsonl")] if (ROOT / "logs" / "llm_calls.jsonl").exists() else []
    mine = [c for c in calls if str(c.get("tag", "")).startswith(ctx.task_id)]
    llm = {}
    for role in ("generator", "judge"):
        ok = [c for c in mine if c["role"] == role and c.get("ok") and not c.get("cached")]
        llm[role] = dict(answered_calls=len(ok), failed_attempts=sum(1 for c in mine if c["role"] == role and c.get("ok") is False),
                         input_tokens=sum((c.get("usage") or {}).get("input_tokens") or 0 for c in ok),
                         output_tokens=sum((c.get("usage") or {}).get("output_tokens") or 0 for c in ok),
                         thinking_tokens=sum((c.get("usage") or {}).get("thinking_tokens") or 0 for c in ok),
                         wall_seconds=round(sum(c.get("seconds") or 0 for c in ok), 1),
                         list_price_usd=round(sum((c.get("usage") or {}).get("list_price_usd") or 0 for c in ok), 4))
    write_json(ctx.dir / "costs.json", dict(collected=now(), plans=plans, llm=llm))
    print(json.dumps(llm, indent=1))


def stage_index(ctx):
    """README 5.5: one index per task listing every preserved file with its checksum (data arrays of the view excluded)."""
    for sub in ("shared_inputs", "plans", "pilot", "tournament", "full", "audit"):
        if (ctx.dir / sub).exists():
            write_index(ctx.dir / sub, note=f"{ctx.task_id}/{sub}")
    entries = []
    for sub in ("shared_inputs", "plans", "pilot", "tournament", "full", "audit"):
        idx = ctx.dir / sub / "index.json"
        if idx.exists():
            for f in read_json(idx)["files"]:
                entries.append(dict(path=f"{sub}/{f['path']}", bytes=f["bytes"], sha256=f["sha256"]))
    write_json(ctx.dir / "index.json", dict(
        created=now(), task_id=ctx.task_id, n_files=len(entries),
        note="The evaluator keeps the unified scores of benchmark 1 in its own directory; they are not listed here and selection methods must not read them.",
        files=entries))
    print(f"indexed {len(entries)} files for {ctx.task_id}")


def dispatch(stage, task_id, rest):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan")
    a = ap.parse_args(rest)
    ctx = Context(task_id)
    if stage == "pilot":
        stage_pilot(ctx, a.plan)
    elif stage == "cards":
        stage_cards(ctx)
    elif stage == "tournament":
        stage_tournament(ctx)
    elif stage == "execute":
        stage_execute(ctx, a.plan)
    elif stage == "audit":
        sys.exit(stage_audit(ctx))
    elif stage == "costs":
        stage_costs(ctx)
    elif stage == "index":
        stage_index(ctx)
    else:
        raise SystemExit(f"unknown stage {stage}")
