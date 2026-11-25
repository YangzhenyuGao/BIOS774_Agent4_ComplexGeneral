#!/usr/bin/env python
"""Self-check for README 9.2, stage 6: the LLM stages reproduce from llm_cache/ without a live call.

    python scripts/replay_check.py            # both tasks; writes outputs/replay_check.json

For each task a scratch project root is built that holds only what the LLM stages read (configuration, prompts,
lessons, the cache, the task view, the data profile, the stored evidence cards, the champion's full-run record and
the collected report inputs). The stages are then re-run there with A4_REPLAY_ONLY=1, where a cache miss is an
error, and what they write is compared with the stored outputs. Nothing under outputs/ is modified.

A stage "replays" when it finishes without a live call; it "matches" when its result equals the stored one.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
VOLATILE = {"compiled", "decided", "created", "written", "finished", "audited", "seconds", "cached", "usage", "cost", "collected", "attempts", "exported"}


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in VOLATILE}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def same(a: Path, b: Path, pick=None):
    if not a.exists() or not b.exists():
        return False
    x, y = json.loads(a.read_text()), json.loads(b.read_text())
    if pick:
        x, y = pick(x), pick(y)
    return strip(x) == strip(y)


def run(scratch, *args):
    env = dict(os.environ, AGENT4_ROOT=str(scratch), A4_REPLAY_ONLY="1", PYTHONPATH=f"{ROOT / 'src'}:{ROOT}", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([PY, "-B", *args], cwd=scratch, env=env, capture_output=True, text=True)
    return p.returncode == 0, (p.stderr or p.stdout).strip().splitlines()[-1:] if p.returncode else []


def link_files(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            (dst / f.name).symlink_to(f)


def check_task(task, work):
    real, scratch = ROOT / "outputs" / task, work / task
    out = scratch / "outputs" / task
    out.mkdir(parents=True)
    for name in ("config", "templates", "knowhow", "llm_cache", "scripts"):
        (scratch / name).symlink_to(ROOT / name)
    (out / "agent_view").symlink_to(real / "agent_view")
    (out / "shared_inputs").mkdir()
    for name in ("profile.json", "difficulties.json"):
        shutil.copy(real / "shared_inputs" / name, out / "shared_inputs" / name)
    res, cli = {}, ("-m", "agent4.cli")

    def stage(name, args, compare):
        ok, err = run(scratch, *args)
        res[name] = dict(replayed=ok, matches=bool(ok and all(compare())), **({"error": err} if err else {}))
        return ok

    plan_ids = json.loads((real / "plans" / "pool.json").read_text())["plan_ids"]
    if stage("objective", [*cli, "objective", "--task", task], lambda: [same(out / "shared_inputs" / "objective.json", real / "shared_inputs" / "objective.json")]) and \
       stage("plans", [*cli, "plans", "--task", task], lambda: [same(out / "plans" / f"{p}.json", real / "plans" / f"{p}.json") for p in plan_ids] +
             [same(out / "plans" / "pool.json", real / "plans" / "pool.json", lambda d: d["plan_ids"])]):
        for p in plan_ids:
            (out / "pilot" / p).mkdir(parents=True)
            for name in ("evidence_card.json", "status.json"):
                shutil.copy(real / "pilot" / p / name, out / "pilot" / p / name)
        if stage("tournament", [*cli, "tournament", "--task", task], lambda: [same(out / "tournament" / "ranking.json", real / "tournament" / "ranking.json",
                                                                                 lambda d: dict(ranking=d["ranking"], wins=d["wins_matrix"], n=d["n_matches"]))]):
            final = json.loads((real / "audit" / "final.json").read_text())
            for step in final["trail"]:
                link_files(real / "full" / step["plan_id"], out / "full" / step["plan_id"])
            stage("audit", [*cli, "audit", "--task", task], lambda: [same(out / "audit" / "final.json", real / "audit" / "final.json", lambda d: (d["accepted_plan"], d["outcome"], d["trail"]))] +
                  [same(out / "audit" / f"{s['plan_id']}.json", real / "audit" / f"{s['plan_id']}.json", lambda d: (d["passed"], d["blocking_failures"], d["review"]["decision"], d["review"]["wording_limits"]))
                   for s in final["trail"]])
    if (real / "report" / "prose.json").exists():
        (out / "report").mkdir(parents=True, exist_ok=True)
        for name in ("collected.json", "named_facts.json"):
            shutil.copy(real / "report" / name, out / "report" / name)
        if json.loads((real / "report" / "prose.json").read_text())["meta"].get("source") == "generator":
            stage("report_prose", ["-m", "agent4.report", "write", "--task", task], lambda: [same(out / "report" / "prose.json", real / "report" / "prose.json", lambda d: d["prose"])])
        else:
            res["report_prose"] = dict(replayed=None, matches=None, note="stored prose is the deterministic fallback; no LLM answer to replay")
    if (real / "alt_question" / "plans" / "pool.json").exists():
        alt_ids = json.loads((real / "alt_question" / "plans" / "pool.json").read_text())["plan_ids"]
        if stage("alt_objective", [*cli, "objective", "--task", task, "--alt"], lambda: [same(out / "alt_question" / "shared_inputs" / "objective.json", real / "alt_question" / "shared_inputs" / "objective.json")]):
            stage("alt_plans", [*cli, "plans", "--task", task, "--alt"], lambda: [same(out / "alt_question" / "plans" / f"{p}.json", real / "alt_question" / "plans" / f"{p}.json") for p in alt_ids])
    log = scratch / "logs" / "llm_calls.jsonl"
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    res["llm_answers_served_from_cache"] = sum(bool(c.get("cached")) for c in calls)
    res["live_calls"] = sum(not c.get("cached") for c in calls)
    return res


def main():
    work = Path(tempfile.mkdtemp(prefix="a4_replay_", dir=os.environ.get("TMPDIR")))
    try:
        result = {task: check_task(task, work) for task in ("task_m", "task_t")}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    stages = [(t, s, v) for t, r in result.items() for s, v in r.items() if isinstance(v, dict)]
    result["all_replayed"] = all(v["replayed"] is not False for _, _, v in stages)
    result["all_match"] = all(v["matches"] is not False for _, _, v in stages)
    (ROOT / "outputs" / "replay_check.json").write_text(json.dumps(result, indent=1) + "\n")
    for t, s, v in stages:
        print(f"{t:7s} {s:14s} replayed={v['replayed']} matches={v['matches']} {v.get('error', v.get('note', ''))}")
    print("all replayed:", result["all_replayed"], "| all match:", result["all_match"])
    return 0 if result["all_replayed"] and result["all_match"] else 1


if __name__ == "__main__":
    sys.exit(main())
