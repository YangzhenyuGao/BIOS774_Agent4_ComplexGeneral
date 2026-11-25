#!/usr/bin/env python
"""One command for the whole pipeline (README 9.2, stages 1-6).

    python scripts/controller.py all                  both tasks, every stage, skipping whatever already exists
    python scripts/controller.py task --task task_m   one task
    python scripts/controller.py status

Heavy steps are Slurm jobs; LLM steps run here. Every step is skipped when its output already exists, so the
command can be re-run after an interruption, and with a warm llm_cache/ (set A4_REPLAY_ONLY=1) it reproduces
the run without keys or login. `--recompute` ignores existing compute outputs.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from agent4.io import read_json  # noqa: E402

PY = "/vast/som/yunligrp/users/ygao/conda/envs/agent3/bin/python"
ENV = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}:{ROOT}", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", AGENT4_ROOT=str(ROOT))
TASKS = {"task_m": dict(bench="bench.task_m", report="reports/generated_report_1.pdf", title="Generated report 1: PBMC Multiome - which genes can ATAC predict beyond cell type?"),
         "task_t": dict(bench="bench.task_t", report="reports/generated_report_2.pdf", title="Generated report 2: TCGA-BRCA - do H&E images add to clinical variables for PAM50?")}
RECOMPUTE = False


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def local(*args, check=True):
    """Run a light stage in this process's environment (LLM calls, bookkeeping)."""
    log("run: " + " ".join(args))
    p = subprocess.run([PY, "-B", *args], cwd=ROOT, env=ENV)
    if check and p.returncode != 0:
        raise SystemExit(f"stage failed ({p.returncode}): {' '.join(args)}")
    return p.returncode


def sbatch(name, args, cpus=16, mem="64G", minutes=120, array=None, partition="small"):
    cmd = ["sbatch", "--parsable", "-J", name, f"--cpus-per-task={cpus}", f"--mem={mem}", f"--time={minutes}", f"--partition={partition}"]
    if array:
        cmd.append(f"--array={array}")
    out = subprocess.run(cmd + ["scripts/slurm/cpu.sbatch", *args], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"sbatch failed: {out.stderr}")
    job = out.stdout.strip().split(";")[0]
    log(f"submitted {name} as job {job}: {' '.join(args)}")
    return job


def wait(jobs, poll=20):
    jobs = [j for j in jobs if j]
    if not jobs:
        return
    while subprocess.run(["squeue", "-h", "-j", ",".join(jobs)], capture_output=True, text=True).stdout.strip():
        time.sleep(poll)
    time.sleep(5)
    acct = subprocess.run(["sacct", "-n", "-X", "-P", "-j", ",".join(jobs), "-o", "JobID,JobName,State,Elapsed"], capture_output=True, text=True).stdout
    bad = [line for line in acct.splitlines() if line and "COMPLETED" not in line]
    if bad:
        raise SystemExit("jobs did not complete:\n" + "\n".join(bad))
    log(f"jobs finished: {', '.join(jobs)}")


def need(path):
    return RECOMPUTE or not Path(ROOT / path).exists()


def visible(path, timeout=300):
    """Wait until a file written by a compute node is visible here (the shared file system caches directory listings for a few seconds)."""
    target, start = Path(ROOT / path), time.time()
    while time.time() - start < timeout:
        os.listdir(target.parent) if target.parent.exists() else None
        if target.exists():
            return
        time.sleep(3)
    raise SystemExit(f"expected output did not appear: {path}")


def run_task(task):
    t, out, private = TASKS[task], ROOT / "outputs" / task, ROOT / "bench_private" / task
    family_m = task == "task_m"
    # ---- stage 1: evaluator data, frozen rules, rulers
    if need(f"outputs/{task}/agent_view/view.json"):
        wait([sbatch(f"{task}-prepare", ["-m", t["bench"], "prepare"], 16, "64G", 60)])
    if family_m and need(f"outputs/{task}/agent_view/rules.json"):
        wait([sbatch(f"{task}-rules", ["-m", t["bench"], "rules"], 32, "128G", 90)])
    if family_m and need(f"bench_private/{task}/rulers.json"):
        wait([sbatch(f"{task}-rulers", ["-m", t["bench"], "rulers"], 32, "128G", 90)])
    # ---- steps 1-4: profile, objective, plans, compliance
    if need(f"outputs/{task}/shared_inputs/profile.json"):
        wait([sbatch(f"{task}-profile", ["-m", "agent4.cli", "profile", "--task", task], 16, "64G", 30)])
    if need(f"outputs/{task}/shared_inputs/objective.json"):
        local("-m", "agent4.cli", "objective", "--task", task)
    if need(f"outputs/{task}/plans/pool.json"):
        local("-m", "agent4.cli", "plans", "--task", task)
    gaps = read_json(out / "plans" / "registry_gaps.json")
    if gaps:
        raise SystemExit(f"plans need components the registry lacks; implement or reject them before going on: {gaps}")
    pool = read_json(out / "plans" / "pool.json")["plan_ids"]
    compliant = [c["plan_id"] for c in read_json(out / "plans" / "compliance.json") if c["passed"]]
    cpus = 32 if family_m else 12
    part = "small" if family_m else "inter"     # two partitions with separate per-user CPU caps, so the tasks do not block each other
    # ---- step 5: pilots
    wait([sbatch(f"{task}-pilot-{p}", ["-m", "agent4.cli", "pilot", "--task", task, "--plan", p], cpus, "128G", 120, partition=part)
          for p in compliant if need(f"outputs/{task}/pilot/{p}/evidence_card.json")])
    # ---- step 6: tournament
    if need(f"outputs/{task}/tournament/ranking.json"):
        local("-m", "agent4.cli", "tournament", "--task", task)
    ranking = read_json(out / "tournament" / "ranking.json")["ranking"]
    # ---- steps 7-8: full run of the champion, audit, next-ranked plan if the audit rejects
    while need(f"outputs/{task}/audit/final.json"):
        code = local("-m", "agent4.cli", "audit", "--task", task, check=False)
        if code == 0:
            break
        if code != 3:
            raise SystemExit("audit stage failed")
        nxt = next(p for p in ranking if not (out / "full" / p / "full_run.json").exists())
        wait([sbatch(f"{task}-full-{nxt}", ["-m", "agent4.cli", "execute", "--task", task, "--plan", nxt], cpus, "160G", 180, partition=part)])
    # ---- benchmark 1: every candidate in full under the unified protocol (after the agent has decided)
    wait([sbatch(f"{task}-full-{p}", ["-m", "agent4.cli", "execute", "--task", task, "--plan", p], cpus, "160G", 180, partition=part)
          for p in compliant if need(f"outputs/{task}/full/{p}/full_run.json")])
    if family_m:
        wait([sbatch(f"{task}-score-{p}", ["-m", t["bench"], "score", "--pred", f"outputs/{task}/full/{p}/pred_hidden.npy", "--name", p], 32, "128G", 60)
              for p in compliant if need(f"bench_private/{task}/true_scores/{p}.json")])
    else:
        jobs = []
        for spec in ["ruler_clinical_floor", "ruler_simple_reference"] + compliant:
            pid = spec
            arg = spec if spec.startswith("ruler_") else f"outputs/{task}/plans/{spec}.json"
            for fold_set, n in (("main", 15), ("site", 5)):
                if need(f"bench_private/{task}/true_scores/{pid}/{fold_set}/r{(n - 1) // 5}_f4.json"):
                    jobs.append(sbatch(f"{task}-bench-{pid}-{fold_set}", ["-m", t["bench"], "fold", "--plan", arg, "--set", fold_set], 8, "48G", 180, array=f"0-{n - 1}", partition=part))
        wait(jobs)
        for pid in ["ruler_clinical_floor", "ruler_simple_reference"] + compliant:
            for fold_set in ("main", "site"):
                if need(f"bench_private/{task}/true_scores/{pid}/true_score_{fold_set}.json"):
                    local("-m", t["bench"], "collect", "--plan-id", pid, "--set", fold_set)
    local("-m", "bench.export", "--task", task)
    # ---- README 4.4: the same data under another question (plans only)
    if need(f"outputs/{task}/alt_question/plans/pool.json"):
        local("-m", "agent4.cli", "objective", "--task", task, "--alt")
        local("-m", "agent4.cli", "plans", "--task", task, "--alt")
    # ---- bookkeeping and report
    local("-m", "agent4.cli", "costs", "--task", task)
    local("-m", "agent4.cli", "index", "--task", task)
    wait([sbatch(f"{task}-report-facts", ["-m", "agent4.report", "facts", "--task", task], 8, "32G", 30)])
    for name in ("collected.json", "named_facts.json"):
        visible(f"outputs/{task}/report/{name}")
    if need(f"outputs/{task}/report/prose.json"):
        local("-m", "agent4.report", "write", "--task", task)
    local("-m", "agent4.report", "render", "--task", task, "--out", t["report"], "--title", t["title"])
    log(f"{task} finished: pool {pool}, ranking {ranking}")


def status():
    for task in TASKS:
        out = ROOT / "outputs" / task
        marks = {name: (out / rel).exists() for name, rel in (
            ("view", "agent_view/view.json"), ("profile", "shared_inputs/profile.json"), ("objective", "shared_inputs/objective.json"),
            ("pool", "plans/pool.json"), ("tournament", "tournament/ranking.json"), ("audit", "audit/final.json"),
            ("benchmark1", "benchmark1/summary.json"), ("report", "report/prose.json"))}
        pilots = len(list((out / "pilot").glob("*/evidence_card.json"))) if (out / "pilot").exists() else 0
        fulls = len(list((out / "full").glob("*/full_run.json"))) if (out / "full").exists() else 0
        print(task, json.dumps(marks), f"pilots={pilots} full_runs={fulls}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["all", "task", "status"])
    ap.add_argument("--task", choices=list(TASKS))
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    RECOMPUTE = a.recompute
    if a.command == "status":
        status()
    elif a.command == "task":
        run_task(a.task)
    else:
        for task in TASKS:
            run_task(task)
