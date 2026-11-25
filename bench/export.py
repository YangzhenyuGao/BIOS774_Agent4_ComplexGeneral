"""Benchmark 1 (README 5.4): put every candidate's unified score next to what the agent saw and chose.

    python -m bench.export --task task_m

Runs only after the agent has finished selecting, executing and auditing. It writes a release copy of the
comparison to outputs/<task>/benchmark1/ for the reports. The agent never reads this directory while it
is deciding, and selection methods compared later must not read it either (only the oracle reference may).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from agent4.io import ROOT, now, read_json, write_json
from bench import paths


def _maybe(path):
    return read_json(path) if path.exists() else None


def _agent_side(task):
    out = ROOT / "outputs" / task
    pool = read_json(out / "plans" / "pool.json")["plan_ids"]
    ranking = read_json(out / "tournament" / "ranking.json")
    final = _maybe(out / "audit" / "final.json") or {}
    rows = {}
    for p in pool:
        plan = read_json(out / "plans" / f"{p}.json")
        rows[p] = dict(plan_id=p, name=plan["name"], complexity=plan["complexity"], component=plan["model"].get("component"),
                       tournament_rank=(ranking["ranking"].index(p) + 1) if p in ranking["ranking"] else None,
                       bt_strength=ranking["bradley_terry_strength"].get(p), pilot=_maybe(out / "pilot" / p / "self_validation.json"),
                       full=_maybe(out / "full" / p / "full_run.json"), audit=_maybe(out / "audit" / f"{p}.json"))
    return out, pool, ranking, final, rows


def _rank(values, higher_is_better=True):
    order = sorted([k for k, v in values.items() if v is not None], key=lambda k: -values[k] if higher_is_better else values[k])
    return {k: i + 1 for i, k in enumerate(order)}


def export_m(task="task_m"):
    out, pool, ranking, final, rows = _agent_side(task)
    private = paths.private_dir(task)
    rulers = read_json(private / "rulers.json")
    ceiling = rulers["rulers"]["ceiling"]["mean_r"]
    table, true = [], {}
    for p in pool:
        ts = _maybe(private / "true_scores" / f"{p}.json")
        true[p] = ts["true_score_mean_r"] if ts else None
        pilot, full = rows[p]["pilot"] or {}, rows[p]["full"] or {}
        confirmed = None
        if ts and (out / "full" / p / "self_validation_per_gene.npz").exists():
            z = np.load(out / "full" / p / "self_validation_per_gene.npz")
            per_gene = pd.read_parquet(private / "true_scores" / f"{p}_per_gene.parquet")
            claimed = set(per_gene.gene.values[np.isin(np.arange(len(per_gene)), np.flatnonzero(z["claimed"]))]) if len(z["claimed"]) == len(per_gene) else set()
            true_beyond = set(per_gene.gene[per_gene.label == 2])
            confirmed = dict(n_claimed=len(claimed), n_confirmed=len(claimed & true_beyond),
                             replication_rate=(len(claimed & true_beyond) / len(claimed)) if claimed else None,
                             n_true_beyond=len(true_beyond))
        table.append(dict(
            plan_id=p, name=rows[p]["name"], complexity=rows[p]["complexity"], component=rows[p]["component"],
            tournament_rank=rows[p]["tournament_rank"], bt_strength=rows[p]["bt_strength"],
            self_score_pilot=pilot.get("self_score_mean_r"), self_score_full=full.get("self_score_mean_r"),
            true_score=true[p], inflation_full=(full.get("self_score_mean_r") - true[p]) if ts and full else None,
            inflation_pilot=(pilot.get("self_score_mean_r") - true[p]) if ts and pilot else None,
            fraction_of_ceiling=ts and ts["fraction_of_ceiling"], n_beyond_cell_type=ts and ts["n_beyond_cell_type"],
            n_not_beyond=ts and ts["n_not_beyond_cell_type"], n_cannot_judge=ts and ts["n_cannot_judge"],
            mean_delta_vs_cluster_mean=ts and ts["mean_delta_vs_cluster_mean"], mean_r_within_cluster=ts and ts["mean_r_within_cluster"],
            beats=ts and ts["beats"], n_claimed_by_plan=full.get("n_genes_claimed_beyond_cell_type"), gene_list_check=confirmed,
            scoring_target=(full.get("validation") or pilot.get("validation") or {}).get("target"),
            audit_passed=(rows[p]["audit"] or {}).get("passed")))
    true_rank = _rank(true)
    for row in table:
        row["true_rank"] = true_rank.get(row["plan_id"])
    selected = final.get("accepted_plan") or ranking["champion"]
    best = max((p for p in pool if true[p] is not None), key=lambda p: true[p])
    summary = dict(
        task=task, exported=now(), primary_score="mean per-gene Pearson r on the hidden B halves, in-scope genes",
        rulers={k: v["mean_r"] for k, v in rulers["rulers"].items()}, n_scope=rulers["n_scope"], n_judgeable=rulers["n_judgeable"],
        tournament_champion=ranking["champion"], selected_plan=selected, best_true_plan=best,
        selected_true_rank=true_rank.get(selected), n_candidates_scored=len(true_rank),
        generation_gap=ceiling - true[best], selection_gap=true[best] - true[selected] if true.get(selected) is not None else None,
        spearman_tournament_vs_true=_spearman([r["tournament_rank"] for r in table], [r["true_rank"] for r in table]),
        spearman_pilot_score_vs_true=_spearman([-(r["self_score_pilot"] or np.nan) for r in table], [r["true_rank"] for r in table]),
        candidates=table)
    dest = out / "benchmark1"
    write_json(dest / "summary.json", summary)
    for p in pool:
        src = private / "true_scores" / f"{p}_per_gene.parquet"
        if src.exists():
            pd.read_parquet(src).to_parquet(dest / f"{p}_per_gene.parquet")
    pd.read_parquet(private / "rulers_per_gene.parquet").to_parquet(dest / "rulers_per_gene.parquet")
    _print(summary, "true_score")
    return summary


def export_t(task="task_t"):
    out, pool, ranking, final, rows = _agent_side(task)
    private = paths.private_dir(task)
    table, true = [], {}
    reference = _maybe(private / "true_scores" / "ruler_simple_reference" / "true_score_main.json")
    for p in pool:
        ts = _maybe(private / "true_scores" / p / "true_score_main.json")
        site = _maybe(private / "true_scores" / p / "true_score_site.json")
        true[p] = ts["true_gain"] if ts else None
        pilot, full = rows[p]["pilot"] or {}, rows[p]["full"] or {}
        table.append(dict(
            plan_id=p, name=rows[p]["name"], complexity=rows[p]["complexity"], component=rows[p]["component"],
            tournament_rank=rows[p]["tournament_rank"], bt_strength=rows[p]["bt_strength"],
            self_gain_pilot=pilot.get("self_gain"), self_ci_pilot=pilot.get("self_ci"),
            self_gain_full=full.get("self_gain"), self_ci_full=full.get("self_ci"), self_conclusion=full.get("conclusion"),
            true_gain=true[p], true_ci=[ts["ci_low"], ts["ci_high"]] if ts else None, true_conclusion=ts and ts["conclusion"],
            macro_f1_plan=ts and ts["macro_f1_plan"], macro_f1_floor=ts and ts["macro_f1_floor"],
            inflation_full=(full.get("self_gain") - true[p]) if ts and full.get("self_gain") is not None else None,
            inflation_pilot=(pilot.get("self_gain") - true[p]) if ts and pilot.get("self_gain") is not None else None,
            recall_change=ts and ts["recall_change"], auroc_gain=ts and ts["auroc_gain"], neg_log_loss_gain=ts and ts["neg_log_loss_gain"],
            site_held_out=site and dict(gain=site["true_gain"], ci=[site["ci_low"], site["ci_high"]]),
            selection=(full.get("validation") or pilot.get("validation") or {}).get("selection"),
            converged_folds=ts and f"{ts['converged_folds']}/{ts['n_folds']}", cpu_core_hours=ts and round(ts["cpu_core_seconds"] / 3600, 2),
            audit_passed=(rows[p]["audit"] or {}).get("passed")))
    true_rank = _rank(true)
    for row in table:
        row["true_rank"] = true_rank.get(row["plan_id"])
    selected = final.get("accepted_plan") or ranking["champion"]
    scored = [p for p in pool if true[p] is not None]
    best = max(scored, key=lambda p: true[p])
    sel_row = next(r for r in table if r["plan_id"] == selected)
    summary = dict(
        task=task, exported=now(), primary_score="paired macro-F1 gain over the evaluator's tuned clinical-only model, repeated nested CV",
        clinical_floor_macro_f1=sel_row["macro_f1_floor"],
        simple_reference=reference and dict(gain=reference["true_gain"], ci=[reference["ci_low"], reference["ci_high"]],
                                            macro_f1=reference["macro_f1_plan"], conclusion=reference["conclusion"]),
        tournament_champion=ranking["champion"], selected_plan=selected, best_true_plan=best,
        selected_true_rank=true_rank.get(selected), n_candidates_scored=len(scored),
        selection_gap=true[best] - true[selected] if true.get(selected) is not None else None,
        n_candidates_positive_gain=sum(1 for p in scored if true[p] > 0),
        n_candidates_ci_above_zero=sum(1 for r in table if r["true_ci"] and r["true_ci"][0] > 0),
        n_candidates_beating_simple_reference=reference and sum(1 for p in scored if true[p] > reference["true_gain"]),
        conclusion_agreement=dict(agent=sel_row["self_conclusion"], evaluator=sel_row["true_conclusion"],
                                  agree=sel_row["self_conclusion"] == sel_row["true_conclusion"]),
        spearman_tournament_vs_true=_spearman([r["tournament_rank"] for r in table], [r["true_rank"] for r in table]),
        spearman_pilot_gain_vs_true=_spearman([-(r["self_gain_pilot"] if r["self_gain_pilot"] is not None else np.nan) for r in table],
                                              [r["true_rank"] for r in table]),
        candidates=table)
    write_json(out / "benchmark1" / "summary.json", summary)
    _print(summary, "true_gain")
    return summary


def _spearman(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    ra, rb = pd.Series(a[ok]).rank().values, pd.Series(b[ok]).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def _print(summary, key):
    print(f"selected {summary['selected_plan']} has true rank {summary['selected_true_rank']} of {summary['n_candidates_scored']}; "
          f"best by true score {summary['best_true_plan']}; selection gap {summary['selection_gap']}")
    for r in sorted(summary["candidates"], key=lambda r: r["true_rank"] or 99):
        print(f"  {r['plan_id']} true_rank={r['true_rank']} tournament_rank={r['tournament_rank']} {key}={r[key]} "
              f"inflation_full={r['inflation_full']} :: {r['name']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    a = ap.parse_args(argv)
    family = read_json(ROOT / "outputs" / a.task / "shared_inputs" / "objective.json")["task_family"]
    {"cross_modal_prediction": export_m, "incremental_value_classification": export_t}[family](a.task)


if __name__ == "__main__":
    sys.exit(main())
