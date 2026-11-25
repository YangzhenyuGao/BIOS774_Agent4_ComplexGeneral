"""Step 9 of README 6.1: the per-dataset report.

    python -m agent4.report facts   --task task_m     gather every number and draw the figures (data job)
    python -m agent4.report write   --task task_m     generator writes the prose from named facts (controller)
    python -m agent4.report render  --task task_m --out reports/generated_report_1.pdf

Prose comes from the generator; numbers never do. The generator may refer to a number only through a
named placeholder, and the text is rejected if it contains a digit of its own. All placeholders are filled
from files written by earlier stages.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys

import numpy as np

from .io import ROOT, now, read_json, write_json
from .llm import client

FAMILY_M, FAMILY_T = "cross_modal_prediction", "incremental_value_classification"


def _maybe(path):
    return read_json(path) if path.exists() else None


def _fmt(x, digits=3, signed=False):
    if x is None:
        return "n/a"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


# ----------------------------------------------------------------------------- gather
def collect(task):
    d = ROOT / "outputs" / task
    shared = d / "shared_inputs"
    objective, profile = read_json(shared / "objective.json"), read_json(shared / "profile.json")
    pool, ranking = read_json(d / "plans" / "pool.json"), read_json(d / "tournament" / "ranking.json")
    final, bench = _maybe(d / "audit" / "final.json") or {}, read_json(d / "benchmark1" / "summary.json")
    plans = {p: read_json(d / "plans" / f"{p}.json") for p in pool["plan_ids"]}
    compliance = {c["plan_id"]: c for c in read_json(d / "plans" / "compliance.json")}
    cards = {p: _maybe(d / "pilot" / p / "evidence_card.json") for p in plans}
    pilots = {p: _maybe(d / "pilot" / p / "self_validation.json") for p in plans}
    fulls = {p: _maybe(d / "full" / p / "full_run.json") for p in plans}
    audits = {p: _maybe(d / "audit" / f"{p}.json") for p in plans}
    selected = final.get("accepted_plan") or ranking["champion"]
    alt = None
    if (d / "alt_question" / "plans" / "pool.json").exists():
        ad = d / "alt_question"
        alt = dict(question=(ad / "shared_inputs" / "question.txt").read_text().strip(), objective=read_json(ad / "shared_inputs" / "objective.json"),
                   plans=[read_json(ad / "plans" / f"{p}.json") for p in read_json(ad / "plans" / "pool.json")["plan_ids"]])
    matches = sorted((d / "tournament" / "matches").glob("*.json"))
    example = None
    for m in matches:
        mm = read_json(m)
        if selected in (mm["first"], mm["second"]) and mm["winner"] and (example is None or len(mm["reason"]) > len(example["reason"])):
            example = {k: mm[k] for k in ("first", "second", "winner", "cited_evidence", "reason")}
    return dict(task=task, family=objective["task_family"], question=(shared / "question.txt").read_text().strip(), objective=objective,
                profile=profile, pool=pool, plans=plans, compliance=compliance, cards=cards, pilots=pilots, fulls=fulls, audits=audits,
                ranking=ranking, final=final, bench=bench, selected=selected, alt=alt, example_verdict=example,
                costs=_maybe(d / "costs.json"), dropped=_maybe(d / "plans" / "dropped.json") or [], collected=now())


def named_facts(c):
    """Flat {key: (formatted value, meaning)} table the generator may cite."""
    b, sel = c["bench"], c["selected"]
    row = next(r for r in b["candidates"] if r["plan_id"] == sel)
    rk = c["ranking"]
    f = {
        "n_candidates": (len(b["candidates"]), "number of candidate plans in the frozen pool"),
        "n_ambitious": (sum(r["complexity"] == "ambitious" for r in b["candidates"]), "candidates classed as ambitious"),
        "selected_plan": (sel, "id of the plan the agent selected and reported"),
        "selected_name": (row["name"], "name of the selected plan"),
        "tournament_champion": (rk["champion"], "plan ranked first by the tournament"),
        "order_consistency": (f"{rk['order_consistency']:.0%}", "share of plan pairs judged the same way in both presentation orders"),
        "n_matches": (rk["n_matches"], "number of judged matches"),
        "selected_true_rank": (b["selected_true_rank"], "rank of the selected plan among all candidates under the evaluator's unified score (1 = best)"),
        "best_true_plan": (b["best_true_plan"], "plan with the best unified score"),
        "selection_gap": (_fmt(b["selection_gap"], 4), "unified score of the best candidate minus that of the selected plan"),
        "audit_outcome": (c["final"].get("outcome", "n/a"), "outcome of the audit"),
    }
    if c["family"] == FAMILY_M:
        r = b["rulers"]
        infl = [x["inflation_full"] for x in b["candidates"] if x["inflation_full"] is not None]
        f.update({
            "n_genes_in_scope": (b["n_scope"], "genes in scope"), "n_genes_judgeable": (b["n_judgeable"], "in-scope genes whose ceiling is above the permutation null on held-out cells"),
            "ceiling_r": (_fmt(r["ceiling"]), "mean per-gene r of the RNA-to-RNA ceiling"),
            "cluster_mean_r": (_fmt(r["floor1_cluster_mean"]), "mean per-gene r of the cell-type (cluster-mean) baseline"),
            "ridge_r": (_fmt(r["floor4_ridge"]), "mean per-gene r of ridge regression on the ATAC latent space"),
            "gene_activity_r": (_fmt(r["floor2_gene_activity_promoter_and_distal"]), "mean per-gene r of the gene-activity score"),
            "depth_r": (_fmt(r["floor3_depth_only"]), "mean per-gene r from ATAC depth alone"),
            "selected_true_r": (_fmt(row["true_score"]), "unified score of the selected plan: mean per-gene r on held-out cells"),
            "selected_self_r": (_fmt(row["self_score_full"]), "score the selected plan reported for itself in its full run"),
            "selected_inflation": (_fmt(row["inflation_full"], 3, True), "self-reported minus unified score of the selected plan"),
            "mean_inflation": (_fmt(float(np.mean(infl)), 3, True) if infl else "n/a", "mean of self-reported minus unified score over candidates"),
            "selected_fraction_of_ceiling": (f"{row['fraction_of_ceiling']:.0%}", "unified score of the selected plan as a share of the ceiling"),
            "selected_delta_vs_cluster": (_fmt(row["mean_delta_vs_cluster_mean"], 4, True), "mean per-gene r of the selected plan minus the cluster-mean baseline on held-out cells"),
            "n_beyond": (row["n_beyond_cell_type"], "genes the selected plan predicts beyond cell type on held-out cells (FDR 5%)"),
            "n_not_beyond": (row["n_not_beyond"], "judgeable genes not beyond the cell-type baseline"),
            "n_cannot_judge": (row["n_cannot_judge"], "genes that cannot be judged"),
            "share_beyond": (f"{row['n_beyond_cell_type'] / max(b['n_judgeable'], 1):.1%}", "genes predicted beyond cell type as a share of judgeable genes"),
            "n_claimed": (row["n_claimed_by_plan"], "genes the plan itself claimed from development cells"),
            "replication_rate": (f"{row['gene_list_check']['replication_rate']:.0%}" if row.get("gene_list_check") and row["gene_list_check"]["replication_rate"] is not None else "n/a",
                                 "share of the plan's claimed genes confirmed on held-out cells"),
            "within_cluster_r": (_fmt(row["mean_r_within_cluster"]), "mean per-gene r of the selected plan inside clusters"),
            "max_beyond_any_plan": (max(x["n_beyond_cell_type"] or 0 for x in b["candidates"]), "largest number of genes beyond cell type achieved by any candidate"),
        })
    else:
        ref = b.get("simple_reference") or {}
        infl = [x["inflation_full"] for x in b["candidates"] if x["inflation_full"] is not None]
        f.update({
            "n_patients": (c["profile"]["summary"]["n_labelled"], "labelled patients"),
            "clinical_f1": (_fmt(b["clinical_floor_macro_f1"]), "macro-F1 of the evaluator's tuned clinical-only model"),
            "selected_f1": (_fmt(row["macro_f1_plan"]), "macro-F1 of the selected plan under the unified protocol"),
            "true_gain": (_fmt(row["true_gain"], 3, True), "unified gain of the selected plan over clinical only"),
            "true_ci_low": (_fmt(row["true_ci"][0], 3, True), "lower bound of the 95% interval of the unified gain"),
            "true_ci_high": (_fmt(row["true_ci"][1], 3, True), "upper bound of the 95% interval of the unified gain"),
            "true_conclusion": (row["true_conclusion"].replace("_", " "), "evaluator's three-way conclusion for the selected plan"),
            "self_gain": (_fmt(row["self_gain_full"], 3, True), "gain the selected plan reported for itself"),
            "self_ci_low": (_fmt((row["self_ci_full"] or [None, None])[0], 3, True), "lower bound of the plan's own interval"),
            "self_ci_high": (_fmt((row["self_ci_full"] or [None, None])[1], 3, True), "upper bound of the plan's own interval"),
            "self_conclusion": (str(row["self_conclusion"]).replace("_", " "), "conclusion the agent drew from the plan's own validation"),
            "conclusions_agree": ("yes" if b["conclusion_agreement"]["agree"] else "no", "whether the agent's and the evaluator's conclusions agree"),
            "selected_inflation": (_fmt(row["inflation_full"], 3, True), "self-reported minus unified gain of the selected plan"),
            "mean_inflation": (_fmt(float(np.mean(infl)), 3, True) if infl else "n/a", "mean of self-reported minus unified gain over candidates"),
            "n_positive": (b["n_candidates_positive_gain"], "candidates with a positive unified gain"),
            "n_ci_above_zero": (b["n_candidates_ci_above_zero"], "candidates whose unified 95% interval lies above zero"),
            "reference_gain": (_fmt(ref.get("gain"), 3, True), "unified gain of the simple reference (clinical plus mean-pooled image features, logistic regression)"),
            "n_beating_reference": (b.get("n_candidates_beating_simple_reference"), "candidates with a larger unified gain than the simple reference"),
        })
        for cls, v in (row.get("recall_change") or {}).items():
            f[f"recall_change_{cls}"] = (_fmt(v, 3, True), f"change in recall for {cls} with images, unified protocol")
        if row.get("site_held_out"):
            f["site_gain"] = (_fmt(row["site_held_out"]["gain"], 3, True), "gain of the selected plan when whole collection sites are held out")
    return {k: (str(v[0]), v[1]) for k, v in f.items()}


# ----------------------------------------------------------------------------- figures
def figures(c, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    b, rk, sel = c["bench"], c["ranking"], c["selected"]
    cand = sorted(b["candidates"], key=lambda r: r["plan_id"])
    ids = [r["plan_id"] for r in cand]
    made = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=170)
        plt.close(fig)
        made.append(name)

    # tournament: who beat whom, and the resulting strengths
    rows = rk["wins_matrix"]["rows"]
    w = np.array(rk["wins_matrix"]["values"])
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.3), gridspec_kw=dict(width_ratios=[1.1, 1]))
    im = ax[0].imshow(w, cmap="Blues", vmin=0, vmax=2)
    ax[0].set_xticks(range(len(rows)), rows), ax[0].set_yticks(range(len(rows)), rows)
    for i in range(len(rows)):
        for j in range(len(rows)):
            if i != j:
                ax[0].text(j, i, f"{w[i, j]:g}", ha="center", va="center", fontsize=8, color="white" if w[i, j] > 1.2 else "black")
    ax[0].set_title("Wins of row plan over column plan\n(two matches per pair, one per order)", fontsize=9)
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    order = rk["ranking"]
    ax[1].barh(order[::-1], [rk["bradley_terry_strength"][p] for p in order[::-1]], color=["#c0392b" if p == sel else "#7f8c8d" for p in order[::-1]])
    ax[1].set_xlabel("Bradley-Terry strength"), ax[1].set_title("Tournament ranking (selected plan in red)", fontsize=9)
    save(fig, "tournament.png")

    key_true, key_self_p, key_self_f, ylabel = (("true_score", "self_score_pilot", "self_score_full", "mean per-gene r") if c["family"] == FAMILY_M
                                                else ("true_gain", "self_gain_pilot", "self_gain_full", "macro-F1 gain over clinical only"))
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    x = np.arange(len(ids))
    for off, key, label, col in ((-0.26, key_self_p, "self-reported, pilot", "#f5b041"), (0.0, key_self_f, "self-reported, full run", "#e67e22"),
                                 (0.26, key_true, "unified score (evaluator)", "#2e86c1")):
        vals = [np.nan if r[key] is None else r[key] for r in cand]
        ax.bar(x + off, vals, 0.25, label=label, color=col)
    if c["family"] == FAMILY_T:
        for i, r in enumerate(cand):
            if r["true_ci"]:
                ax.plot([i + 0.26] * 2, r["true_ci"], color="black", lw=1)
        ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, [f"{p}\n(T{r['tournament_rank']}, U{r['true_rank']})" for p, r in zip(ids, cand)], fontsize=8)
    ax.set_ylabel(ylabel), ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    ax.set_xlabel("candidate (T = tournament rank, U = rank by unified score)", fontsize=8)
    save(fig, "self_vs_true.png")

    if c["family"] == FAMILY_M:
        r = b["rulers"]
        names = ["depth only", "gene activity", "k-NN", "cluster mean", "ridge", "PLS"] + ids + ["ceiling"]
        vals = [r["floor3_depth_only"], r["floor2_gene_activity_promoter_and_distal"], r["floor4_knn"], r["floor1_cluster_mean"], r["floor4_ridge"],
                r["floor4_pls"]] + [x["true_score"] or np.nan for x in cand] + [r["ceiling"]]
        cols = ["#bdc3c7"] * 6 + ["#c0392b" if p == sel else "#2e86c1" for p in ids] + ["#27ae60"]
        fig, ax = plt.subplots(figsize=(7.6, 3.0))
        ax.bar(names, vals, color=cols)
        ax.axhline(r["floor1_cluster_mean"], color="black", lw=0.7, ls="--")
        ax.set_ylabel("mean per-gene r on held-out cells"), ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        ax.set_title("Rulers (grey), candidates (blue; selected in red) and ceiling (green)", fontsize=9)
        save(fig, "rulers.png")
        import pandas as pd
        pg_path = ROOT / "outputs" / c["task"] / "benchmark1" / f"{sel}_per_gene.parquet"
        if pg_path.exists():
            pg = pd.read_parquet(pg_path)
            fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.2))
            ax[0].hexbin(pg.r_cluster_mean, pg.r, gridsize=55, bins="log", cmap="viridis", mincnt=1)
            lim = [min(pg.r_cluster_mean.min(), pg.r.min()), max(pg.r_cluster_mean.max(), pg.r.max())]
            ax[0].plot(lim, lim, color="red", lw=0.8)
            ax[0].set_xlabel("r of cluster-mean baseline"), ax[0].set_ylabel(f"r of {sel}"), ax[0].set_title("Per gene, held-out cells", fontsize=9)
            ax[1].hist(pg.delta_vs_cluster.dropna(), bins=80, color="#2e86c1")
            ax[1].axvline(0, color="black", lw=0.8)
            ax[1].set_xlabel("r(plan) - r(cluster mean)"), ax[1].set_ylabel("genes"), ax[1].set_title("Gain over the cell-type baseline", fontsize=9)
            counts = [int((pg.label == k).sum()) for k in (0, 1, 2)]
            ax[2].bar(["cannot judge", "not beyond\ncell type", "beyond\ncell type"], counts, color=["#bdc3c7", "#7f8c8d", "#c0392b"])
            for i, v in enumerate(counts):
                ax[2].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
            ax[2].set_ylabel("genes"), ax[2].set_title("Three-way classification (FDR 5%)", fontsize=9)
            save(fig, "genes.png")
        fig, ax = plt.subplots(figsize=(7.6, 2.8))
        bottom = np.zeros(len(ids))
        for key, label, col in (("n_cannot_judge", "cannot judge", "#bdc3c7"), ("n_not_beyond", "not beyond cell type", "#7f8c8d"),
                                ("n_beyond_cell_type", "beyond cell type", "#c0392b")):
            vals = np.array([x[key] or 0 for x in cand], dtype=float)
            ax.bar(ids, vals, bottom=bottom, label=label, color=col)
            bottom += vals
        ax.set_ylabel("genes"), ax.legend(fontsize=7, ncol=3), ax.set_title("Gene classification for every candidate (unified protocol)", fontsize=9)
        save(fig, "classes.png")
    else:
        ref = b.get("simple_reference")
        order = sorted(cand, key=lambda r: r["true_gain"] if r["true_gain"] is not None else -9)
        fig, ax = plt.subplots(figsize=(7.6, 3.4))
        labels = []
        for i, r in enumerate(order):
            col = "#c0392b" if r["plan_id"] == sel else "#2e86c1"
            if r["true_ci"]:
                ax.plot(r["true_ci"], [i, i], color=col, lw=2)
            ax.plot(r["true_gain"], i, "o", color=col)
            if r["self_gain_full"] is not None:
                ax.plot(r["self_gain_full"], i, "x", color="#e67e22")
            labels.append(f"{r['plan_id']} {r['name'][:38]}")
        if ref:
            ax.plot(ref["ci"], [len(order)] * 2, color="#7f8c8d", lw=2)
            ax.plot(ref["gain"], len(order), "s", color="#7f8c8d")
            labels.append("simple reference (ruler)")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xlabel("macro-F1 gain over the tuned clinical-only model (95% bootstrap interval)")
        ax.set_title("Unified gains of all candidates (dots; selected in red); x = the plan's own full-run estimate", fontsize=8.5)
        save(fig, "forest.png")
        row = next(r for r in cand if r["plan_id"] == sel)
        if row.get("recall_change"):
            fig, ax = plt.subplots(figsize=(4.6, 2.8))
            cls = list(row["recall_change"])
            ax.bar(cls, [row["recall_change"][k] for k in cls], color=["#c0392b" if v < 0 else "#27ae60" for v in row["recall_change"].values()])
            ax.axhline(0, color="black", lw=0.8)
            ax.set_ylabel("change in recall with images"), ax.set_title(f"{sel}: per-class recall change (unified)", fontsize=9)
            save(fig, "recall.png")

    full = c["fulls"].get(sel) or {}
    curves = full.get("final_fit_curves") or (full.get("training") or {}).get("example_curve")
    if curves:
        fig, ax = plt.subplots(figsize=(4.8, 2.8))
        for k, v in curves.items():
            if isinstance(v, list) and len(v) > 1:
                ax.plot(v, label=k)
        ax.set_xlabel("epoch"), ax.legend(fontsize=7), ax.set_title(f"{sel}: training curves", fontsize=9)
        save(fig, "curves.png")
    return made


# ----------------------------------------------------------------------------- prose
SYSTEM = """You write four sections of an analysis report for a biomedical data-analysis agent: Conclusion, Evidence, Limits, Recommendations.
You receive the scientific question, the compiled objective, what was done, and a table of named facts.

Hard rules
- Every number you mention must be written as a placeholder {{fact_key}} using a key from the table. Never type a digit yourself (plan ids such as P3 and the terms PAM50, H&E are allowed).
- Say only what the facts support. The conclusion must be no stronger than the evaluator's unified result allows, must answer the question that was asked, and must stay inside the stated scope.
- Report unfavourable facts plainly (a selected plan that is not the best, a self-reported score above the unified one, an inconclusive interval, few genes). Do not soften or pad.
- Plain, direct English. No bullet inside the Conclusion. No headings inside your text.

Reply with one JSON object and nothing else:
{"conclusion": [two or three paragraphs], "evidence": [three to five paragraphs], "limits": [four to seven sentences, one per item], "recommendations": [three to five sentences, one per item]}"""


def _check_text(parts, facts):
    text = " ".join(x for v in parts.values() for x in v)
    unknown = sorted(set(re.findall(r"\{\{(\w+)\}\}", text)) - set(facts))
    if unknown:
        raise ValueError(f"unknown placeholders: {unknown}")
    stripped = re.sub(r"\{\{\w+\}\}", "", text)
    stripped = re.sub(r"\bP\d+\b|PAM50|H&E|\bE\d+\b", "", stripped)
    digits = re.findall(r"\S*\d\S*", stripped)
    if digits:
        raise ValueError(f"digits typed outside placeholders: {digits[:8]}")


def write_prose(c, facts, tag):
    sel = c["selected"]
    did = dict(pool=[dict(id=p, name=pl["name"], complexity=pl["complexity"], component=pl["model"].get("component")) for p, pl in c["plans"].items()],
               tournament_ranking=c["ranking"]["ranking"], selected=sel, selected_summary=c["plans"][sel]["summary"],
               audit_trail=c["final"].get("trail"), wording_limits=((c["audits"].get(sel) or {}).get("review") or {}).get("wording_limits"),
               difficulties=[f"{d['id']} {d['tag']}: {d['evidence']}" for d in c["profile"]["difficulties"]])
    prompt = ("SCIENTIFIC QUESTION:\n" + c["question"] + "\n\nOBJECTIVE:\n" +
              json.dumps({k: c["objective"][k] for k in ("estimand", "forbidden_inputs", "conclusion_options", "conclusion_scope")}, indent=1) +
              "\n\nWHAT WAS DONE:\n" + json.dumps(did, indent=1) + "\n\nNAMED FACTS (key = value : meaning):\n" +
              "\n".join(f"{k} = {v[0]} : {v[1]}" for k, v in facts.items()))
    error = None
    for attempt in range(2):
        rec = client.call("generator", SYSTEM, prompt if attempt == 0 else prompt + f"\n\nYOUR PREVIOUS ANSWER BROKE A RULE: {error}\nRewrite it.",
                          tag=f"{tag}-report{'' if attempt == 0 else '-repair'}")
        try:
            parts = {k: list(rec["json"][k]) for k in ("conclusion", "evidence", "limits", "recommendations")}
            _check_text(parts, facts)
            return parts, dict(source="generator", model=rec["model"], cache_key=rec["key"], repair_attempts=attempt)
        except (KeyError, TypeError, ValueError) as exc:
            error = str(exc)[:500]
    return None, dict(source="template_fallback", error=error)


def fallback_prose(c, facts):
    """Rule-based text used only if the generator's prose breaks the number rules twice."""
    if c["family"] == FAMILY_M:
        concl = ["The selected plan {{selected_plan}} reaches a mean per-gene correlation of {{selected_true_r}} on held-out cells against {{cluster_mean_r}} "
                 "for the cell-type baseline and {{ceiling_r}} for the ceiling. It predicts {{n_beyond}} of {{n_genes_judgeable}} judgeable genes beyond cell type."]
    else:
        concl = ["Under the unified protocol the selected plan {{selected_plan}} changes macro-F1 by {{true_gain}} over clinical variables alone "
                 "(interval {{true_ci_low}} to {{true_ci_high}}); the conclusion is: {{true_conclusion}}."]
    return dict(conclusion=concl,
                evidence=["The selected plan ranks {{selected_true_rank}} of {{n_candidates}} candidates by the unified score; the gap to the best candidate is {{selection_gap}}.",
                          "Its self-reported score differs from the unified score by {{selected_inflation}}."],
                limits=["The conclusion applies only within the scope stated in the objective."],
                recommendations=["Repeat the comparison of selection methods on the preserved plan pool and evidence cards."])


# ----------------------------------------------------------------------------- render
CSS = """
@page { size: Letter; margin: 16mm 15mm 16mm 15mm; @bottom-center { content: counter(page); font-size: 8pt; color: #666; } }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 9.2pt; line-height: 1.38; color: #1c1c1c; }
h1 { font-size: 16pt; margin: 0 0 4pt 0; } h2 { font-size: 11.5pt; margin: 13pt 0 4pt 0; border-bottom: 0.6pt solid #999; padding-bottom: 2pt; }
h3 { font-size: 9.8pt; margin: 9pt 0 3pt 0; } p { margin: 0 0 5pt 0; } .q { background: #f1f4f7; padding: 6pt 8pt; border-left: 3pt solid #2e86c1; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 8pt 0; font-size: 7.6pt; } th, td { border: 0.4pt solid #bbb; padding: 2.2pt 3.5pt; vertical-align: top; text-align: left; }
th { background: #eef1f4; } .small { font-size: 7.6pt; color: #444; } img { max-width: 100%; } .fig { text-align: center; margin: 5pt 0 2pt 0; }
.cap { font-size: 7.8pt; color: #333; margin-bottom: 7pt; } .sel { background: #fdecea; } ul { margin: 2pt 0 6pt 14pt; padding: 0; } li { margin-bottom: 2pt; }
"""


def _table(headers, rows, selected_col=None, selected=None):
    out = ["<table><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr>"]
    for r in rows:
        cls = " class='sel'" if selected is not None and r[selected_col] == selected else ""
        out.append(f"<tr{cls}>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in r) + "</tr>")
    return "".join(out) + "</table>"


def _unfinished(fulls):
    """One phrase per plan whose own full run did not finish what it declared, read from the run records."""
    out = []
    for p, fr in fulls.items():
        if not fr:
            continue
        parts = []
        declared, ran = fr.get("n_splits_declared", fr.get("n_folds_declared")), fr.get("n_splits_run", fr.get("n_folds_run"))
        if declared is not None and ran is not None and ran < declared:
            parts.append(f"{ran} of {declared} declared validation splits ran")
        fit = fr.get("final_fit") or {}
        if fit.get("converged") is False:
            parts.append(f"the final fit did not converge (stopped by: {str(fit.get('stopped_by')).replace('_', ' ')})")
        if parts:
            out.append(f"{p}: " + ", ".join(parts))
    return out


def _fill(text, facts):
    return html.escape(re.sub(r"\{\{(\w+)\}\}", lambda m: facts[m.group(1)][0], text))


def render(c, facts, prose, prose_meta, fig_dir, out_pdf, title):
    from weasyprint import HTML
    b, sel, fam = c["bench"], c["selected"], c["family"]
    cand = sorted(b["candidates"], key=lambda r: r["plan_id"])
    shown = []

    def fig(name, cap):
        """A figure with its caption, numbered in order of appearance; nothing if the figure was not produced."""
        if not (fig_dir / name).exists():
            return ""
        shown.append(name)
        return f"<div class='fig'><img src='file://{fig_dir / name}'></div><div class='cap'>Figure {len(shown)}. {html.escape(cap)}</div>"
    h = [f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body><h1>{html.escape(title)}</h1>",
         f"<p class='small'>Generated by Agent 4 on {c['collected'][:10]}. Prose by the plan-generator model from named facts; every number is filled in by code from saved results "
         f"({'generator text' if prose_meta['source'] == 'generator' else 'rule-based fallback text'}).</p>",
         f"<p class='q'><b>Question.</b> {html.escape(c['question'])}</p>", "<h2>1. Conclusion</h2>"]
    h += [f"<p>{_fill(p, facts)}</p>" for p in prose["conclusion"]]
    h.append("<h2>2. Evidence</h2>")
    h += [f"<p>{_fill(p, facts)}</p>" for p in prose["evidence"]]
    if fam == FAMILY_M:
        h.append(fig("rulers.png", "Mean per-gene correlation on held-out cells for the rulers, every candidate plan and the ceiling. The dashed line is the cell-type baseline."))
        h.append(fig("genes.png", "Selected plan on held-out cells: per-gene correlation against the cell-type baseline, the distribution of the gain, and the three-way classification of in-scope genes."))
    else:
        h.append(fig("forest.png", "Gain in macro-F1 over the evaluator's tuned clinical-only model for every candidate, with 95% patient-level bootstrap intervals (repeated nested cross-validation)."))
        h.append(fig("recall.png", "Change in per-class recall when images are added, selected plan, unified protocol."))
    h.append("<h2>3. Limits</h2><ul>" + "".join(f"<li>{_fill(x, facts)}</li>" for x in prose["limits"]) + "</ul>")
    h.append("<h2>4. Recommendations</h2><ul>" + "".join(f"<li>{_fill(x, facts)}</li>" for x in prose["recommendations"]) + "</ul>")

    h.append("<h2>5. How the agent got here</h2><h3>5.1 Measured difficulties of the data</h3>")
    h.append(_table(["id", "difficulty", "severity", "evidence", "what an analysis must do"],
                    [[d["id"], d["tag"].replace("_", " "), d["severity"], d["evidence"], d["implication"]] for d in c["profile"]["difficulties"]]))
    o = c["objective"]
    h.append("<h3>5.2 Compiled objective</h3>" + _table(["field", "content"], [
        ["task family", o["task_family"]], ["analysis unit", o["analysis_unit"]], ["estimand", o["estimand"]],
        ["allowed inputs", "; ".join(o["allowed_inputs"])], ["forbidden inputs", "; ".join(o["forbidden_inputs"])],
        ["required comparators", "; ".join(o["required_comparators"])], ["primary metric", o["primary_metric"]], ["scope of the conclusion", o["conclusion_scope"]]]))
    h.append("<p class='small'>Acceptance criteria: " + html.escape("; ".join(f"{a['id']} {a['check']}" for a in o["acceptance_criteria"])) + ".</p>")
    h.append("<h3>5.3 Candidate plans (frozen pool)</h3>")
    h.append(_table(["id", "name", "kind", "core component", "data", "model", "method", "compliant"],
                    [[p, pl["name"], pl["complexity"].replace("_", " "), pl["model"].get("component"), *(str(pl["diversity"].get(k) or "").replace("_", " ") for k in ("data", "model", "method")),
                      "yes" if c["compliance"][p]["passed"] else "no"] for p, pl in c["plans"].items()], 0, sel))
    rep = c["pool"]["selection_report"]
    h.append(f"<p class='small'>{rep['n_candidates']} candidates were generated, {rep['n_after_dedup']} remained after removing near-duplicates, "
             f"{rep['n_pool']} were kept ({rep['n_ambitious']} ambitious, {rep['n_simple']} simple reference). The selected plan is shaded.</p>")
    h.append("<h3>5.4 Pilot evidence and tournament</h3>")
    if fam == FAMILY_M:
        rows = [[p, _fmt(s["self_score_mean_r"]), _fmt(s["rulers_mean_r"]["cluster_mean"]), _fmt(s["margin_over_cluster_mean"], 4, True), s["validation"]["target"].replace("_", " "),
                 "yes" if s["all_converged"] else "no", f"{s['fit_seconds']:.0f}"] for p, s in c["pilots"].items() if s]
        h.append(_table(["id", "self-reported r (pilot)", "cluster-mean r, same cells", "margin", "scoring target", "converged", "fit seconds"], rows, 0, sel))
    else:
        rows = [[p, _fmt(s["self_gain"], 3, True), f"{_fmt(s['self_ci'][0], 3, True)} to {_fmt(s['self_ci'][1], 3, True)}", _fmt(s["gain_over_simple_reference"], 3, True),
                 s["validation"]["selection"].replace("_", " "), f"{s['n_configs_image_arm']} / {s['n_configs_clinical_arm']}",
                 "n/a" if s["training"]["share_converged"] is None else f"{s['training']['share_converged']:.0%}", f"{s['plan_seconds']:.0f}"]
                for p, s in c["pilots"].items() if s]
        h.append(_table(["id", "self-reported gain (pilot)", "95% interval", "vs simple reference", "configuration choice", "configs image / clinical", "fits converged", "seconds"], rows, 0, sel))
    rk = c["ranking"]
    h.append(fig("tournament.png", f"Round-robin tournament: {rk['n_matches']} matches judged by {', '.join(rk['judge_models'])}, each pair in both presentation orders. "
                                   f"Pairs judged identically in both orders: {rk['order_consistency']:.0%}."))
    if c["example_verdict"]:
        ev = c["example_verdict"]
        h.append(f"<p class='small'><b>One verdict as recorded</b> ({ev['first']} presented first, {ev['second']} second; winner {ev['winner']}; cites {', '.join(ev['cited_evidence'][:8])}): "
                 f"{html.escape(ev['reason'])}</p>")
    h.append("<h3>5.5 Full run and audit of the selected plan</h3>")
    au = c["audits"].get(sel)
    if au:
        short = lambda text: re.sub(r"-?\d+\.\d{5,}", lambda m: f"{float(m.group()):.4f}", str(text))
        h.append(_table(["criterion", "check", "result", "detail"], [[x["id"], x["check"], "passed" if x["passed"] else "FAILED", short(x["detail"])] for x in au["checks"]]))
        h.append(f"<p class='small'>Independent review: {html.escape(au['review']['decision'])}. {html.escape(au['review'].get('reason', ''))}</p>")
    trail = c["final"].get("trail") or []
    if len(trail) > 1:
        h.append("<p class='small'>Audit trail: " + html.escape("; ".join(f"{t['plan_id']} (tournament rank {t['rank']}) {'passed' if t['passed'] else 'rejected'}" for t in trail)) + ".</p>")
    h.append(fig("curves.png", "Training curves of the selected plan."))

    h.append("<h2>6. Benchmark 1: every candidate run in full under the unified protocol</h2>")
    if fam == FAMILY_M:
        rows = [[r["plan_id"], r["tournament_rank"], r["true_rank"], _fmt(r["self_score_pilot"]), _fmt(r["self_score_full"]), _fmt(r["true_score"]),
                 _fmt(r["inflation_full"], 3, True), "n/a" if r["fraction_of_ceiling"] is None else f"{r['fraction_of_ceiling']:.0%}", _fmt(r["n_beyond_cell_type"]),
                 _fmt(r["n_claimed_by_plan"]), "n/a" if not r.get("gene_list_check") or r["gene_list_check"]["replication_rate"] is None else f"{r['gene_list_check']['replication_rate']:.0%}"]
                for r in sorted(cand, key=lambda r: r["true_rank"] or 99)]
        h.append(_table(["id", "tournament rank", "unified rank", "self r (pilot)", "self r (full)", "unified r", "self minus unified", "share of ceiling",
                         "genes beyond cell type", "genes claimed", "claims confirmed"], rows, 0, sel))
        h.append(fig("classes.png", "Three-way gene classification for every candidate under the unified protocol."))
    else:
        rows = [[r["plan_id"], r["tournament_rank"], r["true_rank"], _fmt(r["self_gain_pilot"], 3, True), _fmt(r["self_gain_full"], 3, True), _fmt(r["true_gain"], 3, True),
                 "n/a" if not r["true_ci"] else f"{_fmt(r['true_ci'][0], 3, True)} to {_fmt(r['true_ci'][1], 3, True)}", str(r["true_conclusion"]).replace("_", " "),
                 _fmt(r["inflation_full"], 3, True), r["converged_folds"]] for r in sorted(cand, key=lambda r: r["true_rank"] or 99)]
        h.append(_table(["id", "tournament rank", "unified rank", "self gain (pilot)", "self gain (full)", "unified gain", "95% interval", "conclusion",
                         "self minus unified", "folds converged"], rows, 0, sel))
    notes = _unfinished(c["fulls"])
    if notes:
        h.append("<p class='small'><b>Full runs the time budget cut short</b> (the agent's own runs; the self-reported full-run scores of these plans rest on what was completed): "
                 + html.escape("; ".join(notes)) + ".</p>")
    h.append(fig("self_vs_true.png", "What each plan reported for itself (pilot and full run) against the evaluator's unified score."))
    if c["alt"]:
        a = c["alt"]
        h.append("<h2>7. The same data under a different question</h2>" + f"<p class='q'><b>Alternative question (plans generated, not executed).</b> {html.escape(a['question'])}</p>")
        h.append(_table(["field", "main question", "alternative question"], [
            ["task family", o["task_family"], a["objective"]["task_family"]],
            ["allowed inputs", "; ".join(o["allowed_inputs"]), "; ".join(a["objective"]["allowed_inputs"])],
            ["forbidden inputs", "; ".join(o["forbidden_inputs"]), "; ".join(a["objective"]["forbidden_inputs"])],
            ["required comparators", "; ".join(o["required_comparators"]), "; ".join(a["objective"]["required_comparators"])],
            ["primary metric", o["primary_metric"], a["objective"]["primary_metric"]]]))
        from .tools import registry
        known = set(registry.component_keys(a["objective"]["task_family"]))
        h.append(_table(["plan", "kind", "core component", "component in the registry"],
                        [[p["name"], p["complexity"].replace("_", " "), p["model"].get("component"), "yes" if p["model"].get("component") in known else "no (proposal only)"] for p in a["plans"]]))
    if c["costs"]:
        co = c["costs"]
        h.append("<h2>8. Cost</h2>" + _table(["plan", "pilot CPU core-hours", "full-run CPU core-hours", "GPU hours"],
                                              [[p, (v.get("pilot") or {}).get("cpu_core_hours", "n/a"), (v.get("full") or {}).get("cpu_core_hours", "n/a"),
                                                round(sum((v.get(k) or {}).get("gpu_hours", 0) for k in ("pilot", "full")), 3)] for p, v in co["plans"].items()]))
        h.append(_table(["role", "answered calls", "failed attempts", "input tokens", "output tokens", "thinking tokens", "wall seconds", "list-price USD (estimate)"],
                        [[k, v["answered_calls"], v["failed_attempts"], f"{v['input_tokens']:,}", f"{v['output_tokens']:,}", f"{v['thinking_tokens']:,}", v["wall_seconds"],
                          v["list_price_usd"] or "n/a"] for k, v in co["llm"].items()]))
    h.append("</body></html>")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    HTML(string="".join(h)).write_pdf(out_pdf)
    return out_pdf


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["facts", "write", "render"])
    ap.add_argument("--task", required=True), ap.add_argument("--out"), ap.add_argument("--title")
    a = ap.parse_args(argv)
    rdir = ROOT / "outputs" / a.task / "report"
    if a.command == "facts":
        c = collect(a.task)
        made = figures(c, rdir / "figures")
        facts = named_facts(c)
        write_json(rdir / "collected.json", c)
        write_json(rdir / "named_facts.json", {k: dict(value=v[0], meaning=v[1]) for k, v in facts.items()})
        print(f"{len(facts)} named facts; figures: {made}")
        return
    c = read_json(rdir / "collected.json")
    facts = {k: (v["value"], v["meaning"]) for k, v in read_json(rdir / "named_facts.json").items()}
    if a.command == "write":
        prose, meta = write_prose(c, facts, tag=a.task)
        if prose is None:
            prose = fallback_prose(c, facts)
        write_json(rdir / "prose.json", dict(prose=prose, meta=meta, written=now()))
        print(f"prose source: {meta['source']}")
    else:
        p = read_json(rdir / "prose.json")
        out = render(c, facts, p["prose"], p["meta"], rdir / "figures", ROOT / a.out, a.title or f"Analysis report: {a.task}")
        print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
