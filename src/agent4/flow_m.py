"""Cross-modal prediction: a plan's own validation (pilot and full scale), its evidence card, and the full run.

Everything here uses development cells only. The plan is validated the way the plan itself declared
(README 5.2: "the score a plan reports comes from its own validation design"); the evaluator's unified
score is computed elsewhere and never read here.

Inside its own validation the agent applies the same decision rules the evaluator will apply to held-out
cells (README 4.3): every required comparator is fitted on the same training cells and scored on the same
validation cells - the RNA-to-RNA ceiling, the cell-type (cluster-mean) baseline, gene activity, depth
only, ridge, k-nearest neighbours and partial least squares - and genes are classified into "cannot be
judged", "not beyond the cell-type baseline" and "beyond cell type".
"""
from __future__ import annotations

import os
import time

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

from . import stats
from .io import now, write_json
from .schemas import EvidenceCard, EvidenceItem
from .tools import m_basic as mb
from .tools import registry

FAMILY = "cross_modal_prediction"
PILOT_CELLS, PILOT_GENES = 3000, 1500
# Wall-clock budgets in seconds. A full run (own validation, then the final fit) has to fit one two-hour job (README 6.2).
PILOT_TIME_BUDGET, FULL_VALIDATION_BUDGET, FINAL_FIT_BUDGET = 25 * 60, 60 * 60, 50 * 60
RULERS = ("ceiling", "cluster_mean", "ridge", "pls", "knn", "gene_activity", "depth_only")


def _validation(plan):
    v = plan.get("validation", {}) or {}
    scheme = str(v.get("scheme", "holdout")).lower()
    target = str(v.get("scoring_target", "thinned_half")).lower()
    out = dict(scheme="kfold" if "fold" in scheme else "holdout",
               k=int(min(max(int(v.get("k", 3) or 3), 2), 5)),
               fraction=float(min(max(float(v.get("holdout_fraction", 0.2) or 0.2), 0.1), 0.3)),
               target=target if target in ("thinned_half", "full_depth", "knn_smoothed") else "thinned_half",
               n_seeds=int(min(max(int(v.get("n_seeds", 1) or 1), 1), 3)))
    if out["target"] == "knn_smoothed":
        out["smooth_k"] = int(v.get("smoothing_k", 15) or 15)
    return out


def _truth(view, rna, genes, v, seed):
    """Validation truth as the plan chose to define it. Returns (truth, the other read half or None)."""
    cols = view.scope[genes]
    if v["target"] == "thinned_half":
        half_a, half_b = stats.thin_counts(rna, seed=seed)
        return view.target(half_b, cols), half_a
    full = view.target(rna, cols)
    if v["target"] == "full_depth":
        return full, None
    z = TruncatedSVD(30, random_state=seed).fit_transform(stats.lognorm(rna))
    nn = NearestNeighbors(n_neighbors=min(v["smooth_k"], len(z))).fit(z).kneighbors(z, return_distance=False)
    return full[nn].mean(1).astype(np.float32), None


def _rulers(view, atac_tr, rna_tr, atac_va, genes, seed, half_a_val=None):
    """Every required comparator, fitted on the same training cells and predicted for the same validation cells."""
    cols = view.scope[genes]
    lsi = mb.LSI(seed=seed, **view.rules["baseline_lsi"]).fit(atac_tr)
    z_tr, z_va = lsi.transform(atac_tr), lsi.transform(atac_va)
    y_tr = view.target(rna_tr, cols)
    act, has = mb.gene_activity(atac_va, view.peak_gene, cols, atac_va.shape[1])
    act[:, ~has] = 0.0
    out = dict(
        cluster_mean=mb.ClusterMean(view.cluster_k, seed=seed).fit(z_tr, y_tr).predict(z_va),
        ridge=mb.LatentRidge(10.0).fit(z_tr, y_tr).predict(z_va), pls=mb.LatentPLS(15).fit(z_tr, y_tr).predict(z_va),
        knn=mb.LatentKNN(20).fit(z_tr, y_tr).predict(z_va), gene_activity=act,
        depth_only=mb.DepthOnly().fit(mb.depth_feature(atac_tr), y_tr).predict(mb.depth_feature(atac_va)))
    if half_a_val is not None:
        # Noise ceiling inside the development set: an RNA-to-RNA model sees one read half of the validation
        # cells and predicts the other half, exactly as the evaluator's ceiling does on held-out cells.
        a_tr, b_tr = stats.thin_counts(rna_tr, seed=seed + 5)
        out["ceiling"] = mb.DenoiseCeiling(seed=seed).fit(a_tr, b_tr, cols, counts_full=rna_tr).predict(half_a_val)
    return out


def validate(plan, view, scale, out_dir, seed=0, log=print):
    """Run the plan's own validation and write raw results. Returns the summary dict."""
    v = _validation(plan)
    rng = np.random.default_rng(seed)
    n_dev = view.dev_rna.shape[0]
    cells = np.sort(rng.choice(n_dev, PILOT_CELLS, replace=False)) if scale == "pilot" else np.arange(n_dev)
    genes = np.sort(rng.choice(len(view.scope), PILOT_GENES, replace=False)) if scale == "pilot" else np.arange(len(view.scope))
    budget = PILOT_TIME_BUDGET if scale == "pilot" else FULL_VALIDATION_BUDGET
    splits = []
    for s in range(v["n_seeds"]):
        order = np.random.default_rng(seed + 1000 * s).permutation(cells)
        if v["scheme"] == "kfold":
            for fold in np.array_split(order, v["k"]):
                splits.append((np.setdiff1d(cells, fold), np.sort(fold), s))
        else:
            cut = int(round(v["fraction"] * len(order)))
            splits.append((np.sort(order[cut:]), np.sort(order[:cut]), s))
    n_declared = len(splits)
    if scale == "pilot":
        splits = splits[: max(1, min(len(splits), 2))]
    per_split, reports, t_fit = [], [], 0.0
    # Gene-level tests use the splits of the first seed only, so that no validation cell is counted twice;
    # further seeds contribute to the spread of the mean score.
    pooled = {k: [] for k in ("pred", "truth", *RULERS)}
    t_start = time.time()
    for i, (tr, va, s) in enumerate(splits):
        elapsed = time.time() - t_start
        if i >= 1 and elapsed + elapsed / i > budget:
            log(f"  stopping after {i} of {len(splits)} splits: the next one would exceed the {budget / 60:.0f}-minute validation budget")
            break
        model = registry.build_model(plan, view, scale=scale, seed=seed + i, genes=genes, time_budget=max(budget - elapsed, 120.0))
        t0 = time.time()
        report = model.fit(view.dev_atac[tr], view.dev_rna[tr])
        pred = model.predict(view.dev_atac[va])
        t_fit += time.time() - t0
        truth, half_a = _truth(view, view.dev_rna[va], genes, v, seed + 77 + i)
        primary = s == 0
        rulers = _rulers(view, view.dev_atac[tr], view.dev_rna[tr], view.dev_atac[va], genes, seed + i, half_a if primary else None)
        row = dict(split=i, seed=s, n_train=len(tr), n_val=len(va), plan_mean_r=float(np.mean(stats.colcorr(pred, truth))),
                   **{f"{k}_mean_r": float(np.mean(stats.colcorr(p, truth))) for k, p in rulers.items()})
        per_split.append(row)
        reports.append({k: report.get(k) for k in ("seconds", "device", "epochs", "stopped_by", "converged", "convergence_note", "curves", "n_parameters")})
        if primary:
            pooled["pred"].append(pred), pooled["truth"].append(truth)
            for k, p in rulers.items():
                pooled[k].append(p)
        log(f"  split {i}: plan r={row['plan_mean_r']:.4f} cluster_mean r={row['cluster_mean_mean_r']:.4f} ridge r={row['ridge_mean_r']:.4f} "
            f"({report.get('stopped_by')}, converged={report.get('converged')}, {report.get('seconds')}s)")
    pred, truth, base = (np.vstack(pooled[k]) for k in ("pred", "truth", "cluster_mean"))
    boot = stats.paired_bootstrap_delta_r(pred, base, truth, n_boot=200, seed=seed)
    null = stats.permutation_null(pred, truth, n_perm=50, seed=seed)
    r = stats.colcorr(pred, truth)
    has_ceiling = bool(pooled["ceiling"])
    if has_ceiling:
        ceiling_pred = np.vstack(pooled["ceiling"])
        null_ceiling = stats.permutation_null(ceiling_pred, truth, n_perm=50, seed=seed + 1)
        label, q = stats.classify_genes(r, null_ceiling["p"], boot["p"])
        r_ceiling = stats.colcorr(ceiling_pred, truth)
    else:                      # the plan scores against a target that leaves no independent read half for a ceiling
        claimed, q = stats.bh_reject(boot["p"])
        label, r_ceiling = np.where(claimed, 2, 1), np.full(len(r), np.nan)
    claimed = label == 2
    scores = np.array([row["plan_mean_r"] for row in per_split])
    ruler_means = {k: float(np.mean([row[f"{k}_mean_r"] for row in per_split if f"{k}_mean_r" in row])) for k in RULERS
                   if any(f"{k}_mean_r" in row for row in per_split)}
    summary = dict(
        plan_id=plan["plan_id"], scale=scale, finished=now(), validation=v, n_cells=len(cells), n_genes=len(genes),
        n_splits_declared=n_declared, n_splits_run=len(per_split), n_cells_in_gene_tests=int(len(pred)),
        self_score_mean_r=float(scores.mean()), self_score_sd=float(scores.std(ddof=1)) if len(scores) > 1 else None,
        rulers_mean_r=ruler_means, fraction_of_ceiling=(float(scores.mean()) / ruler_means["ceiling"]) if "ceiling" in ruler_means else None,
        margin_over_cluster_mean=boot["mean_delta"], margin_ci=boot["mean_delta_ci"],   # cell-level bootstrap
        ceiling_computed=has_ceiling, n_cannot_judge=int((label == 0).sum()), n_judgeable=int((label != 0).sum()),
        n_not_beyond_cell_type=int((label == 1).sum()), n_genes_claimed_beyond_cell_type=int(claimed.sum()),
        share_claimed=float(claimed.sum() / max((label != 0).sum(), 1)), n_genes_above_zero=int(stats.bh_reject(null["p"])[0].sum()),
        null_r_q95=null["null_q95"], per_split=per_split, fit_reports=reports, fit_seconds=round(t_fit, 1),
        wall_seconds=round(time.time() - t_start, 1), cpus=int(os.environ.get("A4_THREADS", 1)),
        all_converged=bool(all(rep.get("converged") for rep in reports)),
        resolved_config=getattr(model, "resolved", None), ignored_params=getattr(model, "ignored", []))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "self_validation.json", summary)
    np.savez_compressed(out_dir / "self_validation_per_gene.npz", genes=view.scope[genes], r=r, r_ceiling=r_ceiling, delta_vs_cluster=boot["delta"],
                        delta_se=boot["se"], q_beyond=q, label=label, claimed=claimed)
    return summary


def evidence_card(plan, summary, compliance, limits) -> EvidenceCard:
    v, items = summary["validation"], []

    def add(kind, text, value=None):
        items.append(EvidenceItem(id=f"{plan['plan_id']}-E{len(items) + 1}", kind=kind, text=text, value=value))

    sd = f" (SD {summary['self_score_sd']:.4f} over {len(summary['per_split'])} splits)" if summary["self_score_sd"] is not None else " (one split, no spread available)"
    add("self_score", f"Self-reported score: mean per-gene Pearson r = {summary['self_score_mean_r']:.4f}{sd}, on "
                      f"{summary['n_genes']} genes; {summary['n_cells_in_gene_tests']} distinct validation cells enter the gene-level tests.", summary["self_score_mean_r"])
    rl = summary["rulers_mean_r"]
    ceiling = f"noise ceiling (RNA-to-RNA) {rl['ceiling']:.4f}; " if "ceiling" in rl else "no noise ceiling (the scoring target leaves no independent read half); "
    add("ruler", f"Required comparators on the same training and validation cells, genes and scoring target: {ceiling}cluster-mean baseline {rl['cluster_mean']:.4f}, "
                 f"ridge {rl['ridge']:.4f}, partial least squares {rl['pls']:.4f}, k-nearest neighbours {rl['knn']:.4f}, gene activity {rl['gene_activity']:.4f}, "
                 f"depth only {rl['depth_only']:.4f}." + (f" The plan reaches {summary['fraction_of_ceiling']:.0%} of the ceiling." if summary.get("fraction_of_ceiling") else ""), rl)
    best_simple = max(rl["ridge"], rl["pls"], rl["knn"])
    add("ruler", f"Margin over the cluster-mean baseline: mean per-gene difference in r = {summary['margin_over_cluster_mean']:+.4f} "
                 f"(95% interval {summary['margin_ci'][0]:+.4f} to {summary['margin_ci'][1]:+.4f}, bootstrap over cells); margin over the best simple learner "
                 f"(ridge, PLS, k-NN) {summary['self_score_mean_r'] - best_simple:+.4f}.", summary["margin_over_cluster_mean"])
    add("self_score", f"Three-way gene classification by the evaluator's rule, applied inside the development set: {summary['n_cannot_judge']} genes cannot be judged "
                      f"(ceiling not above the permutation null), {summary['n_not_beyond_cell_type']} are not beyond the cell-type baseline, "
                      f"{summary['n_genes_claimed_beyond_cell_type']} are predicted beyond cell type (paired bootstrap against the cluster-mean baseline, FDR 5%; "
                      f"{summary['share_claimed']:.1%} of judgeable genes). Genes with r above the permutation null: {summary['n_genes_above_zero']}.",
        summary["n_genes_claimed_beyond_cell_type"])
    reps = summary["fit_reports"]
    curve_note = "; ".join(f"fit {i}: {r.get('stopped_by')} after {r.get('epochs')} epochs, converged={r.get('converged')} ({r.get('convergence_note') or 'no note'})" for i, r in enumerate(reps[:4]))
    more = f"; {len(reps) - 4} further fits, all converged={all(r.get('converged') for r in reps[4:])}" if len(reps) > 4 else ""
    add("convergence", f"Training: {curve_note}{more}.", dict(all_converged=summary["all_converged"]))
    target_text = {"thinned_half": "an independent half of each validation cell's reads (unsmoothed, same as the evaluator)",
                   "full_depth": "the unsplit log-normalised counts of validation cells (less noisy than what the evaluator scores against)",
                   "knn_smoothed": "expression pooled over nearest validation cells (smoothed target)"}[v["target"]]
    scheme = f"{v['k']}-fold cross-validation" if v["scheme"] == "kfold" else f"a single hold-out of {v['fraction']:.0%}"
    add("validation_design", f"Validation design: split unit cell; {scheme}; {v['n_seeds']} seed(s); scored against {target_text}. "
                             f"Early stopping used a separate split carved from the training cells.", v)
    add("validation_design", f"Scale of this run: {summary['scale']}; {summary['n_splits_run']} of the {summary['n_splits_declared']} declared train/validation splits were run "
                             f"({summary['n_cells']} development cells, {summary['n_genes']} genes; "
                             f"the full problem has {limits.get('n_dev')} cells and {limits.get('n_scope')} genes).")
    add("cost", f"Cost: {summary['fit_seconds']:.0f} s of model fitting on {summary['cpus']} CPU cores ({reps[0].get('device')}); declared budget for the full run "
                f"{plan.get('budget', {}).get('full_minutes')} min, needs GPU: {plan.get('budget', {}).get('needs_gpu')}.", summary["fit_seconds"])
    add("compliance", "Compliance checks: " + "; ".join(f"{r['rule']} {'passed' if r['passed'] else 'FAILED'}" for r in compliance["rules"]) + ".",
        compliance["passed"])
    if summary.get("ignored_params"):
        add("note", f"Fields of the plan the executor has no implementation for (its defaults were used; some of these are descriptive text rather than settings): "
                    f"{summary['ignored_params']}.")
    return EvidenceCard(plan_id=plan["plan_id"], scale=summary["scale"], items=items, raw_files=["self_validation.json", "self_validation_per_gene.npz"])


def execute(plan, view, out_dir, seed=0, log=print):
    """Full run: the plan's own validation at full scale, then a final fit on all development cells and the
    deliverable for the held-out cells (which have ATAC only)."""
    summary = validate(plan, view, "full", out_dir, seed=seed, log=log)
    model = registry.build_model(plan, view, scale="full", seed=seed, genes=None, time_budget=FINAL_FIT_BUDGET)
    t0 = time.time()
    report = model.fit(view.dev_atac, view.dev_rna)
    pred = model.predict(view.hidden_atac).astype(np.float32)
    seconds = time.time() - t0
    np.save(out_dir / "pred_hidden.npy", pred)
    z = np.load(out_dir / "self_validation_per_gene.npz")
    order = np.argsort(-z["delta_vs_cluster"])
    order = order[z["claimed"][order]]
    write_json(out_dir / "gene_claims.json", dict(
        n_claimed=int(z["claimed"].sum()), rule="judgeable and r above the cluster-mean baseline, paired bootstrap, BH FDR 5%, development cells",
        genes=[dict(gene=str(view.genes.name.values[z["genes"][i]]), r=float(z["r"][i]), delta_vs_cluster=float(z["delta_vs_cluster"][i]),
                    delta_ci=[float(z["delta_vs_cluster"][i] - 1.96 * z["delta_se"][i]), float(z["delta_vs_cluster"][i] + 1.96 * z["delta_se"][i])]) for i in order]))
    n_claimed = summary["n_genes_claimed_beyond_cell_type"]
    result = dict(plan_id=plan["plan_id"], finished=now(), self_score_mean_r=summary["self_score_mean_r"], self_score_sd=summary["self_score_sd"],
                  rulers_mean_r=summary["rulers_mean_r"], fraction_of_ceiling=summary["fraction_of_ceiling"],
                  margin_over_cluster_mean=summary["margin_over_cluster_mean"], margin_ci=summary["margin_ci"],
                  n_genes_in_scope=summary["n_genes"], n_cannot_judge=summary["n_cannot_judge"], n_judgeable=summary["n_judgeable"],
                  n_not_beyond_cell_type=summary["n_not_beyond_cell_type"], n_genes_claimed_beyond_cell_type=n_claimed,
                  share_claimed=summary["share_claimed"], n_genes_above_zero=summary["n_genes_above_zero"], permutation_null_r_q95=summary["null_r_q95"],
                  validation=summary["validation"], n_splits_declared=summary["n_splits_declared"], n_splits_run=summary["n_splits_run"],
                  n_cells_in_gene_tests=summary["n_cells_in_gene_tests"],
                  final_fit={k: report.get(k) for k in ("seconds", "device", "epochs", "stopped_by", "converged", "convergence_note", "n_parameters")},
                  final_fit_curves=report.get("curves"), validation_all_converged=summary["all_converged"],
                  deliverable=dict(file="pred_hidden.npy", shape=list(pred.shape), finite=bool(np.isfinite(pred).all()), gene_claims="gene_claims.json"),
                  seconds_total=round(summary["wall_seconds"] + seconds, 1), cpus=int(os.environ.get("A4_THREADS", 1)),
                  conclusion=("specific genes predicted beyond cell identity" if n_claimed > 0 else "no gene predicted beyond cell identity"),
                  scope="unseen cells of the same donor; development-set estimate, the evaluator scores the held-out cells separately")
    write_json(out_dir / "full_run.json", result)
    return result
