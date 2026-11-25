"""Evaluator for task M (README 4.1): data split, read thinning, rulers, unified scoring.

    python -m bench.task_m prepare   build the agent view and the private hidden halves
    python -m bench.task_m rules     freeze the data-determined rules using development cells only
    python -m bench.task_m rulers    ceiling and floors on the hidden cells
    python -m bench.task_m score --pred <npy> --name <plan id>   true score of one prediction matrix
"""
from __future__ import annotations

import argparse
import sys

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from agent4 import stats
from agent4.data import load_cell_view
from agent4.io import Timer, now, read_json, rel, sha256_file, write_index, write_json
from agent4.tools import m_basic as mb
from bench import paths

TASK = "task_m"
SEED = 774
HIDDEN_FRACTION = 0.30
MIN_DETECTION_FRACTION = 0.01      # numerical-stability prefilter only (README 4.1)
CLUSTER_GRID = (5, 8, 12, 16, 20, 30, 40, 60, 80, 120, 160, 240)
BASELINE_LSI = dict(n_components=50, max_peaks=60000)


def _decode(v):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in v])


def prepare():
    view, private = paths.view_dir(TASK), paths.private_dir(TASK)
    view.mkdir(parents=True, exist_ok=True), private.mkdir(parents=True, exist_ok=True)
    with h5py.File(paths.DATA / "pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5", "r") as f:
        g = f["matrix"]
        m = sparse.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=g["shape"][:]).tocsr()
        barcodes, ftype = _decode(g["barcodes"][:]), _decode(g["features/feature_type"][:])
        fname, fid, interval = (_decode(g[f"features/{k}"][:]) for k in ("name", "id", "interval"))
    is_gene, is_peak = ftype == "Gene Expression", ftype == "Peaks"
    rna, atac = m[np.flatnonzero(is_gene)].T.tocsr(), m[np.flatnonzero(is_peak)].T.tocsr()
    labels = pd.read_csv(paths.DATA / "annotations/pbmc_10x_annotation_FixFlip.csv", index_col=0)
    assert np.array_equal(labels.index.astype(str), barcodes)
    qc = np.flatnonzero((labels.celltype != "filtered").values)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(qc)
    n_hidden = int(round(HIDDEN_FRACTION * len(order)))
    hidden, dev = np.sort(order[:n_hidden]), np.sort(order[n_hidden:])

    # Gene universe: drop genes detected in very few development cells.
    detected = np.asarray((rna[dev] > 0).sum(0)).ravel()
    universe = np.flatnonzero(detected >= MIN_DETECTION_FRACTION * len(dev))
    parsed = pd.Series(interval[is_gene][universe]).str.extract(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
    genes = pd.DataFrame(dict(name=fname[is_gene][universe], gene_id=fid[is_gene][universe], chrom=parsed.chrom.values,
                              tss=((parsed.start.astype(float) + parsed.end.astype(float)) / 2).values,
                              n_cells_dev=detected[universe]))
    pk = pd.Series(fname[is_peak]).str.extract(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
    peaks = pd.DataFrame(dict(name=fname[is_peak], chrom=pk.chrom.values, start=pk.start.astype(int).values,
                              end=pk.end.astype(int).values))

    ann = pd.read_csv(paths.DATA / "pbmc_granulocyte_sorted_10k_atac_peak_annotation.tsv", sep="\t", dtype=str)
    rows = []
    for col in ("gene", "distance", "peak_type"):
        ann[col] = ann[col].fillna("").str.split(";")
    ann = ann.explode(["gene", "distance", "peak_type"])
    peak_pos = pd.Series(np.arange(len(peaks)), index=peaks.name.str.replace(r"[:\-]", "_", regex=True))
    gene_pos = pd.Series(np.arange(len(genes)), index=genes.name).groupby(level=0).first()
    ann = ann[ann.gene.isin(gene_pos.index) & ann.peak.isin(peak_pos.index)]
    peak_gene = pd.DataFrame(dict(peak_idx=peak_pos[ann.peak].values, gene_idx=gene_pos[ann.gene].values,
                                  distance=pd.to_numeric(ann.distance, errors="coerce").fillna(0).astype(int).values,
                                  peak_type=ann.peak_type.values)).drop_duplicates(["peak_idx", "gene_idx"])
    del rows

    obs = pd.DataFrame(dict(celltype=labels.celltype.values, n_rna=np.asarray(rna.sum(1)).ravel(),
                            n_atac=np.asarray(atac.sum(1)).ravel()), index=barcodes)
    sparse.save_npz(view / "dev_rna.npz", rna[dev][:, universe].astype(np.int32).tocsr())
    sparse.save_npz(view / "dev_atac.npz", atac[dev].astype(np.int32).tocsr())
    sparse.save_npz(view / "hidden_atac.npz", atac[hidden].astype(np.int32).tocsr())
    obs.iloc[dev].to_parquet(view / "dev_obs.parquet")
    # Held-out cells: ATAC-derived fields only. The cell-type annotation was derived from RNA as well.
    obs.iloc[hidden][["n_atac"]].to_parquet(view / "hidden_obs.parquet")
    genes.to_parquet(view / "genes.parquet"), peaks.to_parquet(view / "peaks.parquet")
    peak_gene.to_parquet(view / "peak_gene.parquet")
    write_json(view / "view.json", dict(
        adapter="paired_single_cell", task_id=TASK, created=now(),
        n_dev=len(dev), n_hidden=len(hidden), n_genes=len(universe), n_peaks=int(is_peak.sum()),
        modalities=dict(rna="UMI counts, development cells only", atac="peak counts, all cells"),
        note="Held-out cells provide ATAC only. Gene universe: detected in at least "
             f"{MIN_DETECTION_FRACTION:.0%} of development cells."))

    hidden_rna = rna[hidden][:, universe]
    half_a, half_b = stats.thin_counts(hidden_rna, seed=SEED + 1)
    sparse.save_npz(private / "hidden_rna_full.npz", hidden_rna.astype(np.int32).tocsr())
    sparse.save_npz(private / "hidden_rna_A.npz", half_a.astype(np.int32))
    sparse.save_npz(private / "hidden_rna_B.npz", half_b.astype(np.int32))
    obs.iloc[hidden].to_parquet(private / "hidden_obs_full.parquet")
    write_json(private / "split.json", dict(seed=SEED, hidden_fraction=HIDDEN_FRACTION, thinning_seed=SEED + 1,
                                            n_qc_cells=len(qc), dev_barcodes=barcodes[dev].tolist(),
                                            hidden_barcodes=barcodes[hidden].tolist()))
    print(f"prepared: qc={len(qc)} dev={len(dev)} hidden={len(hidden)} genes={len(universe)} peaks={int(is_peak.sum())} "
          f"links={len(peak_gene)}")


def rules():
    """Freeze the data-determined rules from development cells only (README 4.3)."""
    view = load_cell_view(paths.view_dir(TASK))
    n = view.dev_rna.shape[0]
    all_genes = np.arange(view.dev_rna.shape[1])
    rng = np.random.default_rng(SEED + 2)
    order = rng.permutation(n)
    fit, val = np.sort(order[: int(0.7 * n)]), np.sort(order[int(0.7 * n):])
    a_fit, b_fit = stats.thin_counts(view.dev_rna[fit], seed=SEED + 3)
    a_val, b_val = stats.thin_counts(view.dev_rna[val], seed=SEED + 4)

    # Gene scope: genes whose ceiling prediction beats the permutation null inside the development set.
    with Timer() as t:
        ceiling = mb.DenoiseCeiling(seed=SEED).fit(a_fit, b_fit, all_genes, counts_full=view.dev_rna[fit])
        truth_val = np.asarray(stats.lognorm(b_val).todense(), dtype=np.float32)
        null = stats.permutation_null(ceiling.predict(a_val), truth_val, n_perm=100, seed=SEED)
    reject, q = stats.bh_reject(null["p"])
    scope = np.flatnonzero(reject)
    print(f"gene scope: {len(scope)} of {len(all_genes)} universe genes ({t.seconds}s); ceiling rank {ceiling.rank_}")

    # Granularity of the cell-type baseline: 3-fold CV inside the development set, scored on thinned halves.
    folds = np.array_split(rng.permutation(n), 3)
    grid_scores = {k: [] for k in CLUSTER_GRID}
    for i, test in enumerate(folds):
        train = np.setdiff1d(np.arange(n), test)
        lsi = mb.LSI(seed=SEED, **BASELINE_LSI).fit(view.dev_atac[train])
        z_tr, z_te = lsi.transform(view.dev_atac[train]), lsi.transform(view.dev_atac[test])
        y_tr = view.target(view.dev_rna[train], scope)
        _, b_te = stats.thin_counts(view.dev_rna[test], seed=SEED + 10 + i)
        y_te = view.target(b_te, scope)
        for k in CLUSTER_GRID:
            pred = mb.ClusterMean(k, seed=SEED).fit(z_tr, y_tr).predict(z_te)
            grid_scores[k].append(float(np.nanmean(stats.colcorr(pred, y_te))))
    mean_scores = {k: float(np.mean(v)) for k, v in grid_scores.items()}
    cluster_k = max(mean_scores, key=mean_scores.get)
    print("cluster grid (mean r):", {k: round(v, 4) for k, v in mean_scores.items()}, "->", cluster_k)

    out = dict(
        frozen=now(), seed=SEED, gene_scope=scope.tolist(), n_universe=len(all_genes), n_scope=len(scope),
        scope_rule="ceiling r above the permutation null inside the development set, BH FDR 5%",
        ceiling_dev=dict(rank=ceiling.rank_, rank_scores=ceiling.rank_scores_,
                         mean_r_universe=float(np.nanmean(null["r"])), mean_r_scope=float(np.nanmean(null["r"][scope])),
                         null_q95=null["null_q95"], null_sd_stability=null["sd_split_half_rel_diff"], n_perm=100,
                         n_val_cells=len(val)),
        cluster_k=int(cluster_k), cluster_grid_mean_r=mean_scores, cluster_grid_fold_r=grid_scores,
        cluster_rule="k with the best cluster-mean prediction in 3-fold CV inside the development set",
        baseline_lsi=BASELINE_LSI, fdr=stats.FDR, confidence=stats.CONFIDENCE)
    path = write_json(paths.view_dir(TASK) / "rules.json", out)
    write_json(paths.view_dir(TASK) / "rules.sha256.json", dict(file="rules.json", sha256=sha256_file(path), frozen=out["frozen"]))
    write_index(paths.view_dir(TASK), note="agent view for task M; contains no RNA of held-out cells")


def _hidden():
    private = paths.private_dir(TASK)
    return (sparse.load_npz(private / "hidden_rna_A.npz"), sparse.load_npz(private / "hidden_rna_B.npz"))


def rulers():
    """Ceiling and the four floors, fitted on all development cells and scored on the hidden B halves."""
    view, private = load_cell_view(paths.view_dir(TASK)), paths.private_dir(TASK)
    scope = view.scope
    half_a, half_b = _hidden()
    truth = view.target(half_b, scope)
    preds, info = {}, {}

    with Timer() as t:
        a_dev, b_dev = stats.thin_counts(view.dev_rna, seed=SEED + 20)
        ceiling = mb.DenoiseCeiling(seed=SEED).fit(a_dev, b_dev, scope, counts_full=view.dev_rna)
        preds["ceiling"] = ceiling.predict(half_a)
    info["ceiling"] = dict(seconds=t.seconds, rank=ceiling.rank_, sees="A half of the hidden cells' RNA")

    lsi = mb.LSI(seed=SEED, **BASELINE_LSI).fit(view.dev_atac)
    z_dev, z_hid = lsi.transform(view.dev_atac), lsi.transform(view.hidden_atac)
    y_dev = view.target(view.dev_rna, scope)
    cm = mb.ClusterMean(view.cluster_k, seed=SEED).fit(z_dev, y_dev)
    preds["floor1_cluster_mean"] = cm.predict(z_hid)
    clusters_hidden = cm.assign(z_hid)
    info["floor1_cluster_mean"] = dict(k=view.cluster_k)

    for mode in ("promoter_and_distal", "promoter"):
        act, has = mb.gene_activity(view.hidden_atac, view.peak_gene, scope, view.hidden_atac.shape[1], mode=mode)
        act[:, ~has] = 0.0
        preds[f"floor2_gene_activity_{mode}"] = act
        info[f"floor2_gene_activity_{mode}"] = dict(genes_with_peaks=int(has.sum()))
    preds["floor3_depth_only"] = mb.DepthOnly().fit(mb.depth_feature(view.dev_atac), y_dev).predict(
        mb.depth_feature(view.hidden_atac))

    # Floor 4: simple learners on the latent space; one hyper-parameter each, chosen inside the development set.
    rng = np.random.default_rng(SEED + 30)
    order = rng.permutation(len(z_dev))
    tr, va = order[: int(0.8 * len(order))], order[int(0.8 * len(order)):]
    _, b_va = stats.thin_counts(view.dev_rna[np.sort(va)], seed=SEED + 31)
    y_va = view.target(b_va, scope)
    va = np.sort(va)
    for name, cls, grid in (("ridge", mb.LatentRidge, (1.0, 10.0, 100.0, 1000.0)), ("knn", mb.LatentKNN, (10, 20, 50, 100)),
                            ("pls", mb.LatentPLS, (5, 10, 20, 40))):
        scores = {v: float(np.nanmean(stats.colcorr(cls(v).fit(z_dev[tr], y_dev[tr]).predict(z_dev[va]), y_va))) for v in grid}
        best = max(scores, key=scores.get)
        preds[f"floor4_{name}"] = cls(best).fit(z_dev, y_dev).predict(z_hid)
        info[f"floor4_{name}"] = dict(chosen=best, dev_scores=scores)

    null_ceiling = stats.permutation_null(preds["ceiling"], truth, n_perm=100, seed=SEED + 40)
    table = pd.DataFrame(dict(gene=view.genes.name.values[scope], gene_idx=scope, p_ceiling_above_null=null_ceiling["p"]))
    summary = {}
    for name, pred in preds.items():
        r = stats.colcorr(pred, truth)
        table[f"r_{name}"] = r
        summary[name] = dict(mean_r=float(np.nanmean(r)), median_r=float(np.nanmedian(r)), **info.get(name, {}))
        np.save(private / f"pred_{name}.npy", pred.astype(np.float32))
    judgeable, _ = stats.bh_reject(null_ceiling["p"])
    table["judgeable"] = judgeable
    table.to_parquet(private / "rulers_per_gene.parquet")
    np.save(private / "hidden_clusters.npy", clusters_hidden)
    write_json(private / "rulers.json", dict(created=now(), n_hidden=truth.shape[0], n_scope=len(scope),
                                             n_judgeable=int(judgeable.sum()), null_q95=null_ceiling["null_q95"],
                                             null_sd_stability=null_ceiling["sd_split_half_rel_diff"], rulers=summary))
    for k, v in summary.items():
        print(f"{k:38s} mean_r={v['mean_r']:.4f} median_r={v['median_r']:.4f}")
    print("judgeable genes (ceiling above null on hidden cells):", int(judgeable.sum()), "of", len(scope))


def score(pred_path, name, out_dir=None):
    """Unified true score of one prediction matrix (hidden cells x scope genes)."""
    view, private = load_cell_view(paths.view_dir(TASK)), paths.private_dir(TASK)
    scope = view.scope
    _, half_b = _hidden()
    truth = view.target(half_b, scope)
    pred = np.load(pred_path).astype(np.float32)
    assert pred.shape == truth.shape, f"prediction shape {pred.shape} != {truth.shape}"
    assert np.isfinite(pred).all(), "prediction contains non-finite values"
    rulers_tbl = pd.read_parquet(private / "rulers_per_gene.parquet")
    base = np.load(private / "pred_floor1_cluster_mean.npy")
    clusters = np.load(private / "hidden_clusters.npy")
    r = stats.colcorr(pred, truth)
    null = stats.permutation_null(pred, truth, n_perm=100, seed=SEED + 50)
    boot = stats.paired_bootstrap_delta_r(pred, base, truth, n_boot=300, seed=SEED + 51)
    label, q_beyond = stats.classify_genes(r, rulers_tbl.p_ceiling_above_null.values, boot["p"])
    above_zero, _ = stats.bh_reject(null["p"])
    r_ceiling = rulers_tbl.r_ceiling.values
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.clip(r / r_ceiling, None, 1.0)
    table = pd.DataFrame(dict(gene=rulers_tbl.gene, r=r, r_cluster_mean=rulers_tbl.r_floor1_cluster_mean, r_ceiling=r_ceiling,
                              delta_vs_cluster=boot["delta"], delta_se=boot["se"], q_beyond=q_beyond, label=label,
                              above_zero=above_zero, ratio_to_ceiling=ratio, exceeds_ceiling=r > r_ceiling,
                              r_within_cluster=stats.within_group_corr(pred, truth, clusters)))
    floors = {c[2:]: float(np.nanmean(rulers_tbl[c])) for c in rulers_tbl.columns if c.startswith("r_")}
    thresholds = np.round(np.arange(0.0, 0.31, 0.02), 2)
    result = dict(
        plan=name, scored=now(), n_cells=int(truth.shape[0]), n_scope=int(len(scope)),
        true_score_mean_r=float(np.nanmean(r)), median_r=float(np.nanmedian(r)),
        fraction_of_ceiling=float(np.nanmean(r) / floors["ceiling"]),
        mean_delta_vs_cluster_mean=float(np.nanmean(boot["delta"])),
        n_cannot_judge=int((label == 0).sum()), n_not_beyond_cell_type=int((label == 1).sum()),
        n_beyond_cell_type=int((label == 2).sum()), n_above_zero=int(above_zero.sum()),
        n_exceeding_ceiling=int(np.nansum(r > r_ceiling)), mean_r_within_cluster=float(np.nanmean(table.r_within_cluster)),
        beats={k: bool(np.nanmean(r) > v) for k, v in floors.items() if k != "ceiling"}, rulers_mean_r=floors,
        curve_delta_threshold=dict(threshold=thresholds.tolist(),
                                   n_genes=[int(np.nansum((boot["delta"] > t) & (label == 2))) for t in thresholds]),
        curve_fdr=dict(alpha=[0.01, 0.05, 0.10, 0.20],
                       n_genes=[int(np.nansum((q_beyond <= a) & (label != 0))) for a in (0.01, 0.05, 0.10, 0.20)]))
    out_dir = private / "true_scores" if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / f"{name}_per_gene.parquet")
    write_json(out_dir / f"{name}.json", result)
    print(f"{name}: mean_r={result['true_score_mean_r']:.4f} of ceiling={result['fraction_of_ceiling']:.3f} "
          f"beyond={result['n_beyond_cell_type']} not_beyond={result['n_not_beyond_cell_type']} "
          f"cannot_judge={result['n_cannot_judge']}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "rules", "rulers", "score", "score-rulers"])
    ap.add_argument("--pred"), ap.add_argument("--name")
    a = ap.parse_args(argv)
    if a.command == "score":
        score(a.pred, a.name)
    elif a.command == "score-rulers":
        private = paths.private_dir(TASK)
        for p in sorted(private.glob("pred_*.npy")):
            score(p, "ruler_" + p.stem[5:], out_dir=private / "ruler_scores")
    else:
        {"prepare": prepare, "rules": rules, "rulers": rulers}[a.command]()


if __name__ == "__main__":
    sys.exit(main())
