"""Steps 1 of README 6.1: data profile and quantitative diagnostics.

Each diagnostic is a small function that declares which kind of view it applies to, measures something on
the data the agent may see, and returns a Difficulty with the numbers behind it. The set that runs is
decided by the view's adapter (what the data are), never by a dataset name.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from . import stats
from .schemas import DataProfile, Difficulty

_REGISTRY = []


def diagnostic(adapter):
    def wrap(fn):
        _REGISTRY.append((adapter, fn))
        return fn
    return wrap


def _sev(value, medium, high):
    return "high" if value >= high else "medium" if value >= medium else "low"


# ============================================================================= paired single cell
def _cell_cache(view):
    """Quantities several diagnostics share: a development split, read halves and a latent space."""
    if not hasattr(view, "_diag"):
        from .tools import m_basic as mb
        rng = np.random.default_rng(11)
        order = rng.permutation(view.dev_rna.shape[0])
        cut = int(0.7 * len(order))
        tr, te = np.sort(order[:cut]), np.sort(order[cut:])
        half_a, half_b = stats.thin_counts(view.dev_rna[te], seed=12)
        lsi = mb.LSI(seed=0, **view.rules["baseline_lsi"]).fit(view.dev_atac[tr])
        view._diag = dict(tr=tr, te=te, half_a=half_a, half_b=half_b, z_tr=lsi.transform(view.dev_atac[tr]),
                          z_te=lsi.transform(view.dev_atac[te]), y_tr=view.target(view.dev_rna[tr]),
                          y_te_b=view.target(half_b), y_te_a=view.target(half_a))
    return view._diag


@diagnostic("paired_single_cell")
def measurement_noise(view):
    c = _cell_cache(view)
    r = stats.colcorr(c["y_te_a"], c["y_te_b"])
    zeros = 1.0 - view.dev_rna.nnz / np.prod(view.dev_rna.shape)
    med = float(np.nanmedian(r))
    return Difficulty(
        id="", tag="measurement_noise", severity=_sev(1 - med, 0.5, 0.75),
        evidence=(f"RNA counts are {zeros:.1%} zeros (median {int(np.median(np.asarray(view.dev_rna.sum(1)).ravel()))} UMIs per cell). "
                  f"Splitting each cell's reads into two independent halves, a gene's value in one half correlates with "
                  f"the other half at a median r of {med:.3f} across {len(r)} in-scope genes (90th percentile {np.nanquantile(r, 0.9):.3f})."),
        numbers=dict(fraction_zero=zeros, split_half_r_median=med, split_half_r_q90=float(np.nanquantile(r, 0.9)),
                     split_half_r_q10=float(np.nanquantile(r, 0.1))),
        implication="Per-gene correlation with a single noisy reading is bounded well below 1; scores must be read against a noise ceiling, and validating against smoothed or full-depth targets overstates accuracy.")


@diagnostic("paired_single_cell")
def depth_confounding(view):
    from .tools import m_basic as mb
    c = _cell_cache(view)
    n_rna, n_atac = view.dev_obs["n_rna"].values, view.dev_obs["n_atac"].values
    rho = float(np.corrcoef(np.log1p(n_rna), np.log1p(n_atac))[0, 1])
    d_tr, d_te = mb.depth_feature(view.dev_atac[c["tr"]]), mb.depth_feature(view.dev_atac[c["te"]])
    r_depth = float(np.nanmean(stats.colcorr(mb.DepthOnly().fit(d_tr, c["y_tr"]).predict(d_te), c["y_te_b"])))
    return Difficulty(
        id="", tag="technical_confounding", severity=_sev(abs(rho), 0.3, 0.6),
        evidence=(f"ATAC depth and RNA depth are correlated across cells (r = {rho:.2f} on the log scale). A model that sees "
                  f"only ATAC depth reaches a mean per-gene r of {r_depth:.3f}."),
        numbers=dict(depth_correlation=rho, depth_only_mean_r=r_depth),
        implication="Report a depth-only baseline; a predictor must beat it before its signal is called biological.")


@diagnostic("paired_single_cell")
def strong_cell_type_baseline(view):
    from .tools import m_basic as mb
    c = _cell_cache(view)
    k = view.cluster_k
    cm = float(np.nanmean(stats.colcorr(mb.ClusterMean(k, seed=0).fit(c["z_tr"], c["y_tr"]).predict(c["z_te"]), c["y_te_b"])))
    ridge = float(np.nanmean(stats.colcorr(mb.LatentRidge(10.0).fit(c["z_tr"], c["y_tr"]).predict(c["z_te"]), c["y_te_b"])))
    ceiling = float(view.rules["ceiling_dev"]["mean_r_scope"])
    return Difficulty(
        id="", tag="strong_baseline", severity=_sev(cm / max(ridge, 1e-9), 0.8, 0.95),
        evidence=(f"Predicting each cell by the mean expression of its ATAC cluster (k = {k}, chosen by cross-validation inside "
                  f"the development set) gives a mean per-gene r of {cm:.3f}; ridge regression on the ATAC latent space gives "
                  f"{ridge:.3f}; the RNA-to-RNA noise ceiling is {ceiling:.3f}."),
        numbers=dict(cluster_k=k, cluster_mean_r=cm, ridge_r=ridge, ceiling_r=ceiling),
        implication="Most predictable variation is cell identity. A plan answers the question only through what it predicts beyond the cluster-mean baseline, gene by gene, on the same cells.")


@diagnostic("paired_single_cell")
def cell_class_imbalance(view):
    counts = view.dev_obs["celltype"].value_counts()
    return Difficulty(
        id="", tag="class_imbalance", severity=_sev(counts.max() / max(counts.min(), 1), 20, 100),
        evidence=(f"The provided annotation has {len(counts)} cell types in the development set, from {int(counts.max())} cells "
                  f"({counts.idxmax()}) down to {int(counts.min())} ({counts.idxmin()})."),
        numbers=dict(n_types=int(len(counts)), largest=int(counts.max()), smallest=int(counts.min())),
        implication="Rare populations contribute little to a pooled correlation and are easy to underfit; within-cluster correlation should be reported alongside the pooled one.")


@diagnostic("paired_single_cell")
def many_genes(view):
    g = len(view.scope)
    return Difficulty(
        id="", tag="multiple_comparisons", severity="high" if g > 1000 else "medium",
        evidence=f"{g} genes are in scope and each gets its own test.",
        numbers=dict(n_genes=g),
        implication="Gene-level claims need false-discovery-rate control across all in-scope genes.")


@diagnostic("paired_single_cell")
def power_cells(view):
    q95 = float(view.rules["ceiling_dev"]["null_q95"])
    n = int(view.meta["n_hidden"])
    return Difficulty(
        id="", tag="statistical_power", severity="low",
        evidence=(f"With thousands of cells a correlation as small as {q95:.3f} already exceeds 95% of the permutation null "
                  f"({n} held-out cells will be scored)."),
        numbers=dict(null_r_q95=q95, n_scored_cells=n),
        implication="Significance against zero is nearly automatic and cannot be the headline; effect size and the comparison with the cell-type baseline carry the conclusion.")


# ============================================================================= cohort with features
def _cohort_cache(view):
    if not hasattr(view, "_diag"):
        from .tools import t_basic as tb
        ids = view.labelled.index.values
        y = view.y(ids)
        folds = tb.make_folds(y, 5, 1, seed=21)[0]
        k = len(view.classes)
        clin, fused = np.zeros((len(ids), k)), np.zeros((len(ids), k))
        spec_c = dict(model=dict(component="clinical_only", params=dict(tune=False)))
        spec_f = dict(model=dict(component="patient_mean_logreg", params=dict(grid=[dict(encoders=[view.encoders[0]], n_pca=64, C=0.3)])))
        rc, rf = tb.ClinicalOnly(spec_c, view), tb.PatientMeanLogReg(spec_f, view)
        for tr, te in folds:
            clin[te], _ = rc.fit_predict(ids[tr], ids[te], rc.configs()[0], 0)
            fused[te], _ = rf.fit_predict(ids[tr], ids[te], rf.configs()[0], 0)
        view._diag = dict(ids=ids, y=y, clin=clin, fused=fused, folds=folds)
    return view._diag


@diagnostic("cohort_features")
def label_imbalance(view):
    counts = view.labelled["label"].value_counts()
    frac = counts.min() / counts.sum()
    return Difficulty(
        id="", tag="class_imbalance", severity=_sev(1 - frac * len(counts), 0.4, 0.65),
        evidence=(f"Class counts among {int(counts.sum())} labelled patients: " + ", ".join(f"{c} {int(n)}" for c, n in counts.items())
                  + f". The smallest class is {frac:.1%} of the cohort."),
        numbers=dict(counts={c: int(n) for c, n in counts.items()}, smallest_fraction=float(frac)),
        implication="Use a class-balanced metric and class weighting; an unweighted model can score well while missing the smallest class entirely.")


@diagnostic("cohort_features")
def covariate_missingness(view):
    frame = view.labelled
    rows = {}
    for c in ("er", "pr", "her2", "stage", "histological_type", "age"):
        missing = float(frame[c].isna().mean())
        vague = float(frame[c].isin(["equivocal", "indeterminate"]).mean()) if frame[c].dtype == object else 0.0
        rows[c] = dict(missing=missing, not_informative=missing + vague)
    worst = max(rows, key=lambda c: rows[c]["not_informative"])
    return Difficulty(
        id="", tag="missingness", severity=_sev(rows[worst]["not_informative"], 0.1, 0.25),
        evidence=("Share of labelled patients without a definite value: " +
                  ", ".join(f"{c} {v['not_informative']:.1%} (missing {v['missing']:.1%})" for c, v in rows.items()) + "."),
        numbers=rows,
        implication="Handle missing values inside each training fold with an explicit missing level, identically in both comparison arms.")


@diagnostic("cohort_features")
def nested_units(view):
    per_patient = view.tile_index[view.tile_index.patient_id.isin(view.labelled.index)].groupby("patient_id").size()
    multi = int((per_patient > 1).sum())
    tiles = view.tile_index.assign(n=view.tile_index.end - view.tile_index.start).groupby("patient_id").n.sum()
    return Difficulty(
        id="", tag="hierarchy", severity="medium" if multi else "low",
        evidence=(f"{multi} labelled patients have more than one slide; tiles per patient range from {int(tiles.min())} to "
                  f"{int(tiles.max())} (median {int(tiles.median())})."),
        numbers=dict(multi_slide_patients=multi, tiles_min=int(tiles.min()), tiles_median=int(tiles.median()), tiles_max=int(tiles.max())),
        implication="The patient is the unit of analysis: all slides and tiles of a patient must stay on the same side of every split.")


@diagnostic("cohort_features")
def dimensionality(view):
    n = len(view.labelled)
    dims = {e: int(view.patient_features(e, view.labelled.index[:1]).shape[1]) for e in view.encoders}
    return Difficulty(
        id="", tag="small_n_large_p", severity="high",
        evidence=f"{n} labelled patients against image features of " + ", ".join(f"{d} ({e})" for e, d in dims.items()) + " dimensions per encoder.",
        numbers=dict(n=n, dims=dims),
        implication="High-capacity image models overfit easily; regularise, reduce dimension inside folds, and require evidence of convergence and of a gain over a simple reference.")


@diagnostic("cohort_features")
def site_confounding(view):
    frame = view.labelled
    sites = frame["site"].value_counts()
    keep = frame[frame["site"].isin(sites[sites >= 15].index)]
    x = StandardScaler().fit_transform(view.patient_features(view.encoders[0], keep.index))
    x = PCA(32, random_state=0).fit_transform(x)
    pred = cross_val_predict(LogisticRegression(max_iter=2000), x, keep["site"].values, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    acc, chance = float(np.mean(pred == keep["site"].values)), float(sites.max() / len(keep))
    table = pd.crosstab(frame["site"], frame["label"])
    chi2 = float(((table - np.outer(table.sum(1), table.sum(0)) / table.values.sum()) ** 2 / (np.outer(table.sum(1), table.sum(0)) / table.values.sum())).values.sum())
    cramers_v = float(np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1))))
    return Difficulty(
        id="", tag="technical_confounding", severity=_sev(acc - chance, 0.15, 0.35),
        evidence=(f"Labelled patients come from {len(sites)} collection sites (largest {int(sites.max())} patients, "
                  f"{int((sites < 10).sum())} sites with fewer than 10). Image features ({view.encoders[0]}) identify the site of a "
                  f"patient with accuracy {acc:.2f} against {chance:.2f} for always guessing the largest site; site and class "
                  f"are associated (Cramer's V = {cramers_v:.2f})."),
        numbers=dict(n_sites=int(len(sites)), site_accuracy=acc, majority_rate=chance, cramers_v=cramers_v),
        implication="Part of an image-based gain may come from site signatures; a site-held-out sensitivity analysis should accompany the main estimate.")


@diagnostic("cohort_features")
def strong_clinical_baseline(view):
    c = _cohort_cache(view)
    k = len(view.classes)
    f1 = stats.macro_f1(c["y"], c["clin"], k)
    recall = dict(zip(view.classes, stats.per_class_recall(c["y"], c["clin"], k)))
    weakest = min(recall, key=recall.get)
    return Difficulty(
        id="", tag="strong_baseline", severity=_sev(f1, 0.45, 0.6),
        evidence=(f"Clinical variables alone reach macro-F1 {f1:.3f} in 5-fold cross-validation; recall by class: " +
                  ", ".join(f"{cl} {r:.2f}" for cl, r in recall.items()) + f". The weakest class is {weakest}."),
        numbers=dict(clinical_macro_f1=f1, recall=recall),
        implication="The question is about what images add to this baseline, so the clinical arm must be tuned as carefully as the image arm and evaluated on the same patients.")


@diagnostic("cohort_features")
def power_patients(view):
    c = _cohort_cache(view)
    boot = stats.paired_gain_bootstrap(c["y"], [c["fused"]], [c["clin"]], n_boot=1000, seed=0)
    mdd = 2.8 * boot["se"]
    return Difficulty(
        id="", tag="statistical_power", severity=_sev(mdd, 0.03, 0.06),
        evidence=(f"For a simple image-plus-clinical model against clinical only, the paired macro-F1 difference has a bootstrap "
                  f"standard error of {boot['se']:.3f} with one 5-fold split of {len(c['y'])} patients. The smallest difference "
                  f"detectable with 80% power is about {mdd:.3f}."),
        numbers=dict(paired_se=boot["se"], minimal_detectable_difference=mdd, observed_gain_simple=boot["gain"]),
        implication="Gains of a few points cannot be established from one split; use repeated cross-validation and report an interval, not only a point estimate.")


@diagnostic("cohort_features")
def many_model_choices(view):
    return Difficulty(
        id="", tag="multiple_comparisons", severity="medium",
        evidence=f"{len(view.encoders)} image encoders are available, each usable at tile or patient level with several fusion strategies.",
        numbers=dict(n_encoders=len(view.encoders)),
        implication="Choosing the best of many configurations on the evaluation folds inflates the reported gain; selection must be nested or counted.")


# ============================================================================= entry point
def _summary(view):
    if view.adapter == "paired_single_cell":
        return dict(
            n_development_units=int(view.dev_rna.shape[0]), n_held_out_units=int(view.hidden_atac.shape[0]),
            modalities=dict(
                rna=dict(features=int(view.dev_rna.shape[1]), available_for="development cells only",
                         fraction_zero=float(1 - view.dev_rna.nnz / np.prod(view.dev_rna.shape)),
                         median_counts_per_cell=float(np.median(view.dev_obs["n_rna"]))),
                atac=dict(features=int(view.dev_atac.shape[1]), available_for="all cells",
                          fraction_zero=float(1 - view.dev_atac.nnz / np.prod(view.dev_atac.shape)),
                          median_counts_per_cell=float(np.median(view.dev_obs["n_atac"])))),
            pairing="both modalities measured in the same cell for every development cell",
            genes_in_scope=int(len(view.scope)), peak_gene_links=int(len(view.peak_gene)),
            cell_type_baseline_k=view.cluster_k)
    frame = view.labelled
    return dict(
        n_units_with_images=int(len(view.clinical)), n_labelled=int(len(frame)), n_unlabelled=int(len(view.clinical) - len(frame)),
        classes=view.classes, n_slides=int(len(view.tile_index)), n_tiles=int(view.meta["n_tiles"]),
        modalities=dict(
            clinical=dict(variables=["age", "stage", "histological_type", "er", "pr", "her2"]),
            image=dict(encoders={e: int(view.patient_features(e, frame.index[:1]).shape[1]) for e in view.encoders},
                       levels=["tile (up to 128 per slide)", "patient mean"])),
        grouping=dict(column="site", n_groups=int(frame["site"].nunique())))


def profile_and_diagnose(view, task_cfg) -> DataProfile:
    found = []
    for adapter, fn in _REGISTRY:
        if adapter == view.adapter:
            found.append(fn(view))
    # A difficulty that is documented rather than measured: where the labels come from.
    prov = task_cfg["dataset_card"].get("label_provenance")
    if prov:
        found.append(Difficulty(id="", tag="label_provenance", severity="high", evidence=prov["statement"],
                                numbers={}, implication=prov["implication"]))
    order = {"high": 0, "medium": 1, "low": 2}
    found.sort(key=lambda d: order[d.severity])
    for i, d in enumerate(found, 1):
        d.id = f"D{i}"
    return DataProfile(task_id=task_cfg["task_id"], adapter=view.adapter, unit=task_cfg["dataset_card"]["unit"],
                       summary=_summary(view), dataset_card=task_cfg["dataset_card"], difficulties=found)
