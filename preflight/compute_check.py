"""Agent4 preflight: confirm that the Python environment, the input data and each tool family named in
README section 6.5 actually run on this node.

This is a setup check on small subsamples with fixed settings. The numbers it prints show that the
code paths work; they are not analysis results and must not be quoted as such.
"""
import argparse
import json
import os
import platform
import time
import traceback
from pathlib import Path

ROOT = Path("/vast/som/yunligrp/users/ygao/BIOS774Agent")
PBMC = ROOT / "data/pbmc_granulocyte_sorted_10k"
PRE = ROOT / "Agent3_RepresentationLearning/outputs/tcga_brca/preprocess"
BANK = ROOT / "Agent3_RepresentationLearning/outputs/tcga_brca_hw4_feature_bank"
OUT = ROOT / "Agent4_ComplexGeneral/preflight/results"
ENCODERS = ["UNI2-h", "CONCH-Z", "Virchow2", "MUSK", "H-Optimus-1"]
CLASSES = ["LumA", "LumB", "Basal", "Her2"]
RESULTS, STATE = [], {}


def check(name):
    def run(fn):
        start = time.time()
        try:
            detail, status = fn(), "PASS"
        except Exception as exc:  # report and continue so one failure does not hide the rest
            detail, status = {"error": repr(exc)[:500], "trace": traceback.format_exc()[-1500:]}, "FAIL"
        seconds = round(time.time() - start, 1)
        RESULTS.append(dict(check=name, status=status, seconds=seconds, detail=detail))
        print(f"[{status}] {name} ({seconds}s) {json.dumps(detail, default=str)[:900]}", flush=True)
        return fn
    return run


def colcorr(pred, truth):
    """Pearson correlation of each column of pred with the same column of truth."""
    import numpy as np
    p, t = pred - pred.mean(0), truth - truth.mean(0)
    den = np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0))
    den[den == 0] = np.nan
    return (p * t).sum(0) / den


def lognorm(counts):
    import numpy as np
    from scipy import sparse
    lib = np.asarray(counts.sum(1)).ravel().astype(float)
    lib[lib == 0] = 1.0
    return (sparse.diags(1e4 / lib) @ counts).log1p()


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true", help="also run the GPU checks")
parser.add_argument("--scvi-cpu", action="store_true", help="run the MultiVI code path on CPU (no GPU needed)")
args = parser.parse_args()


@check("imports")
def _():
    import importlib.metadata as md
    names = ["numpy", "scipy", "pandas", "scikit-learn", "statsmodels", "scanpy", "anndata", "muon", "mudata",
             "scvi-tools", "torch", "lightning", "lifelines", "h5py", "pyarrow", "matplotlib", "seaborn",
             "weasyprint", "jinja2", "pydantic", "requests", "pyyaml", "pypdf", "umap-learn", "leidenalg", "networkx"]
    versions = {n: md.version(n) for n in names}
    import lifelines, muon, scanpy, scvi, sklearn, statsmodels.api, torch, weasyprint  # noqa: F401
    return versions


@check("pbmc_load")
def _():
    import h5py
    import numpy as np
    import pandas as pd
    from scipy import sparse
    dec = lambda v: np.array([x.decode() for x in v])
    with h5py.File(PBMC / "pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5", "r") as f:
        g = f["matrix"]
        m = sparse.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=g["shape"][:])
        barcodes, ftype = dec(g["barcodes"][:]), dec(g["features/feature_type"][:])
        fname = dec(g["features/name"][:])
    rna, atac = m[ftype == "Gene Expression"].T.tocsr(), m[ftype == "Peaks"].T.tocsr()
    labels = pd.read_csv(PBMC / "annotations/pbmc_10x_annotation_FixFlip.csv", index_col=0)
    ann = pd.read_csv(PBMC / "pbmc_granulocyte_sorted_10k_atac_peak_annotation.tsv", sep="\t")
    assert np.array_equal(labels.index.astype(str), barcodes), "label barcodes differ from matrix barcodes"
    peaks = pd.Series(fname[ftype == "Peaks"]).str.replace(r"[:\-]", "_", regex=True).values
    overlap = float(np.isin(peaks, ann["peak"].astype(str).unique()).mean())
    assert overlap > 0.9, f"only {overlap:.2f} of matrix peaks appear in the peak annotation"
    STATE.update(rna=rna, atac=atac, genes=fname[ftype == "Gene Expression"], peaks=peaks, labels=labels, ann=ann)
    return dict(cells=rna.shape[0], genes=rna.shape[1], peaks=atac.shape[1],
                labelled=int((labels.celltype != "filtered").sum()), cell_types=int(labels.celltype.nunique() - 1),
                smallest_type=int(labels.celltype[labels.celltype != "filtered"].value_counts().min()),
                peak_annotation_rows=len(ann), peaks_in_annotation=round(overlap, 4),
                peak_types=ann.peak_type.value_counts().to_dict())


@check("pbmc_tools")
def _():
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from sklearn.cluster import KMeans
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from statsmodels.stats.multitest import multipletests
    rng = np.random.default_rng(774)
    keep = np.flatnonzero((STATE["labels"].celltype != "filtered").values)
    idx = np.sort(rng.choice(keep, 3000, replace=False))
    counts, atac = STATE["rna"][idx].astype(np.int64), STATE["atac"][idx]
    # Binomial split of every RNA count into two independent halves.
    half_a = counts.copy()
    half_a.data = rng.binomial(counts.data, 0.5)
    half_b = counts - half_a
    assert (half_a + half_b != counts).nnz == 0 and half_b.data.min() >= 0
    genes = np.argsort(-np.asarray((counts > 0).sum(0)).ravel())[:500]
    y_full, y_a, y_b = (lognorm(x)[:, genes].toarray() for x in (counts, half_a, half_b))
    # ATAC representation: TF-IDF then truncated SVD (latent semantic indexing); drop the depth-like first component.
    top_peaks = np.argsort(-np.asarray((atac > 0).sum(0)).ravel())[:20000]
    tfidf = TfidfTransformer().fit_transform((atac[:, top_peaks] > 0).astype(np.float32))
    lsi = TruncatedSVD(31, random_state=774).fit_transform(tfidf)[:, 1:]
    depth = np.log1p(np.asarray(atac.sum(1)).ravel())[:, None]
    order = rng.permutation(3000)
    tr, te = order[:2100], order[2100:]
    r = {}
    r["upper_identity_A_to_B"] = colcorr(y_a[te], y_b[te])
    ridge_pred = Ridge(alpha=10.0).fit(lsi[tr], y_full[tr]).predict(lsi[te])
    r["ridge_lsi"] = colcorr(ridge_pred, y_b[te])
    r["knn_lsi"] = colcorr(KNeighborsRegressor(20).fit(lsi[tr], y_full[tr]).predict(lsi[te]), y_b[te])
    r["pls_lsi"] = colcorr(PLSRegression(10).fit(lsi[tr], y_full[tr]).predict(lsi[te]), y_b[te])
    km = KMeans(15, n_init=3, random_state=774).fit(lsi[tr])
    means = np.vstack([y_full[tr][km.labels_ == k].mean(0) for k in range(15)])
    r["cluster_mean"] = colcorr(means[km.predict(lsi[te])], y_b[te])
    r["depth_only"] = colcorr(LinearRegression().fit(depth[tr], y_full[tr]).predict(depth[te]), y_b[te])
    # Gene activity: sum the counts of every annotated peak of a gene (no learning).
    ann = STATE["ann"].assign(gene=STATE["ann"]["gene"].astype(str).str.split(";")).explode("gene")
    gene_pos = pd.Series(np.arange(len(genes)), index=STATE["genes"][genes])
    peak_pos = pd.Series(np.arange(len(STATE["peaks"])), index=STATE["peaks"])
    ann = ann[ann.gene.isin(gene_pos.index) & ann.peak.isin(peak_pos.index)].drop_duplicates(["peak", "gene"])
    link = sparse.csr_matrix((np.ones(len(ann), dtype=np.float32), (peak_pos[ann.peak].values, gene_pos[ann.gene].values)),
                             shape=(len(peak_pos), len(genes)))
    activity = np.log1p(np.asarray((atac @ link).todense(), dtype=float) / np.exp(depth) * 1e4)
    has_peak = np.asarray(link.sum(0)).ravel() > 0
    r["gene_activity"] = np.where(has_peak, colcorr(activity[te], y_b[te]), np.nan)
    # Null by permuting the cell correspondence, then Benjamini-Hochberg across genes.
    null = np.vstack([colcorr(ridge_pred[rng.permutation(len(te))], y_b[te]) for _ in range(200)])
    pvals = (1 + (null >= r["ridge_lsi"]).sum(0)) / (1 + len(null))
    significant = int(multipletests(pvals, alpha=0.05, method="fdr_bh")[0].sum())
    return dict(note="setup check on 3,000 cells and 500 genes; not a result",
                median_r={k: round(float(np.nanmedian(v)), 4) for k, v in r.items()},
                genes_with_annotated_peaks=int(has_peak.sum()), peak_gene_links=len(ann),
                ridge_genes_passing_fdr5=significant, null_r_95th=round(float(np.quantile(null, 0.95)), 4))


@check("tcga_load")
def _():
    import h5py
    import numpy as np
    import pandas as pd
    clin = pd.read_parquet(PRE / "clinical_cohort.parquet")
    cohort = clin[clin.label.isin(CLASSES)]
    emb = {}
    for enc in ENCODERS:
        with np.load(BANK / f"representations/{enc}/patient_embeddings.npz", allow_pickle=False) as z:
            frame = pd.DataFrame(z["embedding"], index=z["ids"].astype(str))
        emb[enc] = frame.loc[cohort.index].values
        assert np.isfinite(emb[enc]).all() and len(emb[enc]) == len(cohort)
    index = pd.read_parquet(BANK / "feature_index.parquet")
    tile_shapes = {}
    for enc in ENCODERS:
        with h5py.File(index[index.model == enc].path.iloc[0], "r") as f:
            tile_shapes[enc] = list(f["features"].shape)
    slides = index[index.model == "UNI2-h"].groupby("patient_id").size()
    STATE.update(clin=cohort, emb=emb, index=index)
    return dict(patients_with_slides=len(clin), cohort=len(cohort), classes=cohort.label.value_counts().to_dict(),
                sites_all=int(clin.site.nunique()), sites_cohort=int(cohort.site.nunique()),
                her2_values=cohort.her2.fillna("NA").value_counts().to_dict(),
                multi_slide_patients=int((slides > 1).sum()), dims={e: emb[e].shape[1] for e in ENCODERS},
                example_tile_shapes=tile_shapes, feature_files=len(index))


@check("tcga_tools")
def _():
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    clin, emb = STATE["clin"], STATE["emb"]
    y = clin.label.values
    cat, num = ["stage", "histological_type", "er", "pr", "her2"], ["age"]
    x_clin = clin[cat + num].copy()
    x_clin[cat] = x_clin[cat].fillna("missing").astype(str)
    encode = lambda: ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num)])
    cv = StratifiedKFold(5, shuffle=True, random_state=774)
    f1 = lambda p: round(float(f1_score(y, np.array(sorted(set(y)))[p.argmax(1)], average="macro")), 4)
    clinical = make_pipeline(encode(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    p_clin = cross_val_predict(clinical, x_clin, y, cv=cv, method="predict_proba")
    out = dict(note="setup check: one 5-fold split, fixed settings, no tuning; not a result", clinical_only=f1(p_clin))
    for enc in ENCODERS:
        image = make_pipeline(StandardScaler(), PCA(64, random_state=0),
                              LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1))
        p_img = cross_val_predict(image, emb[enc], y, cv=cv, method="predict_proba")
        out[enc] = dict(image_only=f1(p_img), late_fusion=f1((p_clin + p_img) / 2))
    boosted = make_pipeline(encode(), HistGradientBoostingClassifier(max_iter=60, class_weight="balanced", random_state=0))
    x_dense = encode().fit_transform(x_clin)
    x_dense = x_dense.toarray() if hasattr(x_dense, "toarray") else x_dense
    out["gradient_boosting_clinical"] = f1(cross_val_predict(
        HistGradientBoostingClassifier(max_iter=60, class_weight="balanced", random_state=0), x_dense, y, cv=cv,
        method="predict_proba"))
    del boosted
    site_cv = GroupKFold(5)
    out["clinical_only_site_held_out"] = f1(cross_val_predict(clinical, x_clin, y, cv=site_cv, groups=clin.site.values,
                                                              method="predict_proba"))
    return out


@check("pdf_report_toolchain")
def _():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pypdf import PdfReader
    from weasyprint import HTML
    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1, 2], [0.58, 0.65, 0.72], marker="o")
    ax.set_xlabel("candidate"), ax.set_ylabel("macro-F1")
    fig.tight_layout(), fig.savefig(tmp / "preflight.png", dpi=150), plt.close(fig)
    html = (f"<html><body style='font-family: DejaVu Sans, sans-serif'><h1>Agent4 preflight</h1>"
            f"<p>Gain 0.103 (95% CI -0.020 to 0.223); symbols: &ge; &plusmn; &alpha;.</p>"
            f"<img src='file://{tmp / 'preflight.png'}' width='320'>"
            f"<table border='1'><tr><th>arm</th><th>F1</th></tr><tr><td>clinical</td><td>0.620</td></tr></table></body></html>")
    HTML(string=html).write_pdf(tmp / "preflight.pdf")
    reader = PdfReader(tmp / "preflight.pdf")
    text = reader.pages[0].extract_text()
    assert "Agent4 preflight" in text and "0.103" in text, text[:200]
    return dict(pages=len(reader.pages), bytes=(tmp / "preflight.pdf").stat().st_size, images=len(reader.pages[0].images))


if args.gpu:
    @check("torch_gpu_attention_mil")
    def _():
        import h5py
        import numpy as np
        import torch
        assert torch.cuda.is_available(), "CUDA is not available to PyTorch"
        dev = torch.device("cuda")
        sub = STATE["index"][STATE["index"].model == "UNI2-h"].head(64)
        x, mask = torch.zeros(64, 128, 1536), torch.zeros(64, 128, dtype=torch.bool)
        for i, path in enumerate(sub.path):
            with h5py.File(path, "r") as f:
                a = f["features"][:]
            x[i, :len(a)], mask[i, :len(a)] = torch.from_numpy(a), True

        class AttentionMIL(torch.nn.Module):
            def __init__(self, dim=1536, hidden=256, classes=4):
                super().__init__()
                self.proj = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.ReLU())
                self.att_v, self.att_u = torch.nn.Linear(hidden, 128), torch.nn.Linear(hidden, 128)
                self.att_w, self.head = torch.nn.Linear(128, 1), torch.nn.Linear(hidden, classes)

            def forward(self, x, mask):
                h = self.proj(x)
                score = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1)
                weight = torch.softmax(score.masked_fill(~mask, float("-inf")), dim=1)
                return self.head((weight.unsqueeze(-1) * h).sum(1))

        torch.manual_seed(774)
        model, y = AttentionMIL().to(dev), torch.from_numpy(np.arange(64) % 4).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x, mask, losses = x.to(dev), mask.to(dev), []
        for _ in range(40):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = torch.nn.functional.cross_entropy(model(x, mask).float(), y)
            opt.zero_grad(), loss.backward(), opt.step()
            losses.append(float(loss))
        assert losses[-1] < losses[0], "loss did not decrease"
        return dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
                    total_mem_gb=round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1),
                    peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                    loss_first=round(losses[0], 4), loss_last=round(losses[-1], 4))

if args.gpu or args.scvi_cpu:
    @check("scvi_multivi_gpu" if args.gpu else "scvi_multivi_cpu")
    def _():
        import anndata as ad
        import mudata as md
        import numpy as np
        import pandas as pd
        import scvi
        from scipy import sparse
        rng = np.random.default_rng(774)
        keep = np.flatnonzero((STATE["labels"].celltype != "filtered").values)
        idx = np.sort(rng.choice(keep, 2000, replace=False))
        rna, atac = STATE["rna"][idx], STATE["atac"][idx]
        genes = np.argsort(-np.asarray((rna > 0).sum(0)).ravel())[:2000]
        peaks = np.argsort(-np.asarray((atac > 0).sum(0)).ravel())[:5000]
        rna, atac = rna[:, genes].astype(np.float32).tocsr(), (atac[:, peaks] > 0).astype(np.float32).tocsr()

        def mudata(r, a, ids):
            obs = pd.DataFrame(index=ids)
            out = md.MuData({
                "rna": ad.AnnData(r, obs=obs.copy(), var=pd.DataFrame(index=[f"gene_{i}" for i in range(r.shape[1])])),
                "atac": ad.AnnData(a, obs=obs.copy(), var=pd.DataFrame(index=[f"peak_{i}" for i in range(a.shape[1])]))})
            out.obs["_indices"] = np.arange(len(ids))
            return out

        ids = np.array([f"cell_{i}" for i in range(2000)])
        train = mudata(rna[:1800], atac[:1800], ids[:1800])
        scvi.settings.seed = 774
        scvi.model.MULTIVI.setup_mudata(train, modalities={"rna_layer": "rna", "atac_layer": "atac"})
        model = scvi.model.MULTIVI(train, n_genes=2000, n_regions=5000, n_latent=16, n_hidden=128,
                                   gene_likelihood="nb", fully_paired=False, modality_weights="equal")
        model.train(max_epochs=5, batch_size=128, accelerator="gpu" if args.gpu else "cpu", devices=1, early_stopping=False,
                    check_val_every_n_epoch=1, enable_progress_bar=False, enable_checkpointing=False)
        # Keep scvi's default logger: with logger=False model.history stays empty and convergence cannot be audited.
        history = {k: [round(float(x), 2) for x in v.iloc[:, 0].values] for k, v in (model.history or {}).items()
                   if k in ("reconstruction_loss_train", "reconstruction_loss_validation", "elbo_validation")}
        assert history, f"no training history recorded; keys={list((model.history or {}).keys())}"
        # The task M use case: predict expression for cells that supply ATAC only.
        atac_only = mudata(sparse.csr_matrix((200, 2000), dtype=np.float32), atac[1800:], ids[1800:])
        pred = model.get_normalized_expression(atac_only, return_numpy=True)
        latent = model.get_latent_representation(atac_only, modality="accessibility")
        truth = np.asarray(lognorm(rna[1800:]).todense())
        r = colcorr(np.log1p(np.asarray(pred) * 1e4), truth)
        assert np.isfinite(pred).all() and pred.shape == (200, 2000)
        return dict(note="5 epochs on 2,000 cells to prove the code path; far from converged, not a result",
                    scvi=scvi.__version__, pred_shape=list(pred.shape), latent_shape=list(latent.shape),
                    history=history, median_r_after_5_epochs=round(float(np.nanmedian(r)), 4))

summary = dict(host=platform.node(), job=os.environ.get("SLURM_JOB_ID"), partition=os.environ.get("SLURM_JOB_PARTITION"),
               gpu_requested=args.gpu, finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               passed=sum(r["status"] == "PASS" for r in RESULTS), failed=sum(r["status"] == "FAIL" for r in RESULTS),
               checks=RESULTS)
OUT.mkdir(parents=True, exist_ok=True)
target = OUT / f"compute_{'gpu' if args.gpu else 'scvi_cpu' if args.scvi_cpu else 'cpu'}_{summary['job'] or 'login'}.json"
target.write_text(json.dumps(summary, indent=1, default=str))
print(f"SUMMARY passed={summary['passed']} failed={summary['failed']} -> {target}")
raise SystemExit(1 if summary["failed"] else 0)
