"""Executable models for cross-modal expression prediction (README 6.5, task M).

    model = build(spec, view, scale="full", seed=0, genes=None, time_budget=None)
    report = model.fit(atac, rna, val=None)      # CSR counts of the training cells
    pred = model.predict(atac)                   # (cells x modelled genes), log-normalised scale of view.target
    model.resolved, model.ignored                # hyper-parameters actually used / plan parameters not understood

Every transform (LSI, scalers, clusterings, peak selection) is fitted on the training cells passed to fit().
Iterative models carve an internal validation part (about 12% of the training cells) unless one is given,
record their training curves and state whether training converged. A pilot never shortens training; the
caller shrinks the data instead.
"""
from __future__ import annotations

import copy
import os
import re
import time

import numpy as np
from scipy import sparse

from ..stats import colcorr, lognorm
from . import m_basic as mb

THREADS = int(os.environ.get("A4_THREADS", 8))
INPUT_TYPES = ("lsi", "tfidf_peaks", "gene_activity", "cis_peaks", "depth")
VAL_FRACTION = 0.12


# ============================================================================= parameter handling
def _cast(v, kind):
    if kind is None:
        return v
    if kind == "int":
        return int(round(float(v)))
    if kind == "float":
        return float(v)
    if kind == "ints":
        return [int(round(float(v)))] if isinstance(v, (int, float, str)) else [int(round(float(x))) for x in v]
    if kind == "str":
        return str(v).strip().lower()
    if kind == "bool":
        return v.strip().lower() in ("true", "yes", "1") if isinstance(v, str) else bool(v)
    raise ValueError(kind)


class Params:
    """Dictionary of plan parameters that remembers which keys were understood."""

    def __init__(self, d, flatten=False):
        self.d = {}
        for k, v in (d.items() if isinstance(d, dict) else []):
            self.d[str(k)] = v
            if flatten and isinstance(v, dict):          # {"early_stopping": {"patience": 10}} -> early_stopping_patience, patience
                for kk, vv in v.items():
                    self.d.setdefault(f"{k}_{kk}", vv)
                    self.d.setdefault(str(kk), vv)
        self.used = set()

    def get(self, name, default=None, kind=None, aliases=()):
        for n in (name, *aliases):
            if n in self.d and self.d[n] is not None:
                try:
                    value = _cast(self.d[n], kind)
                except (TypeError, ValueError):
                    continue
                self.used.add(n)
                return value
        return default

    def unused(self, prefix=""):
        nested = {k for k, v in self.d.items() if isinstance(v, dict)}
        derived = {f"{k}_{kk}" for k in nested for kk in self.d[k]} | {str(kk) for k in nested for kk in self.d[k]}
        return sorted(prefix + k for k in self.d if k not in self.used and k not in derived and k not in nested)


INPUT_ALIASES = {"tfidf": "tfidf_peaks", "tf_idf": "tfidf_peaks", "tf_idf_peaks": "tfidf_peaks", "peaks_tfidf": "tfidf_peaks",
                 "latent_semantic_indexing": "lsi", "gene_activity_score": "gene_activity", "gene_activities": "gene_activity",
                 "cis": "cis_peaks", "cis_window": "cis_peaks", "cis_peak_window": "cis_peaks",
                 "atac_depth": "depth", "log_depth": "depth", "sequencing_depth": "depth"}


def _input_type(name):
    """Canonical input type for a name used in a plan, or None."""
    if not isinstance(name, str):
        return None
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    key = INPUT_ALIASES.get(key, key)
    return key if key in INPUT_TYPES else None


def parse_inputs(inputs):
    """Find input representations anywhere in the plan's `inputs` field. Returns {type: params}."""
    found = {}

    def walk(o):
        if isinstance(o, dict):
            t = _input_type(o.get("type"))
            if t:
                found.setdefault(t, {}).update({k: v for k, v in o.items() if k != "type"})
            for k, v in o.items():
                kt = _input_type(k)
                if kt and k != "type":
                    if isinstance(v, dict):
                        found.setdefault(kt, {}).update(v)
                    elif v not in (False, None):
                        found.setdefault(kt, {})
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif _input_type(o):
            found.setdefault(_input_type(o), {})

    walk(inputs)
    return found


# ============================================================================= input representations
class TfidfPeaks:
    """Sparse TF-IDF over the most accessible peaks of the training cells."""

    def __init__(self, max_peaks=30000):
        self.max_peaks = int(max_peaks)

    def fit(self, atac):
        binary = (atac > 0).astype(np.float32)
        detected = np.asarray(binary.sum(0)).ravel()
        self.peaks_ = np.sort(np.argsort(-detected)[: self.max_peaks])
        self.idf_ = np.log1p(binary.shape[0] / (1.0 + detected[self.peaks_])).astype(np.float32)
        return self

    def transform(self, atac):
        binary = (atac[:, self.peaks_] > 0).astype(np.float32)
        tf = sparse.diags(1.0 / np.maximum(np.asarray(binary.sum(1)).ravel(), 1.0)) @ binary
        return (tf @ sparse.diags(self.idf_ * 1e4)).log1p().tocsr().astype(np.float32)


def cis_links(view, gene_idx, window_kb=100.0, max_peaks_per_gene=20):
    """Peaks within the window around each gene's start site on the same chromosome, nearest first.

    Returns (gene_pos, peak_idx): gene_pos indexes into gene_idx, peak_idx into the view's peaks.
    """
    genes = view.genes.iloc[np.asarray(gene_idx)]
    g_chrom, g_tss = genes["chrom"].astype(str).values, genes["tss"].astype(float).values
    p_chrom = view.peaks["chrom"].astype(str).values
    p_center = ((view.peaks["start"].values + view.peaks["end"].values) / 2.0).astype(float)
    half = float(window_kb) * 1000.0
    rows, cols = [], []
    for chrom in np.unique(g_chrom):
        pidx = np.flatnonzero(p_chrom == chrom)
        if not len(pidx):
            continue
        pidx = pidx[np.argsort(p_center[pidx])]
        centers = p_center[pidx]
        for gpos in np.flatnonzero(g_chrom == chrom):
            tss = g_tss[gpos]
            if not np.isfinite(tss):
                continue
            lo, hi = np.searchsorted(centers, tss - half, "left"), np.searchsorted(centers, tss + half, "right")
            cand = pidx[lo:hi]
            if len(cand) > max_peaks_per_gene:
                cand = cand[np.argsort(np.abs(centers[lo:hi] - tss))[: int(max_peaks_per_gene)]]
            rows.append(np.full(len(cand), gpos, dtype=np.int64))
            cols.append(cand.astype(np.int64))
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(rows), np.concatenate(cols)


class Features:
    """Builds the inputs a plan asked for. Fitted on training cells only."""

    def __init__(self, view, gene_idx, want, seed=0):
        self.view, self.gene_idx, self.seed = view, np.asarray(gene_idx), seed
        self.want = {k: Params(v) for k, v in want.items()}
        self.resolved = {}

    def fit(self, atac):
        n, p = atac.shape
        if "lsi" in self.want:
            w = self.want["lsi"]
            nc = min(w.get("n_components", 50, "int"), n - 2, p - 2)
            self.lsi = mb.LSI(n_components=nc, max_peaks=w.get("max_peaks", 60000, "int"), seed=self.seed).fit(atac)
            self.resolved["lsi"] = dict(n_components=nc, max_peaks=self.lsi.max_peaks)
        if "tfidf_peaks" in self.want:
            self.tfidf = TfidfPeaks(self.want["tfidf_peaks"].get("max_peaks", 30000, "int")).fit(atac)
            self.resolved["tfidf_peaks"] = dict(max_peaks=len(self.tfidf.peaks_))
        if "gene_activity" in self.want:
            mode = self.want["gene_activity"].get("mode", "promoter_and_distal", "str")
            self.ga_mode = mode if mode in ("promoter", "promoter_and_distal") else "promoter_and_distal"
            act, self.ga_has = mb.gene_activity(atac, self.view.peak_gene, self.gene_idx, p, self.ga_mode)
            self.ga_scale = float(act.std()) or 1.0
            self.resolved["gene_activity"] = dict(mode=self.ga_mode, genes_with_peaks=int(self.ga_has.sum()))
        if "depth" in self.want:
            d = mb.depth_feature(atac)
            self.depth_mu, self.depth_sd = float(d.mean()), float(d.std()) or 1.0
            self.resolved["depth"] = {}
        if "cis_peaks" in self.want:
            w = self.want["cis_peaks"]
            window, cap = w.get("window_kb", 100.0, "float"), w.get("max_peaks_per_gene", 20, "int")
            gpos, pidx = cis_links(self.view, self.gene_idx, window, cap)
            self.cis_union, local = np.unique(pidx, return_inverse=True)
            self.cis_gene, self.cis_local = gpos, local
            raw = self._cis_raw(atac)
            self.cis_scale = float(raw.data.mean()) if raw.nnz else 1.0
            self.resolved["cis_peaks"] = dict(window_kb=window, max_peaks_per_gene=cap, n_links=int(len(gpos)),
                                              n_peaks=int(len(self.cis_union)),
                                              genes_with_cis_peaks=int(len(np.unique(gpos))))
        return self

    def _cis_raw(self, atac):
        depth = np.maximum(np.asarray(atac.sum(1)).ravel(), 1.0)
        sub = atac[:, self.cis_union].astype(np.float32)
        return (sparse.diags((1e4 / depth).astype(np.float32)) @ sub).log1p().tocsr()

    def dense(self, atac):
        parts = []
        if "lsi" in self.want:
            parts.append(self.lsi.transform(atac))
        if "gene_activity" in self.want:
            act, _ = mb.gene_activity(atac, self.view.peak_gene, self.gene_idx, atac.shape[1], self.ga_mode)
            parts.append(act / self.ga_scale)
        if "depth" in self.want:
            parts.append((mb.depth_feature(atac) - self.depth_mu) / self.depth_sd)
        return np.hstack(parts).astype(np.float32) if parts else None

    def own_activity(self, atac):
        """Gene activity of each modelled gene, as an extra per-gene feature for per-gene learners."""
        if "gene_activity" not in self.want:
            return None
        act, _ = mb.gene_activity(atac, self.view.peak_gene, self.gene_idx, atac.shape[1], self.ga_mode)
        return (act / self.ga_scale).astype(np.float32)

    def sparse(self, atac):
        return self.tfidf.transform(atac) if "tfidf_peaks" in self.want else None

    def cis(self, atac):
        return (self._cis_raw(atac) / self.cis_scale).tocsr().astype(np.float32) if "cis_peaks" in self.want else None

    def unused(self):
        return [u for k, w in self.want.items() for u in w.unused(f"inputs.{k}.")]


# ============================================================================= shared helpers
def _carve(n, seed, fraction=VAL_FRACTION):
    order = np.random.default_rng(seed + 9173).permutation(n)
    cut = max(1, int(round(fraction * n)))
    return np.sort(order[cut:]), np.sort(order[:cut])


def _convergence(curve, stopped_by, lower_is_better=True, what="validation loss"):
    """README 6.3: converged if early stopping fired, or the last fifth of the monitored curve improved by under 1%."""
    curve = [float(c) for c in curve if c is not None and np.isfinite(c)]
    if stopped_by == "time_budget":
        return False, "Training was cut by the time budget, so convergence cannot be claimed."
    if stopped_by == "early_stopping":
        return True, f"Early stopping fired after {len(curve)} epochs: the {what} had stopped improving."
    if len(curve) < 5:
        return False, f"Only {len(curve)} epochs were run; too few to judge convergence."
    k = max(1, int(round(0.2 * len(curve))))
    sign = 1.0 if lower_is_better else -1.0
    before, after = min(sign * np.array(curve[:-k])), min(sign * np.array(curve[-k:]))
    rel = (before - after) / max(abs(before), 1e-12)
    if rel < 0.01:
        return True, f"Reached max_epochs; the {what} improved by {max(rel, 0):.2%} over the last {k} epochs (under 1%)."
    return False, f"Reached max_epochs while the {what} was still improving ({rel:.2%} over the last {k} epochs)."


def _denoise_spec(train: Params):
    raw = train.get("target_denoising", None, aliases=("denoising", "target_denoise"))
    if raw is None or raw is False:
        return 0
    if isinstance(raw, dict):
        kind = str(raw.get("type", raw.get("method", "knn_pool"))).lower()
        return int(raw.get("k", 10)) if "knn" in kind else 0
    text = str(raw).lower()
    if "knn" not in text and "pool" not in text:
        return 0
    digits = re.findall(r"\d+", text)
    return int(digits[0]) if digits else int(train.get("k", 10, "int", aliases=("denoise_k", "knn_k", "target_denoising_k")))


def _knn_pool(z, rna, k):
    """Sum the counts of each training cell with those of its k nearest training cells in latent space."""
    from sklearn.neighbors import NearestNeighbors
    k = min(int(k), len(z) - 1)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(z, return_distance=False)
    rows = np.repeat(np.arange(len(z)), k + 1)
    adj = sparse.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, idx.ravel())), shape=(len(z), len(z)))
    return (adj @ rna.astype(np.float32)).tocsr()


class MModel:
    """Common fit / predict wrapper. Subclasses implement _fit and _predict."""

    iterative = True

    def __init__(self, spec, view, scale="full", seed=0, genes=None, time_budget=None):
        self.spec, self.view, self.scale, self.seed = spec, view, scale, int(seed)
        self.component = spec["model"]["component"]
        self.gpos = np.arange(len(view.scope)) if genes is None else np.asarray(genes, dtype=int)
        self.gene_idx = np.asarray(view.scope)[self.gpos]
        self.p = Params(spec.get("model", {}).get("params", {}))
        self.t = Params(spec.get("training", {}), flatten=True)
        self.inputs = parse_inputs(spec.get("inputs", {}))
        self.time_budget = time_budget
        self.resolved, self.extra_ignored = dict(component=self.component, seed=self.seed, scale=scale,
                                                 n_genes=int(len(self.gene_idx))), []

    def target(self, rna):
        return self.view.target(rna, self.gene_idx)

    @property
    def ignored(self):
        out = self.p.unused("model.params.") + self.extra_ignored
        if self.iterative:
            out += self.t.unused("training.")
        if hasattr(self, "feats"):
            out += self.feats.unused()
        return sorted(set(out))

    def fit(self, atac, rna, val=None):
        import torch
        torch.set_num_threads(THREADS)
        t0 = time.time()
        self.deadline = t0 + self.time_budget if self.time_budget else None
        atac, rna = sparse.csr_matrix(atac), sparse.csr_matrix(rna)
        if val is None and self.iterative:
            tr, va = _carve(atac.shape[0], self.seed)
            val = (atac[va], rna[va])
            atac, rna = atac[tr], rna[tr]
        report = dict(device="cpu", epochs=0, stopped_by="closed_form", converged=True,
                      convergence_note="Closed-form or non-iterative fit; nothing to converge.", curves={}, n_parameters=0)
        report.update(self._fit(atac, rna, val) or {})
        report.update(seconds=round(time.time() - t0, 2), n_train_cells=int(atac.shape[0]),
                      n_val_cells=int(val[0].shape[0]) if val is not None else 0, n_genes=int(len(self.gene_idx)))
        self.report = report
        return report

    def predict(self, atac):
        out = np.asarray(self._predict(sparse.csr_matrix(atac)), dtype=np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=20.0, neginf=0.0)

    def out_of_time(self):
        return self.deadline is not None and time.time() > self.deadline


# ============================================================================= basic tier
class LatentModel(MModel):
    """cluster_mean, latent_ridge, latent_knn, latent_pls on LSI components."""

    iterative = False

    def _fit(self, atac, rna, val):
        frozen = self.component == "cluster_mean" and str(self.p.get("k", "frozen")).lower() == "frozen"
        w = Params(self.inputs.get("lsi", {}))
        base = self.view.rules.get("baseline_lsi", dict(n_components=50, max_peaks=60000))
        nc = base["n_components"] if frozen else w.get("n_components", base["n_components"], "int")
        mp = base["max_peaks"] if frozen else w.get("max_peaks", base["max_peaks"], "int")
        nc = min(nc, atac.shape[0] - 2, atac.shape[1] - 2)
        self.lsi = mb.LSI(n_components=nc, max_peaks=mp, seed=self.seed).fit(atac)
        z, y = self.lsi.transform(atac), self.target(rna)
        if self.component == "cluster_mean":
            k = self.view.cluster_k if frozen else self.p.get("k", self.view.cluster_k, "int")
            self.m = mb.ClusterMean(min(k, len(z)), seed=self.seed).fit(z, y)
            hp = dict(k=int(min(k, len(z))), frozen=frozen)
        elif self.component == "latent_ridge":
            hp = dict(alpha=self.p.get("alpha", 10.0, "float"))
            self.m = mb.LatentRidge(hp["alpha"]).fit(z, y)
        elif self.component == "latent_knn":
            hp = dict(k=min(self.p.get("k", 20, "int"), len(z)))
            self.m = mb.LatentKNN(hp["k"]).fit(z, y)
        else:
            hp = dict(n_components=self.p.get("n_components", 15, "int"))
            self.m = mb.LatentPLS(hp["n_components"]).fit(z, y)
        self.resolved.update(lsi=dict(n_components=nc, max_peaks=mp), **hp)
        self.extra_ignored += [f"inputs.{k}" for k in self.inputs if k != "lsi"]
        return {}

    def _predict(self, atac):
        return self.m.predict(self.lsi.transform(atac))


class GeneActivityScore(MModel):
    """Ruler floor 2: the gene activity score itself is the prediction."""

    iterative = False

    def _fit(self, atac, rna, val):
        mode = self.p.get("mode", Params(self.inputs.get("gene_activity", {})).get("mode", "promoter_and_distal", "str"), "str")
        self.mode = mode if mode in ("promoter", "promoter_and_distal") else "promoter_and_distal"
        self.resolved.update(mode=self.mode)
        return dict(convergence_note="No learning: the gene activity score is used as the prediction.")

    def _predict(self, atac):
        act, has = mb.gene_activity(atac, self.view.peak_gene, self.gene_idx, atac.shape[1], self.mode)
        act[:, ~has] = 0.0
        return act


# ============================================================================= neural network
def _torch():
    import torch
    return torch


def _device():
    torch = _torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_net(d_in, hidden, latent, n_genes, decoder, dec_hidden, dropout, factor_dim=None, cis=None, input_dropout=0.0):
    torch = _torch()
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            layers, prev = ([nn.Dropout(input_dropout)] if input_dropout > 0 else []), d_in
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
                prev = h
            self.encoder = nn.Sequential(*layers)
            self.to_latent = nn.Linear(prev, latent)
            if decoder == "mlp":
                self.decoder = nn.Sequential(nn.Linear(latent, dec_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(dec_hidden, n_genes))
            elif decoder == "factorized":        # latent -> gene-embedding space -> genes (shared low-rank structure)
                self.decoder = nn.Sequential(nn.Linear(latent, factor_dim, bias=False), nn.Linear(factor_dim, n_genes))
            else:
                self.decoder = nn.Linear(latent, n_genes)
            self.has_cis = cis is not None
            if self.has_cis:
                gene, local, n_cis = cis
                self.register_buffer("cis_gene", torch.as_tensor(np.asarray(gene), dtype=torch.long))
                self.register_buffer("cis_local", torch.as_tensor(np.asarray(local), dtype=torch.long))
                self.cis_w = nn.Parameter(torch.zeros(len(gene)))

        @property
        def out_layer(self):
            return self.decoder[-1] if isinstance(self.decoder, nn.Sequential) else self.decoder

        def embed(self, x):
            return self.to_latent(self.encoder(x))

        def forward(self, x, x_cis=None):
            out = self.decoder(self.embed(x))
            if self.has_cis:
                # Masked linear layer over peak-gene links: gather each link's peak value, weigh it, add it to its gene.
                # (torch.sparse.mm is avoided: its gradient for the sparse factor materialises a dense genes x peaks matrix.)
                out = out.index_add(1, self.cis_gene, x_cis.index_select(1, self.cis_local) * self.cis_w)
            return out

    return Net()


class _Batcher:
    """Turns row indices into device tensors; sparse blocks are densified per batch."""

    def __init__(self, dense, sp, cis, device):
        self.dense, self.sp, self.cis, self.device = dense, sp, cis, device
        self.n = (dense if dense is not None else sp if sp is not None else cis).shape[0]
        self.d_in = (dense.shape[1] if dense is not None else 0) + (sp.shape[1] if sp is not None else 0)

    def x(self, idx):
        torch = _torch()
        parts = []
        if self.dense is not None:
            parts.append(self.dense[idx])
        if self.sp is not None:
            parts.append(np.asarray(self.sp[idx].todense(), dtype=np.float32))
        x = torch.from_numpy(np.ascontiguousarray(np.hstack(parts) if len(parts) > 1 else parts[0])).to(self.device)
        xc = None
        if self.cis is not None:
            xc = torch.from_numpy(np.asarray(self.cis[idx].todense(), dtype=np.float32)).to(self.device)
        return x, xc


def _loss_and_pred(out, target, likelihood, base=None, lib=None, log_theta=None, scale=None):
    """Loss against the batch target and the prediction on the log-normalised scale.

    MSE: the network works in per-gene standardised units (scale = (mean, sd) of the training target), so every
    gene weighs the same in the loss, as it does in the per-gene correlation the plans are scored with.
    """
    torch = _torch()
    if likelihood == "mse":
        mu, sd = scale
        pred = mu + sd * out
        if base is not None:
            pred = pred + base
        return torch.mean(((pred - target) / sd) ** 2), pred
    o = torch.clamp(out, -18.0, 12.0)                       # log of the normalised rate per 10,000 counts
    mu = lib[:, None] / 1e4 * torch.exp(o) + 1e-8
    if likelihood == "poisson":
        loss = torch.mean(mu - target * torch.log(mu))
    else:                                                   # negative binomial with a per-gene inverse dispersion
        theta = torch.exp(torch.clamp(log_theta, -6.0, 8.0))[None, :]
        loss = -torch.mean(torch.lgamma(target + theta) - torch.lgamma(theta) - torch.lgamma(target + 1.0)
                           + theta * torch.log(theta / (theta + mu)) + target * torch.log(mu / (theta + mu)))
    return loss, torch.log1p(torch.exp(o))


class NetModel(MModel):
    """One configurable network behind translation_network, residual_over_cluster_mean (mlp), multitask_lowrank,
    cis_regulatory (masked_nn) and contrastive_pretrain_head."""

    def _config(self):
        c, p = self.component, self.p
        cfg = dict(residual=False, cis=False, contrastive=False, head="mlp", knn_k=30, factor_dim=None, zero_init=False)
        if c == "translation_network":
            cfg.update(default_inputs={"tfidf_peaks": {"max_peaks": 30000}}, hidden=p.get("encoder_hidden", [1024, 256], "ints", ("hidden",)),
                       latent=p.get("latent_dim", 64, "int"), decoder=p.get("decoder", "mlp", "str"),
                       likelihood=p.get("likelihood", "poisson", "str"), dropout=p.get("dropout", 0.1, "float"))
        elif c == "residual_over_cluster_mean":
            cfg.update(default_inputs={"lsi": {"n_components": 50}, "depth": {}}, hidden=p.get("hidden", [512, 256], "ints"),
                       latent=p.get("latent_dim", 128, "int"), decoder="linear", likelihood="mse",
                       dropout=p.get("dropout", 0.2, "float"), residual=True, zero_init=True)
            p.get("learner", "mlp", "str")
        elif c == "multitask_lowrank":
            cfg.update(default_inputs={"lsi": {"n_components": 100}}, hidden=p.get("trunk_hidden", [512], "ints", ("hidden",)),
                       latent=p.get("rank", 32, "int"), decoder="factorized", factor_dim=p.get("gene_embedding_dim", 64, "int"),
                       likelihood=p.get("likelihood", "mse", "str"), dropout=p.get("dropout", 0.1, "float"))
        elif c == "cis_regulatory":
            gdim = p.get("global_state_dim", 30, "int")
            cis_in = dict(self.inputs.get("cis_peaks", {}))
            cis_in.setdefault("window_kb", p.get("window_kb", 100.0, "float"))
            cis_in.setdefault("max_peaks_per_gene", p.get("max_peaks_per_gene", 20, "int"))
            self.inputs["cis_peaks"] = cis_in
            cfg.update(default_inputs={"lsi": {"n_components": gdim}}, hidden=p.get("hidden", [256], "ints"),
                       latent=p.get("latent_dim", 64, "int"), decoder="linear", likelihood=p.get("likelihood", "mse", "str"),
                       dropout=p.get("dropout", 0.1, "float"), cis=True)
            p.get("learner", "masked_nn", "str")
        elif c == "contrastive_pretrain_head":
            cfg.update(default_inputs={"lsi": {"n_components": 100}}, hidden=p.get("encoder_hidden", [512], "ints", ("hidden",)),
                       latent=p.get("embed_dim", 64, "int"), decoder="mlp", likelihood="mse", dropout=p.get("dropout", 0.1, "float"),
                       contrastive=True, head=p.get("head", "mlp", "str"), temperature=p.get("temperature", 0.1, "float"),
                       pretrain_epochs=p.get("pretrain_epochs", 60, "int"), knn_k=p.get("knn_k", 30, "int", ("k",)))
        else:
            raise KeyError(c)
        if cfg["likelihood"] not in ("poisson", "nb", "mse"):
            cfg["likelihood"] = "mse"
        if cfg["decoder"] not in ("linear", "mlp", "factorized"):
            cfg["decoder"] = "linear"
        cfg["dec_hidden"] = p.get("decoder_hidden", 256, "int")
        t = self.t
        cfg.update(max_epochs=t.get("max_epochs", 200, "int", ("epochs",)),
                   patience=t.get("early_stopping_patience", 15, "int", ("patience",)),
                   lr=t.get("lr", 1e-3, "float", ("learning_rate",)), batch_size=t.get("batch_size", 256, "int"),
                   weight_decay=t.get("weight_decay", 1e-4, "float"), denoise_k=_denoise_spec(t),
                   monitor=t.get("early_stopping_metric", "val_mean_r", "str", ("monitor", "early_stopping_monitor")))
        if cfg["monitor"] not in ("val_mean_r", "val_loss"):
            cfg["monitor"] = "val_mean_r"
        t.get("loss", None)
        dense_types = [k for k in self.inputs if k in ("lsi", "tfidf_peaks", "gene_activity", "depth")]
        if not dense_types:
            self.inputs.update(cfg["default_inputs"])
        return cfg

    def _fit(self, atac, rna, val):
        torch = _torch()
        cfg = self.cfg = self._config()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        device = _device()
        atac_va, rna_va = val
        self.feats = Features(self.view, self.gene_idx, self.inputs, self.seed).fit(atac)
        tr = _Batcher(self.feats.dense(atac), self.feats.sparse(atac), self.feats.cis(atac) if cfg["cis"] else None, device)
        va = _Batcher(self.feats.dense(atac_va), self.feats.sparse(atac_va), self.feats.cis(atac_va) if cfg["cis"] else None, device)
        y_va = self.target(rna_va)
        y_tr_plain = self.target(rna)

        need_lsi = cfg["residual"] or cfg["denoise_k"] > 0
        if need_lsi:
            base_cfg = self.view.rules.get("baseline_lsi", dict(n_components=50, max_peaks=60000))
            self.base_lsi = mb.LSI(n_components=min(base_cfg["n_components"], atac.shape[0] - 2, atac.shape[1] - 2),
                                   max_peaks=base_cfg["max_peaks"], seed=self.seed).fit(atac)
            z_tr = self.base_lsi.transform(atac)
        base_tr = base_va = None
        if cfg["residual"]:
            k_raw = self.p.get("cluster_k", "frozen")
            k = self.view.cluster_k if str(k_raw).lower() == "frozen" else int(k_raw)
            self.base_model = mb.ClusterMean(min(k, len(z_tr)), seed=self.seed).fit(z_tr, y_tr_plain)
            base_tr, base_va = self.base_model.predict(z_tr), self.base_model.predict(self.base_lsi.transform(atac_va))
            cfg["cluster_k"] = int(min(k, len(z_tr)))

        counts = cfg["likelihood"] != "mse"
        self.scale = None
        rna_fit = _knn_pool(z_tr, rna, cfg["denoise_k"]) if cfg["denoise_k"] > 0 else rna
        if counts:
            tgt_tr, lib_tr = rna_fit[:, self.gene_idx].astype(np.float32).tocsr(), np.asarray(rna_fit.sum(1)).ravel().astype(np.float32)
            tgt_va, lib_va = rna_va[:, self.gene_idx].astype(np.float32).tocsr(), np.asarray(rna_va.sum(1)).ravel().astype(np.float32)
            rate = np.asarray(tgt_tr.sum(0)).ravel() / max(float(lib_tr.sum()), 1.0) * 1e4
            bias = np.log(rate + 1e-4)
        else:
            tgt_tr = self.target(rna_fit) if cfg["denoise_k"] > 0 else y_tr_plain
            tgt_va, lib_tr, lib_va = y_va, None, None
            bias = np.zeros(len(self.gene_idx), dtype=np.float32)
            sd = tgt_tr.std(0)
            sd = np.maximum(sd, 0.1 * float(np.median(sd)) + 1e-6).astype(np.float32)
            mu = np.zeros_like(sd) if cfg["residual"] else tgt_tr.mean(0).astype(np.float32)
            self.scale = (torch.as_tensor(mu, device=device), torch.as_tensor(sd, device=device))

        cis = (self.feats.cis_gene, self.feats.cis_local, len(self.feats.cis_union)) if cfg["cis"] else None
        # Dropout on the raw input only where it is high-dimensional and sparse (TF-IDF peaks).
        cfg["input_dropout"] = self.p.get("input_dropout", 0.25 if tr.sp is not None else 0.0, "float")
        net = _make_net(tr.d_in, cfg["hidden"], cfg["latent"], len(self.gene_idx), cfg["decoder"], cfg["dec_hidden"],
                        cfg["dropout"], cfg["factor_dim"], cis, cfg["input_dropout"]).to(device)
        with torch.no_grad():
            net.out_layer.bias.copy_(torch.as_tensor(bias, dtype=torch.float32))
            if cfg["zero_init"]:
                net.out_layer.weight.zero_()
        params = list(net.parameters())
        log_theta = None
        if cfg["likelihood"] == "nb":
            log_theta = torch.nn.Parameter(torch.zeros(len(self.gene_idx), device=device))
            params.append(log_theta)
        self.net, self.log_theta, self.device = net, log_theta, device
        curves = {}

        if cfg["contrastive"]:
            stop = self._pretrain(net, tr, va, rna, rna_va, cfg, rng, curves)
            if cfg["head"] == "knn_transfer":
                self._knn_bank = (self._rna_embed(rna), y_tr_plain)
                self.resolved.update(self._resolved(cfg, tr.d_in))
                conv, note = _convergence(curves["pretrain_val_loss"], stop)
                pred = self._predict_knn(va)
                curves["val_mean_r"] = [float(np.nanmean(colcorr(pred, y_va)))]
                curves.update(train_loss=curves["pretrain_train_loss"], val_loss=curves["pretrain_val_loss"])
                return dict(device=str(device), epochs=len(curves["pretrain_val_loss"]), stopped_by=stop, converged=conv,
                            convergence_note="Contrastive alignment; expression transferred from nearest RNA neighbours. " + note,
                            curves=curves, n_parameters=int(sum(q.numel() for q in net.parameters())))

        def to_t(a, idx):
            if sparse.issparse(a):
                a = np.asarray(a[idx].todense(), dtype=np.float32)
            else:
                a = a[idx]
            return torch.from_numpy(np.ascontiguousarray(a)).to(device)

        def evaluate():
            net.eval()
            losses, preds = [], []
            with torch.no_grad():
                for s in range(0, va.n, 512):
                    idx = np.arange(s, min(s + 512, va.n))
                    x, xc = va.x(idx)
                    loss, pred = _loss_and_pred(net(x, xc), to_t(tgt_va, idx), cfg["likelihood"],
                                                None if base_va is None else to_t(base_va, idx),
                                                None if lib_va is None else to_t(lib_va, idx), log_theta, self.scale)
                    losses.append(loss.item() * len(idx))
                    preds.append(pred.cpu().numpy())
            return sum(losses) / va.n, float(np.nanmean(colcorr(np.vstack(preds), y_va)))

        opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        # Early stopping watches the mean per-gene correlation on the internal validation cells by default: it is the
        # quantity plans are scored with, and a pooled loss can keep falling on a few high-variance genes while it stalls.
        train_loss, val_loss, val_r = [], [], []
        best, best_state, stale, stopped_by = -np.inf, None, 0, "max_epochs"
        for epoch in range(cfg["max_epochs"]):
            net.train()
            order, running = rng.permutation(tr.n), 0.0
            for s in range(0, tr.n, cfg["batch_size"]):
                idx = np.sort(order[s:s + cfg["batch_size"]])
                x, xc = tr.x(idx)
                loss, _ = _loss_and_pred(net(x, xc), to_t(tgt_tr, idx), cfg["likelihood"],
                                         None if base_tr is None else to_t(base_tr, idx),
                                         None if lib_tr is None else to_t(lib_tr, idx), log_theta, self.scale)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                running += loss.item() * len(idx)
            vl, vr = evaluate()
            train_loss.append(running / tr.n), val_loss.append(vl), val_r.append(vr)
            score = vr if cfg["monitor"] == "val_mean_r" else -vl
            if np.isfinite(score) and score > best + 1e-7:
                best, stale = score, 0
                best_state = (copy.deepcopy(net.state_dict()), None if log_theta is None else log_theta.detach().clone())
            else:
                stale += 1
            if stale >= cfg["patience"]:
                stopped_by = "early_stopping"
                break
            if self.out_of_time():
                stopped_by = "time_budget"
                break
        if best_state is not None:
            net.load_state_dict(best_state[0])
            if log_theta is not None:
                with torch.no_grad():
                    log_theta.copy_(best_state[1])
        net.eval()
        curves.update(train_loss=train_loss, val_loss=val_loss, val_mean_r=val_r)
        by_r = cfg["monitor"] == "val_mean_r"
        converged, note = _convergence(val_r if by_r else val_loss, stopped_by, lower_is_better=not by_r,
                                       what="validation mean per-gene correlation" if by_r else "validation loss")
        self.resolved.update(self._resolved(cfg, tr.d_in))
        best_epoch = int(np.nanargmax(val_r) if by_r else np.argmin(val_loss)) + 1
        return dict(device=str(device), epochs=len(val_loss), stopped_by=stopped_by, converged=converged, convergence_note=note,
                    curves=curves, n_parameters=int(sum(q.numel() for q in params)), monitor=cfg["monitor"], best_epoch=best_epoch,
                    best_val_loss=float(val_loss[best_epoch - 1]), best_val_mean_r=float(val_r[best_epoch - 1]))

    def _resolved(self, cfg, d_in):
        keep = ("hidden", "latent", "decoder", "dec_hidden", "factor_dim", "likelihood", "dropout", "input_dropout", "residual", "cis", "contrastive",
                "head", "max_epochs", "patience", "monitor", "lr", "batch_size", "weight_decay", "denoise_k", "cluster_k", "temperature",
                "pretrain_epochs", "knn_k")
        return dict(network={k: cfg[k] for k in keep if k in cfg}, input_dim=int(d_in), inputs=self.feats.resolved,
                    val_fraction=VAL_FRACTION)

    # ---- contrastive alignment ----
    def _rna_features(self, rna, fit=False):
        from sklearn.decomposition import TruncatedSVD
        x = lognorm(rna).astype(np.float32)
        if fit:
            n_comp = min(50, x.shape[0] - 2, x.shape[1] - 2)
            self._rna_svd = TruncatedSVD(n_comp, random_state=self.seed, n_iter=5).fit(x)
            self._rna_scale = self._rna_svd.transform(x).std(0) + 1e-8
        return (self._rna_svd.transform(x) / self._rna_scale).astype(np.float32)

    def _rna_embed(self, rna):
        torch = _torch()
        with torch.no_grad():
            z = self.rna_enc(torch.from_numpy(self._rna_features(rna)).to(self.device))
            return torch.nn.functional.normalize(z, dim=1).cpu().numpy()

    def _pretrain(self, net, tr, va, rna, rna_va, cfg, rng, curves):
        torch = _torch()
        nn, F = torch.nn, torch.nn.functional
        r_tr, r_va = self._rna_features(rna, fit=True), self._rna_features(rna_va)
        self.rna_enc = nn.Sequential(nn.Linear(r_tr.shape[1], 256), nn.GELU(), nn.Linear(256, cfg["latent"])).to(self.device)
        enc_params = list(net.encoder.parameters()) + list(net.to_latent.parameters()) + list(self.rna_enc.parameters())
        opt = torch.optim.AdamW(enc_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

        def nce(x, r):
            za, zr = F.normalize(net.embed(x), dim=1), F.normalize(self.rna_enc(r), dim=1)
            logits = za @ zr.t() / cfg["temperature"]
            labels = torch.arange(len(x), device=self.device)
            return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2

        tl, vl, best, best_state, stale, stop = [], [], np.inf, None, 0, "max_epochs"
        for epoch in range(cfg["pretrain_epochs"]):
            net.train(), self.rna_enc.train()
            order, running, seen = rng.permutation(tr.n), 0.0, 0
            for s in range(0, tr.n, cfg["batch_size"]):
                idx = np.sort(order[s:s + cfg["batch_size"]])
                if len(idx) < 8:
                    continue
                x, _ = tr.x(idx)
                loss = nce(x, torch.from_numpy(r_tr[idx]).to(self.device))
                opt.zero_grad(), loss.backward(), opt.step()
                running, seen = running + loss.item() * len(idx), seen + len(idx)
            net.eval(), self.rna_enc.eval()
            with torch.no_grad():
                losses = []
                for s in range(0, va.n, cfg["batch_size"]):          # same batch size, so the loss is comparable to training
                    idx = np.arange(s, min(s + cfg["batch_size"], va.n))
                    if len(idx) >= 8:
                        losses.append(float(nce(va.x(idx)[0], torch.from_numpy(r_va[idx]).to(self.device))))
            tl.append(running / max(seen, 1)), vl.append(float(np.mean(losses)) if losses else float("nan"))
            if vl[-1] < best - 1e-7:
                best, stale = vl[-1], 0
                best_state = (copy.deepcopy(net.state_dict()), copy.deepcopy(self.rna_enc.state_dict()))
            else:
                stale += 1
            if stale >= cfg["patience"]:
                stop = "early_stopping"
                break
            if self.out_of_time():
                stop = "time_budget"
                break
        if best_state is not None:
            net.load_state_dict(best_state[0]), self.rna_enc.load_state_dict(best_state[1])
        net.eval(), self.rna_enc.eval()
        curves.update(pretrain_train_loss=tl, pretrain_val_loss=vl)
        return stop

    def _predict_knn(self, batcher):
        torch = _torch()
        bank_z, bank_y = self._knn_bank
        k = min(self.cfg["knn_k"], len(bank_z))
        out = []
        with torch.no_grad():
            bz = torch.from_numpy(bank_z).to(self.device)
            for s in range(0, batcher.n, 128):
                x, _ = batcher.x(np.arange(s, min(s + 128, batcher.n)))
                za = torch.nn.functional.normalize(self.net.embed(x), dim=1)
                sim, idx = torch.topk(za @ bz.t(), k, dim=1)
                w = torch.softmax(sim / self.cfg["temperature"], dim=1).cpu().numpy()
                out.append(np.einsum("nk,nkg->ng", w, bank_y[idx.cpu().numpy()]))
        return np.vstack(out).astype(np.float32)

    def _predict(self, atac):
        torch = _torch()
        cfg = self.cfg
        b = _Batcher(self.feats.dense(atac), self.feats.sparse(atac), self.feats.cis(atac) if cfg["cis"] else None, self.device)
        if cfg["contrastive"] and cfg["head"] == "knn_transfer":
            return self._predict_knn(b)
        base = self.base_model.predict(self.base_lsi.transform(atac)) if cfg["residual"] else None
        out = []
        self.net.eval()
        with torch.no_grad():
            for s in range(0, b.n, 512):
                idx = np.arange(s, min(s + 512, b.n))
                x, xc = b.x(idx)
                o = self.net(x, xc)
                if cfg["likelihood"] == "mse":
                    o = (self.scale[0] + self.scale[1] * o).cpu().numpy() + (0.0 if base is None else base[idx])
                else:
                    o = torch.log1p(torch.exp(torch.clamp(o, -18.0, 12.0))).cpu().numpy()
                out.append(o)
        return np.vstack(out)


# ============================================================================= per-gene learners
def _fit_gene_chunk(x_global, extras, y, learner, hp, seed):
    """Fit one model per gene of a chunk. extras[j] is that gene's own feature block (or None).

    x_global may be None (gene-specific features only). A gene without any feature gets no model.
    """
    from threadpoolctl import threadpool_limits
    out = []
    with threadpool_limits(1):
        for j in range(y.shape[1]):
            blocks = [b for b in (x_global, extras[j]) if b is not None and b.shape[1] > 0]
            if not blocks:
                out.append((None, 0, None))
                continue
            x = blocks[0] if len(blocks) == 1 else np.hstack(blocks)
            if learner == "gbm":
                from sklearn.ensemble import HistGradientBoostingRegressor
                m = HistGradientBoostingRegressor(
                    max_iter=hp["max_iter"], learning_rate=hp["learning_rate"], max_depth=hp["max_depth"], min_samples_leaf=20,
                    l2_regularization=1.0, early_stopping=True, validation_fraction=VAL_FRACTION,
                    n_iter_no_change=hp["patience"], random_state=seed).fit(x, y[:, j])
                curve = (-np.asarray(m.validation_score_)).tolist()
                out.append((m, int(m.n_iter_), curve))
            else:
                import warnings
                from sklearn.linear_model import ElasticNetCV
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # float64 inputs: with float32 the precomputed Gram matrix fails sklearn's consistency check.
                    m = ElasticNetCV(l1_ratio=hp["l1_ratio"], alphas=hp["n_alphas"], cv=3, max_iter=2000,   # sklearn >= 1.7: alphas=int
                                     random_state=seed).fit(x.astype(np.float64), y[:, j].astype(np.float64))
                out.append((m if np.any(m.coef_ != 0) else None, int(m.n_iter_), None))
    return out


class PerGeneModel(MModel):
    """residual_over_cluster_mean (gbm) and cis_regulatory (gbm | elastic_net): one model per gene, in parallel.

    Both are two-stage. A base prediction carries what is shared across genes: the cluster mean (residual model)
    or a ridge regression on the global cell state (cis model, cross-fitted so the residual is out of sample).
    The per-gene learner then models what the base leaves unexplained, from the gene's own cis peaks and
    activity score (and, for boosting, the global state as well). A gene whose learner finds nothing keeps the
    base prediction, so no gene is ever predicted by a constant.
    """

    iterative = False

    def _fit(self, atac, rna, val):
        from joblib import Parallel, delayed
        from sklearn.preprocessing import StandardScaler
        p = self.p
        self.learner = p.get("learner", "gbm", "str")
        if self.learner not in ("gbm", "elastic_net"):
            self.learner = "gbm"
        self.is_cis = self.component == "cis_regulatory"
        if self.is_cis:
            cis_in = dict(self.inputs.get("cis_peaks", {}))
            cis_in.setdefault("window_kb", p.get("window_kb", 100.0, "float"))
            cis_in.setdefault("max_peaks_per_gene", p.get("max_peaks_per_gene", 20, "int"))
            self.inputs["cis_peaks"] = cis_in
            self.inputs.setdefault("lsi", {"n_components": p.get("global_state_dim", 30, "int")})
        elif not any(k in self.inputs for k in ("lsi", "depth")):
            self.inputs.update({"lsi": {"n_components": 50}, "depth": {}})
        if "tfidf_peaks" in self.inputs:
            self.inputs.pop("tfidf_peaks")
            self.extra_ignored.append("inputs.tfidf_peaks (per-gene learners take dense features)")
        own_ga = "gene_activity" in self.inputs
        self.feats = Features(self.view, self.gene_idx, self.inputs, self.seed).fit(atac)
        # Global features shared by every gene exclude gene activity: each gene gets only its own activity score.
        self._global = copy.copy(self.feats)
        self._global.want = {k: v for k, v in self.feats.want.items() if k != "gene_activity"}
        x_global = self._global.dense(atac)
        if x_global is None:
            x_global = mb.depth_feature(atac)
        self.scaler = StandardScaler().fit(x_global)
        x_global = self.scaler.transform(x_global).astype(np.float32)
        own = self.feats.own_activity(atac) if own_ga else None
        cis = self.feats.cis(atac) if self.is_cis else None
        y = self.target(rna)
        n, g = y.shape

        if self.is_cis:
            alpha = p.get("global_alpha", 10.0, "float")
            self.base_model = mb.LatentRidge(alpha).fit(x_global, y)
            base = np.zeros_like(y)
            for test in np.array_split(np.random.default_rng(self.seed + 17).permutation(n), 3):
                train = np.setdiff1d(np.arange(n), test)
                base[test] = mb.LatentRidge(alpha).fit(x_global[train], y[train]).predict(x_global[test])
            self.base_kind = f"ridge on the global cell state (alpha {alpha}), cross-fitted in 3 folds"
        else:
            base_cfg = self.view.rules.get("baseline_lsi", dict(n_components=50, max_peaks=60000))
            self.base_lsi = mb.LSI(n_components=min(base_cfg["n_components"], atac.shape[0] - 2, atac.shape[1] - 2),
                                   max_peaks=base_cfg["max_peaks"], seed=self.seed).fit(atac)
            z = self.base_lsi.transform(atac)
            k_raw = p.get("cluster_k", "frozen")
            k = self.view.cluster_k if str(k_raw).lower() == "frozen" else int(k_raw)
            self.base_model = mb.ClusterMean(min(k, len(z)), seed=self.seed).fit(z, y)
            base = self.base_model.predict(z)
            self.base_kind = f"cluster mean (k {min(k, len(z))})"
        resid = (y - base).astype(np.float32)
        # Elastic net on the cis model: the global state is already in the base, so the learner sees gene-specific features only.
        self.use_global = not (self.is_cis and self.learner == "elastic_net")

        t = self.t
        hp = dict(max_iter=t.get("max_epochs", p.get("max_iter", 150, "int"), "int", ("max_iter",)),
                  learning_rate=p.get("learning_rate", t.get("lr", 0.1, "float"), "float"), max_depth=p.get("max_depth", 3, "int"),
                  patience=t.get("early_stopping_patience", 10, "int", ("patience",)), l1_ratio=p.get("l1_ratio", 0.5, "float"),
                  n_alphas=p.get("n_alphas", 8, "int"))
        cis_cols = [np.zeros(0, dtype=int)] * g
        if self.is_cis:
            order = np.argsort(self.feats.cis_gene, kind="stable")
            gene_sorted, local_sorted = self.feats.cis_gene[order], self.feats.cis_local[order]
            bounds = np.searchsorted(gene_sorted, np.arange(g + 1))
            cis_cols = [local_sorted[bounds[j]:bounds[j + 1]] for j in range(g)]
            cis_csc = cis.tocsc()
        self.cis_cols = cis_cols

        def extras_for(j):
            parts = []
            if self.is_cis and len(cis_cols[j]):
                parts.append(np.asarray(cis_csc[:, cis_cols[j]].todense(), dtype=np.float32))
            if own is not None:
                parts.append(own[:, [j]])
            return np.hstack(parts) if parts else None

        chunk = 8 if self.learner == "gbm" else 16
        chunks = [np.arange(s, min(s + chunk, g)) for s in range(0, g, chunk)]
        self.models, n_iter, curves_sample = [None] * g, np.zeros(g, dtype=int), []
        n_jobs, done, stopped_by = max(1, THREADS), 0, "closed_form"
        round_size = n_jobs * 4
        xg = x_global if self.use_global else None
        with Parallel(n_jobs=n_jobs, backend="loky") as par:
            for s in range(0, len(chunks), round_size):
                batch = chunks[s:s + round_size]
                results = par(delayed(_fit_gene_chunk)(xg, [extras_for(j) for j in c], resid[:, c], self.learner, hp, self.seed)
                              for c in batch)
                for c, res in zip(batch, results):
                    for j, (m, it, curve) in zip(c, res):
                        self.models[j], n_iter[j] = m, it
                        if curve is not None and len(curves_sample) < 200:
                            curves_sample.append(curve)
                    done += len(c)
                if self.out_of_time() and done < g:
                    stopped_by = "time_budget"
                    break
        self.n_fitted = done
        n_active = int(sum(m is not None for m in self.models))
        self.resolved.update(learner=self.learner, hyper_parameters=hp, inputs=self.feats.resolved, base=self.base_kind,
                             learner_sees_global_state=bool(self.use_global), own_gene_activity_feature=bool(own_ga),
                             genes_fitted=int(done), genes_with_a_nonzero_model=n_active, n_jobs=n_jobs)
        curves = {}
        if self.learner == "gbm":
            hit_cap = float(np.mean(n_iter[:done] >= hp["max_iter"])) if done else 1.0
            if curves_sample:
                length = max(len(c) for c in curves_sample)
                padded = np.array([c + [c[-1]] * (length - len(c)) for c in curves_sample])
                curves["val_loss"] = padded.mean(0).tolist()
            note = (f"Base: {self.base_kind}. One boosted model per gene on the residual, early stopping on an internal "
                    f"{VAL_FRACTION:.0%} split; {1 - hit_cap:.1%} of {done} fitted genes stopped before the cap of {hp['max_iter']} "
                    f"iterations (median {int(np.median(n_iter[:done])) if done else 0} iterations).")
            converged = stopped_by != "time_budget" and hit_cap <= 0.10
            if stopped_by != "time_budget":
                stopped_by = "early_stopping" if hit_cap <= 0.10 else "max_epochs"
        else:
            note = (f"Base: {self.base_kind}. One cross-validated elastic net per gene on the residual (coordinate descent to "
                    f"tolerance); {n_active} of {done} genes kept at least one non-zero coefficient, the others keep the base prediction.")
            converged = stopped_by != "time_budget"
        if stopped_by == "time_budget":
            note += f" The time budget stopped fitting after {done} of {g} genes; the rest keep the base prediction."
        return dict(epochs=int(np.median(n_iter[:done])) if done else 0, stopped_by=stopped_by, converged=bool(converged),
                    convergence_note=note, curves=curves, n_parameters=0)

    def _predict(self, atac):
        x_global = self._global.dense(atac)
        if x_global is None:
            x_global = mb.depth_feature(atac)
        x_global = self.scaler.transform(x_global).astype(np.float32)
        own = self.feats.own_activity(atac) if "gene_activity" in self.feats.want else None
        cis_csc = self.feats.cis(atac).tocsc() if self.is_cis else None
        if self.is_cis:
            out = self.base_model.predict(x_global).copy()
        else:
            out = self.base_model.predict(self.base_lsi.transform(atac)).copy()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(1):          # thousands of tiny predictions: thread start-up would dominate
            for j, m in enumerate(self.models):
                if m is None:
                    continue
                parts = [x_global] if self.use_global else []
                if self.is_cis and len(self.cis_cols[j]):
                    parts.append(np.asarray(cis_csc[:, self.cis_cols[j]].todense(), dtype=np.float32))
                if own is not None:
                    parts.append(own[:, [j]])
                out[:, j] += m.predict(parts[0] if len(parts) == 1 else np.hstack(parts))
        return out


# ============================================================================= scvi-tools models
def _scvi_stopper(patience, deadline, monitor="reconstruction_loss_validation"):
    """Early stopping on held-out reconstruction with best-weights restore, plus the wall-clock budget.

    scvi's own early stopping is not used: MULTIVI.train hard-codes a patience of 50, and the reason training
    stopped has to be known to state convergence honestly.
    """
    from lightning.pytorch.callbacks import Callback

    class Stopper(Callback):
        def __init__(self):
            self.best, self.state, self.stale, self.reason = float("inf"), None, 0, "max_epochs"

        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            metric = trainer.callback_metrics.get(monitor)
            if metric is None:
                return
            value = float(metric)
            if value < self.best - 1e-6:
                self.best, self.stale, self.state = value, 0, copy.deepcopy(pl_module.module.state_dict())
            else:
                self.stale += 1
            if self.stale >= patience:
                trainer.should_stop, self.reason = True, "early_stopping"
            elif deadline is not None and time.time() > deadline:
                trainer.should_stop, self.reason = True, "time_budget"

        def on_train_end(self, trainer, pl_module):
            if self.state is not None:
                pl_module.module.load_state_dict(self.state, strict=True)

    return Stopper()


def _scvi_train_kwargs(n_train, n_val, max_epochs, lr, batch_size, stopper):
    torch = _torch()
    # lr is passed as train(lr=...): MULTIVI/PEAKVI overwrite plan_kwargs["lr"] with that argument.
    # scvi's default logger is kept: with logger=False the loss history is empty and convergence cannot be audited.
    return dict(max_epochs=max_epochs, lr=lr, batch_size=batch_size, accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1, early_stopping=False, check_val_every_n_epoch=1,
                datasplitter_kwargs={"external_indexing": [np.arange(n_train), np.arange(n_train, n_train + n_val), np.array([], dtype=int)]},
                callbacks=[stopper], enable_progress_bar=False, enable_checkpointing=False)


def _scvi_history(model, keys):
    hist = model.history or {}
    return {k: [float(x) for x in hist[k].iloc[:, 0].values] for k in keys if k in hist and len(hist[k])}


class MultiVIModel(MModel):
    """MultiVI trained with early stopping; expression of ATAC-only cells is imputed from the joint latent space."""

    def _mudata(self, rna_block, atac_block, prefix):
        import anndata as ad
        import mudata as md
        import pandas as pd
        ids = np.array([f"{prefix}_{i}" for i in range(atac_block.shape[0])])
        obs = pd.DataFrame(index=ids)
        out = md.MuData({
            "rna": ad.AnnData(rna_block, obs=obs.copy(), var=pd.DataFrame(index=[f"gene_{i}" for i in range(rna_block.shape[1])])),
            "atac": ad.AnnData(atac_block, obs=obs.copy(), var=pd.DataFrame(index=[f"peak_{i}" for i in range(atac_block.shape[1])]))})
        out.obs["_indices"] = np.arange(len(ids))
        return out

    def _fit(self, atac, rna, val):
        import scvi
        p, t = self.p, self.t
        atac_va, rna_va = val
        n_peaks = min(p.get("n_peaks", 30000, "int"), atac.shape[1])
        detected = np.asarray((atac > 0).sum(0)).ravel()
        self.peaks = np.sort(np.argsort(-detected)[:n_peaks])
        hp = dict(n_latent=p.get("n_latent", 20, "int"), n_hidden=p.get("n_hidden", 256, "int"), n_peaks=int(n_peaks),
                  max_epochs=p.get("max_epochs", t.get("max_epochs", 300, "int", ("epochs",)), "int"),
                  patience=p.get("early_stopping_patience", t.get("early_stopping_patience", 20, "int", ("patience",)), "int"),
                  lr=t.get("lr", 1e-3, "float", ("learning_rate",)), batch_size=t.get("batch_size", 256, "int"))
        scvi.settings.seed = self.seed
        rna_all = sparse.vstack([rna[:, self.gene_idx], rna_va[:, self.gene_idx]]).astype(np.float32).tocsr()
        atac_all = (sparse.vstack([atac[:, self.peaks], atac_va[:, self.peaks]]) > 0).astype(np.float32).tocsr()
        mdata = self._mudata(rna_all, atac_all, "fit")
        scvi.model.MULTIVI.setup_mudata(mdata, modalities={"rna_layer": "rna", "atac_layer": "atac"})
        self.m = scvi.model.MULTIVI(mdata, n_genes=len(self.gene_idx), n_regions=int(n_peaks), n_latent=hp["n_latent"],
                                    n_hidden=hp["n_hidden"], gene_likelihood="nb", fully_paired=False, modality_weights="equal")
        t0 = time.time()
        stopper = _scvi_stopper(hp["patience"], self.deadline)
        self.m.train(**_scvi_train_kwargs(atac.shape[0], atac_va.shape[0], hp["max_epochs"], hp["lr"], hp["batch_size"], stopper))
        hist = _scvi_history(self.m, ("reconstruction_loss_train", "reconstruction_loss_validation", "elbo_validation"))
        val_curve = hist.get("reconstruction_loss_validation", [])
        epochs, stopped_by = len(val_curve), stopper.reason
        # Share of a cell's counts that falls on the modelled genes: converts MultiVI's normalised expression
        # (which sums to one over modelled genes) to the per-10,000 scale used by view.target.
        self.fraction = float(rna[:, self.gene_idx].sum() / max(rna.sum(), 1))
        converged, note = _convergence(val_curve, stopped_by)
        curves = dict(train_loss=hist.get("reconstruction_loss_train", []), val_loss=val_curve)
        if "elbo_validation" in hist:
            curves["elbo_validation"] = hist["elbo_validation"]
        curves["val_mean_r"] = [float(np.nanmean(colcorr(self._predict(atac_va), self.target(rna_va))))]
        self.resolved.update(hyper_parameters=hp, likelihood="nb", fully_paired=False, fraction_counts_on_modelled_genes=self.fraction,
                             train_seconds=round(time.time() - t0, 1))
        return dict(device="cuda" if _torch().cuda.is_available() else "cpu", epochs=epochs, stopped_by=stopped_by, converged=converged,
                    convergence_note="MultiVI held-out reconstruction loss. " + note, curves=curves,
                    n_parameters=int(sum(q.numel() for q in self.m.module.parameters())))

    def _predict(self, atac):
        out = []
        for s in range(0, atac.shape[0], 2000):
            block = (atac[s:s + 2000][:, self.peaks] > 0).astype(np.float32).tocsr()
            empty = sparse.csr_matrix((block.shape[0], len(self.gene_idx)), dtype=np.float32)
            scale = self.m.get_normalized_expression(self._mudata(empty, block, f"new{s}"), return_numpy=True)
            out.append(np.log1p(np.asarray(scale, dtype=np.float64) * 1e4 * self.fraction))
        return np.vstack(out).astype(np.float32)


class PeakVIHead(MModel):
    """PeakVI latent representation of accessibility followed by a supervised head (ridge or small MLP)."""

    def _adata(self, atac_block, prefix):
        import anndata as ad
        import pandas as pd
        obs = pd.DataFrame(index=[f"{prefix}_{i}" for i in range(atac_block.shape[0])])
        return ad.AnnData(atac_block, obs=obs, var=pd.DataFrame(index=[f"peak_{i}" for i in range(atac_block.shape[1])]))

    def _latent(self, atac):
        block = (atac[:, self.peaks] > 0).astype(np.float32).tocsr()
        z = self.m.get_latent_representation(self._adata(block, f"new{atac.shape[0]}"))
        return ((np.asarray(z) - self.z_mu) / self.z_sd).astype(np.float32)

    def _fit(self, atac, rna, val):
        import scvi
        p, t = self.p, self.t
        atac_va, rna_va = val
        n_peaks = min(p.get("n_peaks", 30000, "int"), atac.shape[1])
        detected = np.asarray((atac > 0).sum(0)).ravel()
        self.peaks = np.sort(np.argsort(-detected)[:n_peaks])
        hp = dict(n_latent=p.get("n_latent", 30, "int"), n_peaks=int(n_peaks), head=p.get("head", "ridge", "str"),
                  max_epochs=p.get("max_epochs", t.get("max_epochs", 300, "int", ("epochs",)), "int"),
                  patience=p.get("early_stopping_patience", t.get("early_stopping_patience", 20, "int", ("patience",)), "int"),
                  lr=t.get("lr", 1e-3, "float", ("learning_rate",)), batch_size=t.get("batch_size", 256, "int"),
                  ridge_alpha=p.get("alpha", 10.0, "float"))
        if hp["head"] not in ("ridge", "mlp"):
            hp["head"] = "ridge"
        scvi.settings.seed = self.seed
        block = (sparse.vstack([atac[:, self.peaks], atac_va[:, self.peaks]]) > 0).astype(np.float32).tocsr()
        adata = self._adata(block, "fit")
        scvi.model.PEAKVI.setup_anndata(adata)
        self.m = scvi.model.PEAKVI(adata, n_latent=hp["n_latent"])
        stopper = _scvi_stopper(hp["patience"], self.deadline)
        self.m.train(**_scvi_train_kwargs(atac.shape[0], atac_va.shape[0], hp["max_epochs"], hp["lr"], hp["batch_size"], stopper))
        hist = _scvi_history(self.m, ("reconstruction_loss_train", "reconstruction_loss_validation"))
        val_curve = hist.get("reconstruction_loss_validation", [])
        epochs, stopped_by = len(val_curve), stopper.reason
        self.z_mu, self.z_sd = 0.0, 1.0
        z_raw = self._latent(atac)
        self.z_mu, self.z_sd = z_raw.mean(0), z_raw.std(0) + 1e-8
        z_tr, z_va = self._latent(atac), self._latent(atac_va)
        y_tr, y_va = self.target(rna), self.target(rna_va)
        curves = dict(train_loss=hist.get("reconstruction_loss_train", []), val_loss=val_curve)
        converged, note = _convergence(val_curve, stopped_by)
        note = "PeakVI held-out reconstruction loss. " + note
        if hp["head"] == "mlp":
            self.head, head_curves, head_stop = _fit_dense_head(z_tr, y_tr, z_va, y_va, self.seed, self)
            curves.update(head_train_loss=head_curves["train_loss"], head_val_loss=head_curves["val_loss"])
            curves["head_val_mean_r"] = head_curves["val_mean_r"]
            head_conv, head_note = _convergence(head_curves["val_mean_r"], head_stop, lower_is_better=False,
                                                what="validation mean per-gene correlation")
            converged, note = converged and head_conv, note + " Head: " + head_note
            if head_stop == "time_budget":
                stopped_by = "time_budget"
        else:
            ridge = mb.LatentRidge(hp["ridge_alpha"]).fit(z_tr, y_tr)
            self.head = ridge.predict
        curves["val_mean_r"] = [float(np.nanmean(colcorr(self.head(z_va), y_va)))]
        self.resolved.update(hyper_parameters=hp)
        return dict(device="cuda" if _torch().cuda.is_available() else "cpu", epochs=epochs, stopped_by=stopped_by, converged=converged,
                    convergence_note=note, curves=curves, n_parameters=int(sum(q.numel() for q in self.m.module.parameters())))

    def _predict(self, atac):
        return self.head(self._latent(atac))


def _fit_dense_head(z_tr, y_tr, z_va, y_va, seed, owner, hidden=256, max_epochs=200, patience=15, lr=1e-3):
    """Small MLP from a fixed latent representation to all modelled genes.

    Per-gene standardised MSE; early stopping on the mean per-gene correlation of the internal validation cells.
    """
    torch = _torch()
    torch.manual_seed(seed)
    device = _device()
    mu, sd = y_tr.mean(0), y_tr.std(0)
    sd = np.maximum(sd, 0.1 * float(np.median(sd)) + 1e-6)
    net = torch.nn.Sequential(torch.nn.Linear(z_tr.shape[1], hidden), torch.nn.GELU(), torch.nn.Dropout(0.1),
                              torch.nn.Linear(hidden, y_tr.shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    xt, yt = torch.from_numpy(z_tr).to(device), torch.from_numpy(((y_tr - mu) / sd).astype(np.float32)).to(device)
    xv, yv = torch.from_numpy(z_va).to(device), torch.from_numpy(((y_va - mu) / sd).astype(np.float32)).to(device)
    rng = np.random.default_rng(seed)
    tl, vl, vr, best, best_state, stale, stop = [], [], [], -np.inf, None, 0, "max_epochs"
    for epoch in range(max_epochs):
        net.train()
        order, running = rng.permutation(len(xt)), 0.0
        for s in range(0, len(xt), 256):
            idx = torch.from_numpy(order[s:s + 256]).to(device)
            loss = torch.mean((net(xt[idx]) - yt[idx]) ** 2)
            opt.zero_grad(), loss.backward(), opt.step()
            running += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            out = net(xv)
            v = float(torch.mean((out - yv) ** 2))
            r = float(np.nanmean(colcorr(out.cpu().numpy(), y_va)))
        tl.append(running / len(xt)), vl.append(v), vr.append(r)
        if np.isfinite(r) and r > best + 1e-7:
            best, stale, best_state = r, 0, copy.deepcopy(net.state_dict())
        else:
            stale += 1
        if stale >= patience:
            stop = "early_stopping"
            break
        if owner.out_of_time():
            stop = "time_budget"
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    def predict(z):
        with torch.no_grad():
            out = net(torch.from_numpy(np.asarray(z, dtype=np.float32)).to(device)).cpu().numpy()
        return (out * sd + mu).astype(np.float32)

    return predict, dict(train_loss=tl, val_loss=vl, val_mean_r=vr), stop


# ============================================================================= stacking
class StackingEnsemble(MModel):
    """Out-of-fold stacking of heterogeneous members with a per-gene or a global linear meta-learner."""

    iterative = False

    def _member_specs(self):
        members = self.p.get("members", [])
        specs = []
        for m in members if isinstance(members, list) else []:
            if isinstance(m, str):
                m = {"component": m}
            if not isinstance(m, dict):
                continue
            key = m.get("component") or (m.get("model", {}) or {}).get("component")
            if key in (None, "stacking_ensemble") or key not in COMPONENTS:
                self.extra_ignored.append(f"model.params.members[{key}] (not an executable member)")
                continue
            params = m.get("params") or (m.get("model", {}) or {}).get("params") or {}
            specs.append(dict(model=dict(component=key, params=params), inputs=m.get("inputs", self.spec.get("inputs", {})),
                              training=m.get("training", self.spec.get("training", {}))))
        if not specs:                       # a usable default so a mis-specified plan still runs, and says so
            self.extra_ignored.append("model.params.members (none usable; default members used)")
            specs = [dict(model=dict(component="latent_ridge", params={}), inputs={"lsi": {"n_components": 50}}, training={}),
                     dict(model=dict(component="residual_over_cluster_mean", params={"learner": "mlp"}), inputs={}, training={})]
        return specs

    def _fit(self, atac, rna, val):
        specs = self._member_specs()
        k = max(2, min(self.p.get("oof_folds", 3, "int"), 5))
        if self.scale == "pilot":
            k = min(k, 3)
        meta = self.p.get("meta", "per_gene_ridge", "str")
        meta = meta if meta in ("per_gene_ridge", "global_weights") else "per_gene_ridge"
        n, g, m = atac.shape[0], len(self.gene_idx), len(specs)
        folds = np.array_split(np.random.default_rng(self.seed + 31).permutation(n), k)
        n_fits = (k + 1) * m
        per_fit = None if self.deadline is None else max(30.0, (self.deadline - time.time()) / n_fits)
        y = self.target(rna)
        oof = [np.zeros((n, g), dtype=np.float32) for _ in specs]
        oof_reports = []
        for f, test in enumerate(folds):
            train = np.setdiff1d(np.arange(n), test)
            for i, spec in enumerate(specs):
                member = build(spec, self.view, self.scale, self.seed + 100 * f + i, self.gpos, per_fit)
                rep = member.fit(atac[train], rna[train])
                oof[i][test] = member.predict(atac[test])
                oof_reports.append(dict(fold=f, member=i, component=spec["model"]["component"], seconds=rep["seconds"],
                                        converged=rep["converged"], stopped_by=rep["stopped_by"]))
        # Meta-learner: gene by gene, y ~ sum_i w_i * member_i + intercept, solved in closed form.
        cols = oof + [np.ones((n, g), dtype=np.float32)]
        a = np.empty((g, m + 1, m + 1))
        b = np.empty((g, m + 1))
        for i in range(m + 1):
            b[:, i] = np.einsum("ng,ng->g", cols[i], y, dtype=np.float64)
            for j in range(i, m + 1):
                a[:, i, j] = a[:, j, i] = np.einsum("ng,ng->g", cols[i], cols[j], dtype=np.float64)
        # Global weights (one set for all genes, per-gene intercepts), from the pooled normal equations.
        w_global = np.linalg.lstsq(a[:, :m, :m].sum(0) - np.einsum("gi,gj->ij", a[:, :m, m], a[:, :m, m] / n),
                                   (b[:, :m] - a[:, :m, m] * (b[:, [m]] / n)).sum(0), rcond=None)[0]
        shrink = self.p.get("meta_shrinkage", 0.2, "float")
        if meta == "per_gene_ridge":
            # Per-gene weights shrunk towards the global weights: single-cell targets are noisy, and unshrunk
            # per-gene weights fit that noise.
            lam = shrink * np.mean(a[:, np.arange(m), np.arange(m)], axis=1) + 1e-8
            reg = np.zeros((g, m + 1, m + 1))
            reg[:, np.arange(m), np.arange(m)] = lam[:, None]
            rhs = b.copy()
            rhs[:, :m] += lam[:, None] * w_global[None, :]
            self.weights = np.linalg.solve(a + reg, rhs[..., None])[..., 0]
        else:
            intercept = (b[:, m] - a[:, :m, m] @ w_global) / n
            self.weights = np.column_stack([np.tile(w_global, (g, 1)), intercept])
        blended = sum(self.weights[:, i] * oof[i] for i in range(m)) + self.weights[:, m]
        oof_r = dict(stack=float(np.nanmean(colcorr(blended, y))),
                     **{f"member{i}_{s['model']['component']}": float(np.nanmean(colcorr(oof[i], y))) for i, s in enumerate(specs)})
        del oof, cols
        self.members, member_reports, curves = [], [], {}
        for i, spec in enumerate(specs):
            member = build(spec, self.view, self.scale, self.seed + i, self.gpos, per_fit)
            rep = member.fit(atac, rna)
            self.members.append(member)
            member_reports.append(rep)
            for name, curve in rep.get("curves", {}).items():
                curves[f"member{i}_{spec['model']['component']}/{name}"] = curve
            self.extra_ignored += [f"member{i}:{x}" for x in member.ignored]
        all_conv = all(r["converged"] for r in member_reports)
        timed = any(r["stopped_by"] == "time_budget" for r in member_reports) or any(r["stopped_by"] == "time_budget" for r in oof_reports)
        self.resolved.update(meta=meta, meta_shrinkage_towards_global=shrink, global_weights=[float(x) for x in w_global],
                             oof_folds=k, members=[mm.resolved for mm in self.members], oof_mean_r_full_depth=oof_r,
                             mean_weights={f"member{i}": float(np.mean(self.weights[:, i])) for i in range(m)},
                             oof_fits=oof_reports)
        note = ("Stacked " + ", ".join(s["model"]["component"] for s in specs) + f" over {k} out-of-fold splits. "
                + " | ".join(f"{s['model']['component']}: {r['convergence_note']}" for s, r in zip(specs, member_reports)))
        iterative_members = [r for r in member_reports if r["stopped_by"] != "closed_form"]
        return dict(device=member_reports[0]["device"], epochs=int(max([r["epochs"] for r in member_reports] + [0])),
                    stopped_by="time_budget" if timed else ("early_stopping" if iterative_members and all(
                        r["stopped_by"] == "early_stopping" for r in iterative_members) else "max_epochs" if iterative_members else "closed_form"),
                    converged=bool(all_conv and not timed), convergence_note=note, curves=curves,
                    n_parameters=int(sum(r.get("n_parameters", 0) for r in member_reports)))

    def _predict(self, atac):
        m = len(self.members)
        out = np.tile(self.weights[:, m].astype(np.float32), (atac.shape[0], 1))
        for i, member in enumerate(self.members):
            out += self.weights[:, i].astype(np.float32) * member.predict(atac)
        return out


# ============================================================================= registry of builders
def _residual(spec, *a, **k):
    learner = str(spec.get("model", {}).get("params", {}).get("learner", "mlp")).lower()
    return PerGeneModel(spec, *a, **k) if learner in ("gbm", "gradient_boosting", "boosting") else NetModel(spec, *a, **k)


def _cis(spec, *a, **k):
    learner = str(spec.get("model", {}).get("params", {}).get("learner", "masked_nn")).lower()
    return PerGeneModel(spec, *a, **k) if learner in ("gbm", "elastic_net") else NetModel(spec, *a, **k)


COMPONENTS = {
    "cluster_mean": LatentModel, "latent_ridge": LatentModel, "latent_knn": LatentModel, "latent_pls": LatentModel,
    "gene_activity_score": GeneActivityScore, "residual_over_cluster_mean": _residual, "cis_regulatory": _cis,
    "translation_network": NetModel, "multitask_lowrank": NetModel, "contrastive_pretrain_head": NetModel,
    "joint_latent_multivi": MultiVIModel, "peakvi_latent_head": PeakVIHead, "stacking_ensemble": StackingEnsemble,
}


def build(spec, view, scale="full", seed=0, genes=None, time_budget=None):
    """Executable model for one cross-modal prediction plan (or one stacking member)."""
    key = spec.get("model", {}).get("component")
    if key not in COMPONENTS:
        raise KeyError(f"component '{key}' is not in the task M registry; known: {sorted(COMPONENTS)}")
    return COMPONENTS[key](spec, view, scale, seed, genes, time_budget)
