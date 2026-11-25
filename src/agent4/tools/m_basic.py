"""Basic tier for cross-modal expression prediction (README 6.5, task M).

These are the preprocessing steps, the simple learners and the rulers. They serve as baselines, as the
single "simplest valid plan", and as building blocks of the ambitious components. Every predictor maps
ATAC counts of cells to predicted log-normalised expression for a fixed list of genes.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

from ..stats import lognorm


# ----------------------------------------------------------------------------- representations
class LSI:
    """TF-IDF on binarised peaks followed by truncated SVD. The first component tracks depth and is dropped."""

    def __init__(self, n_components=50, max_peaks=60000, seed=0):
        self.n_components, self.max_peaks, self.seed = n_components, max_peaks, seed

    def fit(self, atac):
        binary = (atac > 0).astype(np.float32)
        detected = np.asarray(binary.sum(0)).ravel()
        self.peaks_ = np.sort(np.argsort(-detected)[: self.max_peaks])
        binary = binary[:, self.peaks_]
        self.idf_ = np.log1p(binary.shape[0] / (1.0 + detected[self.peaks_])).astype(np.float32)
        self.svd_ = TruncatedSVD(self.n_components + 1, random_state=self.seed, n_iter=7)
        z = self.svd_.fit_transform(self._tfidf(binary))[:, 1:]
        self.scale_ = z.std(0) + 1e-8
        return self

    def _tfidf(self, binary):
        tf = sparse.diags(1.0 / np.maximum(np.asarray(binary.sum(1)).ravel(), 1.0)) @ binary
        return (tf @ sparse.diags(self.idf_ * 1e4)).log1p()   # log(1 + TF * IDF * 1e4), as in Signac

    def transform(self, atac):
        binary = (atac[:, self.peaks_] > 0).astype(np.float32)
        return (self.svd_.transform(self._tfidf(binary))[:, 1:] / self.scale_).astype(np.float32)

    def tfidf(self, atac):
        """Sparse TF-IDF matrix restricted to the peaks kept at fit time (input for neural encoders)."""
        return self._tfidf((atac[:, self.peaks_] > 0).astype(np.float32)).tocsr()


def depth_feature(atac):
    return np.log1p(np.asarray(atac.sum(1)).ravel())[:, None].astype(np.float32)


def gene_activity(atac, peak_gene, gene_idx, n_peaks, mode="promoter_and_distal", max_distance=None):
    """Sum of ATAC counts over the peaks annotated to each gene, depth-normalised and log-transformed.

    mode: "promoter" uses promoter peaks only; "promoter_and_distal" uses every annotated peak.
    Returns (cells x len(gene_idx)) and a mask of genes that have at least one linked peak.
    """
    pg = peak_gene[peak_gene.gene_idx.isin(gene_idx)]
    if mode == "promoter":
        pg = pg[pg.peak_type == "promoter"]
    if max_distance is not None:
        pg = pg[pg.distance.abs() <= max_distance]
    col = pd_index(gene_idx)[pg.gene_idx.values]
    link = sparse.csr_matrix((np.ones(len(pg), dtype=np.float32), (pg.peak_idx.values, col)),
                             shape=(n_peaks, len(gene_idx)))
    depth = np.maximum(np.asarray(atac.sum(1)).ravel(), 1.0)
    act = np.asarray((atac @ link).todense(), dtype=np.float32) / depth[:, None] * 1e4
    return np.log1p(act), np.asarray(link.sum(0)).ravel() > 0


def pd_index(gene_idx):
    """Lookup array mapping a universe gene index to its position inside gene_idx (-1 if absent)."""
    out = np.full(int(np.max(gene_idx)) + 1, -1, dtype=np.int64)
    out[np.asarray(gene_idx)] = np.arange(len(gene_idx))
    return out


# ----------------------------------------------------------------------------- simple predictors
class ClusterMean:
    """Floor 1: cluster cells on the ATAC latent space, predict each cell by its cluster's mean expression."""

    def __init__(self, k, seed=0):
        self.k, self.seed = int(k), seed

    def fit(self, z, y):
        km = KMeans(self.k, n_init=5, random_state=self.seed) if len(z) < 20000 else \
            MiniBatchKMeans(self.k, n_init=5, random_state=self.seed)
        self.km_ = km.fit(z)
        overall = y.mean(0)
        self.means_ = np.vstack([y[self.km_.labels_ == c].mean(0) if np.any(self.km_.labels_ == c) else overall
                                 for c in range(self.k)]).astype(np.float32)
        return self

    def assign(self, z):
        return self.km_.predict(z)

    def predict(self, z):
        return self.means_[self.km_.predict(z)]


class LatentRidge:
    def __init__(self, alpha=10.0):
        self.alpha = alpha

    def fit(self, z, y):
        self.m_ = Ridge(alpha=self.alpha).fit(z, y)
        return self

    def predict(self, z):
        return self.m_.predict(z).astype(np.float32)


class LatentKNN:
    def __init__(self, k=20):
        self.k = k

    def fit(self, z, y):
        self.m_ = KNeighborsRegressor(self.k, weights="distance").fit(z, y)
        return self

    def predict(self, z):
        return self.m_.predict(z).astype(np.float32)


class LatentPLS:
    def __init__(self, n_components=15):
        self.n = n_components

    def fit(self, z, y):
        self.m_ = PLSRegression(min(self.n, z.shape[1])).fit(z, y)
        return self

    def predict(self, z):
        return np.asarray(self.m_.predict(z), dtype=np.float32)


class DepthOnly:
    """Floor 3: predict expression from ATAC depth alone (expects the depth feature as z)."""

    def fit(self, z, y):
        x = np.hstack([z, z ** 2, np.ones_like(z)])
        self.w_, *_ = np.linalg.lstsq(x, y, rcond=None)
        return self

    def predict(self, z):
        return (np.hstack([z, z ** 2, np.ones_like(z)]) @ self.w_).astype(np.float32)


# ----------------------------------------------------------------------------- ceiling ruler
class DenoiseCeiling:
    """RNA-to-RNA predictor used as the ceiling: sees one half of a cell's reads, predicts the other half.

    Low-rank part: gene programmes are learned from full-depth training cells (truncated SVD); the input half
    is projected onto them and a ridge maps the projection to the target half. Per-gene part: a second ridge
    on [low-rank prediction, the gene's own input value], so that well-measured genes benefit from their own
    reading. The rank is chosen on an internal hold-out of training cells.
    """

    def __init__(self, ranks=(10, 15, 20, 30, 45, 60, 90), alpha=10.0, seed=0):
        self.ranks, self.alpha, self.seed = tuple(sorted(ranks)), alpha, seed

    def _basis(self, basis):
        return TruncatedSVD(max(self.ranks), random_state=self.seed, n_iter=7).fit(basis).components_.astype(np.float32)

    def fit(self, counts_a, counts_b, gene_idx, counts_full=None):
        """counts_a / counts_b: the two read halves of the training cells (cells x universe genes).
        counts_full: their unsplit counts, used only to learn the gene programmes."""
        from ..stats import colcorr
        self.gene_idx_ = np.asarray(gene_idx)
        xa = lognorm(counts_a).astype(np.float32)
        basis = lognorm(counts_full).astype(np.float32) if counts_full is not None else xa
        yb = np.asarray(lognorm(counts_b)[:, self.gene_idx_].todense(), dtype=np.float32)
        own = np.asarray(xa[:, self.gene_idx_].todense(), dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(xa.shape[0])
        cut = int(0.8 * len(order))
        tr, va = np.sort(order[:cut]), np.sort(order[cut:])
        comp = self._basis(basis[tr])
        z_tr, z_va = np.asarray(xa[tr] @ comp.T), np.asarray(xa[va] @ comp.T)
        scores, low_va = {}, {}
        for rank in self.ranks:
            ridge = Ridge(alpha=self.alpha).fit(z_tr[:, :rank], yb[tr])
            low_va[rank] = ridge.predict(z_va[:, :rank])
            scores[rank] = float(np.nanmean(colcorr(low_va[rank], yb[va])))
        self.rank_ = max(scores, key=scores.get)
        self.rank_scores_ = scores
        # Per-gene blend of the low-rank prediction and the gene's own half reading, fitted on the hold-out
        # cells, whose low-rank predictions come from a model that never saw them.
        low = low_va[self.rank_]
        self.blend_ = np.zeros((3, len(self.gene_idx_)), dtype=np.float32)
        for j in range(len(self.gene_idx_)):
            design = np.column_stack([low[:, j], own[va, j], np.ones(len(va))])
            self.blend_[:, j] = np.linalg.solve(design.T @ design + 1e-3 * np.eye(3), design.T @ yb[va, j])
        self.comp_ = self._basis(basis)[: self.rank_]
        self.ridge_ = Ridge(alpha=self.alpha).fit(np.asarray(xa @ self.comp_.T), yb)
        return self

    def predict(self, counts_a):
        xa = lognorm(counts_a).astype(np.float32)
        low = self.ridge_.predict(np.asarray(xa @ self.comp_.T))
        own = np.asarray(xa[:, self.gene_idx_].todense(), dtype=np.float32)
        return (low * self.blend_[0] + own * self.blend_[1] + self.blend_[2]).astype(np.float32)
