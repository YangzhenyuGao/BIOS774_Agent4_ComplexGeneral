"""Statistical rules used by both the agent (on data it may see) and the evaluator (on hidden data).

Everything here is deterministic given a seed and uses only the arrays passed in. The decision rules of
README 4.3 live in this file so that both sides apply exactly the same ones.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse, stats as sps

FDR = 0.05          # convention kept from README 4.3
CONFIDENCE = 0.95   # convention kept from README 4.3


# ----------------------------------------------------------------------------- expression helpers
def lognorm(counts, scale=1e4):
    """log1p of counts scaled to `scale` per cell. Accepts sparse or dense; returns the same kind."""
    lib = np.asarray(counts.sum(1)).ravel().astype(np.float64)
    lib[lib == 0] = 1.0
    if sparse.issparse(counts):
        return (sparse.diags(scale / lib) @ counts).log1p().tocsr()
    return np.log1p(counts * (scale / lib)[:, None])


def thin_counts(counts, seed, p=0.5):
    """Binomial split of every count into two independent parts A and B (A + B == counts)."""
    counts = sparse.csr_matrix(counts).astype(np.int64)
    rng = np.random.default_rng(seed)
    a = counts.copy()
    a.data = rng.binomial(counts.data, p)
    b = counts - a
    a.eliminate_zeros(), b.eliminate_zeros()
    return a.tocsr(), b.tocsr()


def colcorr(pred, truth, constant=0.0):
    """Pearson correlation of each column of `pred` with the same column of `truth`.

    A column that is constant in either matrix carries no information about the other, so its correlation
    is reported as `constant` (0 by default). Returning NaN there and averaging with nanmean would silently
    drop exactly the genes a method failed on and inflate its mean.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    p = p - p.mean(0)
    t = t - t.mean(0)
    den = np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (p * t).sum(0) / den
    r[~(den > 1e-12)] = constant
    return r


def within_group_corr(pred, truth, groups, min_cells=30):
    """Per-gene correlation after removing each group's mean, pooled over groups with enough cells."""
    p = np.asarray(pred, dtype=np.float64).copy()
    t = np.asarray(truth, dtype=np.float64).copy()
    keep = np.zeros(len(p), dtype=bool)
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= min_cells:
            p[m] -= p[m].mean(0)
            t[m] -= t[m].mean(0)
            keep |= m
    return colcorr(p[keep], t[keep])


# ----------------------------------------------------------------------------- multiple testing
def bh_reject(pvalues, alpha=FDR):
    """Benjamini-Hochberg. NaN p-values are never rejected. Returns (reject mask, adjusted p)."""
    p = np.asarray(pvalues, dtype=float)
    ok = np.isfinite(p)
    adj = np.full(p.shape, np.nan)
    m = int(ok.sum())
    if m:
        order = np.argsort(p[ok])
        ranked = p[ok][order] * m / np.arange(1, m + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(m)
        out[order] = np.minimum(ranked, 1.0)
        adj[ok] = out
    return (adj <= alpha) & ok, adj


# ----------------------------------------------------------------------------- task M rules
def permutation_null(pred, truth, n_perm=100, seed=0):
    """Null distribution of per-gene r when the cell correspondence is destroyed.

    Returns per-gene one-sided p-values (r above null) from a normal approximation with the per-gene
    permutation SD, plus the null SD and the pooled 95th percentile of null r.
    """
    rng = np.random.default_rng(seed)
    pred = np.asarray(pred, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    null = np.empty((n_perm, pred.shape[1]), dtype=np.float32)
    for i in range(n_perm):
        null[i] = colcorr(pred[rng.permutation(len(pred))], truth)
    sd = np.nanstd(null, axis=0, ddof=1)
    r = colcorr(pred, truth)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = r / sd
    p = sps.norm.sf(z)
    p[~np.isfinite(z)] = np.nan
    half = n_perm // 2
    stability = float(np.nanmedian(np.abs(np.nanstd(null[:half], 0, ddof=1) - np.nanstd(null[half:], 0, ddof=1)) / sd))
    return dict(r=r, p=p, null_sd=sd, null_q95=float(np.nanquantile(null, 0.95)), sd_split_half_rel_diff=stability)


def _weighted_corr(w, x, y):
    """Correlation of x and y columns under bootstrap count weights w (B x n). Returns B x G."""
    n = w.sum(1, keepdims=True)
    sx, sy = w @ x, w @ y
    sxx, syy, sxy = w @ (x * x), w @ (y * y), w @ (x * y)
    cov = sxy - sx * sy / n
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / np.sqrt(vx * vy)


def paired_bootstrap_delta_r(pred, base, truth, n_boot=300, seed=0, chunk=2000):
    """Per-gene difference r(pred, truth) - r(base, truth), with a cell-level bootstrap SE.

    One-sided p-value (pred better than base) from a normal approximation with the bootstrap SE; the
    bootstrap resamples cells, keeping the three matrices paired.
    """
    rng = np.random.default_rng(seed)
    n, g = truth.shape
    w = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float32)
    delta_boot = np.empty((n_boot, g), dtype=np.float32)
    for start in range(0, g, chunk):
        sl = slice(start, start + chunk)
        x, b, y = (np.asarray(m[:, sl], dtype=np.float32) for m in (pred, base, truth))
        delta_boot[:, sl] = _weighted_corr(w, x, y) - _weighted_corr(w, b, y)
    delta = colcorr(pred, truth) - colcorr(base, truth)
    se = np.nanstd(delta_boot, axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = delta / se
    p = sps.norm.sf(z)
    p[~np.isfinite(z)] = np.nan
    zc = sps.norm.ppf(0.5 + CONFIDENCE / 2)
    # Interval for the mean difference over genes: the same cell resamples, averaged over genes.
    mean_boot = np.nanmean(delta_boot, axis=1)
    lo, hi = np.quantile(mean_boot, [(1 - CONFIDENCE) / 2, 1 - (1 - CONFIDENCE) / 2])
    return dict(delta=delta, se=se, p=p, ci_low=delta - zc * se, ci_high=delta + zc * se,
                mean_delta=float(np.nanmean(delta)), mean_delta_ci=[float(lo), float(hi)])


def classify_genes(r_plan, p_ceiling, p_beyond, alpha=FDR):
    """README 4.1 three-way classification of in-scope genes.

    0 = cannot judge (ceiling not above null); 1 = not beyond the cell-type baseline;
    2 = predicts variation beyond cell type.
    """
    judgeable, _ = bh_reject(p_ceiling, alpha)
    beyond, q = bh_reject(np.where(judgeable, p_beyond, np.nan), alpha)
    label = np.where(~judgeable, 0, np.where(beyond, 2, 1))
    return label, q


# ----------------------------------------------------------------------------- task T rules
def macro_f1(y, proba, n_classes=None):
    y = np.asarray(y)
    pred = np.asarray(proba).argmax(1)
    k = n_classes or int(max(y.max(), pred.max()) + 1)
    f = []
    for c in range(k):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def per_class_recall(y, proba, n_classes):
    pred = np.asarray(proba).argmax(1)
    return [float(np.mean(pred[y == c] == c)) if np.any(y == c) else float("nan") for c in range(n_classes)]


def log_loss(y, proba, eps=1e-7):
    p = np.clip(np.asarray(proba)[np.arange(len(y)), y], eps, 1.0)
    return float(-np.mean(np.log(p)))


def ovr_auroc(y, proba):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))


def paired_gain_bootstrap(y, proba_plan, proba_base, n_boot=2000, seed=0, metric=macro_f1):
    """Gain of plan over baseline with a patient-level bootstrap interval.

    `proba_plan` and `proba_base` are lists with one out-of-fold probability matrix per CV repeat (all for
    the same patients in the same order). The statistic is the metric difference averaged over repeats;
    each bootstrap replicate resamples patients once and reuses the sample for every repeat.
    """
    y = np.asarray(y)
    k = proba_plan[0].shape[1]
    rng = np.random.default_rng(seed)

    def stat(idx):
        return float(np.mean([metric(y[idx], pp[idx], k) - metric(y[idx], pb[idx], k)
                              for pp, pb in zip(proba_plan, proba_base)]))

    point = stat(np.arange(len(y)))
    boots = np.array([stat(rng.integers(0, len(y), len(y))) for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [(1 - CONFIDENCE) / 2, 1 - (1 - CONFIDENCE) / 2])
    return dict(gain=point, ci_low=float(lo), ci_high=float(hi), se=float(boots.std(ddof=1)),
                frac_boot_positive=float(np.mean(boots > 0)))


def three_way_conclusion(ci_low, ci_high):
    """README 4.2: the conclusion depends only on the 95% interval of the gain."""
    if ci_low > 0:
        return "improves"
    if ci_high < 0:
        return "worse"
    return "insufficient_evidence"
