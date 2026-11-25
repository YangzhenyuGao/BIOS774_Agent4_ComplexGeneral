"""Basic tier for incremental-value classification on a cohort (README 6.5, task T).

A "runner" is the executable form of one model family. Every runner exposes

    configs()                                   candidate configurations (dicts) the family may choose from
    fit_predict(train_ids, test_ids, config, seed) -> (probabilities for test_ids, report dict)

and is always fitted inside the training patients it is given: imputation, scaling, PCA and any tuning
never see the test patients. The cross-validation engines at the bottom are shared by the agent's own
validation and by the evaluator's unified protocol.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ..stats import macro_f1

CLINICAL_CATEGORICAL = ["stage", "histological_type", "er", "pr", "her2"]
CLINICAL_NUMERIC = ["age"]
RECEPTORS = ["er", "pr", "her2"]


class ClinicalEncoder:
    """One-hot encoding with an explicit 'missing' level and rare-level pooling; numeric median imputation."""

    def __init__(self, variables=None, min_count=8):
        variables = variables or CLINICAL_CATEGORICAL + CLINICAL_NUMERIC
        self.cat = [v for v in variables if v in CLINICAL_CATEGORICAL]
        self.num = [v for v in variables if v in CLINICAL_NUMERIC]
        self.min_count = min_count

    def fit(self, frame: pd.DataFrame):
        self.levels_ = {}
        for c in self.cat:
            counts = frame[c].fillna("missing").astype(str).value_counts()
            self.levels_[c] = [lv for lv, n in counts.items() if n >= self.min_count]
        self.median_ = {c: float(frame[c].median()) for c in self.num}
        self.scale_ = {c: float(frame[c].std() or 1.0) for c in self.num}
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.cat:
            values = frame[c].fillna("missing").astype(str).values
            for lv in self.levels_[c]:
                cols.append((values == lv).astype(np.float32))
            cols.append((~np.isin(values, self.levels_[c])).astype(np.float32))   # pooled rare / unseen level
        for c in self.num:
            x = frame[c].astype(float)
            cols.append(x.isna().astype(np.float32).values)
            cols.append(((x.fillna(self.median_[c]) - self.median_[c]) / self.scale_[c]).astype(np.float32).values)
        return np.column_stack(cols) if cols else np.zeros((len(frame), 0), dtype=np.float32)


def clinical_matrix(view, train_ids, test_ids, variables=None):
    enc = ClinicalEncoder(variables).fit(view.clinical.loc[train_ids])
    return enc.transform(view.clinical.loc[train_ids]), enc.transform(view.clinical.loc[test_ids])


def image_matrix(view, encoders, train_ids, test_ids, n_pca=64, seed=0):
    """Patient-level image features: standardise and reduce each encoder inside the training patients."""
    tr_parts, te_parts = [], []
    for enc in encoders:
        xtr, xte = view.patient_features(enc, train_ids), view.patient_features(enc, test_ids)
        sc = StandardScaler().fit(xtr)
        xtr, xte = sc.transform(xtr), sc.transform(xte)
        if n_pca and n_pca < min(xtr.shape):
            pca = PCA(n_pca, random_state=seed).fit(xtr)
            xtr, xte = pca.transform(xtr), pca.transform(xte)
            scale = xtr.std(0).mean() or 1.0
            xtr, xte = xtr / scale, xte / scale
        tr_parts.append(xtr), te_parts.append(xte)
    return np.hstack(tr_parts).astype(np.float32), np.hstack(te_parts).astype(np.float32)


def _full_proba(model, x, n_classes):
    proba = np.zeros((len(x), n_classes), dtype=np.float64)
    proba[:, model.classes_] = model.predict_proba(x)
    return proba


class ClinicalOnly:
    """Clinical variables only. With tune=True this is the evaluator's floor: a wide, fixed grid."""

    name = "clinical_only"

    def __init__(self, spec, view, scale="full", seed=0):
        self.view, self.seed = view, seed
        self.variables = spec.get("inputs", {}).get("clinical", {}).get("variables")
        p = spec.get("model", {}).get("params", {})
        self.tune = p.get("tune", True)
        self.fixed = dict(kind="logreg", C=p.get("C", 1.0), class_weight=p.get("class_weight", "balanced"))

    def configs(self):
        if not self.tune:
            return [self.fixed]
        grid = [dict(kind="logreg", C=c, class_weight=w) for c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
                for w in ("balanced", None)]
        grid += [dict(kind="gbm", max_iter=m, learning_rate=lr, class_weight="balanced")
                 for m, lr in ((60, 0.05), (120, 0.05), (60, 0.1))]
        return grid

    def fit_predict(self, train_ids, test_ids, config, seed):
        t0 = time.time()
        xtr, xte = clinical_matrix(self.view, train_ids, test_ids, self.variables)
        y = self.view.y(train_ids)
        k = len(self.view.classes)
        if config["kind"] == "logreg":
            m = LogisticRegression(C=config["C"], class_weight=config["class_weight"], max_iter=3000).fit(xtr, y)
        else:
            m = HistGradientBoostingClassifier(max_iter=config["max_iter"], learning_rate=config["learning_rate"],
                                               class_weight=config["class_weight"], max_depth=3, random_state=seed).fit(xtr, y)
        return _full_proba(m, xte, k), dict(seconds=round(time.time() - t0, 2), converged=True)


class PatientMeanLogReg:
    """Simple reference: clinical variables plus mean-pooled image features, multinomial logistic regression."""

    name = "patient_mean_logreg"

    def __init__(self, spec, view, scale="full", seed=0):
        self.view, self.seed = view, seed
        inp = spec.get("inputs", {})
        self.encoders = inp.get("image", {}).get("encoders") or view.encoders
        self.variables = inp.get("clinical", {}).get("variables")
        self.use_clinical = inp.get("clinical", {}).get("use", True)
        p = spec.get("model", {}).get("params", {})
        as_list = lambda x, default: list(x) if isinstance(x, (list, tuple)) and x else ([x] if x not in (None, [], ()) else list(default))
        encoders = [e for e in as_list(p.get("encoder") or p.get("encoders"), self.encoders) if e in view.encoders] or self.encoders
        # A plan may give its grid as lists of values (encoder, n_pca, C); the product is the candidate set.
        self.grid = p.get("grid") or [dict(encoders=[e], n_pca=int(n), C=float(c)) for e in encoders
                                      for n in as_list(p.get("n_pca"), (16, 64)) for c in as_list(p.get("C"), (0.03, 0.3))]
        # No trimming at pilot scale: these fits take under a second, and cutting the grid to an arbitrary subset
        # would make the pilot score depend on which configurations happened to survive.

    def configs(self):
        return self.grid

    def fit_predict(self, train_ids, test_ids, config, seed):
        t0 = time.time()
        xi_tr, xi_te = image_matrix(self.view, config["encoders"], train_ids, test_ids, config.get("n_pca", 64), seed)
        if self.use_clinical:
            xc_tr, xc_te = clinical_matrix(self.view, train_ids, test_ids, self.variables)
            xi_tr, xi_te = np.hstack([xc_tr, xi_tr]), np.hstack([xc_te, xi_te])
        m = LogisticRegression(C=config.get("C", 0.3), class_weight="balanced", max_iter=3000).fit(xi_tr, self.view.y(train_ids))
        return _full_proba(m, xi_te, len(self.view.classes)), dict(seconds=round(time.time() - t0, 2), converged=True)


# ----------------------------------------------------------------------------- cross-validation engines
def make_folds(y, k, repeats, seed):
    """List of repeats; each repeat is a list of (train_idx, test_idx) index arrays, stratified by class."""
    return [[(tr, te) for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed + r).split(np.zeros(len(y)), y)]
            for r in range(repeats)]


def select_config(runner, ids, y, configs, inner_k, seed):
    """Inner cross-validation inside `ids`: return the index of the best configuration and all scores."""
    if len(configs) == 1:
        return 0, [None]
    scores = []
    inner = make_folds(y, inner_k, 1, seed)[0]
    for cfg in configs:
        oof = np.zeros((len(ids), len(runner.view.classes)))
        for tr, te in inner:
            oof[te], _ = runner.fit_predict(ids[tr], ids[te], cfg, seed)
        scores.append(macro_f1(y, oof, oof.shape[1]))
    return int(np.argmax(scores)), scores


def run_outer_fold(runner, ids, y, train_idx, test_idx, selection="nested", inner_k=3, seed=0):
    """One outer fold. selection: 'nested' (configuration chosen inside the training patients) or 'first'."""
    configs = runner.configs()
    t0 = time.time()
    if selection == "nested":
        best, inner_scores = select_config(runner, ids[train_idx], y[train_idx], configs, inner_k, seed)
    else:
        best, inner_scores = 0, [None]
    proba, report = runner.fit_predict(ids[train_idx], ids[test_idx], configs[best], seed)
    return dict(proba=proba, chosen=best, config=configs[best], inner_scores=inner_scores, report=report,
                seconds=round(time.time() - t0, 2), n_configs=len(configs))


def cross_validate(runner, ids, y, folds, selection="nested", inner_k=3, seed=0, log=print):
    """Out-of-fold probabilities for every repeat.

    selection='nested'      configuration chosen by inner CV inside each training fold (unbiased)
    selection='same_folds'  every configuration is run on the folds and the best one is reported (optimistic)
    selection='first'       no tuning; the first configuration is used
    """
    k = len(runner.view.classes)
    configs = runner.configs()
    out = dict(proba=[], chosen=[], reports=[], selection=selection, n_configs=len(configs))
    for r, repeat in enumerate(folds):
        if selection == "same_folds" and len(configs) > 1:
            best_oof, best_score, best_cfg, reports = None, -1.0, 0, []
            for ci, cfg in enumerate(configs):
                oof, reps = np.zeros((len(ids), k)), []
                for tr, te in repeat:
                    oof[te], rep = runner.fit_predict(ids[tr], ids[te], cfg, seed + r)
                    reps.append(rep)
                s = macro_f1(y, oof, k)
                if s > best_score:
                    best_oof, best_score, best_cfg, reports = oof, s, ci, reps
            out["proba"].append(best_oof), out["chosen"].append([best_cfg] * len(repeat)), out["reports"].append(reports)
        else:
            oof, chosen, reports = np.zeros((len(ids), k)), [], []
            for f, (tr, te) in enumerate(repeat):
                res = run_outer_fold(runner, ids, y, tr, te, "nested" if selection == "nested" else "first", inner_k, seed + r)
                oof[te] = res["proba"]
                chosen.append(res["chosen"]), reports.append(res["report"])
                log(f"  repeat {r} fold {f}: chosen config {res['chosen']} of {res['n_configs']} ({res['seconds']}s)")
            out["proba"].append(oof), out["chosen"].append(chosen), out["reports"].append(reports)
    return out
