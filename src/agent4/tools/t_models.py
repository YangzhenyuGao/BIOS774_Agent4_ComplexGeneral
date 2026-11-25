"""Ambitious tier for incremental-value classification on a cohort (README 6.5, task T).

Every component is a "runner" with the interface of tools/t_basic.py:

    runner.configs()                                        candidate settings of the plan (dict overrides)
    runner.fit_predict(train_ids, test_ids, config, seed)   -> (probabilities for test_ids, report)
    runner.resolved / runner.ignored                        hyper-parameters actually used / plan fields not understood

Everything that is learned is learned inside train_ids: clinical encoding, tile scaling and PCA, prototypes,
the clinical offset model, self-supervised pre-adaptation (which may additionally see unlabelled patients)
and the early-stopping split. Test patients are only predicted, one bag at a time, so a test patient's
prediction never depends on any other test patient.

Neural components share one configurable network (FusionNet): per-encoder projection, fusion across
encoders, a tile aggregator, a clinical branch, a fusion rule and a flat or hierarchical head.
"""
from __future__ import annotations

import copy
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from . import t_basic as tb

torch.set_num_threads(int(os.environ.get("A4_THREADS", 8)))

NEURAL = ("attention_mil", "multi_encoder_fusion", "conditional_fusion", "incremental_offset", "hierarchical",
          "ssl_preadapted_mil")
COMPONENT_DEFAULTS = {
    "attention_mil": dict(level="tile", fusion="early_concat"),
    "multi_encoder_fusion": dict(level="patient", fusion="early_concat", encoders="all"),
    "conditional_fusion": dict(level="tile", fusion="film"),
    "incremental_offset": dict(level="tile", fusion="offset"),
    "hierarchical": dict(level="tile", fusion="early_concat", head="hierarchical"),
    "ssl_preadapted_mil": dict(level="tile", fusion="early_concat"),
}
_AGG = {"gatedattention": "gated_attention", "abmil": "gated_attention", "attention": "gated_attention", "gated": "gated_attention",
        "multiheadattention": "multihead_attention", "multihead": "multihead_attention", "transformer": "transformer",
        "transmil": "transformer", "transformercls": "transformer", "selfattention": "transformer", "mean": "mean",
        "meanpool": "mean", "average": "mean"}
_ENC_FUSION = {"attention": "attention", "gated": "gated", "gating": "gated", "gate": "gated", "concat": "concat",
               "concatenate": "concat", "concatenation": "concat"}
_FUSION = {"earlyconcat": "early_concat", "early": "early_concat", "concat": "early_concat", "earlyfusion": "early_concat",
           "lateaverage": "late_average", "late": "late_average", "latefusion": "late_average", "average": "late_average",
           "offset": "offset", "incremental": "offset", "incrementaloffset": "offset", "residual": "offset", "film": "film",
           "crossattention": "cross_attention", "crossattn": "cross_attention", "none": "none", "imageonly": "none"}
_TRAIN_KEYS = {"max_epochs", "epochs", "n_epochs", "early_stopping_patience", "patience", "lr", "weight_decay", "class_weighting",
               "class_weight", "batch_size", "tile_resample", "tiles_per_bag", "mixup_alpha", "mixup", "tile_dropout",
               "n_seeds_ensemble", "n_seeds", "val_fraction"}


# ----------------------------------------------------------------------------- plan parsing
def _key(v):
    return re.sub(r"[^a-z0-9]", "", str(v).lower()) if v is not None else ""


def _take(d, names, default=None):
    """Pop the first present, non-null name from d (and drop the synonyms); return default if absent."""
    out = default
    found = False
    for n in names:
        if n in d:
            v = d.pop(n)
            if not found and v is not None:
                out, found = v, True
    return out


def _num(v, default, lo, hi, cast=float):
    """Coerce a plan value to a bounded number; lists give their first element, junk gives the default."""
    if isinstance(v, bool):
        return default
    if isinstance(v, (list, tuple)) and v:
        v = v[0]
    try:
        return cast(min(max(float(v), lo), hi))
    except (TypeError, ValueError):
        return default


def _bool(v, default):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "on")
    return default if v is None else bool(v)


def _match_encoders(requested, available):
    table = {_key(e): e for e in available}
    if requested is None or (isinstance(requested, str) and _key(requested) in ("all", "any", "")):
        return list(available), []
    if isinstance(requested, str):
        requested = [requested]
    out, unknown = [], []
    for r in requested:
        k = _key(r)
        hit = table.get(k) or next((e for kk, e in table.items() if k and (kk.startswith(k) or k.startswith(kk))), None)
        if hit and hit not in out:
            out.append(hit)
        elif not hit:
            unknown.append(str(r))
    return out, unknown


def _groups(value, classes):
    """Class-index groups for a hierarchical head; None when the request does not partition the classes."""
    k = len(classes)
    index = {_key(c): i for i, c in enumerate(classes)}
    groups = None
    if isinstance(value, (list, tuple)) and value:
        try:
            groups = [[index[_key(c)] if _key(c) in index else int(c) for c in (g if isinstance(g, (list, tuple)) else [g])]
                      for g in value]
        except (KeyError, TypeError, ValueError):
            groups = None
    if groups is None or sorted(i for g in groups for i in g) != list(range(k)):
        named = [[index.get("luma"), index.get("lumb")], [index.get("basal")], [index.get("her2")]]
        if k == 4 and all(i is not None for g in named for i in g):
            groups = named
        else:
            groups = [[0, 1]] + [[i] for i in range(2, k)] if k >= 3 else None
    return groups


def _grid(spec, scale):
    """Candidate settings a plan asks to be selected among (at most 4; at most 2 at pilot scale)."""
    training = spec.get("training") or {}
    params = (spec.get("model") or {}).get("params") or {}
    places = [(training.get("tuning") or {}).get("grid") if isinstance(training.get("tuning"), dict) else None,
              training.get("tuning.grid"), training.get("grid"),
              (spec.get("tuning") or {}).get("grid") if isinstance(spec.get("tuning"), dict) else None,
              (params.get("tuning") or {}).get("grid") if isinstance(params.get("tuning"), dict) else None, params.get("grid")]
    grid = next((g for g in places if isinstance(g, list) and g and all(isinstance(x, dict) for x in g)), None)
    grid = [dict(g) for g in (grid or [{}])][:4]
    return grid[:2] if scale == "pilot" else grid


def _route(overrides, params, training, image, clinical, top):
    for k, v in (overrides or {}).items():
        key = str(k).split(".")[-1]
        if key in ("model", "params") and isinstance(v, dict):
            params.update(v.get("params", v) if isinstance(v.get("params", v), dict) else v)
        elif key in ("training", "augmentation") and isinstance(v, dict):
            training.update(v)
        elif key == "inputs" and isinstance(v, dict):
            image.update(v.get("image") or {}), clinical.update(v.get("clinical") or {})
        elif key == "image" and isinstance(v, dict):
            image.update(v)
        elif key == "encoder":
            image["encoders"] = [v]
        elif key in ("encoders", "reduce"):
            image[key] = v
        elif key == "fusion":
            top["fusion"] = v
        elif key in _TRAIN_KEYS:
            training[key] = v
        else:
            params[key] = v


def resolve(spec, view, scale="full", overrides=None):
    """Turn a plan (plus one candidate setting) into the full set of hyper-parameters. Returns (cfg, ignored)."""
    spec = spec or {}
    model = spec.get("model") or {}
    component = model.get("component")
    params = dict(model.get("params") or {})
    training = dict(spec.get("training") or {})
    inputs = spec.get("inputs") if isinstance(spec.get("inputs"), dict) else {}
    image = dict(inputs.get("image") or {}) if isinstance(inputs.get("image"), dict) else {}
    clinical = dict(inputs.get("clinical") or {}) if isinstance(inputs.get("clinical"), dict) else {}
    top = dict(fusion=spec.get("fusion"))
    for nest in ("image_branch", "then", "mil", "aggregator_params"):
        if isinstance(params.get(nest), dict):
            for k, v in params.pop(nest).items():
                params.setdefault(k, v)
    for nest in ("augmentation", "early_stopping"):
        block = training.pop(nest, None)
        if isinstance(block, dict):
            for k, v in block.items():
                training.setdefault("early_stopping_patience" if (nest == "early_stopping" and k == "patience") else k, v)
    for k in ("tuning", "tuning.grid", "grid"):
        training.pop(k, None), params.pop(k, None)
    _route(overrides, params, training, image, clinical, top)

    d = dict(COMPONENT_DEFAULTS.get(component, {}))
    c = dict(component=component)
    notes = []
    # ---- inputs
    enc_candidates = [_take(params, ["encoders", "encoder"], None), image.pop("encoders", None), image.pop("encoder", None)]
    enc_request = next((e for e in enc_candidates if e), None)
    if enc_request is None and d.get("encoders") != "all" and component in NEURAL + ("morphology_prototypes",):
        enc_request = [view.encoders[0]]
    encoders, unknown = _match_encoders(enc_request, view.encoders)
    if unknown:
        notes.append(f"unknown encoders dropped: {unknown}")
    c["encoders"] = encoders or [view.encoders[0]]
    level_candidates = [_take(params, ["level"], None), image.pop("level", None), d.get("level"), "patient"]
    level = _key(next(v for v in level_candidates if v))
    c["level"] = "tile" if level.startswith("tile") or component == "ssl_preadapted_mil" else "patient"
    reduce = image.pop("reduce", None)
    r_type = _key(reduce.get("type")) if isinstance(reduce, dict) else _key(reduce)
    r_dim = reduce.get("dim") if isinstance(reduce, dict) else None
    r_dim = r_dim or image.pop("dim", None) or image.pop("n_pca", None)
    c["reduce_type"] = "raw" if r_type in ("none", "linearprojection", "raw") else "pca"
    c["reduce_dim"] = _num(r_dim, 128 if c["level"] == "tile" else 64, 4, 512, int)
    c["max_tiles"] = _num(_take(params, ["max_tiles", "bag_size", "max_bag_size"]), 256, 8, 512, int)
    if c["level"] == "tile" and c["reduce_type"] == "raw":
        total = sum(int(view.tiles(e).shape[1]) for e in c["encoders"])
        if total > 3000:
            c["reduce_type"], c["reduce_dim"] = "pca", 256
            notes.append(f"raw tile features of {len(c['encoders'])} encoders ({total} dims) do not fit the per-fit budget; "
                         "reduced by in-fold PCA to 256 per encoder")
    c["clinical_use"] = _bool(clinical.pop("use", None), True)
    variables = clinical.pop("variables", None)
    c["clinical_variables"] = [v for v in variables if v in tb.CLINICAL_CATEGORICAL + tb.CLINICAL_NUMERIC] if isinstance(variables, list) else None
    # ---- architecture
    mode = _FUSION.get(_key(_take(params, ["mode"], None)))
    param_fusion = _FUSION.get(_key(_take(params, ["fusion", "clinical_fusion"], None)))
    fusion = _FUSION.get(_key(top["fusion"])) or param_fusion
    if component == "conditional_fusion":
        fusion = fusion if fusion in ("film", "cross_attention") else (mode if mode in ("film", "cross_attention") else "film")
    elif component == "incremental_offset":
        fusion = "offset"
    c["fusion"] = fusion or d.get("fusion", "early_concat")
    if not c["clinical_use"]:
        c["fusion"] = "none"
    c["aggregator"] = _AGG.get(_key(_take(params, ["aggregator", "aggregation", "pooling"])), "gated_attention")
    c["encoder_fusion"] = _ENC_FUSION.get(_key(_take(params, ["encoder_fusion", "fusion_over_encoders"])), "attention")
    c["embed_dim"] = _num(_take(params, ["embed_dim", "embedding_dim", "hidden_dim", "hidden", "d_model"]), 128, 16, 256, int)
    c["clin_dim"] = _num(_take(params, ["clin_dim", "clinical_dim"]), 32, 8, 128, int)
    c["dropout"] = _num(_take(params, ["dropout"]), 0.25, 0.0, 0.6)
    c["heads"] = _num(_take(params, ["heads", "n_heads", "num_heads", "attention_heads"]), 4, 1, 8, int)
    c["layers"] = _num(_take(params, ["layers", "n_layers", "num_layers", "depth"]), 1, 1, 3, int)
    head = _key(_take(params, ["head"], None) or d.get("head") or "flat")
    c["head"] = "hierarchical" if head.startswith("hier") or component == "hierarchical" else "flat"
    c["groups"] = _groups(_take(params, ["groups", "coarse_groups"]), view.classes) if c["head"] == "hierarchical" else None
    if c["head"] == "hierarchical" and c["groups"] is None:
        c["head"] = "flat"
    fine = _key(_take(params, ["fine_inputs"], "image+clinical"))
    c["fine_inputs"] = "image" if fine in ("image", "imageonly") else "image+clinical"
    c["init_zero"] = _bool(_take(params, ["init_zero", "zero_init"]), True)
    c["clinical_C"] = _num(_take(params, ["clinical_C", "clinical_c"]), 1.0, 1e-3, 100.0)
    # ---- self-supervised pre-adaptation
    c["pretrain_epochs"] = _num(_take(params, ["pretrain_epochs", "ssl_epochs"]), 20 if component == "ssl_preadapted_mil" else 0, 0, 100, int)
    c["uses_unlabelled"] = _bool(_take(params, ["uses_unlabelled", "use_unlabelled", "uses_unlabeled", "use_unlabeled"]), True)
    c["temperature"] = _num(_take(params, ["temperature"]), 0.2, 0.02, 2.0)
    # ---- non-neural components
    c["n_prototypes"] = _num(_take(params, ["n_prototypes", "prototypes", "n_clusters", "k"]), 32, 4, 128, int)
    c["classifier"] = "gbm" if _key(_take(params, ["classifier"])) in ("gbm", "gradientboosting", "boosting", "hgb") else "logreg"
    c["proto_pca"] = _num(_take(params, ["proto_pca", "pca_dim"]), 32, 4, 128, int)
    c["C"] = _num(_take(params, ["C", "c"]), 1.0, 1e-3, 100.0)
    c["n_pca"] = _num(_take(params, ["n_pca"]), 16, 2, 128, int)
    c["max_iter"] = _num(_take(params, ["max_iter", "n_estimators"]), 100, 10, 500, int)
    c["learning_rate"] = _num(_take(params, ["learning_rate"]), 0.05, 0.005, 0.5)
    c["max_depth"] = _num(_take(params, ["max_depth"]), 3, 1, 8, int)
    members = _take(params, ["members"], None)
    c["members"] = [m for m in members if isinstance(m, dict) and m.get("component")] if isinstance(members, list) else []
    _take(params, ["meta"])
    c["inner_k"] = _num(_take(params, ["inner_k", "oof_folds", "inner_folds", "folds"]), 3, 2, 5, int)
    if scale == "pilot":
        c["inner_k"] = 2
    c["meta_C"] = _num(_take(params, ["meta_C", "meta_c"]), 1.0, 1e-3, 100.0)
    # ---- training
    c["max_epochs"] = _num(_take(training, ["max_epochs", "epochs", "n_epochs"]), 60, 1, 300, int)
    c["patience"] = _num(_take(training, ["early_stopping_patience", "patience"]), 10, 2, 50, int)
    default_lr = 3e-4 if (c["aggregator"] == "transformer" and c["level"] == "tile" and component in NEURAL) else 1e-3
    c["lr"] = _num(_take(training, ["lr", "learning_rate"]), default_lr, 1e-5, 1e-2)
    c["weight_decay"] = _num(_take(training, ["weight_decay"]), 1e-2, 0.0, 0.3)
    c["class_weighting"] = "none" if _key(_take(training, ["class_weighting", "class_weight", "class_weights"])) in ("none", "false", "unweighted") else "balanced"
    c["batch_size"] = _num(_take(training, ["batch_size"]), 64, 8, 256, int)
    resample = _take(training, ["tile_resample", "tiles_per_bag", "tile_resampling"])
    resample = c["max_tiles"] if resample is False else (128 if resample is True or resample is None else resample)
    c["tile_resample"] = _num(resample, 128, 4, c["max_tiles"], int)
    mix = _take(training, ["mixup_alpha", "mixup"])
    c["mixup_alpha"] = _num(0.2 if mix is True else mix, 0.0, 0.0, 1.0)
    c["tile_dropout"] = _num(_take(training, ["tile_dropout"]), 0.0, 0.0, 0.8)
    seeds = _take(training, ["n_seeds_ensemble", "n_seeds", "ensemble_seeds", "seeds"])
    c["n_seeds_ensemble"] = 1 if scale == "pilot" else _num(len(seeds) if isinstance(seeds, (list, tuple)) else seeds, 1, 1, 3, int)
    c["val_fraction"] = _num(_take(training, ["val_fraction", "validation_fraction"]), 0.15, 0.05, 0.3)
    c["notes"] = notes
    ignored = ([f"model.params.{k}" for k in params] + [f"training.{k}" for k in training] +
               [f"inputs.image.{k}" for k in image] + [f"inputs.clinical.{k}" for k in clinical])
    return c, sorted(ignored)


# ----------------------------------------------------------------------------- tile storage
class TileStore:
    """Tiles of the requested patients held in RAM as one compact float32 array per encoder.

    Bags larger than max_tiles are cut once with a fixed seed, so every fit sees the same bags.
    """

    def __init__(self, view, ids, max_tiles, seed=0):
        self.view, self.ids, self.max_tiles = view, np.asarray(ids), int(max_tiles)
        rng = np.random.default_rng(seed)
        rows, counts = [], []
        known = set(view.tile_index["patient_id"].unique())
        with_tiles = [i for i in self.ids if i in known]
        per_patient = dict(zip(with_tiles, view.tile_rows(with_tiles))) if with_tiles else {}
        for pid in self.ids:
            r = np.asarray(per_patient.get(pid, np.zeros(0, dtype=np.int64)))
            if len(r) > self.max_tiles:
                r = np.sort(rng.choice(r, self.max_tiles, replace=False))
            rows.append(r), counts.append(len(r))
        self.counts = np.asarray(counts)
        self.rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)])
        self.position = {pid: i for i, pid in enumerate(self.ids)}
        n = len(self.rows)
        self.pad_index = np.full((len(self.ids), max(int(self.counts.max()), 1)), n, dtype=np.int64)   # n = zero row
        for i, cnt in enumerate(self.counts):
            self.pad_index[i, :cnt] = np.arange(self.offsets[i], self.offsets[i + 1])
        self._raw = {}

    def has(self, ids):
        return all(i in self.position for i in ids)

    def pos(self, ids):
        return np.array([self.position[i] for i in ids], dtype=np.int64)

    def raw(self, encoder):
        if encoder not in self._raw:
            order = np.argsort(self.rows, kind="stable")
            out = np.empty((len(self.rows), self.view.tiles(encoder).shape[1]), dtype=np.float32)
            out[order] = np.asarray(self.view.tiles(encoder)[self.rows[order]], dtype=np.float32)
            self._raw[encoder] = out
        return self._raw[encoder]

    def tile_positions(self, patient_pos):
        """Rows of the compact arrays that belong to the given patients."""
        if len(patient_pos) == 0:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in patient_pos])

    def reduce(self, encoder, train_patient_pos, kind, dim, seed):
        """Scale (and PCA-reduce) every tile of one encoder with statistics from training tiles only."""
        raw = self.raw(encoder)
        rng = np.random.default_rng(seed)
        train_rows = self.tile_positions(train_patient_pos)
        sub = np.sort(rng.choice(train_rows, min(len(train_rows), 20000), replace=False))
        xs = raw[sub]
        mu, sd = xs.mean(0), xs.std(0) + 1e-6
        if kind != "pca":
            return (raw - mu) / sd
        xs = (xs - mu) / sd
        dim = int(min(dim, xs.shape[1], max(xs.shape[0] - 1, 1)))
        pca = PCA(dim, svd_solver="randomized", random_state=seed).fit(xs)
        scale = float(np.sqrt(pca.explained_variance_.mean())) or 1.0
        w = (pca.components_ / sd).T / scale                                    # raw -> reduced in one product
        b = ((mu / sd + pca.mean_) @ pca.components_.T) / scale
        return (raw @ w.astype(np.float32) - b.astype(np.float32)).astype(np.float32)

    def padded(self, reduced_list, patient_pos):
        """(patients, tiles, features) tensor with a validity mask, features of all encoders concatenated."""
        idx = self.pad_index[patient_pos]
        t = max(int(self.counts[patient_pos].max()), 1)
        idx = idx[:, :t]
        parts = []
        for z in reduced_list:
            ext = np.vstack([z, np.zeros((1, z.shape[1]), dtype=np.float32)])
            parts.append(ext[idx])
        mask = idx < len(self.rows)
        mask[:, 0] = True                                                      # a patient without tiles keeps one zero tile
        return np.concatenate(parts, axis=-1), mask


# ----------------------------------------------------------------------------- network
class _Attention(nn.Module):
    """Gated attention pooling (Ilse et al.); with several heads the pooled vectors are concatenated and mixed."""

    def __init__(self, dim, heads=1):
        super().__init__()
        hidden = max(32, dim // 2)
        self.v, self.u, self.w = nn.Linear(dim, hidden), nn.Linear(dim, hidden), nn.Linear(hidden, heads)
        self.mix = nn.Identity() if heads == 1 else nn.Linear(heads * dim, dim)

    def forward(self, h, mask):
        a = self.w(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h))).masked_fill(~mask.unsqueeze(-1), float("-inf"))
        pooled = torch.einsum("btk,bte->bke", torch.softmax(a, dim=1), h)
        return self.mix(pooled.flatten(1))


class _Transformer(nn.Module):
    def __init__(self, dim, heads, layers, dropout):
        super().__init__()
        heads = max(h for h in range(1, heads + 1) if dim % h == 0)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=2 * dim, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, mask):
        x = torch.cat([self.cls.expand(len(h), -1, -1), h], dim=1)
        pad = torch.cat([torch.zeros(len(h), 1, dtype=torch.bool, device=h.device), ~mask], dim=1)
        out = self.encoder(x, src_key_padding_mask=pad)
        m = mask.unsqueeze(-1).float()
        return self.norm(out[:, 0] + (out[:, 1:] * m).sum(1) / m.sum(1).clamp(min=1.0))


class _Mean(nn.Module):
    def forward(self, h, mask):
        m = mask.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


class _Head(nn.Module):
    """Flat or two-stage head returning class log-probabilities; an optional offset is added at each stage."""

    def __init__(self, feat_dim, fine_dim, n_classes, groups, zero_init):
        super().__init__()
        self.groups, self.k = groups, n_classes
        if groups is None:
            self.flat = nn.Linear(feat_dim, n_classes)
        else:
            self.coarse = nn.Linear(feat_dim, len(groups))
            self.fine = nn.ModuleDict({str(i): nn.Linear(fine_dim, len(g)) for i, g in enumerate(groups) if len(g) > 1})
        if zero_init:
            for p in self.parameters():
                nn.init.zeros_(p)

    def forward(self, feat, fine, offset=None):
        if self.groups is None:
            z = self.flat(feat)
            return F.log_softmax(z if offset is None else z + offset, dim=-1)
        zc = self.coarse(feat)
        if offset is not None:
            zc = zc + torch.stack([torch.logsumexp(offset[:, g], dim=-1) for g in self.groups], dim=-1)
        lc = F.log_softmax(zc, dim=-1)
        cols = [None] * self.k
        for i, g in enumerate(self.groups):
            if len(g) == 1:
                cols[g[0]] = lc[:, i]
            else:
                zf = self.fine[str(i)](fine)
                lf = F.log_softmax(zf if offset is None else zf + offset[:, g], dim=-1)
                for j, cls in enumerate(g):
                    cols[cls] = lc[:, i] + lf[:, j]
        return torch.stack(cols, dim=-1)


class FusionNet(nn.Module):
    def __init__(self, dims, n_clin, n_classes, c):
        super().__init__()
        self.dims, self.c, e, p = list(dims), c, c["embed_dim"], c["dropout"]
        self.tile = c["level"] == "tile"
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(d, e), nn.LayerNorm(e), nn.GELU(), nn.Dropout(p)) for d in dims])
        n = len(dims)
        self.enc_fusion = c["encoder_fusion"] if n > 1 else "single"
        if self.enc_fusion == "concat":
            self.enc_mix = nn.Linear(n * e, e)
        elif self.enc_fusion == "attention":
            self.enc_score = nn.Linear(e, 1)
        elif self.enc_fusion == "gated":
            self.enc_gate = nn.Linear(n * e, n)
        self.fusion = c["fusion"] if n_clin > 0 else ("offset" if c["fusion"] == "offset" else "none")
        self.use_clin = self.fusion in ("early_concat", "late_average", "film", "cross_attention")
        ec = c["clin_dim"]
        if self.use_clin:
            self.clin = nn.Sequential(nn.Linear(n_clin, ec), nn.GELU(), nn.Dropout(p))
        if self.fusion == "film":
            self.film = nn.Linear(ec, 2 * e)
            nn.init.zeros_(self.film.weight), nn.init.zeros_(self.film.bias)
        if self.fusion == "cross_attention":
            heads = max(h for h in range(1, c["heads"] + 1) if e % h == 0)
            self.query, self.xattn, self.xnorm = nn.Linear(ec, e), nn.MultiheadAttention(e, heads, dropout=p, batch_first=True), nn.LayerNorm(e)
        if self.tile:
            self.agg = {"gated_attention": lambda: _Attention(e, 1), "multihead_attention": lambda: _Attention(e, c["heads"]),
                        "transformer": lambda: _Transformer(e, c["heads"], c["layers"], p), "mean": _Mean}[c["aggregator"]]()
        self.drop = nn.Dropout(p)
        feat = e + (ec if self.fusion in ("early_concat", "film", "cross_attention") else 0)
        self.fine_image = c["fine_inputs"] == "image"
        self.head = _Head(feat, e if self.fine_image else feat, n_classes, c["groups"] if c["head"] == "hierarchical" else None,
                          zero_init=self.fusion == "offset" and c["init_zero"])
        if self.fusion == "late_average":
            self.clin_head = nn.Linear(ec, n_classes)
        self.ssl_head = nn.Linear(e, e)

    def image_embed(self, x, mask, clin_emb=None):
        hs = [proj(part) for proj, part in zip(self.proj, torch.split(x, self.dims, dim=-1))]
        if self.enc_fusion == "single":
            h = hs[0]
        elif self.enc_fusion == "concat":
            h = self.enc_mix(torch.cat(hs, dim=-1))
        else:
            stack = torch.stack(hs, dim=-2)
            if self.enc_fusion == "attention":
                h = (torch.softmax(self.enc_score(stack), dim=-2) * stack).sum(-2)
            else:
                gate = torch.sigmoid(self.enc_gate(torch.cat(hs, dim=-1))).unsqueeze(-1)
                h = (gate * stack).sum(-2) / (gate.sum(-2) + 1e-6)
        if self.fusion == "film" and clin_emb is not None:
            gamma, beta = self.film(clin_emb).chunk(2, dim=-1)
            h = h * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1) if self.tile else h * (1 + gamma) + beta
        if self.tile:
            emb = self.agg(h, mask)
            tokens, pad = h, ~mask
        else:
            emb, tokens, pad = h, torch.stack(hs, dim=1), None
        if self.fusion == "cross_attention" and clin_emb is not None:
            out, _ = self.xattn(self.query(clin_emb).unsqueeze(1), tokens, tokens, key_padding_mask=pad, need_weights=False)
            emb = self.xnorm(emb + out.squeeze(1))
        return self.drop(emb)

    def embed(self, x, mask, clin):
        clin_emb = self.clin(clin) if self.use_clin else None
        return self.image_embed(x, mask, clin_emb), clin_emb

    def classify(self, emb, clin_emb, offset):
        feat = torch.cat([emb, clin_emb], dim=-1) if self.fusion in ("early_concat", "film", "cross_attention") else emb
        lp = self.head(feat, emb if self.fine_image else feat, offset if self.fusion == "offset" else None)
        if self.fusion == "late_average":
            lp = F.log_softmax(0.5 * (lp + F.log_softmax(self.clin_head(clin_emb), dim=-1)), dim=-1)
        return lp

    def forward(self, x, mask, clin, offset):
        return self.classify(*self.embed(x, mask, clin), offset)


def _subsample_tiles(x, mask, k, gen):
    """Draw at most k valid tiles per bag (uniformly, without replacement)."""
    if x.dim() != 3 or k >= x.shape[1]:
        return x, mask
    score = torch.rand(mask.shape, generator=gen).to(x.device).masked_fill(~mask, -1.0)
    top = score.topk(k, dim=1).indices
    return x.gather(1, top.unsqueeze(-1).expand(-1, -1, x.shape[-1])), mask.gather(1, top)


def _trim(x, mask):
    if x.dim() != 3:
        return x, mask
    t = int(mask.sum(1).max().item())
    return x[:, :t], mask[:, :t]


def _predict(model, data, rows, device, batch=128):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            r = rows[i:i + batch]
            x, mask = _trim(data["x"][r], data["mask"][r]) if data["mask"] is not None else (data["x"][r], None)
            out.append(model(x.to(device), None if mask is None else mask.to(device), data["clin"][r].to(device),
                             data["offset"][r].to(device)))
    return torch.cat(out).float()


def _ssl_pretrain(model, data, rows, c, gen, device):
    """Bag-level contrastive pre-adaptation: two disjoint random halves of a patient's tiles should agree."""
    params = [p for n, p in model.named_parameters() if n.split(".")[0] in ("proj", "enc_mix", "enc_score", "enc_gate", "agg", "ssl_head")]
    opt = torch.optim.AdamW(params, lr=c["lr"], weight_decay=c["weight_decay"])
    curve = []
    for _ in range(c["pretrain_epochs"]):
        model.train()
        perm = rows[torch.randperm(len(rows), generator=gen)]
        total, n = 0.0, 0
        for i in range(0, len(perm), 128):
            r = perm[i:i + 128]
            if len(r) < 4:
                continue
            x, mask = _subsample_tiles(*_trim(data["x"][r], data["mask"][r]), c["tile_resample"], gen)
            score = torch.rand(mask.shape, generator=gen).masked_fill(~mask, 2.0)
            rank = score.argsort(1).argsort(1)
            half = (mask.sum(1) // 2).clamp(min=1).unsqueeze(1)
            m1 = mask & (rank < half)
            m2 = mask & ~m1
            m2 = torch.where(m2.any(1, keepdim=True), m2, m1)
            x = x.to(device)
            z1 = F.normalize(model.ssl_head(model.image_embed(x, m1.to(device))), dim=-1)
            z2 = F.normalize(model.ssl_head(model.image_embed(x, m2.to(device))), dim=-1)
            logits = z1 @ z2.T / c["temperature"]
            target = torch.arange(len(r), device=device)
            loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
            opt.zero_grad(), loss.backward(), opt.step()
            total, n = total + loss.item() * len(r), n + len(r)
        curve.append(round(total / max(n, 1), 5))
    return curve


def _train(model, data, y, fit_rows, val_rows, c, gen, device, weight):
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    y = y.to(device)
    best, best_state, stale, curves, stopped = float("inf"), None, 0, dict(train_loss=[], val_loss=[]), "max_epochs"
    for _ in range(c["max_epochs"]):
        model.train()
        perm = fit_rows[torch.randperm(len(fit_rows), generator=gen)]
        total = 0.0
        for i in range(0, len(perm), c["batch_size"]):
            r = perm[i:i + c["batch_size"]]
            if len(r) < 2:
                continue
            if data["mask"] is not None:
                x, mask = _subsample_tiles(*_trim(data["x"][r], data["mask"][r]), c["tile_resample"], gen)
                if c["tile_dropout"] > 0:
                    kept = mask & (torch.rand(mask.shape, generator=gen) >= c["tile_dropout"])
                    mask = torch.where(kept.any(1, keepdim=True), kept, mask)
                mask = mask.to(device)
            else:
                x, mask = data["x"][r], None
            emb, clin_emb = model.embed(x.to(device), mask, data["clin"][r].to(device))
            offset, target = data["offset"][r].to(device), y[r]
            if c["mixup_alpha"] > 0:
                lam = float(torch.distributions.Beta(c["mixup_alpha"], c["mixup_alpha"]).sample())
                lam = max(lam, 1 - lam)
                sw = torch.randperm(len(r), generator=gen).to(device)
                emb = lam * emb + (1 - lam) * emb[sw]
                clin_emb = None if clin_emb is None else lam * clin_emb + (1 - lam) * clin_emb[sw]
                offset = lam * offset + (1 - lam) * offset[sw]
                lp = model.classify(emb, clin_emb, offset)
                loss = lam * F.nll_loss(lp, target, weight=weight) + (1 - lam) * F.nll_loss(lp, target[sw], weight=weight)
            else:
                loss = F.nll_loss(model.classify(emb, clin_emb, offset), target, weight=weight)
            opt.zero_grad(), loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(r)
        val = float(F.nll_loss(_predict(model, data, val_rows, device), y[val_rows], weight=weight))
        curves["train_loss"].append(round(total / len(fit_rows), 5)), curves["val_loss"].append(round(val, 5))
        if val < best - 1e-4:
            best, best_state, stale = val, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= c["patience"]:
                stopped = "early_stopping"
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return curves, stopped


def convergence(curves, stopped):
    """README 6.3: converged when early stopping fired, or the validation loss has stopped improving at the end."""
    val = curves.get("val_loss") or []
    if stopped == "early_stopping":
        return True, f"early stopping after {len(val)} epochs; best validation loss {min(val):.4f} at epoch {int(np.argmin(val)) + 1}"
    if len(val) < 5:
        return False, f"only {len(val)} epochs were run"
    tail = max(1, int(round(0.2 * len(val))))
    before, after = min(val[:-tail]), min(val[-tail:])
    gain = (before - after) / max(abs(before), 1e-9)
    return bool(gain < 0.01), (f"reached max_epochs ({len(val)}); best validation loss improved by {gain:.1%} over the last "
                               f"{tail} epochs")


def _val_split(y, fraction, seed):
    idx = np.arange(len(y))
    if np.bincount(y).min() >= 2 and len(y) * fraction >= len(np.unique(y)):
        fit, val = next(StratifiedShuffleSplit(1, test_size=fraction, random_state=seed).split(idx, y))
    else:
        perm = np.random.default_rng(seed).permutation(idx)
        cut = max(1, int(round(fraction * len(idx))))
        fit, val = perm[cut:], perm[:cut]
    return np.sort(fit), np.sort(val)


# ----------------------------------------------------------------------------- runners
class _Base:
    def __init__(self, spec, view, scale="full", seed=0):
        self.spec, self.view, self.scale, self.seed = spec, view, scale, seed
        self.name = (spec.get("model") or {}).get("component")
        self.resolved, self.ignored = resolve(spec, view, scale)
        self._stores = {}

    def configs(self):
        return _grid(self.spec, self.scale)

    def _store(self, ids, c, need_unlabelled=False):
        key = (c["max_tiles"], need_unlabelled)
        if key not in self._stores or not self._stores[key].has(ids):
            base = self.view.clinical.index.values if need_unlabelled else self.view.labelled.index.values
            base_set = set(base)
            extra = [i for i in dict.fromkeys(ids) if i not in base_set]
            self._stores[key] = TileStore(self.view, np.concatenate([base, np.asarray(extra, dtype=base.dtype)]) if extra else base,
                                          c["max_tiles"], seed=self.seed)
        return self._stores[key]

    def _clinical(self, c, train_ids, test_ids, extra_ids=None):
        if not c["clinical_use"]:
            z = np.zeros((0, 0), dtype=np.float32)
            return (np.zeros((len(train_ids), 0), np.float32), np.zeros((len(test_ids), 0), np.float32),
                    np.zeros((0 if extra_ids is None else len(extra_ids), 0), np.float32)), z
        enc = tb.ClinicalEncoder(c["clinical_variables"]).fit(self.view.clinical.loc[train_ids])
        parts = [enc.transform(self.view.clinical.loc[ids]) for ids in (train_ids, test_ids)]
        parts.append(enc.transform(self.view.clinical.loc[extra_ids]) if extra_ids is not None and len(extra_ids) else
                     np.zeros((0, parts[0].shape[1]), np.float32))
        return tuple(parts), enc

    @staticmethod
    def _closed_form_report(t0, n_parameters=0, **extra):
        return dict(seconds=round(time.time() - t0, 2), device="cpu", epochs=0, stopped_by="closed_form", converged=True,
                    convergence_note="closed-form or tree model; no iterative training curve", curves={},
                    n_parameters=int(n_parameters), **extra)


class NeuralRunner(_Base):
    """attention_mil, multi_encoder_fusion, conditional_fusion, incremental_offset, hierarchical, ssl_preadapted_mil."""

    def _clinical_offset(self, c, x_tr, x_te, y, seed):
        """Log-probabilities of an in-fold clinical model: cross-fitted for training patients, full fit for test."""
        k = len(self.view.classes)
        make = lambda: LogisticRegression(C=c["clinical_C"], class_weight="balanced" if c["class_weighting"] == "balanced" else None, max_iter=3000)
        if x_tr.shape[1] == 0:
            return np.zeros((len(x_tr), k), np.float32), np.zeros((len(x_te), k), np.float32)
        oof = np.full((len(y), k), 1.0 / k)
        folds = min(5, int(np.bincount(y).min()))
        if folds >= 2:
            for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(x_tr, y):
                m = make().fit(x_tr[tr], y[tr])
                oof[te] = 1e-6
                oof[np.ix_(te, m.classes_)] = m.predict_proba(x_tr[te])
        m = make().fit(x_tr, y)
        full = np.full((len(x_te), k), 1e-6)
        full[:, m.classes_] = m.predict_proba(x_te)
        return np.log(np.clip(oof, 1e-6, 1)).astype(np.float32), np.log(np.clip(full, 1e-6, 1)).astype(np.float32)

    def fit_predict(self, train_ids, test_ids, config, seed):
        t0 = time.time()
        c, _ = resolve(self.spec, self.view, self.scale, config)
        train_ids, test_ids = np.asarray(train_ids), np.asarray(test_ids)
        y = self.view.y(train_ids)
        k = len(self.view.classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ssl = c["pretrain_epochs"] > 0 and c["level"] == "tile"
        extra_ids = np.zeros(0, dtype=train_ids.dtype)
        if ssl and c["uses_unlabelled"]:
            labelled = set(self.view.labelled.index)
            extra_ids = np.array([i for i in self.view.clinical.index if i not in labelled], dtype=train_ids.dtype)
        (clin_tr, clin_te, clin_ex), _ = self._clinical(c, train_ids, test_ids, extra_ids)
        off_tr, off_te = (self._clinical_offset(c, clin_tr, clin_te, y, seed) if c["fusion"] == "offset" else
                          (np.zeros((len(train_ids), k), np.float32), np.zeros((len(test_ids), k), np.float32)))
        n_tr, n_te = len(train_ids), len(test_ids)
        all_ids = np.concatenate([train_ids, test_ids, extra_ids])
        if c["level"] == "tile":
            store = self._store(all_ids, c, need_unlabelled=len(extra_ids) > 0)
            pos = store.pos(all_ids)
            reduced = [store.reduce(e, pos[:n_tr], c["reduce_type"], c["reduce_dim"], seed) for e in c["encoders"]]
            x, mask = store.padded(reduced, pos)
            dims = [z.shape[1] for z in reduced]
            del reduced
            x, mask = torch.from_numpy(x), torch.from_numpy(mask)
        else:
            parts, dims = [], []
            for e in c["encoders"]:
                xtr = self.view.patient_features(e, train_ids)
                sc = StandardScaler().fit(xtr)
                xtr, xall = sc.transform(xtr), sc.transform(self.view.patient_features(e, all_ids))
                if c["reduce_type"] == "pca" and c["reduce_dim"] < min(xtr.shape):
                    pca = PCA(c["reduce_dim"], random_state=seed).fit(xtr)
                    xall = pca.transform(xall) / (float(np.sqrt(pca.explained_variance_.mean())) or 1.0)
                parts.append(xall.astype(np.float32)), dims.append(xall.shape[1])
            x, mask = torch.from_numpy(np.hstack(parts)), None
        data = dict(x=x, mask=mask, clin=torch.from_numpy(np.vstack([clin_tr, clin_te, clin_ex]).astype(np.float32)),
                    offset=torch.from_numpy(np.vstack([off_tr, off_te, np.zeros((len(extra_ids), k), np.float32)])))
        fit_idx, val_idx = _val_split(y, c["val_fraction"], seed)
        weight = None
        if c["class_weighting"] == "balanced":
            counts = np.bincount(y[fit_idx], minlength=k).astype(np.float64)
            weight = torch.tensor(np.where(counts > 0, len(fit_idx) / (k * np.maximum(counts, 1)), 0.0), dtype=torch.float32, device=device)
        y_all = torch.from_numpy(np.concatenate([y, np.zeros(n_te + len(extra_ids), dtype=np.int64)]))
        fit_rows, val_rows = torch.from_numpy(fit_idx), torch.from_numpy(val_idx)
        test_rows = torch.arange(n_tr, n_tr + n_te)
        ssl_rows = torch.cat([torch.arange(n_tr), torch.arange(n_tr + n_te, len(all_ids))])
        proba, members = np.zeros((n_te, k)), []
        for m in range(c["n_seeds_ensemble"]):
            torch.manual_seed(1000 * int(seed) + m)
            gen = torch.Generator().manual_seed(1000 * int(seed) + m)
            model = FusionNet(dims, clin_tr.shape[1], k, c).to(device)
            ssl_curve = _ssl_pretrain(model, data, ssl_rows, c, gen, device) if ssl else None
            curves, stopped = _train(model, data, y_all, fit_rows, val_rows, c, gen, device, weight)
            proba += torch.exp(_predict(model, data, test_rows, device)).cpu().numpy() / c["n_seeds_ensemble"]
            ok, note = convergence(curves, stopped)
            if ssl_curve is not None:
                curves["ssl_loss"] = ssl_curve
            members.append(dict(curves=curves, stopped_by=stopped, converged=ok, note=note, epochs=len(curves["val_loss"])))
        proba = proba / proba.sum(1, keepdims=True)
        first = members[0]
        report = dict(seconds=round(time.time() - t0, 2), device=str(device), epochs=first["epochs"], stopped_by=first["stopped_by"],
                      converged=bool(all(mm["converged"] for mm in members)), convergence_note=first["note"], curves=first["curves"],
                      n_parameters=int(sum(p.numel() for p in model.parameters()) - model.ssl_head.weight.numel() - model.ssl_head.bias.numel()),
                      ensemble_epochs=[mm["epochs"] for mm in members], n_train=int(len(fit_idx)), n_val=int(len(val_idx)),
                      overrides=dict(config or {}))
        return proba, report


class PrototypeRunner(_Base):
    """morphology_prototypes: tiles clustered into prototypes inside the training fold; patients described by prototype shares."""

    def fit_predict(self, train_ids, test_ids, config, seed):
        t0 = time.time()
        c, _ = resolve(self.spec, self.view, self.scale, config)
        train_ids, test_ids = np.asarray(train_ids), np.asarray(test_ids)
        all_ids = np.concatenate([train_ids, test_ids])
        store = self._store(all_ids, c)
        pos = store.pos(all_ids)
        n_tr = len(train_ids)
        feats = []
        for e in c["encoders"]:
            z = store.reduce(e, pos[:n_tr], "pca", c["proto_pca"], seed)
            rng = np.random.default_rng(seed)
            train_rows = store.tile_positions(pos[:n_tr])
            sub = rng.choice(train_rows, min(len(train_rows), 50000), replace=False)
            km = MiniBatchKMeans(c["n_prototypes"], n_init=3, batch_size=4096, random_state=seed).fit(z[sub])
            owner = np.repeat(np.arange(len(pos)), store.counts[pos])
            shares = np.zeros((len(all_ids), c["n_prototypes"]), dtype=np.float32)
            np.add.at(shares, (owner, km.predict(z[store.tile_positions(pos)])), 1.0)
            feats.append(np.sqrt(shares / np.maximum(store.counts[pos], 1)[:, None]))
        x = np.hstack(feats)
        x = StandardScaler().fit(x[:n_tr]).transform(x)
        (clin_tr, clin_te, _), _ = self._clinical(c, train_ids, test_ids)
        x_tr, x_te = np.hstack([clin_tr, x[:n_tr]]), np.hstack([clin_te, x[n_tr:]])
        y = self.view.y(train_ids)
        if c["classifier"] == "gbm":
            m = HistGradientBoostingClassifier(max_iter=c["max_iter"], learning_rate=c["learning_rate"], max_depth=c["max_depth"],
                                               class_weight="balanced", random_state=seed).fit(x_tr, y)
            n_par = 0
        else:
            m = LogisticRegression(C=c["C"], class_weight="balanced", max_iter=3000).fit(x_tr, y)
            n_par = m.coef_.size + m.intercept_.size
        return tb._full_proba(m, x_te, len(self.view.classes)), self._closed_form_report(t0, n_par, overrides=dict(config or {}))


class BoostingRunner(_Base):
    """gradient_boosting: boosted trees on clinical variables plus PCA-reduced patient-level image features."""

    def fit_predict(self, train_ids, test_ids, config, seed):
        t0 = time.time()
        c, _ = resolve(self.spec, self.view, self.scale, config)
        xi_tr, xi_te = tb.image_matrix(self.view, c["encoders"], train_ids, test_ids, c["n_pca"], seed)
        (clin_tr, clin_te, _), _ = self._clinical(c, train_ids, test_ids)
        m = HistGradientBoostingClassifier(max_iter=c["max_iter"], learning_rate=c["learning_rate"], max_depth=c["max_depth"],
                                           class_weight="balanced", random_state=seed).fit(np.hstack([clin_tr, xi_tr]), self.view.y(train_ids))
        return tb._full_proba(m, np.hstack([clin_te, xi_te]), len(self.view.classes)), self._closed_form_report(t0, overrides=dict(config or {}))


class StackingRunner(_Base):
    """stacking_ensemble: logistic meta-learner on out-of-fold class probabilities of heterogeneous members."""

    DEFAULT_MEMBERS = [dict(component="patient_mean_logreg"), dict(component="attention_mil"), dict(component="morphology_prototypes")]

    def __init__(self, spec, view, scale="full", seed=0):
        super().__init__(spec, view, scale, seed)
        from .registry import build_runner
        members = self.resolved["members"] or self.DEFAULT_MEMBERS
        self.members = []
        for m in members:
            sub = dict(inputs=m.get("inputs") or spec.get("inputs") or {}, model=dict(component=m["component"], params=dict(m.get("params") or {})),
                       fusion=m.get("fusion"), training=m.get("training") or {k: v for k, v in (spec.get("training") or {}).items() if k != "tuning"})
            self.members.append(build_runner(sub, view, scale=scale, seed=seed))
        self.resolved["members"] = [dict(component=r.name, resolved=getattr(r, "resolved", None)) for r in self.members]
        self.ignored = sorted(self.ignored + [f"member[{i}].{k}" for i, r in enumerate(self.members) for k in getattr(r, "ignored", [])])

    def fit_predict(self, train_ids, test_ids, config, seed):
        t0 = time.time()
        c, _ = resolve(self.spec, self.view, self.scale, config)
        train_ids, test_ids = np.asarray(train_ids), np.asarray(test_ids)
        y = self.view.y(train_ids)
        k = len(self.view.classes)
        n_splits = max(2, min(c["inner_k"], int(np.bincount(y, minlength=k).min())))
        try:
            folds = list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(train_ids, y))
        except ValueError:                      # a class with a single training patient: fall back to plain folds
            from sklearn.model_selection import KFold
            folds = list(KFold(n_splits, shuffle=True, random_state=seed).split(train_ids))
        oof, test, reports = [], [], []
        for r in self.members:
            cfg = r.configs()[0]
            o = np.zeros((len(train_ids), k))
            for tr, te in folds:
                o[te], _ = r.fit_predict(train_ids[tr], train_ids[te], cfg, seed)
            p, rep = r.fit_predict(train_ids, test_ids, cfg, seed)
            oof.append(o), test.append(p)
            reports.append(dict(component=r.name, seconds=rep.get("seconds"), converged=bool(rep.get("converged", True)),
                                epochs=rep.get("epochs", 0), stopped_by=rep.get("stopped_by", "closed_form"), curves=rep.get("curves", {})))
        logit = lambda parts: np.log(np.clip(np.hstack(parts), 1e-6, 1.0))
        meta = LogisticRegression(C=c["meta_C"], class_weight="balanced", max_iter=3000).fit(logit(oof), y)
        curves = next((r["curves"] for r in reports if r["curves"]), {})
        report = dict(seconds=round(time.time() - t0, 2), device="cuda" if torch.cuda.is_available() else "cpu",
                      epochs=int(max(r["epochs"] for r in reports)), stopped_by="stacked",
                      converged=bool(all(r["converged"] for r in reports)),
                      convergence_note="; ".join(f"{r['component']}: {'converged' if r['converged'] else 'NOT converged'} ({r['stopped_by']})" for r in reports),
                      curves=curves, n_parameters=int(meta.coef_.size + meta.intercept_.size),
                      members=[{k2: v for k2, v in r.items() if k2 != "curves"} for r in reports],
                      meta_weights_norm=[float(np.abs(meta.coef_[:, i * k:(i + 1) * k]).mean()) for i in range(len(self.members))],
                      inner_folds=len(folds), overrides=dict(config or {}))
        return tb._full_proba(meta, logit(test), k), report


def build(spec, view, scale="full", seed=0):
    key = (spec.get("model") or {}).get("component")
    if key in NEURAL:
        return NeuralRunner(spec, view, scale, seed)
    if key == "morphology_prototypes":
        return PrototypeRunner(spec, view, scale, seed)
    if key == "gradient_boosting":
        return BoostingRunner(spec, view, scale, seed)
    if key == "stacking_ensemble":
        return StackingRunner(spec, view, scale, seed)
    if key == "clinical_only":
        return tb.ClinicalOnly(spec, view, scale, seed)
    if key == "patient_mean_logreg":
        return tb.PatientMeanLogReg(spec, view, scale, seed)
    raise KeyError(f"component '{key}' is not implemented for incremental-value classification")
