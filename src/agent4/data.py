"""Read-only access to the data an agent is allowed to see (the "agent view").

The evaluator writes one view directory per task. Nothing in this module knows where hidden data lives:
a view simply does not contain it. Two adapters exist, chosen by the `adapter` field of view.json:

paired_single_cell  cells measured in two count modalities, plus cells with one modality only
cohort_features     patients with tabular covariates, precomputed image features and a class label
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .io import read_json
from .stats import lognorm


# ----------------------------------------------------------------------------- paired single cell
@dataclass
class CellView:
    root: Path
    meta: dict
    dev_rna: sparse.csr_matrix       # development cells x universe genes, raw counts
    dev_atac: sparse.csr_matrix      # development cells x peaks, raw counts
    hidden_atac: sparse.csr_matrix   # held-out cells x peaks; their RNA is not in the view
    dev_obs: pd.DataFrame
    hidden_obs: pd.DataFrame
    genes: pd.DataFrame              # name, gene_id, chrom, tss, n_cells_dev
    peaks: pd.DataFrame              # name, chrom, start, end
    peak_gene: pd.DataFrame          # peak_idx, gene_idx, distance, peak_type (annotation links)
    rules: dict = field(default_factory=dict)

    adapter = "paired_single_cell"

    @property
    def scope(self) -> np.ndarray:
        """Indices (into universe genes) of the genes every plan must predict."""
        if "gene_scope" not in self.rules:
            raise RuntimeError("gene scope has not been frozen yet (rules.json missing in the view)")
        return np.asarray(self.rules["gene_scope"], dtype=int)

    @property
    def cluster_k(self) -> int:
        return int(self.rules["cluster_k"])

    def target(self, counts, gene_idx=None) -> np.ndarray:
        """Log-normalised expression (library size over all universe genes) for the requested genes."""
        gene_idx = self.scope if gene_idx is None else gene_idx
        return np.asarray(lognorm(counts)[:, gene_idx].todense(), dtype=np.float32)


def load_cell_view(root) -> CellView:
    root = Path(root)
    rules = read_json(root / "rules.json") if (root / "rules.json").exists() else {}
    return CellView(
        root=root, meta=read_json(root / "view.json"),
        dev_rna=sparse.load_npz(root / "dev_rna.npz").tocsr(),
        dev_atac=sparse.load_npz(root / "dev_atac.npz").tocsr(),
        hidden_atac=sparse.load_npz(root / "hidden_atac.npz").tocsr(),
        dev_obs=pd.read_parquet(root / "dev_obs.parquet"), hidden_obs=pd.read_parquet(root / "hidden_obs.parquet"),
        genes=pd.read_parquet(root / "genes.parquet"), peaks=pd.read_parquet(root / "peaks.parquet"),
        peak_gene=pd.read_parquet(root / "peak_gene.parquet"), rules=rules)


# ----------------------------------------------------------------------------- cohort with features
@dataclass
class CohortView:
    root: Path
    meta: dict
    clinical: pd.DataFrame           # every patient with image features; label is NaN when unlabelled
    tile_index: pd.DataFrame         # patient_id, slide_id, start, end rows into the tile arrays
    encoders: list
    classes: list

    adapter = "cohort_features"

    def __post_init__(self):
        self._patient, self._tiles = {}, {}
        self._row = pd.Series(np.arange(len(self.clinical)), index=self.clinical.index)

    @property
    def labelled(self) -> pd.DataFrame:
        return self.clinical[self.clinical["label"].isin(self.classes)]

    def y(self, ids) -> np.ndarray:
        return self.clinical.loc[ids, "label"].map({c: i for i, c in enumerate(self.classes)}).values.astype(int)

    def patient_features(self, encoder, ids) -> np.ndarray:
        """Mean-pooled image features (patient level) for the requested patients."""
        if encoder not in self._patient:
            self._patient[encoder] = np.load(self.root / f"patient_{encoder}.npy", mmap_mode="r")
        return np.asarray(self._patient[encoder][self._row[ids].values], dtype=np.float32)

    def tiles(self, encoder) -> np.ndarray:
        """All tile features of one encoder as a memory-mapped array; rows follow tile_index."""
        if encoder not in self._tiles:
            self._tiles[encoder] = np.load(self.root / f"tiles_{encoder}.npy", mmap_mode="r")
        return self._tiles[encoder]

    def tile_rows(self, ids) -> list:
        """For each patient, the row indices of all of that patient's tiles (all slides concatenated)."""
        if not hasattr(self, "_rows_by_patient"):
            self._rows_by_patient = {
                pid: np.concatenate([np.arange(s, e) for s, e in zip(g["start"], g["end"])])
                for pid, g in self.tile_index.groupby("patient_id")}
        return [self._rows_by_patient[i] for i in ids]


def load_cohort_view(root) -> CohortView:
    root = Path(root)
    meta = read_json(root / "view.json")
    return CohortView(root=root, meta=meta, clinical=pd.read_parquet(root / "clinical.parquet"),
                      tile_index=pd.read_parquet(root / "tile_index.parquet"),
                      encoders=meta["encoders"], classes=meta["classes"])


def load_view(root):
    adapter = read_json(Path(root) / "view.json")["adapter"]
    return {"paired_single_cell": load_cell_view, "cohort_features": load_cohort_view}[adapter](root)
