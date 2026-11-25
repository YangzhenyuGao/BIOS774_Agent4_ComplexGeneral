"""Fast checks of the task T components on a tiny synthetic cohort (no real data is read)."""
import numpy as np
import pandas as pd
import pytest

from agent4.data import load_cohort_view
from agent4.io import write_json
from agent4.tools import t_basic as tb
from agent4.tools import t_models as tm
from agent4.tools.registry import build_runner

CLASSES = ["LumA", "LumB", "Basal", "Her2"]
DIMS = {"encA": 24, "encB": 16}
N, N_LABELLED = 76, 64


def make_view(root, changed=()):
    """Write a small cohort view. Patients listed in `changed` get different image features."""
    rng = np.random.default_rng(0)
    ids = np.array([f"P{i:03d}" for i in range(N)])
    y = np.arange(N) % 4
    counts = rng.integers(5, 21, N)
    counts[1] = 40                                           # one bag above the cap used in the tests
    clinical = pd.DataFrame(dict(
        site=rng.choice(["S1", "S2", "S3"], N), age=rng.normal(58, 12, N),
        stage=rng.choice(["I", "II", "III", None], N), histological_type=rng.choice(["ductal", "lobular"], N),
        er=np.where(y < 2, "positive", "negative"), pr=rng.choice(["positive", "negative", None], N),
        her2=np.where(y == 3, "positive", rng.choice(["negative", "equivocal", None], N)),
        label=[CLASSES[c] if i < N_LABELLED else None for i, c in enumerate(y)]), index=pd.Index(ids, name="patient_id"))
    clinical.to_parquet(root / "clinical.parquet")
    starts = np.concatenate([[0], np.cumsum(counts)])
    pd.DataFrame(dict(patient_id=ids, slide_id=[f"s{i}" for i in range(N)], start=starts[:-1], end=starts[1:])).to_parquet(root / "tile_index.parquet")
    owner = np.repeat(np.arange(N), counts)
    for enc, d in DIMS.items():
        centres = rng.normal(0, 1, (4, d))
        tiles = centres[y[owner]] * 0.8 + rng.normal(0, 1, (len(owner), d))
        for pid in changed:
            rows = owner == int(pid[1:])
            tiles[rows] = tiles[rows] * -3.0 + 5.0
        np.save(root / f"tiles_{enc}.npy", tiles.astype(np.float16))
        np.save(root / f"patient_{enc}.npy", np.vstack([tiles[owner == i].mean(0) for i in range(N)]).astype(np.float32))
    write_json(root / "view.json", dict(adapter="cohort_features", encoders=list(DIMS), classes=CLASSES, n_tiles=int(len(owner))))
    return load_cohort_view(root)


@pytest.fixture()
def view(tmp_path):
    return make_view(tmp_path)


def split(view):
    ids = view.labelled.index.values
    return ids[:48], ids[48:]


FAST = dict(max_epochs=4, early_stopping_patience=3, batch_size=16)
INPUT_ONE = dict(image=dict(encoders=["encA"], level="tile", reduce=dict(type="pca", dim=8)), clinical=dict(use=True))
INPUT_TWO = dict(image=dict(encoders=["encA", "encB"], reduce=dict(type="pca", dim=8)), clinical=dict(use=True))


def spec(component, params=None, inputs=INPUT_ONE, fusion=None, training=None):
    return dict(inputs=inputs, model=dict(component=component, params=dict(embed_dim=16, max_tiles=24, **(params or {}))),
                fusion=fusion, training=dict(FAST, **(training or {})))


SPECS = {
    "gated": spec("attention_mil"),
    "multihead": spec("attention_mil", dict(aggregator="multihead_attention", heads=2)),
    "transformer": spec("attention_mil", dict(aggregator="transformer", heads=2, layers=1)),
    "late": spec("attention_mil", fusion="late_average"),
    "image_only": spec("attention_mil", inputs=dict(image=dict(encoders=["encA"], level="tile"), clinical=dict(use=False))),
    "fusion_patient": spec("multi_encoder_fusion", dict(level="patient", encoder_fusion="attention"), INPUT_TWO),
    "fusion_tile_gated": spec("multi_encoder_fusion", dict(level="tile", encoder_fusion="gated"), INPUT_TWO),
    "fusion_tile_concat": spec("multi_encoder_fusion", dict(level="tile", encoder_fusion="concat", aggregator="mean"), INPUT_TWO),
    "prototypes": spec("morphology_prototypes", dict(n_prototypes=6, proto_pca=6)),
    "prototypes_gbm": spec("morphology_prototypes", dict(n_prototypes=6, proto_pca=6, classifier="gbm", max_iter=10)),
    "film": spec("conditional_fusion", dict(mode="film")),
    "cross": spec("conditional_fusion", dict(mode="cross_attention", heads=2)),
    "cross_patient": spec("conditional_fusion", dict(mode="cross_attention", heads=2, level="patient"), INPUT_TWO),
    "offset": spec("incremental_offset"),
    "offset_hier": spec("incremental_offset", dict(head="hierarchical")),
    "hierarchical": spec("hierarchical", dict(groups=[["LumA", "LumB"], ["Basal"], ["Her2"]], fine_inputs="image")),
    "ssl": spec("ssl_preadapted_mil", dict(pretrain_epochs=2, uses_unlabelled=True)),
    "boosting": spec("gradient_boosting", dict(n_pca=4, max_iter=10), INPUT_TWO),
    "stacking": spec("stacking_ensemble", dict(members=[
        dict(component="patient_mean_logreg", params=dict(grid=[dict(encoders=["encA"], n_pca=4, C=0.3)])),
        dict(component="incremental_offset", params=dict(embed_dim=16, max_tiles=24)),
        dict(component="morphology_prototypes", params=dict(n_prototypes=6, proto_pca=6))], oof_folds=2)),
    "augmented": spec("attention_mil", training=dict(augmentation=dict(mixup_alpha=0.3, tile_dropout=0.3, tile_resample=8), n_seeds_ensemble=2)),
}
REPORT_KEYS = {"seconds", "device", "epochs", "stopped_by", "converged", "convergence_note", "curves", "n_parameters"}


@pytest.mark.parametrize("name", sorted(SPECS))
def test_component_runs_and_reports(view, name):
    train, test = split(view)
    runner = build_runner(SPECS[name], view, scale="full", seed=3)
    proba, report = runner.fit_predict(train, test, runner.configs()[0], seed=3)
    assert proba.shape == (len(test), len(CLASSES))
    assert np.isfinite(proba).all() and np.allclose(proba.sum(1), 1.0, atol=1e-5) and (proba >= 0).all()
    assert REPORT_KEYS <= set(report), REPORT_KEYS - set(report)
    assert isinstance(report["converged"], bool) and isinstance(runner.resolved, dict)
    assert runner.ignored == [], runner.ignored
    if report["stopped_by"] in ("early_stopping", "max_epochs"):
        assert len(report["curves"]["train_loss"]) == len(report["curves"]["val_loss"]) == report["epochs"] > 0


def test_same_seed_gives_same_predictions(view):
    train, test = split(view)
    a, _ = build_runner(SPECS["augmented"], view, seed=5).fit_predict(train, test, {}, seed=5)
    b, _ = build_runner(SPECS["augmented"], view, seed=5).fit_predict(train, test, {}, seed=5)
    c, _ = build_runner(SPECS["augmented"], view, seed=5).fit_predict(train, test, {}, seed=6)
    assert np.allclose(a, b, atol=1e-6)
    assert not np.allclose(a, c, atol=1e-6)


def test_unknown_and_oddly_named_parameters_are_tolerated(view):
    train, test = split(view)
    s = spec("attention_mil", dict(aggregator="ABMIL (gated)", hidden_dim=[16, 8], wavelet_order=3, encoder="ENC-a"),
             inputs=dict(image=dict(level="Tile-level", colour_normalisation="macenko"), clinical=dict(use="true", imputation="mice")),
             training=dict(epochs="4", optimizer="lion", class_weight="Balanced", early_stopping=dict(patience=2), mixup=True))
    runner = build_runner(s, view, seed=1)
    proba, report = runner.fit_predict(train, test, {"model.params.dropout": 0.1, "training": {"lr": 0.002}, "galaxy": 7}, seed=1)
    assert proba.shape == (len(test), 4) and np.isfinite(proba).all()
    assert set(runner.ignored) == {"model.params.wavelet_order", "inputs.image.colour_normalisation", "inputs.clinical.imputation",
                                   "training.optimizer"}
    r = runner.resolved
    assert (r["aggregator"], r["embed_dim"], r["max_epochs"], r["patience"], r["mixup_alpha"], r["encoders"], r["level"]) == \
        ("gated_attention", 16, 4, 3, 0.2, ["encA"], "tile")
    cfg, ignored = tm.resolve(s, view, "full", {"model.params.dropout": 0.1, "training": {"lr": 0.002}, "galaxy": 7})
    assert cfg["dropout"] == 0.1 and cfg["lr"] == 0.002 and "model.params.galaxy" in ignored


def test_candidate_grid_is_read_and_trimmed_at_pilot_scale(view):
    grid = [dict(embed_dim=16), dict(embed_dim=32), dict(dropout=0.4), dict(lr=1e-4), dict(lr=1e-2)]
    s = spec("attention_mil", training=dict(tuning=dict(grid=grid)))
    assert build_runner(s, view, scale="full").configs() == grid[:4]
    assert build_runner(s, view, scale="pilot").configs() == grid[:2]
    assert build_runner(spec("attention_mil"), view).configs() == [{}]
    assert build_runner(spec("attention_mil", training=dict(n_seeds_ensemble=3)), view, scale="pilot").resolved["n_seeds_ensemble"] == 1


@pytest.mark.parametrize("name", ["gated", "offset", "ssl", "prototypes", "stacking", "fusion_patient", "boosting"])
def test_test_labels_are_never_used(view, name):
    train, test = split(view)
    a, _ = build_runner(SPECS[name], view, seed=2).fit_predict(train, test, {}, seed=2)
    view.clinical.loc[test, "label"] = np.roll(view.clinical.loc[test, "label"].values, 1)
    b, _ = build_runner(SPECS[name], view, seed=2).fit_predict(train, test, {}, seed=2)
    assert np.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("name", ["gated", "transformer", "fusion_tile_gated", "offset", "ssl", "prototypes", "stacking", "cross"])
def test_a_test_patient_does_not_influence_other_test_patients(tmp_path, name):
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    view_a = make_view(tmp_path / "a")
    train, test = split(view_a)
    view_b = make_view(tmp_path / "b", changed=list(test[8:]))
    a, _ = build_runner(SPECS[name], view_a, seed=4).fit_predict(train, test, {}, seed=4)
    b, _ = build_runner(SPECS[name], view_b, seed=4).fit_predict(train, test, {}, seed=4)
    assert np.allclose(a[:8], b[:8], atol=1e-5), "predictions for untouched test patients changed"
    assert not np.allclose(a[8:], b[8:], atol=1e-3), "changing a patient's images did not change its prediction"


def test_offset_model_starts_from_the_clinical_model(view):
    """With a zero-initialised image branch and no training, the offset model reproduces the in-fold clinical model."""
    train, test = split(view)
    s = spec("incremental_offset", training=dict(max_epochs=1, lr=1e-5))
    runner = build_runner(s, view, seed=1)
    proba, _ = runner.fit_predict(train, test, {}, seed=1)
    x_tr, x_te = tb.clinical_matrix(view, train, test)
    from sklearn.linear_model import LogisticRegression
    ref = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(x_tr, view.y(train)).predict_proba(x_te)
    assert np.abs(proba - ref).max() < 0.05


def test_convergence_rule():
    ok, _ = tm.convergence(dict(val_loss=[1.0, 0.8, 0.7, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69]), "max_epochs")
    still_improving, _ = tm.convergence(dict(val_loss=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]), "max_epochs")
    stopped, _ = tm.convergence(dict(val_loss=[1.0, 0.9, 0.95]), "early_stopping")
    assert ok and stopped and not still_improving


def test_cross_validation_engine_accepts_the_runners(view):
    ids = view.labelled.index.values
    y = view.y(ids)
    folds = tb.make_folds(y, 3, 1, seed=0)
    s = spec("attention_mil", training=dict(tuning=dict(grid=[dict(embed_dim=16), dict(embed_dim=24)])))
    out = tb.cross_validate(build_runner(s, view, seed=0), ids, y, folds, selection="nested", inner_k=2, seed=0, log=lambda *_: None)
    assert out["proba"][0].shape == (len(ids), 4) and np.allclose(out["proba"][0].sum(1), 1.0, atol=1e-5)
