"""Stage-2 self-check (README 9.2): hand-written plans must run through the plan executor at pilot scale on
both tasks and produce evidence cards. Writes under outputs/_smoke/handwritten/. Not part of the formal run.

    python scripts/handwritten_check.py task_t|task_m [plan index]
"""
import sys

import yaml

from agent4 import flow_m, flow_t
from agent4.data import load_view
from agent4.io import ROOT, write_json

PLANS = {
    "task_t": [
        dict(plan_id="H1", name="Gated attention MIL, early concatenation", complexity="ambitious", analysis_unit="patient",
             inputs=dict(image=dict(encoders=["H-Optimus-1"], level="tile", reduce=dict(type="pca", dim=128)), clinical=dict(use=True)),
             model=dict(component="attention_mil", params=dict(aggregator="gated_attention", embed_dim=128, dropout=0.25)), fusion="early_concat",
             training=dict(max_epochs=60, early_stopping_patience=8, class_weighting="balanced", tuning=dict(grid=[{"lr": 1e-3}, {"lr": 3e-4}])),
             validation=dict(scheme="kfold", k=5, split_unit="patient", selection="same_folds", clinical_comparator=dict(tuned=False)),
             budget=dict(needs_gpu=False, pilot_minutes=10, full_minutes=60, cpus=16), diversity=dict(data="tile", model="mil", method="early")),
        dict(plan_id="H2", name="Incremental offset over the clinical model", complexity="ambitious", analysis_unit="patient",
             inputs=dict(image=dict(encoders=["Virchow2", "H-Optimus-1"], level="patient", reduce=dict(type="pca", dim=64)), clinical=dict(use=True)),
             model=dict(component="incremental_offset", params=dict(init_zero=True, embed_dim=64)), fusion="offset",
             training=dict(max_epochs=80, early_stopping_patience=10, class_weighting="balanced"),
             validation=dict(scheme="repeated_kfold", k=5, repeats=2, split_unit="patient", selection="nested", inner_k=3, clinical_comparator=dict(tuned=True)),
             budget=dict(needs_gpu=False, pilot_minutes=10, full_minutes=60, cpus=16), diversity=dict(data="patient", model="mlp", method="offset")),
    ],
    "task_m": [
        dict(plan_id="H1", name="Translation network with Poisson likelihood", complexity="ambitious", analysis_unit="cell",
             inputs=dict(atac_representation=[dict(type="tfidf_peaks", max_peaks=30000)]),
             model=dict(component="translation_network", params=dict(encoder_hidden=[512], latent_dim=64, decoder="linear", likelihood="poisson", dropout=0.2)),
             training=dict(max_epochs=150, early_stopping_patience=10, lr=1e-3, batch_size=256),
             validation=dict(scheme="holdout", holdout_fraction=0.2, split_unit="cell", scoring_target="thinned_half", n_seeds=1),
             budget=dict(needs_gpu=False, pilot_minutes=10, full_minutes=60, cpus=32), diversity=dict(data="peaks", model="autoencoder", method="joint")),
        dict(plan_id="H2", name="Residual over the cluster mean", complexity="ambitious", analysis_unit="cell",
             inputs=dict(atac_representation=[dict(type="lsi", n_components=50), dict(type="depth")]),
             model=dict(component="residual_over_cluster_mean", params=dict(learner="mlp", hidden=[256, 128], dropout=0.2, cluster_k="frozen")),
             training=dict(max_epochs=150, early_stopping_patience=10, lr=1e-3, batch_size=256),
             validation=dict(scheme="kfold", k=3, split_unit="cell", scoring_target="thinned_half", n_seeds=1),
             budget=dict(needs_gpu=False, pilot_minutes=10, full_minutes=40, cpus=32), diversity=dict(data="lsi", model="mlp", method="residual")),
    ],
}

task = sys.argv[1]
which = [int(sys.argv[2])] if len(sys.argv) > 2 else range(len(PLANS[task]))
cfg = yaml.safe_load((ROOT / "config" / f"{task}.yaml").read_text())
view = load_view(ROOT / cfg["view_dir"])
flow = flow_m if view.adapter == "paired_single_cell" else flow_t
limits = dict(cfg["limits"])
if view.adapter == "paired_single_cell":
    limits.update(n_dev=view.dev_rna.shape[0], n_scope=len(view.scope))
for i in which:
    plan = PLANS[task][i]
    out = ROOT / "outputs" / "_smoke" / "handwritten" / task / plan["plan_id"]
    summary = flow.validate(plan, view, "pilot", out)
    compliance = dict(plan_id=plan["plan_id"], passed=True, rules=[dict(rule="hand-written check", passed=True, detail="not run through the compliance stage")])
    card = flow.evidence_card(plan, summary, compliance, limits)
    write_json(out / "evidence_card.json", card.model_dump())
    print(f"=== {task} {plan['plan_id']}: {plan['name']}")
    for it in card.items:
        print(f"[{it.id}] ({it.kind}) {it.text}")
