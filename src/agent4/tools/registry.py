"""Tool registry (README 6.5): what a plan may be built from, and how a plan becomes something executable.

The registry has two tiers per task family. Tier "basic" holds preprocessing, simple learners and rulers;
tier "ambitious" holds the components candidate plans are mainly built from. The descriptions below are
shown verbatim to the plan generator; the builders at the bottom map a validated plan to code.

A plan may propose a component that is not listed (field new_components). Before the pool is frozen the
builder either implements it and adds it here, or records why it was rejected. After the freeze the
registry is closed: nothing is generated or executed that is not in this file.
"""
from __future__ import annotations

FAMILIES = {
    "cross_modal_prediction": dict(
        what="Predict one measured modality from another, feature by feature, for units that provide only the source modality.",
        executable=True),
    "incremental_value_classification": dict(
        what="Estimate how much a second modality adds to a baseline modality for predicting a class label.",
        executable=True),
    "single_modality_classification": dict(
        what="Estimate how well one modality alone predicts a class label.", executable=False),
    "cross_modal_retrieval": dict(
        what="Match each unit's profile in one modality to its own profile in another modality among many candidates.",
        executable=False),
}

# ----------------------------------------------------------------------------- cross-modal prediction
M_INPUTS = [
    dict(type="lsi", params="n_components 20-150; max_peaks 20000-108377",
         what="TF-IDF on binarised peaks then truncated SVD; first (depth) component dropped. Compact global cell state."),
    dict(type="tfidf_peaks", params="max_peaks 10000-60000",
         what="Sparse TF-IDF over the most accessible peaks; high-dimensional input for a neural encoder."),
    dict(type="gene_activity", params="mode promoter | promoter_and_distal",
         what="Per-gene sum of ATAC counts over annotated peaks, depth-normalised. One feature per in-scope gene."),
    dict(type="cis_peaks", params="window_kb 50-500; max_peaks_per_gene 5-50",
         what="For each gene, the individual peaks within the window around its start site (gene-specific inputs)."),
    dict(type="depth", params="-", what="Log ATAC depth of the cell as a covariate."),
]
M_COMPONENTS = [
    dict(key="cluster_mean", tier="basic", params="k: 'frozen' (the frozen cell-type baseline) or an integer",
         what="Cluster cells on LSI, predict each cell by its cluster's mean expression. This is ruler floor 1.", cost="seconds, CPU"),
    dict(key="latent_ridge", tier="basic", params="alpha", what="Ridge regression from LSI components to all genes.", cost="seconds, CPU"),
    dict(key="latent_knn", tier="basic", params="k", what="k-nearest-neighbour regression in LSI space.", cost="seconds, CPU"),
    dict(key="latent_pls", tier="basic", params="n_components", what="Partial least squares from LSI to all genes.", cost="seconds, CPU"),
    dict(key="gene_activity_score", tier="basic", params="mode", what="Use the gene activity score itself as the prediction (no learning). Ruler floor 2.", cost="seconds, CPU"),
    dict(key="residual_over_cluster_mean", tier="ambitious",
         params="learner mlp | gbm; hidden [..]; dropout; cluster_k 'frozen' | int",
         what="Start from the cluster-mean prediction and train a non-linear model on what is left: the deviation of each cell from its cluster mean. Aims directly at variation beyond cell type.",
         cost="mlp: minutes on CPU; gbm: one boosted model per gene, tens of minutes on 32 CPUs"),
    dict(key="cis_regulatory", tier="ambitious",
         params="learner masked_nn | gbm | elastic_net; window_kb; max_peaks_per_gene; global_state_dim; hidden",
         what="Gene-specific model: each gene is predicted from the peaks near it plus a shared global cell state. masked_nn trains all genes jointly with a sparse peak-to-gene layer; gbm / elastic_net fit one model per gene.",
         cost="masked_nn: minutes on CPU; gbm: tens of minutes on 32 CPUs"),
    dict(key="translation_network", tier="ambitious",
         params="encoder_hidden [..]; latent_dim; decoder linear | mlp; likelihood poisson | nb | mse; dropout",
         what="Encoder-decoder network from the ATAC profile to all genes at once, trained with a count likelihood on raw UMI counts (library size as offset) or MSE on log-normalised values.",
         cost="5-20 minutes on CPU, faster on GPU"),
    dict(key="joint_latent_multivi", tier="ambitious",
         params="n_latent; n_hidden; max_epochs; early_stopping_patience; n_peaks",
         what="MultiVI joint variational model of both modalities, trained to convergence with early stopping on held-out reconstruction; expression of ATAC-only cells is imputed from the joint latent space.",
         cost="30-100 minutes; GPU preferred, CPU possible with a peak subset"),
    dict(key="peakvi_latent_head", tier="ambitious", params="n_latent; head mlp | ridge",
         what="PeakVI latent representation of accessibility followed by a supervised prediction head.", cost="20-60 minutes"),
    dict(key="contrastive_pretrain_head", tier="ambitious",
         params="embed_dim; temperature; pretrain_epochs; head mlp | knn_transfer",
         what="First align the two modalities on development cells with a contrastive (CLIP-style) objective, then predict expression from the aligned ATAC embedding with a head, or by transferring expression from nearest RNA neighbours.",
         cost="10-30 minutes on CPU"),
    dict(key="multitask_lowrank", tier="ambitious", params="rank; trunk_hidden [..]; gene_embedding_dim",
         what="Joint prediction of all genes through a low-rank bottleneck with learned gene embeddings, so genes share statistical strength.",
         cost="minutes on CPU"),
    dict(key="stacking_ensemble", tier="ambitious",
         params="members [list of {component, params, inputs}]; meta per_gene_ridge | global_weights; oof_folds 3-5",
         what="Combine heterogeneous models with out-of-fold predictions: a meta-learner weighs the members gene by gene. Members are refitted on all training cells.",
         cost="(oof_folds + 1) x the summed cost of the members"),
]
M_TRAINING = ("max_epochs; early_stopping_patience (epochs without improvement on an internal validation split); lr; batch_size; "
              "weight_decay; target_denoising none | knn_pool (k) - train against RNA pooled over the k nearest training cells "
              "(training cells only; scoring always uses unsmoothed values unless the validation design says otherwise)")
M_VALIDATION = ("scheme holdout | kfold; holdout_fraction 0.1-0.3; k 3-5; split_unit cell; "
                "scoring_target thinned_half (score against an independent half of each validation cell's reads, as the evaluator does) "
                "| full_depth (score against the unsplit log-normalised counts) | knn_smoothed (score against neighbour-pooled expression); "
                "n_seeds 1-3; compare_with [rulers to compute in the same split]; "
                "gene_claim_rule: how the plan decides which genes it claims are predicted beyond cell type")

# ----------------------------------------------------------------------------- incremental-value classification
T_INPUTS = [
    dict(type="image", params="encoders [subset of the available encoders]; level tile | patient; reduce {type pca | linear_projection | none, dim}",
         what="Frozen foundation-model features. Tile level keeps up to 128 tiles per slide (all slides of a patient pooled into one bag); patient level is the mean over tiles."),
    dict(type="clinical", params="variables [age, stage, histological_type, er, pr, her2]; use true | false",
         what="Clinical covariates, one-hot encoded with an explicit missing level inside each training fold."),
]
T_COMPONENTS = [
    dict(key="clinical_only", tier="basic", params="tune true | false", what="Clinical variables only (logistic regression / boosting). The comparison arm.", cost="seconds"),
    dict(key="patient_mean_logreg", tier="basic", params="grid over encoder, n_pca, C",
         what="Clinical variables concatenated with PCA-reduced mean-pooled image features, multinomial logistic regression. The simple reference.", cost="seconds per fit"),
    dict(key="attention_mil", tier="ambitious",
         params="aggregator gated_attention | multihead_attention | transformer; heads; layers; embed_dim; dropout",
         what="Multiple-instance learning over tiles: a learned attention or transformer aggregator turns a patient's tiles into one embedding.",
         cost="1-3 minutes per fit on 16 CPUs"),
    dict(key="multi_encoder_fusion", tier="ambitious",
         params="level tile | patient; encoder_fusion attention | gated | concat; aggregator (tile level); embed_dim",
         what="Fuse several foundation models: attention or gating over encoders, per tile (tiles are aligned across encoders) or per patient.",
         cost="2-6 minutes per fit on 16 CPUs"),
    dict(key="morphology_prototypes", tier="ambitious", params="n_prototypes 8-64; encoder; classifier logreg | gbm",
         what="Cluster training tiles into morphological prototypes; describe each patient by the share of tiles per prototype (plus clinical variables).",
         cost="under a minute per fit"),
    dict(key="conditional_fusion", tier="ambitious", params="mode film | cross_attention; aggregator; embed_dim",
         what="Let clinical variables condition the image branch: feature-wise modulation of tile embeddings (FiLM) or clinical-query cross-attention over tiles.",
         cost="1-3 minutes per fit on 16 CPUs"),
    dict(key="incremental_offset", tier="ambitious", params="image_branch {level, aggregator, embed_dim}; init_zero true | false",
         what="Fit the clinical model first, freeze its class logits as an offset, and train the image branch only on what the clinical model leaves unexplained. Aims directly at incremental value.",
         cost="1-3 minutes per fit on 16 CPUs"),
    dict(key="hierarchical", tier="ambitious", params="groups [[LumA, LumB], [Basal], [Her2]]; fine_inputs image | image+clinical; image_branch {..}",
         what="Two-stage classifier: separate the coarse groups first, then a dedicated image-driven model for the pair clinical variables separate worst.",
         cost="2-4 minutes per fit on 16 CPUs"),
    dict(key="ssl_preadapted_mil", tier="ambitious", params="pretrain_epochs; uses_unlabelled true | false; then the attention_mil params",
         what="Self-supervised pre-adaptation of the tile aggregator (two random tile subsets of one patient should agree) on training-fold and unlabelled patients, then supervised training. Pre-adaptation is redone inside every training fold.",
         cost="3-8 minutes per fit on 16 CPUs"),
    dict(key="gradient_boosting", tier="ambitious", params="n_pca; max_iter; learning_rate",
         what="Gradient-boosted trees on clinical variables plus PCA-reduced patient-level image features.", cost="seconds per fit"),
    dict(key="stacking_ensemble", tier="ambitious", params="members [list of {component, params, inputs, fusion}]; meta logreg",
         what="Stack heterogeneous models through out-of-fold class probabilities with a logistic meta-learner fitted inside the training fold.",
         cost="(inner folds + 1) x the summed cost of the members"),
]
T_FUSION = ("How clinical information joins the image branch (field 'fusion'): early_concat | late_average | offset | film | cross_attention | none. "
            "Any neural component may set, in model.params: aggregator, encoder_fusion, embed_dim, dropout, head flat | hierarchical.")
T_TRAINING = ("max_epochs; early_stopping_patience; lr; weight_decay; class_weighting balanced | none; "
              "augmentation {tile_resample: tiles drawn per bag per epoch, mixup_alpha, tile_dropout}; n_seeds_ensemble 1-3; "
              "tuning.grid: up to 4 alternative settings of model/training parameters the plan wants selected")
T_VALIDATION = ("scheme kfold | repeated_kfold; k 3-5; repeats 1-3; split_unit patient | slide | tile; "
                "selection nested (configuration chosen inside each training fold) | same_folds (configuration chosen on the folds that are also reported) | none; "
                "inner_k 2-3; clinical_comparator {tuned true | false}; interval bootstrap")

# ----------------------------------------------------------------------------- plan-only families
RETRIEVAL_COMPONENTS = [
    dict(key="contrastive_alignment", tier="ambitious", params="embed_dim; temperature", what="CLIP-style paired encoders; retrieve by cosine similarity."),
    dict(key="joint_latent_multivi", tier="ambitious", params="n_latent", what="Embed both modalities in a shared variational latent space; nearest neighbours across modalities."),
    dict(key="cca_latent", tier="basic", params="n_components", what="Canonical correlation between RNA PCA and ATAC LSI spaces."),
    dict(key="gene_activity_bridge", tier="basic", params="mode", what="Convert ATAC to gene activity and correlate with expression directly."),
    dict(key="optimal_transport_matching", tier="ambitious", params="epsilon", what="Solve a one-to-one assignment across the candidate pool instead of independent nearest neighbours."),
]


def catalogue(family):
    """Structured catalogue for one task family (used to build prompts and to validate plans)."""
    if family == "cross_modal_prediction":
        return dict(inputs=M_INPUTS, components=M_COMPONENTS, training=M_TRAINING, validation=M_VALIDATION, fusion=None)
    if family in ("incremental_value_classification", "single_modality_classification"):
        return dict(inputs=T_INPUTS, components=T_COMPONENTS, training=T_TRAINING, validation=T_VALIDATION, fusion=T_FUSION)
    if family == "cross_modal_retrieval":
        return dict(inputs=M_INPUTS, components=RETRIEVAL_COMPONENTS, training=M_TRAINING,
                    validation="scheme holdout | kfold; candidate_pool_size; metrics recall@k, FOSCTTM; split_unit cell", fusion=None)
    raise KeyError(family)


def describe(family) -> str:
    """Plain-text catalogue shown to the plan generator."""
    c = catalogue(family)
    lines = [f"TASK FAMILY: {family} - {FAMILIES[family]['what']}", "", "INPUT REPRESENTATIONS (field inputs):"]
    lines += [f"  - {i['type']}: {i['what']} Parameters: {i['params']}." for i in c["inputs"]]
    for tier, title in (("basic", "BASIC TIER (baselines, the single simple reference, preprocessing)"),
                        ("ambitious", "AMBITIOUS TIER (what candidate plans are mainly built from)")):
        lines += ["", title + ":"]
        lines += [f"  - {x['key']}: {x['what']} Parameters: {x['params']}." + (f" Cost: {x['cost']}." if x.get("cost") else "")
                  for x in c["components"] if x["tier"] == tier]
    if c["fusion"]:
        lines += ["", "FUSION: " + c["fusion"]]
    lines += ["", "TRAINING OPTIONS (field training): " + c["training"], "", "VALIDATION DESIGN OPTIONS (field validation): " + c["validation"]]
    return "\n".join(lines)


def component_keys(family):
    return {x["key"] for x in catalogue(family)["components"]}


def tier_of(family, key):
    for x in catalogue(family)["components"]:
        if x["key"] == key:
            return x["tier"]
    return None


# ----------------------------------------------------------------------------- builders
def build_runner(spec, view, scale="full", seed=0):
    """Executable runner for an incremental-value classification plan (see tools/t_basic.py for the interface)."""
    key = spec["model"]["component"]
    if key == "clinical_only":
        from .t_basic import ClinicalOnly
        return ClinicalOnly(spec, view, scale, seed)
    if key == "patient_mean_logreg":
        from .t_basic import PatientMeanLogReg
        return PatientMeanLogReg(spec, view, scale, seed)
    from . import t_models
    return t_models.build(spec, view, scale, seed)


def build_model(spec, view, scale="full", seed=0, genes=None, time_budget=None):
    """Executable model for a cross-modal prediction plan (see tools/m_models.py for the interface).

    genes: positions inside view.scope to model (None = every in-scope gene). time_budget: seconds allowed for fit()."""
    from . import m_models
    return m_models.build(spec, view, scale=scale, seed=seed, genes=genes, time_budget=time_budget)
