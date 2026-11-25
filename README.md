# Agent 4: Matching Analysis Plans and Conclusions to the Scientific Question and the Evidence

[GitHub repository](https://github.com/YangzhenyuGao/BIOS774_Agent4_ComplexGeneral) · [Project report](https://github.com/YangzhenyuGao/BIOS774_Agent4_ComplexGeneral/blob/main/reports/report.pdf) · [Study 1: PBMC Multiome](https://github.com/YangzhenyuGao/BIOS774_Agent4_ComplexGeneral/blob/main/reports/generated_report_1.pdf) · [Study 2: TCGA-BRCA](https://github.com/YangzhenyuGao/BIOS774_Agent4_ComplexGeneral/blob/main/reports/generated_report_2.pdf) · [Presentation](https://github.com/YangzhenyuGao/BIOS774_Agent4_ComplexGeneral/blob/main/reports/presentation.pdf)

**Repository contents.** This repository includes source code, configuration and prompts, tests, documentation, cached LLM responses, final objectives and candidate plans, pilot evidence cards, tournament and audit records, evaluation summaries, figures, and the deliverables in `reports/` and `submission/`. Publication follows the inclusion rules in `scripts/package_submission.py`; `submission/manifest.json` lists the files and SHA256 checksums in the two submission archives. Raw data, model weights, cell- and patient-level inputs, hidden evaluation data, and large prediction arrays remain on the server. Credentials are excluded. Reports can be read directly online; rerunning the analyses requires the data and environment described below.

**Document status: design document, updated November 25, 2026. The system was built from this design, and both agent workflows and Benchmark 1 completed on Sycamore on November 25, 2026. Section 9.6 records the actual results, departures from the design, and output locations. Step-by-step execution records, job IDs, checks, and unresolved issues are in `docs/PROGRESS.md`. Other sections retain the original planning context; whenever their numbers conflict with Section 9.6, the actual run record takes precedence.**

This document introduces the problem, tasks, evaluation, system design, available resources, and original six-day implementation plan for new readers and reviewers. Status labels distinguish:

- **[Confirmed]**: approved by the project owner.
- **[Default]**: a proposed setting or procedure to use initially, subject to adjustment.
- **[Available]**: an existing resource or a verifiable observation.
- **[To verify]**: a fact to check when implementation begins.
- **[Deferred]**: work outside the main assignment workflow, to be undertaken separately by the project owner.

**Execution policy [Confirmed].** Build the system and then run both datasets through to complete results and reports without seeking confirmation at every stage. Subsequent comparison experiments (Sections 5.5 and 9.5) are separate work after these results have been frozen.

---

## 1. Two levels of objectives [Confirmed]

**Near-term objective: the course assignment.** BIOS 774 Project 4, "Building an AI Agent for Complex Biomedical Data Analysis," is due November 30, 2026, at 23:59 EST, followed by an eight-minute class presentation. The agent must inspect data, interpret the objective, identify analytical difficulties, plan, choose and execute methods, evaluate, visualize, and report autonomously. The assignment requires at least two multimodal datasets and disallows a fixed, manually specified pipeline. Evaluation covers autonomy, strategy quality, method selection and integration, visualization and interpretation, robustness across datasets, and software design and reproducibility.

**Long-term objective: a research paper.** The assignment establishes the system and evaluation framework. Later work will expand them toward a conference submission at the level of NeurIPS. Assignment artifacts should be reusable in that research.

---

## 2. The problem [Confirmed]

**Can an automated analysis agent select an effective analysis plan for a scientific question and make conclusions that match the evidence, instead of always pursuing the highest score?**

Two observations motivate this question.

**First, the same result supports different conclusions under different questions.** Project 3 predicted PAM50 subtypes in TCGA breast cancer. Its test set contained 121 patients:

| Approach | Macro-F1 |
|---|---|
| Clinical variables only: age, stage, histology, ER, PR, HER2 | 0.620 |
| Pathology images only: attention-based multiple-instance learning | 0.645 |
| Images + clinical variables, late fusion | 0.723 |
| Images + clinical variables, excluding receptors | 0.583 |

Receptor status is strongly associated with PAM50 but is not identical to it; immunohistochemistry-based surrogate subtyping and PAM50 can disagree. Correlation alone therefore does not make receptors leakage. Receptors are forbidden inputs when asking what images alone can predict, but belong in the baseline when asking what images add to known clinical information. The question determines the appropriate analysis.

**Second, plans can overstate their own performance.** Earlier projects repeatedly encountered selection driven by seed noise, models that had not converged, and objective requirements that were never checked (Section 7). Each plan also brings its own validation design. Examples of permissive validation include splitting by slide rather than patient, selecting the best of many configurations against a single baseline, and evaluating against smoothed targets. Selecting solely by self-reported scores can favor plans with the weakest validation.

**When this mechanism is useful [Confirmed].** A question with one standard solution, such as conventional differential expression analysis of bulk RNA, does not need multiple plans, pilots, and a tournament. The intended setting has many substantially different feasible approaches, with failures that a score alone may conceal: incomplete training, small-sample overfitting, or invalid validation. Both candidate pools therefore emphasize ambitious methods, with simple methods serving as baselines and references (Sections 6.2 and 6.5).

---

## 3. Related work and the possible gap [Available; requires rechecking]

These notes came from more than a dozen keyword searches on November 19, 2026. The search was not exhaustive; most papers were assessed through abstracts or secondary summaries, and the Biomni full text was unavailable. **The descriptions below have not been checked against every original paper and are not established literature claims. They must be revisited before a research submission.**

| Category | Examples | Our preliminary understanding |
|---|---|---|
| Pairwise debate, Elo ranking, and evolution | Co-Scientist (Nature 2026); DEEVO | Candidates debate; a judge determines winners and rankings |
| Score-based search | AIDE, SELA, MLE-STAR; Google ERA | Generate and execute candidates, then retain high-scoring ones |
| Ranking with experimental feedback | MOOSE-Chem3 (NeurIPS 2025) | Use tested hypotheses to rank remaining hypotheses |
| Multiple analysis paths | Agentic Garden of Forking Paths; Branchpoint | Quantify variation in conclusions across analytical choices |
| General biomedical agents | Biomni (Science 2026) | Tool retrieval, code as actions, and a knowledge library |
| Elimination after small pilots | Successive halving, Hyperband | Run low-budget trials and eliminate poor performers |

Diagnostic work considered in the design includes *Silent failures in agentic data science*, on downstream scores concealing leakage and invalid validation, and *BaitBench*, on agents taking shortcuts that inflate scores. Preliminary notes also include unfavorable evidence: reported LLM-judge accuracy near chance for research-agent trajectories and an audit in which pairwise comparison was the weakest judging approach. These observations carry the same verification caveat as the table.

**Assessment.** Neither tournaments nor pilot-based selection are new. We did not find a study combining executed pilot evidence with question-conditioned judging and testing its benefit, but failure to find one does not establish novelty. A defensible contribution would be a testable claim and a calibrated evaluation: how much self-reported performance is inflated in real biomedical tasks, and which selection methods choose plans that perform well under a common protocol.

---

## 4. Two tasks [Confirmed]

Both tasks ask a scientific question and provide reference benchmarks: a ceiling for Task M and a clinical baseline for Task T. The input is data plus a natural-language question; the agent determines the analysis.

### 4.1 Task M: PBMC Multiome

> Learn cross-modal relationships from paired RNA and ATAC data. For previously unseen cells from the same donor, with only ATAC available, which genes' RNA expression variation can be predicted reliably?

**Data split, owned by the evaluator [Default]**

- Randomly split quality-controlled cells into 70% development and 30% hidden evaluation cells using a fixed seed.
- For development cells, the agent sees RNA and ATAC and chooses its internal training and validation design.
- For hidden cells, the agent sees only ATAC; hidden RNA belongs to the evaluator.
- Binomially thin each hidden cell's RNA reads into halves A and B. **Every method predicts B.**

**Reference benchmarks**

| Reference | Definition | Purpose |
|---|---|---|
| Ceiling | An RNA-to-RNA model observes hidden-cell half A and predicts half B | Approximate the predictability allowed by measurement noise |
| Floor 1 | Mean expression within ATAC-derived clusters | Measure what cell-type structure alone can predict |
| Floor 2 | Gene activity from promoter and nearby peak counts | Provide a conventional method without learned prediction |
| Floor 3 | Prediction from ATAC sequencing depth alone | Separate depth effects from other predictive information |
| Floor 4 | Latent-space ridge regression, k-nearest-neighbor regression, and partial least squares | Establish elementary learned baselines; these do not otherwise occupy candidate slots |

The ceiling is not a rejection threshold: ATAC may contain less information than RNA. Floors 1 and 2 are the minimum comparators; failure to beat them calls for reconsideration. The ceiling describes the fraction of empirically predictable variation recovered.

**Primary metric.** For each gene, Pearson correlation across hidden cells between the prediction and measured half-B expression after log normalization, denoted by r.

**Definition of reliable prediction [Confirmed: the primary conclusion concerns improvement over the cell-type baseline; see Section 4.3 for threshold rules]**

Why not define reliability as r significantly above zero? With thousands of cells, weak correlations can become significant. In a Sycamore setup check using 3,000 cells and 500 genes—not a final experiment—ridge passed that criterion for 475 genes. Meanwhile, median correlation was 0.28 for cluster means and 0.27–0.30 for ridge, k-NN, and partial least squares. This suggested that much of the predictable RNA variation may simply reflect cell identity. The primary question is therefore: **for which genes can ATAC predict variation beyond the cell-type baseline?**

- Construct a per-gene null distribution of r by repeatedly permuting the correspondence between predicted and observed cells.
- Assign each gene to one of three categories:
  - **Unjudgeable:** ceiling r is not significantly above the null; the gene cannot be measured reliably enough for this comparison.
  - **Not beyond the cell-type baseline:** the plan's r is not significantly greater than cluster-mean r. Genes with no predictive signal also belong here.
  - **Beyond the cell-type baseline:** the plan's r significantly exceeds cluster-mean r under a paired bootstrap by gene and multiple-testing correction across genes at 5% false discovery rate. These genes support the primary conclusion.
- Select cluster granularity by development-only cross-validation to maximize the cluster baseline's own performance, then freeze it. This makes the comparison conservative; granularity cannot be adjusted after seeing hidden results.
- Define the gene scope using development-cell read thinning: retain genes whose empirical ceiling is above the null. Remove extremely rarely detected genes beforehand only for numerical stability.
- Also report descriptively: the number of genes with r above zero, each gene's r and ceiling ratio, within-cluster r, and gene-count sensitivity curves across thresholds. Conclusions should not depend on one cutoff.

Finding few or no genes beyond the baseline is a valid outcome. If all six candidates perform similarly, validity and cost may dominate selection.

**Agent deliverables:** predictions for hidden cells; a list of genes claimed to be predictable beyond the baseline; and the agent's own correlation estimates and confidence intervals.

### 4.2 Task T: TCGA-BRCA

> Given existing clinical information, including ER, PR, and HER2, does adding H&E pathology improve four-class PAM50 prediction? How large is the improvement, and how certain is the evidence?

**Cohort.** There are 802 patients with four-class labels and diagnostic slides: LumA 410, LumB 190, Basal 137, and Her2 65. Both comparison arms must use the same patients and the same missing-data treatment.

| Reference | Definition | Purpose |
|---|---|---|
| Clinical floor | A clinical-only model fixed and thoroughly tuned by the evaluator | Comparator for incremental gain |
| Simple reference | Clinical variables plus patient-mean image features, with logistic regression | Establish what elementary fusion achieves |
| Historical reference | Project 3 late fusion, macro-F1 0.723 | Context, not ground truth |

A shuffled-image negative control is outside the main assignment workflow for the computational reasons below and is deferred to Section 9.5.

**Evaluation [Default]**

- Repeated nested cross-validation over all 802 patients: five outer folds and three repeats, with model and configuration selection restricted to inner folds.
- Primary metric: paired macro-F1 difference, image-plus-clinical minus clinical-only, with patient-level bootstrap confidence intervals.
- Secondary metrics: log-loss and one-vs-rest AUROC differences, changes in per-class recall with emphasis on LumA and LumB, and sensitivity to holding out collection sites.
- Three possible conclusions, based on the 95% confidence interval for gain:
  - **Improvement:** lower bound above zero.
  - **Insufficient evidence:** interval includes zero.
  - **Deterioration:** upper bound below zero.
- Report gains and intervals for all six candidates, not just the champion.
- Describe incremental predictive value; do not infer an independent biological mechanism.

**The champion's estimate may be optimistic; the assignment does not statistically correct this [Confirmed].**

Selection and evaluation use the same 802 patients. Selecting among six noisy estimates can favor a result partly due to chance, making the selected gain and improvement conclusion optimistic.

A rigorous negative-control approach would shuffle image features, rerun all six candidates, and use the distribution of maximum gains. That requires six candidates × 15 outer folds × inner selection × many permutations. The original plan anticipated GPU use for tile-based methods and hours of queueing, making this infeasible within the assignment deadline. It is deferred to the paper stage.

The assignment instead shows every candidate and explicitly acknowledges selection optimism. Broadly positive gains across candidates provide stronger support; an isolated positive champion should be treated as insufficient evidence.

**Existing context.** Project 3's single split gave macro-F1 gain +0.103 (95% CI −0.020 to +0.223) and AUROC gain +0.087 (+0.042 to +0.139). A 121-patient test set cannot establish the gain precisely, motivating repeated cross-validation. Clinical-only LumB recall was 0.18, suggesting that image gains might concentrate on distinguishing LumA from LumB.

### 4.3 Rules for data-derived thresholds [Confirmed]

| Threshold | Calculation | Data used |
|---|---|---|
| Task M: unjudgeable genes | Cell-permutation null and multiple-testing correction | The cells on which predictions are evaluated |
| Task M: beyond the cell-type baseline | Paired bootstrap of the per-gene correlation difference, with multiple-testing correction | The cells on which predictions are evaluated |
| Task M: cluster granularity | Development-only cross-validation maximizing baseline performance | Development data only |
| Task M: gene scope | Whether the development-set ceiling exceeds the null | Development data only |
| Task T: conclusion | Position of the 95% gain interval relative to zero | The same patients and splits in both arms |

Three constraints apply:

1. **Specify rules before seeing results, derive values from data, then freeze them.** Store frozen values, computation records, and checksums.
2. **Do not tune thresholds after inspecting final results.** A change requires changing the rule, rerunning the analysis, and documenting it.
3. **Report sensitivity.** Show how conclusions vary with thresholds.

The only retained conventional levels are 5% FDR and 95% confidence. During objective compilation, the agent applies the same rules to the data it can see; the evaluator computes them independently on hidden data.

### 4.4 Robustness demonstration [Default]

Each dataset also receives an alternative question: image-only PAM50 prediction for TCGA, and RNA-to-ATAC retrieval of the same cell for PBMC. Run only through plan generation, without execution, to show how permitted inputs, comparators, and metrics change with the question.

---

## 5. Evaluation design [Confirmed framework; default settings]

### 5.1 Three levels of evaluation

| Level | Example | Verifiability |
|---|---|---|
| Procedural compliance | Forbidden inputs, patient-level splitting, access to hidden data | Directly checkable |
| Predictive accuracy | Predictions of held-out labels or RNA measurements | Assessable against observations |
| Scientific validity | Whether images make an independent biological contribution | Usually not identifiable from these data alone |

Quantitative conclusions are restricted to the first two levels.

### 5.2 Scores under a common protocol

Each candidate is a complete executable workflow. The evaluator executes each under a common protocol to obtain its evaluation score, called the "true score" in the saved records:

- Task M: fit on development cells and score against half B of hidden cells.
- Task T: execute within the evaluator's repeated nested cross-validation, using the fixed clinical comparator.

Self-reported scores come from each plan's own validation design. Their difference from the common-protocol scores measures inflation.

### 5.3 Metrics

| Metric | Definition | Question answered |
|---|---|---|
| Position relative to references | Task M: fraction of ceiling and improvement over floors. Task T: gain and confidence interval | How good is the candidate itself? |
| Generation gap | Task M: ceiling minus the best candidate's evaluation score | How good is the candidate pool? |
| Selection gap | Best candidate's evaluation score minus the selected candidate's score | How well did selection work? |
| Inflation | Self-reported score minus evaluation score | Is reported performance calibrated? |
| Gene-list replication | Fraction of Task M's claimed genes confirmed on hidden cells | Are the claimed findings reliable? |
| Agreement across candidates | Task T: numbers with positive gains and with lower confidence bounds above zero | Is the champion's result isolated? |
| Conclusion agreement | Agreement between the agent's three-way conclusion and the evaluator's | Does the conclusion match the evidence? |
| Cost | GPU time, CPU time, LLM calls, and tokens | What does the workflow cost? |

A poor candidate pool is visible through a large generation gap or failure to beat the floors; a small selection gap cannot hide it.

### 5.4 Benchmark 1: full evaluation of every candidate [Confirmed; part of the main run]

After the agent has finished, the evaluator runs every candidate at full scale under the common protocol. This determines whether the agent selected the best candidate and, if not, how far behind it was.

Benchmark results are generated only after selection and are not fed back into the agent's decisions. The evaluator shares the executor with the agent; this benchmark adds computation rather than a different modeling implementation.

### 5.5 Benchmark 2: comparison with other selection methods [Deferred]

The candidate pools, task constraints, and evidence cards are frozen and saved. These comparisons can therefore be conducted later using additional LLM calls without rerunning the numerical analyses.

**Records required from the main workflow [Confirmed]**

Each task directory contains an index listing artifact paths and checksums.

| Record | Contents |
|---|---|
| Common selector inputs | Original question, compiled objective, difficulty list, and task constraints; identical for every selector |
| Candidate descriptions | Complete structured fields, a text-only description without pilot results, and values along the three diversity dimensions |
| Provenance | Generation prompts, model identifiers, raw responses, discarded duplicates, and reasons for removal |
| Executable configuration | All settings and seeds needed to rerun the plan |
| Compliance checks | Outcomes for each rule, including rejected plans |
| Evidence cards | Numbered entries and underlying pilot outputs: fold scores, training curves, and timings |
| Self-reported scores | Scores and variation under each plan's own validation design |
| Benchmark 1 scores | Common-protocol full results and confidence intervals; per-sample predictions, including Task T out-of-fold predictions and Task M prediction matrices, for rescoring later |
| Tournament records | Both presentation orders for every pair: complete inputs, verdicts, cited evidence IDs, reasoning, model identifier, tokens, cost, and time; final ranking |
| Compute costs | GPU and CPU time for every candidate's pilot and full run |

Keep all candidates, not only the champion. Store evaluation scores separately from selector inputs. Selection methods must not read these scores, except for the explicitly labeled oracle reference.

| Selection method | Purpose |
|---|---|
| Random selection | Lower reference |
| LLM selects directly from plan text | Is execution evidence necessary? |
| Highest pilot score | Conventional score-based selection |
| Fixed rule checks, then highest score | Is an LLM judge necessary? |
| LLM debate without evidence | Co-Scientist-style comparison |
| LLM selects after seeing all evidence cards at once | Does pairwise competition add anything beyond evidence? |
| Evidence-judged tournament | Proposed mechanism |
| Highest common-protocol evaluation score | Oracle upper reference |

Use the same frozen candidate pool and task constraints, with access to evidence determined by each comparison condition. This stage also assesses judge reliability under changed presentation order and changed judge model. Order-reversal records already exist from the main workflow.

### 5.6 Limitations of this evaluation

1. **The scale cannot establish general tournament superiority.** Two tasks with six candidates each can show the selected rank and gap, but different selectors may choose the same plan.
2. **Task T pilots and evaluation scores use the same patients.** No extra hidden set is reserved because the cohort is small and contains only 65 Her2 patients. Correlated fluctuations may favor score-based selection. An external cohort is deferred to the paper stage.
3. **Task M's ceiling is empirical.** It is produced by an RNA-to-RNA model, not a theoretical bound. If an ATAC model exceeds it for a gene, cap the reported fraction at one and flag that gene.
4. **Task M splits cells within one donor.** Conclusions concern unseen cells from that donor, not cross-donor generalization.
5. **Threshold rules remain human choices.** Values are data-derived, but permutation tests, cluster-mean comparisons, and granularity selection are methodological choices. State them and report sensitivity, correlation magnitudes, and ceiling ratios.
6. **Task T's champion is optimistically selected, without a negative control.** Showing all candidates exposes but does not correct selection bias. Shuffling has not established that the full workflow avoids spurious discoveries.
7. **Beyond-cell-type findings depend on baseline strength.** Finer clusters can yield stronger baselines. Selecting the strongest development-set granularity is conservative and may understate additional ATAC information.
8. **Plans combine registered tools.** This supports execution but limits diversity.

---

## 6. Agent design

The same overall workflow handles both datasets without branching on dataset names; differences are driven by the data profile and the question.

### 6.1 Workflow [Confirmed]

| Step | Action | Responsible component | Output |
|---|---|---|---|
| 1. Profile and diagnose | Measure modality size, sparsity, missingness, and pairing; run quantitative diagnostics with evidence and severity | Deterministic code | Data profile and difficulty list |
| 2. Compile objective | Specify analysis unit, estimand, allowed and forbidden inputs, comparators, primary metric, acceptance criteria, and scope | LLM with schema checks and rule fallback | Structured objective |
| 3. Generate plans | Initially propose 8–10 candidates, deduplicate to six complete plans, retrieve relevant know-how, then freeze the pool | LLM generation and code-based deduplication | Candidate pool |
| 4. Check compliance | Reject forbidden inputs, incorrect splitting units, and budget violations | Deterministic code | Compliance records |
| 5. Run pilots | Execute every plan at reduced scale | Deterministic code | Evidence cards |
| 6. Select | Pairwise tournament; every verdict must cite numbered evidence | An LLM different from the generator | Ranking and verdicts |
| 7. Execute in full | Run the champion | Deterministic code | Results |
| 8. Audit | Check criteria and reference benchmarks; fall back to the next plan or regenerate once; if still unsuccessful, report that available plans cannot answer the question | Deterministic code and LLM | Audit record |
| 9. Report | Organize conclusions, evidence, limitations, and recommendations; populate and verify numbers from saved results | LLM prose and code-supplied numbers | PDF report |

### 6.2 What constitutes a plan [Confirmed]

A plan specifies the analysis unit, permitted inputs, preprocessing, model, validation design, metrics, and allowable conclusions. Differences should concern these analytical choices, rather than simply changing model names.

**Pool size and diversity [Confirmed: default six candidates]**

Retain six plans per task, configurable up to eight. Diversity spans three dimensions:

| Dimension | Meaning | Task M examples | Task T examples |
|---|---|---|---|
| Data | Inputs, representations, preprocessing | LSI, gene activity, peaks near a gene | Encoder choice or combinations; tile- versus patient-level features |
| Model | Model class and capacity | Translation networks, MultiVI, cis-regulatory and residual models | Attention or Transformer MIL, multi-encoder fusion, hierarchical classification |
| Strategy | Overall analytical approach | Separate versus joint gene models; ensembles | Early concatenation, late fusion, clinical-offset models, validation design |

Generation rules:

- **Emphasize ambitious plans [Confirmed].** At least four of six must use deep, nonlinear, joint-latent, multicomponent, or combined approaches. The generator is asked to design at the level of a methods paper, beyond introductory textbook methods.
- **Keep exactly one simplest valid reference.** It establishes what complexity buys. Ridge, k-NN, and PLS otherwise remain baselines, as does patient-mean logistic fusion in Task T.
- Include at least two distinct settings along each diversity dimension.
- Keep only one of two plans whose structured fields are too similar, following the idea of a similarity check.
- If fewer than six remain after deduplication, request another round; if still short, proceed with the available count and record it.

Three constraints apply to ambitious plans:

1. **Complexity earns no credit by itself.** Judges must use evidence, not how advanced a method sounds.
2. **Provide evidence of adequate training.** Include curves, convergence judgments, and direct comparison with the simple reference. Earlier projects failed on this point (Section 7.1).
3. **Declare the budget.** State pilot and full-run GPU/CPU requirements. The default cap is a full run fitting within one two-hour short-queue job.

With only 802 labeled TCGA patients, ambitious methods may not beat the simple reference. That is a valid result to report. Six candidates yield 15 pairs, judged in both orders for 30 judge calls.

### 6.3 Evidence-card contents [Default]

- Score and variation under the plan's validation design.
- Comparisons with all required reference benchmarks.
- Convergence evidence, including whether the end of the learning curve is still improving.
- Validation facts: splitting unit, fold count, matched sample sizes across arms, tuning effort per arm, and whether targets were smoothed.
- Time and resource consumption.

Every entry has an ID that the judge must cite.

### 6.4 Difficulties diagnosed [Default]

| Difficulty | Task M | Task T |
|---|---|---|
| Measurement noise | Sparse RNA counts constrain predictability | Not applicable |
| Technical confounding | Correlation between ATAC and RNA depth | Image features can predict collection site; 39 sites in the source cohort |
| Class imbalance | Smallest annotated cell type has 18 cells | Her2 accounts for 8% |
| Small sample, many features | Less prominent | 802 patients versus 768–2560 features |
| Missingness | None | Approximately 16% missing HER2 |
| Hierarchy | None | 66 patients have multiple slides |
| Strong baseline | Cell type explains much of the predictable variation | Receptors are already highly predictive |
| Label provenance | Cell-type annotations derive from joint clustering and cannot independently validate cross-modal prediction | PAM50 derives from RNA; RNA-derived predictors are forbidden |
| Multiple comparisons | Thousands of genes | Encoders multiplied by fusion strategies |
| Statistical power | Relatively ample | Limited; estimate the minimum detectable difference beforehand |

### 6.5 Tool registry [Default]

The registry has two tiers. Candidates primarily use the ambitious tier; basic tools support preprocessing, baselines, and the simple reference.

**Task M**

| Tier | Components |
|---|---|
| Basic | TF-IDF and LSI; gene activity; cluster-mean prediction; ridge, PLS, and k-NN regression |
| Ambitious | Residual models: predict cluster means, then learn deviations with a nonlinear model to target variation beyond cell type |
| | Cis-regulatory models: nearby peaks plus global cell state, using boosting or small neural networks |
| | Cross-modal translation: ATAC encoder and RNA decoder, jointly trained across genes with a count likelihood |
| | Joint latent models: adequately trained MultiVI for ATAC-only expression imputation, or PeakVI latents with a prediction head |
| | Contrastive pretraining on development cells followed by prediction |
| | Denoising targets pooled from neighboring training cells; evaluation still uses unsmoothed half B |
| | Multitask outputs with gene embeddings or low-rank structure |
| | Stacking different model families using out-of-fold predictions |

**Task T**

| Tier | Components |
|---|---|
| Basic | Tile-, slide-, and patient-level features from five encoders; patient means; PCA; multinomial logistic regression; class weighting; fold-local imputation; patient and site splits |
| Ambitious | Gated-attention, multi-head, and Transformer MIL over tiles |
| | Attention or gated fusion of five foundation-model encoders at tile or patient level |
| | Morphology prototypes: tile clusters and patient-level proportions |
| | Clinical-conditioned image fusion through cross-attention or feature modulation |
| | Incremental models with clinical predictions as an offset, learning residual information from images |
| | Hierarchical classification with a specialized LumA-versus-LumB branch |
| | Self-supervised aggregator pre-adaptation using unlabeled patients, within folds or with explicit disclosure of included patients |
| | Feature-level augmentation, tile resampling, mixing, gradient boosting, and stacking |

**Registry expansion [Confirmed].** Before pool freeze, Gemini may propose unregistered components. The implementer may add them within budget; otherwise record the proposal and rejection reason. No components are added after freeze. The LLM may not freely generate and execute code at runtime.

### 6.6 Scope of the three principal components [Default]

This is narrower than the earlier proposal to include all three components in full.

| Component | Assignment | Paper stage |
|---|---|---|
| Difficulty diagnosis and audit | Full implementation | Ablation studies |
| Evidence-judged tournament | Round robin, Bradley–Terry ranking, both presentation orders | Multiple evolutionary rounds and plan merging |
| Know-how library | Static read-only failures from prior projects, retrieved by difficulty tags | Automatic write-back and assessment of whether prior answers make evaluation effectively open-book |

Free-form code generation and automatic literature-based tool discovery are outside this scope.

### 6.7 LLM use [Confirmed roles; availability tested]

Use different vendors for generation and judging to reduce preference for a model's own writing.

| Role | Model | Interface | Fallback |
|---|---|---|---|
| Objective compilation, plan generation, report prose | Gemini 4 Carbon (`gemini-4-carbon`), described in the original planning notes as Google's strongest reasoning model | Gemini API | No alternate generator; retry after a six-minute timeout |
| Tournament judging and audit review | Claude Opus 5.5 (`claude-opus-5-5`) | Noninteractive Claude Code using the existing local login | `Qwen/Qwen3.8-2.4T-A95B` or `deepseek-ai/DeepSeek-V4-Pro` through the Hugging Face router |

Longleaf checks on November 24, 2026 returned successfully for every listed model: approximately 11 seconds for Gemini and four seconds for Claude. The previously mentioned `Qwen3-VL-8B` is a vision model and is not used.

Sycamore checks on November 25 returned successfully for Claude and both Hugging Face models in 1–8 seconds. Gemini 4 Carbon was intermittent: seven of 15 calls returned in 4–48 seconds; eight did not return, including one observed for 400 seconds. Other Gemini models using the same key returned within seconds, suggesting a model-service issue rather than Sycamore connectivity.

**Gemini calling rules [Confirmed]**

- Use only Gemini 4 Carbon for compilation, planning, and report prose. Preserve the selected reasoning level rather than automatically switching models.
- Wait at most six minutes per attempt, then retry until a response arrives. Log every timeout and retry.
- Estimated planning cost: with roughly half the test calls hanging, about 30 generation calls across both tasks could entail hours of waiting.

**Claude judge setup**

- No Anthropic API key or corresponding SDK was available. Use Claude Code's noninteractive CLI, tested on Longleaf and rechecked on Sycamore before the formal run.
- Supply a custom system prompt, disable all tools, omit user settings and project memory, disable session persistence, and run from an empty directory. The judge should see only the supplied plans and evidence.
- Calls require an existing Claude Code login. The login-node controller makes LLM calls while submitting numerical computation through Slurm.
- An official SDK can replace the CLI interface when an Anthropic key becomes available.
- The selected Claude model does not accept a temperature parameter; assess stability by reversing presentation order for each pair.

**Common rules**

- Route all calls through one client with retries, backoff, and prompt-hash caching. Fallback models apply only to judging.
- Submit cached answers with the code so LLM responses can be replayed without keys or login. Numerical execution still requires the stated data and environment.
- Freeze prompts before the formal run.
- Send only aggregate statistics, plan text, and evidence cards to external services, never patient- or cell-level records.
- The original estimate was about 45 calls per task, including 30 tournament judgments, or approximately 100 calls overall.

---

## 7. Existing resources [Available]

### 7.1 Previous projects

| Project | Task | Data | Final outcome |
|---|---|---|---|
| Agent 1 | Dimensionality reduction, ten methods | PBMC3k, PathMNIST | Selected PCA, MDS, and t-SNE for PBMC3k; PCA, MDS, t-SNE, and UMAP for PathMNIST |
| Agent 2 | Five generative models | PathMNIST, PBMC3k | Synthetic data not recommended for PathMNIST; cautious use of scVI for PBMC3k |
| Agent 3 | Multimodal representation learning | PBMC Multiome, TCGA-BRCA | Selected dual-tower CLIP for PBMC, macro-F1 0.925; late fusion for TCGA, macro-F1 0.723 |

**Inherited strengths:** typed profiles and objectives, method registries, pilot runs, frozen plans before testing, one controlled test-data entry point, report numbers tied to saved evidence, single-command execution, and acceptance of negative conclusions.

**Recurring problems Agent 4 seeks to address**

| Problem | Source | Observation |
|---|---|---|
| Noise-driven selection | Agent 1, first version | Identical inputs produced different selections because runtime differed by less than a second |
| | Agent 2, first version | A 0.01 decision threshold was used despite variation of about 0.25 across seeds |
| Inadequate training | Agent 2, first version | Four generative models trained for about one minute in total and produced noise |
| | Agent 3 | MultiVI trained for only 50 epochs without early stopping |
| Unchecked objective requirements | Agent 3 | Joint structure was required to match the better unimodal representation, but selected CLIP had clustering ARI 0.528 versus RNA's 0.613 |
| Labeling or implementation errors | Agent 3 | The purported ATAC-decoder baseline returned RNA principal components; WNN graph overlap with Seurat was only 0.195 |
| Unhandled imbalance | Agent 3 | The fusion head used unweighted cross-entropy and achieved Her2 recall 0/10 |
| Planning reduced to filtering | All three projects | Rule-based applicability checks, no LLM, and no comparison of different analytical strategies |
| Reports assembled evidence without interpretation | All three projects | Numbers were present but explanations and conclusions were weak |
| Excessive engineering | Agent 3 | Runtime replacement of third-party functions and dataset-specific workflow scripts |

### 7.2 Data

| Property | PBMC Multiome | TCGA-BRCA |
|---|---|---|
| Analysis unit | Cell | Patient |
| Source-cohort size | 11,909 cells from one donor | 1,061 patients, 1,132 slides, 39 collection sites |
| Modalities | Paired RNA: 36,601 genes; ATAC: 108,377 peaks | H&E whole-slide images and six clinical variables |
| Labels | 19 cell types, manually annotated from joint clustering | Four PAM50 classes for 802 patients |
| Additional files | Peak annotation: promoter/distal status, gene, distance | Survival information |

Agent 3's patient-level clinical table and slide manifest can be reused read-only.

### 7.3 Pathology feature bank

Location: `../Agent3_RepresentationLearning/outputs/tcga_brca_hw4_feature_bank/`. Extraction and validation completed November 22, 2026.

| Encoder | Dimensions |
|---|---|
| UNI2-h | 1536 |
| CONCH-Z | 768 |
| Virchow2 | 2560 |
| MUSK | 1024 |
| H-Optimus-1 | 1536 |

All five encoders use identical tile coordinates and ordering: 135,911 tiles, at most 128 per slide, median 123, minimum six. Tile features, slide means, and patient means are available. At the time of the original design, the three newly added encoders had not yet been used in downstream models.

### 7.4 Environment and computation

- Reuse Project 3's conda environment: Python 3.11, scanpy, muon, scvi-tools, PyTorch, scikit-learn, WeasyPrint, and related packages. No LLM client libraries are installed; use existing HTTP libraries. No new dependencies were initially needed. If necessary, install new packages into that environment under the rules in Section 9.0.
- The original plan anticipated fewer than ten GPU-hours on Sycamore H100s. Task T mostly uses CPU with cached features; Task M neural models were expected to need one GPU. Actual compute use is recorded in Section 9.6.
- Gemini and Hugging Face credentials were tested successfully. Service availability was intermittent, with large models sometimes returning HTTP 503 or timing out.

---

## 8. Relationship between the assignment and the paper [Confirmed]

| Aspect | Assignment, by November 30, 2026 | Later paper |
|---|---|---|
| System | Complete Section 6 workflow | Same system |
| Tasks | Two real tasks with reference benchmarks | Additional datasets, donors, external cohorts, and task families |
| Main demonstration | Question-dependent planning and interpretable analyses; self-reported versus evaluation scores | Conditions under which evidence-based selection is more reliable than strong baselines |
| Selection evaluation | Benchmark 1: run all candidates and locate the selected plan in the ranking | Same design at larger scale |
| Selector comparison | Deferred Benchmark 2 | Multiple LLMs, repeated runs, statistical tests, and blinded human assessment |

**Paper-stage extensions**

| Extension | Question tested |
|---|---|
| Direct selection versus pairwise comparison with identical evidence | Does the tournament format add value? |
| Identical candidates with and without executed evidence | Does the benefit come from execution feedback? |
| With and without objective diagnosis or know-how retrieval | Does each component contribute independently? |
| Pools of valid, mixed, and entirely invalid candidates | Can the system choose well without rejecting everything? |
| Controlled traps and matched clean tasks | Can it detect specific errors? Traps must have observable clues or be specified in the objective |
| Simulated data | How do signal strength, redundancy, complementarity, and confounding affect selection? |
| Different judges, orders, and budgets | Are gains stable and worth the cost? |

Two useful indicators for deciding whether to pursue a paper are whether rule checks followed by highest-score selection perform as well as the tournament, and whether reversing presentation order changes judgments. The latter is measured in the main run; the full selector comparison remains deferred.

---

## 9. Implementation plan [Default]

### 9.0 Handoff notes [Confirmed: run on Sycamore]

The original design was written on Longleaf for implementation by a Claude session on Sycamore, where H100 GPUs are available. Both clusters share the same `/vast` paths.

| Item | Setting |
|---|---|
| Working directory | `/vast/som/yunligrp/users/ygao/BIOS774Agent/Agent4_ComplexGeneral` |
| Cluster conventions | Follow Project 3's README and `docs/`: Slurm account `h100_harlin`; partitions `h100_short` (up to two hours), `h100_priority`, and `h100_preempt` |
| GPU limits | Follow Project 3's accounting convention: set a total budget, log usage, and use at most two GPUs simultaneously |
| Python environment | `/vast/som/yunligrp/users/ygao/conda/envs/agent3`; set `PYTHONNOUSERSITE=1`. Install necessary additions here, not in a new environment or user site. Check that additions do not change existing package versions, because Project 3 depends on them, and record additions |
| R environment | `module load r/4.6.0`; install additional R packages into `~/R/x86_64-pc-linux-gnu-library/4.6`. Current plans do not use R |
| LLM credentials | `~/api.env` on Sycamore, containing `GEMINI_API_KEY` and `HF_TOKEN`. Read only; never print values |
| PBMC data | `../data/pbmc_granulocyte_sorted_10k/` |
| TCGA clinical table and slide manifest | `../Agent3_RepresentationLearning/outputs/tcga_brca/preprocess/`, read-only |
| Pathology features | `../Agent3_RepresentationLearning/outputs/tcga_brca_hw4_feature_bank/`, read-only |

Initial setup checks:

1. Locate Claude Code on Sycamore, test `claude-opus-5-5` as described in Section 6.7, and record time and cost.
2. Test `gemini-4-carbon` using the local credentials.
3. Verify outbound network access from compute nodes.
4. Verify that the listed data paths are readable.

If a check fails, use the applicable fallback in Section 6.7 and document the issue while continuing other work.

### 9.1 Directory structure

The original design separates responsibilities as follows; Section 9.6 lists the actual output locations.

```text
Agent4_ComplexGeneral/
  README.md  pyproject.toml  Makefile
  config/        Dataset descriptions, questions, LLM and budget configuration
  knowhow/       Static experience entries
  src/agent4/
    schemas      Types for profiles, objectives, plans, evidence, and verdicts
    profile/     Data profiling
    diagnose/    Difficulty diagnosis
    objective/   Objective compilation
    planning/    Plan generation and compliance
    tools/       Registry: preprocessing, models, fusion, validation
    pilot/       Pilot execution and evidence cards
    select/      Tournament and ranking; interfaces for later selectors
    execute/     Full execution
    audit/       Final checks and review
    report/      Report generation
    llm/         Client, caching, and replay
  bench/         Splits, read thinning, references, common scoring, comparisons
  scripts/       Single-command entry points and Slurm scripts
  templates/  tests/  outputs/  reports/  submission/
```

The evaluator and agent are separate packages. Tests check that agent code does not import the evaluator or name hidden-data locations.

### 9.2 Execution order

Run continuously rather than requesting stage-by-stage approval. Each phase has checks; resolve failures before proceeding to dependent work.

| Phase | Work | Check |
|---|---|---|
| 1 | Skeleton and schemas; evaluator splits, read thinning, reference models, and threshold rules | Task M ceiling and floors available, cluster granularity frozen, Task T clinical floor available, hidden-data separation checked |
| 2 | Registries, executor, pilots, and evidence cards | Handwritten plans run on both tasks and produce cards |
| 3 | LLM client and cache, objective compilation, generation and deduplication, know-how, tournament, audit | Both tasks run end to end |
| 4 | Formal agent runs | Champion results and audits complete |
| 5 | Benchmark 1: all candidates under the common protocol | Every candidate has evaluation scores and inflation estimates |
| 6 | Two study reports and figures, project report of at most four pages, slides, packaging | Deliverables complete and cached responses replayable |

The original target was to finish phases 1–5 before November 28, reserving two days for reports, slides, and contingencies.

**Scope reductions if behind schedule:** first the alternative-question demonstrations, then site-held-out sensitivity, then unused registry components. Retain both tasks' reference benchmarks, six mostly ambitious candidates, the tournament, and Benchmark 1. If GPU queues become the bottleneck, reduce ambitious pilots rather than replacing the methods with simple ones.

### 9.3 Deliverables

- Source code with single-command execution.
- For each dataset: analysis plans, quantitative evaluation, visualizations, selected-method descriptions, and a final report. `generated_report_1` is PBMC; `generated_report_2` is TCGA.
- `report`, at most four pages: architecture, difficulty identification, strategy selection, tools and models, both datasets' results, strengths, and limitations.
- `presentation` for an eight-minute talk.

### 9.4 Presentation narrative

1. **Objective:** the problem, tasks, and reference benchmarks.
2. **Architecture and intended contribution:** diagnosis, multiple plans, pilot evidence, judging, audit, and relationship to prior work.
3. **Performance:** Task M's genes beyond the baseline and fraction of ceiling; Task T's gain, uncertainty, all six candidates, and class-specific changes.
4. **Benchmark 1:** the selected plan's rank, self-reported versus evaluation scores, and an actual verdict excerpt.
5. **Benchmark 2:** selector comparison, to be added when completed.
6. Limitations and next steps.

### 9.5 Deferred work outside the main assignment

| Item | Requirements | Already saved |
|---|---|---|
| Benchmark 2: eight selectors from Section 5.5 | Additional LLM calls | Frozen pool, evidence cards, evaluation scores |
| Debate without evidence | Additional LLM calls | Plan text |
| Agreement after changing the judge | Additional LLM calls | Every matchup input and Claude verdict |
| Multiple evolutionary rounds and plan merging | New code and computation | Rankings and reasoning |
| Automatic know-how write-back and assessment of answer leakage | New code | Audit records |
| Task T negative controls: shuffle images, rerun six plans, retain maximum gains to calibrate selection bias and spurious discovery | Substantial computation; tile methods may need GPU, while patient-level methods can be assessed first on CPU | Executable configurations, seeds, and splits |
| Controlled traps and simulated data | New tasks | Evaluator framework |
| External cohorts, donors, and task families | New data | Reusable code |
| Full-text verification of related work | Reading | Section 3's reference list |

### 9.6 Actual run record: November 25, 2026, Sycamore

Phases 1–6 are complete. Benchmark 2 was not run, as planned; its required inputs were retained (Section 5.5 and the checklist in `docs/PROGRESS.md`). That progress log contains the detailed chronology, job IDs, and checks.

**Results**

| Item | Task M: PBMC Multiome | Task T: TCGA-BRCA |
|---|---|---|
| Pool | Six compliant candidates: five ambitious, one simple reference | Six compliant candidates: five ambitious, one simple reference |
| Tournament ranking | P1 > P6 > P5 > P3 > P4 > P2; 100% agreement across presentation orders | P1 > P3 > P6 > P2 > P5 > P4; 100% agreement across presentation orders |
| Selected plan | P1, latent-space ridge regression; audit passed | P1, patient-mean features with logistic regression; audit passed |
| Common-protocol result | Hidden-set mean per-gene r 0.147; ceiling 0.193, recovering 76%; cluster mean 0.134, a gain of 0.013. Of 10,741 judgeable genes, 2,570 exceeded the cell-type baseline, or 24% | Clinical-only macro-F1 0.601; gain +0.097, 95% CI +0.063 to +0.132. Conclusion: improvement, agreeing with the agent. Site-held-out gain +0.094 |
| Rank among six | Second; P6 stacking was first, ahead by 0.003 | Second; P3 five-encoder gated fusion was first, ahead by approximately 0.0005; no meaningful separation established |
| Selected plan: self-report versus evaluation | 0.145 versus 0.147; difference −0.002 | Gain +0.090 versus +0.097; difference −0.007 |
| Self-report minus evaluation, all candidates | Differences between −0.004 and −0.002; no inflation | −0.007 to +0.018 |
| Rank correlation with evaluation ranking | Tournament 0.71; pilot-score ordering also 0.71 | Tournament 0.83; pilot-gain ordering 0.94 |
| Other findings | Of P1's 4,536 development-set gene claims, 1,787 replicated on hidden cells, or 39% | Five of six candidates had positive gains with intervals above zero; no complex plan statistically surpassed the simple reference |

**Departures from the design and execution qualifications**

- **Generation:** one plan per call, with eight calls per task and later calls seeing earlier plans, instead of generating nine at once. Generating several plans at the highest reasoning level exceeded Gemini's six-minute limit. The model and reasoning level were retained.
- **Computation:** formal analyses ran on CPU nodes. All 16 H100s were occupied, queues lasted about four hours, and CPU execution was feasible under the allocated limits. GPU execution was exercised only in preflight checks.
- **Audit fallback:** only next-ranked-plan fallback was implemented, not regeneration. Both champions in the final runs passed their audits immediately, so this fallback was not invoked there.
- **Task M ran twice:** the first audit rejected the champion because the full-run executor omitted the ceiling, permutation null, and two required floors. This was an executor defect rather than a plan defect. The executor was completed and the workflow rerun. Superseded records remain locally in `outputs/_superseded/`.
- **Task T's tournament was judged twice:** an evidence-card sentence initially suggested that a candidate's pretraining had not occurred. Correcting the card and rejudging left the ranking unchanged.
- **Time limits:** Task M allowed 25 minutes per pilot, 60 minutes for full validation, and 50 minutes for final fitting. Task T allowed 30 minutes per pilot, 75 minutes for a full run, and 35 minutes for site-held-out analysis. Task M P2 completed eight of ten declared validation splits; P6 completed two of nine and its final fit did not converge. Task T P4's agent-side full run completed 13 of 15 folds. The evaluator independently reran Task T without those limits and completed all 15 folds for every candidate. Task M evaluation scored each candidate's saved final predictions, so P6's evaluation score reflects its unconverged fit.
- **No permissively validated candidates:** all 12 plans declared strict validation, including nested selection and unsmoothed targets. Consequently, Benchmark 1 did not test the mechanism against candidates designed with weak validation that could inflate self-reported scores.
- **Alternative questions:** these stopped at planning. Three of six Task M plans and one Task T plan proposed unregistered components; these remain proposals, not implementations.
- **LLM usage:** 37 successful generator calls plus 50 attempts that reached the six-minute timeout and were retried, totaling about eight hours of waiting. The judge made 123 calls, including superseded rounds, consuming about 1.52 million input tokens. List-price-equivalent cost was $13.82; calls used subscription access, with no fallback judge invoked.

**Output locations**

| Content | Location |
|---|---|
| Dataset reports | `reports/generated_report_1.pdf` for PBMC; `reports/generated_report_2.pdf` for TCGA |
| Project report and slides | `reports/report.pdf`, three pages; `reports/presentation.pdf`, 11 pages |
| Per-task records | `outputs/task_m/` and `outputs/task_t/`: `shared_inputs`, `plans`, `pilot`, `tournament`, `full`, `audit`, `benchmark1`, `alt_question`, `report`, `costs.json`, and `index.json` |
| Evaluator data and scores | Local `bench_private/task_m/` and `bench_private/task_t/`, separate from agent inputs and excluded from GitHub |
| LLM answers and call log | `llm_cache/` and `logs/llm_calls.jsonl` |
| Submission archives | `submission/source_code.zip`, `submission/generated_outputs.zip`, and `submission/manifest.json` |
| Execution history | `docs/PROGRESS.md` |

**Reproduction**

- `make test`: 81 tests covering hidden-data separation, decision rules, generation and compliance, and model interfaces.
- `python scripts/replay_check.py`: rerun LLM stages from cache and compare with saved outputs. On November 25, all 14 stages matched without live calls; see `outputs/replay_check.json`.
- `make replay`: run the controller with cached LLM responses, skipping completed stages.
- `make reports` and `make package`: rebuild the project report, slides, and archives.

---

## 10. Risks and setup checks

**Risks considered in the design**

| Risk | Planned response |
|---|---|
| The selected plan is not best in Benchmark 1 | Report its rank and gap and inspect the verdict reasoning |
| LLM service interruption | Cache all responses. Retry Gemini after the six-minute limit without changing models; retry the judge and then use its fallback if necessary |
| Claude judging depends on local login | Cache responses; retain Hugging Face fallback; support an official SDK when a key is available |
| Insufficient candidate diversity | Enforce coverage and deduplication across three dimensions; quantify diversity |
| Complex models are undertrained | Require learning curves and convergence judgments; do not approve unconverged candidates at final audit |
| Complex models require scarce GPU time | Declare and cap budgets, reduce pilot size during congestion, and submit early |
| Complex models do not beat the simple reference | Report this as the answer to what additional complexity buys |
| Task T yields insufficient evidence | Accept that outcome; also report more sensitive secondary metrics and minimum detectable differences |
| Task M cannot beat cluster means | State that predictability is primarily at the cell-type level; report gene categories and within-cluster correlations |
| Only Task T's champion shows gain | Treat evidence as insufficient and explain selection bias |
| Deadline pressure | Reduce scope in the order given in Section 9.2 |

**Setup checks resolved by November 25, 2026**

- **Claude CLI location and noninteractive access:** available. The binary is inside the VS Code extension directory; `scripts/claude_cli.sh` resolves the version-dependent path dynamically.
- **Compute-node internet access:** available on both CPU and GPU nodes. LLM calls still originate from the login-node controller because the judge depends on local login.
- **Task M gene scope and null stability:** 11,282 in-scope genes, of which 10,741 were judgeable on hidden cells. The implementation estimates a permutation-derived standard deviation and uses a normal approximation. It did not conduct a permutation-count stability study.
- **Judge cost:** approximately $0.11 list-price equivalent per pairwise verdict after using a custom system prompt with tools disabled, with roughly 12,000 input tokens per call. Section 9.6 gives total usage. Calls used subscription access rather than additional per-call billing.
- **Slide format:** PDF generated with WeasyPrint.

---

## 11. Constraints [Confirmed]

- Treat the other project's raw TCGA directory as read-only: do not modify or move it, and do not adopt that project's cohort splits.
- Treat shared data, model-weight stores, and the first three project directories as read-only.
- The original project constraints specify CC-BY-NC-ND restrictions for the five pathology encoders and prohibit redistribution of weights and their derivatives. Submission archives exclude weights and raw data.
- Keep API keys in local credential files, never in code, logs, or reports.
- Do not send patient- or cell-level records to external services.
- Submit heavy computation through Slurm rather than running it on login nodes.
- Publish to GitHub only when requested; Canvas submission is separate. GitHub publication was explicitly requested and completed for this repository.

---

## 12. References

These are the original design's reading list, subject to the verification limitations in Section 3.

- [Accelerating scientific discovery with Co-Scientist (Nature 2026)](https://www.nature.com/articles/s41586-026-10644-y)
- Biomni (Science 2026, DOI 10.1126/science.adz4351)
- [Tournament of Prompts / DEEVO](https://arxiv.org/pdf/2506.00178)
- [MOOSE-Chem3](https://arxiv.org/pdf/2505.17873)
- [MLE-STAR](https://arxiv.org/pdf/2506.15692)
- [KompeteAI, including comparisons of AIDE, SELA, and MLE-STAR](https://arxiv.org/pdf/2508.10177)
- [Google ERA: AI-powered empirical software](https://www.research.google/blog/accelerating-scientific-discovery-with-ai-powered-empirical-software)
- [The Agentic Garden of Forking Paths](https://arxiv.org/abs/2607.01507)
- [Branchpoint: Structured agentic multiverse analysis](https://neurips.cc/virtual/2026/156553)
- [Silent failures in agentic data science](https://papers.cool/arxiv/2610.14115)
- [BaitBench](https://alphaxiv.org/abs/2608.30724)
- [Hidden pitfalls in AI scientist systems](https://arxiv.org/html/2509.08713v2)
- [Can We Trust LLM Judges for Evidence-based Research Agents?](https://arxiv.org/html/2605.19196)
- [Cross-paradigm audit of LLM judges for clinical text](https://www.nature.com/articles/s41746-026-03443-2)
- [Evaluating Agentic Bioinformatics through Function, Evidence, and Validation](https://arxiv.org/abs/2607.27556)

Code and reports from the previous projects are in `../Agent1_DR`, `../Agent2_Generative`, and `../Agent3_RepresentationLearning`.

---

## Appendix: evaluation designs set aside during planning

**Twelve tasks with synthetic labels.** An initial proposal retained real features but generated labels from hand-written formulas to create known answers and theoretical ceilings. It was rejected because:

- Generating labels from one modality does not make the information exclusive to it when modalities are correlated.
- It implicitly assumed that a particular model should win, without guaranteeing that outcome.
- Every task reduced to binary-label prediction, assessing predictors rather than analytical plan selection.
- Synthetic labels lacked biological meaning and did not engage domain knowledge.
- On clean tasks without analytical traps, a fixed nested comparison could answer the question without creating a meaningful role for competing plans.

Simulated data and controlled traps remain possible second and third layers of evidence for a paper.

**Deliberately inserted traps with a hidden test set.** Not included in the assignment. The retained lesson is that a trap must have observable clues or be stated in the objective, and must have a matched clean version.
