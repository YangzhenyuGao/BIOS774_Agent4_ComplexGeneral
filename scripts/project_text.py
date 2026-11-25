"""Text of the project report and the slides. Numbers come from the facts dict built in build_project_report.py;
sentences that depend on an outcome are chosen from those numbers."""


def _num(x, signed=False):
    """Three decimals, or four when the value would otherwise print as zero."""
    digits = 3 if abs(x) >= 0.001 or x == 0 else 4
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def _sel_sentence(b, ordinal, what):
    n, rank = b["n_candidates_scored"], b["selected_true_rank"]
    if rank == 1:
        return f"Under the unified protocol the selected plan is the best of the {n} candidates ({what})."
    return (f"Under the unified protocol the selected plan ranks {ordinal(rank)} of {n} candidates; the best one is {b['best_true_plan']}, "
            f"ahead by {_num(b['selection_gap'])} ({what}).")


def _unfinished_note(m, plan_id):
    """A sentence about a full run that the time budget cut short, read from the run record; empty when it finished."""
    fr = m["full"].get(plan_id) or {}
    fit = fr.get("final_fit") or {}
    if fit.get("converged") is not False and fr.get("n_splits_run") == fr.get("n_splits_declared"):
        return ""
    parts = []
    if fit.get("stopped_by") == "time_budget":
        parts.append("its final fit was stopped by the time budget before it converged")
    if fr.get("n_splits_run") != fr.get("n_splits_declared"):
        parts.append(f"only {fr.get('n_splits_run')} of its {fr.get('n_splits_declared')} declared validation splits ran")
    return f" For {plan_id}, " + " and ".join(parts) + ", so its own validation evidence is incomplete." if parts else ""


def _difficulty_rows(profile, tags):
    by = {d["tag"]: d for d in profile["difficulties"]}
    return [[t.replace("_", " "), by[t]["severity"], by[t]["evidence"]] for t in tags if t in by]


def report(f, table, img, e, bench_table_m, bench_table_t, FIG, ordinal):
    m, t, bm, bt, sm, st = f["m"], f["t"], f["bm"], f["bt"], f["sm"], f["st"]
    rm, rkm, rkt = bm["rulers"], m["ranking"], t["ranking"]
    ref = bt.get("simple_reference") or {}
    n_amb_m = sum(r["complexity"] == "ambitious" for r in bm["candidates"])
    n_amb_t = sum(r["complexity"] == "ambitious" for r in bt["candidates"])
    best_complex_m = max((r for r in bm["candidates"] if r["complexity"] == "ambitious"), key=lambda r: r["true_score"])
    best_complex_t = max((r for r in bt["candidates"] if r["complexity"] == "ambitious"), key=lambda r: r["true_gain"])
    simple_m = next(r for r in bm["candidates"] if r["complexity"] == "simple_reference")
    simple_t = next(r for r in bt["candidates"] if r["complexity"] == "simple_reference")
    gl = sm.get("gene_list_check") or {}
    h = []
    h.append("<h1>Agent 4: matching analysis plans and conclusions to the scientific question and the evidence</h1>")
    repo = "https://github.com/YangzhenyuGao/BIOS774_Agent4_ComplexGeneral"
    h.append(f"<div class='project-links'><b>Code and generated reports</b><br>"
             f"<a href='{repo}'>{repo}</a><br>"
             f"<a href='{repo}/blob/main/reports/generated_report_1.pdf'>Study 1: PBMC Multiome report (ATAC to RNA)</a>"
             " &nbsp; | &nbsp; "
             f"<a href='{repo}/blob/main/reports/generated_report_2.pdf'>Study 2: TCGA-BRCA report (H&amp;E + clinical)</a></div>")
    h.append("<p class='small'>BIOS 774, Project 4 (Building an AI agent for complex biomedical data analysis). Datasets: 10x PBMC Multiome (RNA + ATAC) and TCGA-BRCA (H&amp;E + clinical). "
             "Every number below is read from files written by the pipeline; the two dataset reports give the full tables.</p>")
    h.append("<div class='two'>")
    h.append("<h2>1. Problem</h2>")
    h.append("<p>An agent that analyses data on its own has to choose among several feasible analyses. Choosing by the highest score is unsafe for two reasons. "
             "First, the same result supports different conclusions under different questions: receptor status is a legitimate baseline when the question is what images add to clinical information, "
             "and a forbidden shortcut when the question is what images alone contain. Second, each analysis reports its score under its own validation, and the looser the validation, the higher the score. "
             "Our earlier course projects failed in exactly these ways: selections decided by seed noise, complex models that were never trained to convergence, and requirements stated in the objective that nobody checked.</p>"
             "<p>Agent 4 therefore treats a <i>plan</i> as the unit of choice - inputs, model, validation design and the conclusion it would permit - and selects among plans with executed pilot evidence judged against the question. "
             "The machinery is aimed at questions with many feasible and unequal approaches; a question with one standard method does not need it.</p>")
    h.append("<h2>2. System</h2>")
    h.append("<p>One code path serves both datasets; nothing branches on a dataset name. Differences come from the measured data profile and from the question. "
             "Deterministic code measures, executes and checks. The generator (Gemini 4 Carbon) compiles the objective, designs plans and writes report prose. "
             "The judge (Claude Opus 5.5, a different vendor) decides tournament matches and reviews the final result. "
             "The evaluator is a separate package that owns the hidden data and scores every candidate after the agent has decided; a test fails if agent code imports it or names its directories.</p>")
    h.append("</div>")
    h.append(img(FIG / "architecture.png", "Figure 1. The nine steps of the agent and the separate evaluator. A rejected result goes back to the next-ranked plan."))
    h.append("<h2>3. How the agent identifies difficulties</h2>")
    h.append(f"<p>Step 1 runs quantitative diagnostics chosen by the kind of data (paired count modalities; a cohort with features and labels). Each returns a severity, the measured numbers and what an analysis must do about it "
             f"({len(m['profile']['difficulties'])} findings for PBMC, {len(t['profile']['difficulties'])} for TCGA). Examples, as written by the diagnostics:</p>")
    h.append(table(["data", "difficulty", "severity", "measured evidence"],
                   [["PBMC"] + r for r in _difficulty_rows(m["profile"], ["measurement_noise", "strong_baseline"])] +
                   [["TCGA"] + r for r in _difficulty_rows(t["profile"], ["technical_confounding", "statistical_power"])]))
    h.append("<div class='two'>")
    h.append("<h2>4. How it chooses a strategy</h2>")
    h.append(f"<p><b>Objective.</b> The generator turns the question into an estimand, allowed and forbidden inputs, required comparators and acceptance criteria tied to deterministic checks. "
             f"For the alternative TCGA question (images alone) the compiled objective forbids ER, PR and HER2; for the main question it requires them in both arms. "
             f"<b>Plans.</b> The generator designs candidates one per call at its highest reasoning level, sees what was already proposed, and must keep one simplest valid plan and make the rest ambitious. "
             f"Both pools hold six plans ({n_amb_m} and {n_amb_t} ambitious). <b>Compliance</b> checks structured fields: inputs come from the task view, split unit, scoring target, budget. "
             f"<b>Pilots.</b> Each plan runs at reduced scale under its own validation design and produces a numbered evidence card: its score with spread, every required comparator on the same units, "
             f"convergence, the facts of its validation design, and cost. <b>Tournament.</b> All pairs are judged twice, once per presentation order ({rkm['n_matches']} matches per task); "
             f"the judge must cite evidence ids, and Bradley-Terry strengths give the ranking. Pairs judged identically in both orders: {rkm['order_consistency']:.0%} (PBMC) and {rkt['order_consistency']:.0%} (TCGA). "
             f"<b>Audit.</b> The champion is run in full, each acceptance criterion is checked, and the judge reviews the summary; a rejected result is not reported.</p>")
    h.append("<p><b>Tools and models.</b> The registry has two tiers. Basic: LSI, gene activity, cluster-mean, ridge, PLS, k-NN; clinical encoding, mean-pooled features, logistic regression. "
             "Ambitious: residual-over-cluster-mean models, cis-regulatory masked networks and per-gene boosting, count-likelihood translation networks, MultiVI and PeakVI, contrastive pre-training, low-rank multi-task networks, stacking; "
             "attention, multi-head and transformer multiple-instance learning over tiles, multi-encoder fusion of five pathology foundation models, FiLM and cross-attention conditioning, a clinical-offset incremental model, "
             "hierarchical heads, self-supervised pre-adaptation, morphology prototypes, boosting. Plans are built from registered components; no code is generated at run time.</p>")
    h.append("</div>")

    h.append("<h2>5. Results</h2><div class='two'>")
    h.append(f"<p><b>PBMC Multiome - for which genes can ATAC predict RNA beyond cell type?</b> Rulers on {bm['n_scope']:,} in-scope genes and 3,124 held-out cells (mean per-gene r against an independent half of each cell's reads): "
             f"ceiling {rm['ceiling']:.3f}, cluster mean {rm['floor1_cluster_mean']:.3f}, ridge {rm['floor4_ridge']:.3f}, gene activity {rm['floor2_gene_activity_promoter_and_distal']:.3f}, depth only {rm['floor3_depth_only']:.3f}. "
             f"The agent selected {bm['selected_plan']} ({e(sm['name'])}). Its unified score is {sm['true_score']:.3f}, {sm['fraction_of_ceiling']:.0%} of the ceiling and {sm['mean_delta_vs_cluster_mean']:+.3f} above the cluster mean. "
             f"On held-out cells {sm['n_beyond_cell_type']:,} of {bm['n_judgeable']:,} judgeable genes are predicted beyond cell type at a 5% false discovery rate. "
             f"The plan itself had claimed {sm['n_claimed_by_plan']:,} genes from development cells"
             + (f"; {gl['n_confirmed']:,} of them ({gl['replication_rate']:.0%}) were confirmed on held-out cells." if gl.get("replication_rate") is not None else ".") +
             f" The best ambitious plan ({best_complex_m['plan_id']}) scores {best_complex_m['true_score']:.3f} against {simple_m['true_score']:.3f} for the simple reference. "
             f"{_sel_sentence(bm, ordinal, 'mean per-gene r')}{_unfinished_note(m, bm['best_true_plan'])}</p>")
    h.append(f"<p><b>TCGA-BRCA - do H&amp;E images add to clinical variables for PAM50?</b> The evaluator's tuned clinical-only model reaches macro-F1 {bt['clinical_floor_macro_f1']:.3f} in repeated nested cross-validation on 802 patients; "
             f"the simple-reference ruler gains {ref.get('gain', float('nan')):+.3f}. The agent selected {bt['selected_plan']} ({e(st['name'])}). Its unified gain is {st['true_gain']:+.3f} "
             f"(95% interval {st['true_ci'][0]:+.3f} to {st['true_ci'][1]:+.3f}); conclusion: <b>{str(st['true_conclusion']).replace('_', ' ')}</b>. "
             f"The agent's own full run gave {st['self_gain_full']:+.3f} and the conclusion '{str(st['self_conclusion']).replace('_', ' ')}'"
             f" ({'the two conclusions agree' if bt['conclusion_agreement']['agree'] else 'the two conclusions differ'}). "
             f"{bt['n_candidates_positive_gain']} of {bt['n_candidates_scored']} candidates have a positive unified gain and {bt['n_candidates_ci_above_zero']} have an interval above zero, "
             f"so the finding does not rest on the selected plan alone. The best ambitious plan ({best_complex_t['plan_id']}) gains {best_complex_t['true_gain']:+.4f} against {simple_t['true_gain']:+.4f} for the simple reference plan. "
             f"{_sel_sentence(bt, ordinal, 'macro-F1 gain')} The estimate for the selected plan is optimistic by construction (it was chosen among six on the same patients) and no negative control was run.</p>")
    h.append("</div>")
    h.append("<div class='row'><div>" + img(m["figs"] / "rulers.png", "Figure 2. PBMC: rulers, candidates and ceiling on held-out cells.") + "</div><div>" +
             img(t["figs"] / "forest.png", "Figure 3. TCGA: unified gain of every candidate with 95% intervals.") + "</div></div>")
    h.append("<h2 style='break-before: page'>6. Benchmark 1: every candidate run in full under the unified protocol</h2>")
    h.append(bench_table_m(bm))
    h.append(bench_table_t(bt))
    h.append(f"<p class='small'>tourn. = tournament rank; unified = rank by the evaluator's score; shaded = selected plan. Self-reported minus unified score, mean over candidates: "
             f"{f['infl_m'][0]:+.3f} (PBMC, full runs; {f['infl_mp'][0]:+.3f} for pilots) and {f['infl_t'][0]:+.3f} (TCGA, full runs; {f['infl_tp'][0]:+.3f} for pilots); "
             f"largest absolute difference {f['infl_m'][1]:.3f} and {f['infl_t'][1]:.3f}. "
             f"Rank correlation with the unified ranking: tournament {bm['spearman_tournament_vs_true']:.2f} (PBMC) and {bt['spearman_tournament_vs_true']:.2f} (TCGA); "
             f"ordering by pilot score alone {bm['spearman_pilot_score_vs_true']:.2f} and {bt['spearman_pilot_gain_vs_true']:.2f}"
             + (", so on these two pools the tournament did not rank better than the pilot score." if bm['spearman_tournament_vs_true'] <= bm['spearman_pilot_score_vs_true'] + 1e-9
                and bt['spearman_tournament_vs_true'] <= bt['spearman_pilot_gain_vs_true'] + 1e-9 else ".") + "</p>")
    g, j = f["llm"]["generator"], f["llm"]["judge"]
    h.append(f"<p class='small'>Cost of the whole run, both tasks: generator {g['answered_calls']} answered calls ({g['failed_attempts']} attempts timed out at the 6-minute limit and were re-sent to the same model; "
             f"{g['thinking_tokens']:,} reasoning tokens); judge {j['answered_calls']} calls, {j['input_tokens']:,} input tokens; {f['core_hours']:,.0f} CPU core-hours for the agent's pilots and full runs (the evaluator's unified runs not counted), no GPU.</p>")
    h.append("<div class='two'><h2>7. Strengths</h2><ul>"
             "<li>The question drives the analysis: allowed inputs, comparators and the permitted conclusion change when the question changes, without code changes.</li>"
             "<li>Selection uses executed evidence with the validation design on the card, not plan text and not the score alone; every verdict cites evidence ids and is stored.</li>"
             "<li>The audit caught a real gap: in the first PBMC pass it rejected the champion because the noise ceiling, the permutation null and two required floors were missing from the run, which was an executor fault. The executor was completed and the task rerun.</li>"
             "<li>Negative and modest answers are reported as such; all candidates are shown, not only the winner.</li>"
             "<li>Hidden data are isolated by construction and by test; all LLM answers are cached, so the LLM stages replay without keys.</li></ul>"
             "<h2>8. Limitations</h2><ul>"
             "<li>Two tasks and six candidates each: benchmark 1 says where the selected plan ranks here; it cannot show that the tournament is better than simpler selection rules. That comparison (benchmark 2) is prepared but not run.</li>"
             "<li>All twelve generated plans declared strict validation (nested selection, unsmoothed targets), so the pool contains no lax-validation plan and self-reported scores differ from unified ones mainly through scale and noise.</li>"
             "<li>Pilots are small and reduce what a complex plan can show; the judge counted reduced seeds, trimmed grids and unimplemented plan fields against complex plans.</li>"
             "<li>TCGA: selection and evaluation share the same 802 patients, the champion's gain is optimistic, and the shuffled-image negative control was left out for compute reasons.</li>"
             "<li>PBMC: one donor, random cell split; the ceiling is an empirical model; 'beyond cell type' depends on the strength of the cluster baseline.</li>"
             "<li>Plans are compositions of registered components; several free-text plan fields had no executable counterpart. All computation ran on CPU nodes because the GPUs were fully booked; GPU code was exercised only in a setup check.</li></ul></div>")
    return "".join(h)


def slides(f, table, img, e, bench_table_m, bench_table_t, FIG, ordinal):
    m, t, bm, bt, sm, st = f["m"], f["t"], f["bm"], f["bt"], f["sm"], f["st"]
    rm, rkm, rkt = bm["rulers"], m["ranking"], t["ranking"]
    ref = bt.get("simple_reference") or {}
    gl = sm.get("gene_list_check") or {}
    S = []

    def slide(title, body):
        S.append(f"<div class='slide'><h1>{e(title)}</h1>{body}</div>")

    slide("Agent 4: matching analysis plans and conclusions to the question and the evidence",
          "<p class='big'>BIOS 774 Project 4 - an AI agent for complex biomedical data analysis</p>"
          "<ul><li>Two multimodal datasets: PBMC Multiome (RNA + ATAC), TCGA-BRCA (H&amp;E + clinical)</li>"
          "<li>One code path, no dataset-specific branches</li><li>Generator: Gemini 4 Carbon. Judge: Claude Opus 5.5. Everything else: deterministic code</li></ul>"
          "<p class='note'>All numbers in these slides are filled from saved result files.</p>")
    slide("Goal: choose a valid analysis, not the highest score",
          "<ul><li><b>The same result supports different conclusions under different questions.</b> Receptors are a required baseline for 'what do images add?', a forbidden shortcut for 'what do images alone contain?'</li>"
          "<li><b>Self-reported scores depend on the plan's own validation.</b> Seed noise, unconverged models and unchecked requirements sank parts of our earlier projects</li></ul>"
          "<div class='cols'><div class='col'><p class='q'><b>Task M (PBMC).</b> For held-out cells with ATAC only, for which genes can RNA variation be predicted reliably?<br><span class='note'>Ruler: a noise ceiling above, a cell-type baseline below. 'Reliably' = beyond cell type.</span></p></div>"
          "<div class='col'><p class='q'><b>Task T (TCGA).</b> Given clinical variables incl. ER/PR/HER2, do H&amp;E images improve PAM50 prediction? How much, how certain?<br><span class='note'>Ruler: a tuned clinical-only model; the answer is a gain with a 95% interval.</span></p></div></div>")
    slide("System", img(FIG / "architecture.png") +
          "<ul><li>The unit of choice is a whole plan: inputs, model, validation design, permitted conclusion</li>"
          "<li>What is different from score-driven search or text-only debate: plans are executed at small scale and judged on evidence cards against the question</li>"
          "<li>The evaluator is a separate package with the hidden data; it scores every candidate after the agent has decided</li></ul>")
    dm = {d["tag"]: d for d in m["profile"]["difficulties"]}
    dt = {d["tag"]: d for d in t["profile"]["difficulties"]}
    slide("Step 1-2: measured difficulties drive the objective",
          "<div class='cols'><div class='col'><h2>PBMC Multiome</h2><ul>" +
          "".join(f"<li>{e(dm[k]['evidence'])}</li>" for k in ("measurement_noise", "strong_baseline") if k in dm) + "</ul></div>"
          "<div class='col'><h2>TCGA-BRCA</h2><ul>" + "".join(f"<li>{e(dt[k]['evidence'])}</li>" for k in ("technical_confounding", "statistical_power") if k in dt) + "</ul></div></div>"
          "<p class='note'>The compiled objective lists the estimand, allowed and forbidden inputs, required comparators and acceptance criteria tied to deterministic checks. Change the question and these change.</p>")
    slide("Step 3-5: six plans per task, mostly ambitious, each with pilot evidence",
          "<div class='cols'><div class='col'>" + table(["id", "PBMC plan", "core component"], [[p, f["short"](pl["name"], 48), pl["model"].get("component")] for p, pl in m["plans"].items()], bm["selected_plan"]) + "</div>"
          "<div class='col'>" + table(["id", "TCGA plan", "core component"], [[p, f["short"](pl["name"], 48), pl["model"].get("component")] for p, pl in t["plans"].items()], bt["selected_plan"]) + "</div></div>"
          "<p class='note'>One plan per generator call at the highest reasoning level; P1 is the single simple reference. Evidence card: score with spread, every required comparator on the same units, convergence, validation-design facts, cost. Shaded = selected.</p>")
    slide("Step 6-8: evidence-judged tournament, then audit",
          "<div class='cols'><div class='col'>" + img(m["figs"] / "tournament.png") + "<p class='note'>PBMC</p></div><div class='col'>" + img(t["figs"] / "tournament.png") + "<p class='note'>TCGA</p></div></div>"
          f"<ul><li>{rkm['n_matches']} matches per task, each pair in both orders; same winner in both orders for {rkm['order_consistency']:.0%} (PBMC) and {rkt['order_consistency']:.0%} (TCGA) of pairs</li>"
          "<li>In both tasks the judge ranked the simple reference first: in the pilots no ambitious plan had shown a gain over it with an interval above zero</li>"
          "<li>Audit, first PBMC pass: champion rejected - noise ceiling, permutation null and two floors were missing from the run. An executor fault; fixed and rerun</li></ul>")
    share = sm["n_beyond_cell_type"] / max(bm["n_judgeable"], 1)
    slide(f"Result, task M: a small gain beyond cell type, detectable for {share:.0%} of genes",
          "<div class='cols'><div class='col'>" + img(m["figs"] / "rulers.png") + "</div><div class='col'>" + img(m["figs"] / "genes.png") + "</div></div>"
          f"<ul><li>Selected {bm['selected_plan']}: mean per-gene r {sm['true_score']:.3f} on held-out cells; cluster mean {rm['floor1_cluster_mean']:.3f}; ceiling {rm['ceiling']:.3f} ({sm['fraction_of_ceiling']:.0%} of the ceiling)</li>"
          f"<li>{sm['n_beyond_cell_type']:,} of {bm['n_judgeable']:,} judgeable genes predicted beyond cell type (FDR 5%)" +
          (f"; the plan had claimed {sm['n_claimed_by_plan']:,} from development cells, {gl['replication_rate']:.0%} of those confirmed" if gl.get("replication_rate") is not None else "") + "</li></ul>")
    slide("Result, task T: what images add to clinical variables",
          "<div class='cols'><div class='col'>" + img(t["figs"] / "forest.png") + "</div><div class='col'>" + img(t["figs"] / "recall.png") + "</div></div>"
          f"<ul><li>Clinical only: macro-F1 {bt['clinical_floor_macro_f1']:.3f}. Selected {bt['selected_plan']}: gain {st['true_gain']:+.3f} (95% interval {st['true_ci'][0]:+.3f} to {st['true_ci'][1]:+.3f}) - {str(st['true_conclusion']).replace('_', ' ')}</li>"
          f"<li>{bt['n_candidates_positive_gain']} of {bt['n_candidates_scored']} candidates gain; {bt['n_candidates_ci_above_zero']} have an interval above zero. Simple-reference ruler: {ref.get('gain', float('nan')):+.3f}</li>"
          "<li>The champion's estimate is optimistic (chosen among six on the same patients); no shuffled-image control was run</li></ul>")
    slide("Benchmark 1, task M: every candidate run in full and scored by the evaluator",
          bench_table_m(bm, 44) +
          f"<ul><li>{_sel_sentence(bm, ordinal, 'mean per-gene r')}{_unfinished_note(m, bm['best_true_plan'])}</li>"
          f"<li>Self-reported minus unified score: {f['infl_m'][0]:+.3f} on average for full runs, largest {f['infl_m'][1]:.3f}</li>"
          f"<li>Rank correlation with the unified ranking: tournament {bm['spearman_tournament_vs_true']:.2f}, pilot score alone {bm['spearman_pilot_score_vs_true']:.2f}</li></ul>")
    slide("Benchmark 1, task T: every candidate run in full and scored by the evaluator",
          bench_table_t(bt, 40) +
          f"<ul><li>{_sel_sentence(bt, ordinal, 'macro-F1 gain')}</li>"
          f"<li>Self-reported minus unified gain: {f['infl_t'][0]:+.3f} on average for full runs, largest {f['infl_t'][1]:.3f}; agent and evaluator "
          f"{'reach the same conclusion' if bt['conclusion_agreement']['agree'] else 'reach different conclusions'}</li>"
          f"<li>Rank correlation with the unified ranking: tournament {bt['spearman_tournament_vs_true']:.2f}, pilot gain alone {bt['spearman_pilot_gain_vs_true']:.2f}</li></ul>")
    slide("Benchmark 2 (to do), limits, next",
          "<ul><li><b>Benchmark 2:</b> eight selection methods on the same frozen pool, constraints and evidence (random, text-only, top pilot score, rules + top score, debate without evidence, all cards at once, tournament, oracle). Everything needed is saved; only LLM calls are required</li>"
          "<li><b>Limits:</b> two tasks, six candidates each; all generated plans chose strict validation, so the pool has no lax-validation plan; pilots handicap complex plans; TCGA champion estimate optimistic and no negative control; one PBMC donor; CPU only (GPUs fully booked)</li>"
          "<li><b>Next:</b> negative controls, trap tasks with clean twins, external cohorts and new donors, judge reliability across models, multi-round evolution of plans</li></ul>")
    return "".join(S)
