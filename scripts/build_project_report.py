#!/usr/bin/env python
"""Project report (at most 4 pages) and slides (8-minute talk) from saved results. README 9.3, 9.4.

    python scripts/build_project_report.py

Every number is read from files written by the pipeline (outputs/<task>/benchmark1/summary.json and the
records it points to). The surrounding prose is written by the builder of the system, not by an LLM;
sentences that depend on an outcome are chosen from the numbers, so the text cannot disagree with them.
"""
import html
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from agent4.io import read_json  # noqa: E402

OUT = ROOT / "reports"
FIG = OUT / "figures"


def load(task):
    d = ROOT / "outputs" / task
    return dict(bench=read_json(d / "benchmark1" / "summary.json"), ranking=read_json(d / "tournament" / "ranking.json"),
                final=read_json(d / "audit" / "final.json"), profile=read_json(d / "shared_inputs" / "profile.json"),
                objective=read_json(d / "shared_inputs" / "objective.json"), pool=read_json(d / "plans" / "pool.json"),
                costs=read_json(d / "costs.json"), plans={p: read_json(d / "plans" / f"{p}.json") for p in read_json(d / "plans" / "pool.json")["plan_ids"]},
                full={p.parent.name: read_json(p) for p in sorted((d / "full").glob("*/full_run.json"))},
                audit={p.stem: read_json(p) for p in sorted((d / "audit").glob("P*.json"))},
                figs=d / "report" / "figures")


def row(b, pid):
    return next(r for r in b["candidates"] if r["plan_id"] == pid)


def ordinal(n):
    return {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}.get(n, f"{n}th")


def architecture_figure(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(10.5, 3.5))
    ax.set_xlim(0, 105), ax.set_ylim(0, 35), ax.axis("off")
    steps = [("1 Profile and\ndiagnose", "code"), ("2 Compile\nobjective", "gen"), ("3 Generate\n6 plans", "gen"), ("4 Compliance", "code"), ("5 Pilot runs,\nevidence cards", "code"),
             ("6 Tournament", "judge"), ("7 Full run", "code"), ("8 Audit", "judge"), ("9 Report", "gen")]
    colour = dict(code="#d6eaf8", gen="#fdebd0", judge="#f5b7b1")
    for i, (label, kind) in enumerate(steps):
        x = 1 + i * 11.5
        ax.add_patch(FancyBboxPatch((x, 19), 10, 9, boxstyle="round,pad=0.2", fc=colour[kind], ec="#555", lw=0.8))
        ax.text(x + 5, 23.5, label, ha="center", va="center", fontsize=7.6)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 11.4, 23.5), xytext=(x + 10.3, 23.5), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("", xy=(75, 18.6), xytext=(98, 18.6), arrowprops=dict(arrowstyle="<-", lw=0.7, ls="--", connectionstyle="arc3,rad=-0.35"))
    ax.text(86.5, 12.8, "rejected: next-ranked plan", ha="center", fontsize=6.8, style="italic")
    ax.add_patch(FancyBboxPatch((1, 2), 60, 8, boxstyle="round,pad=0.2", fc="#e8f8f5", ec="#555", lw=0.8))
    ax.text(31, 6, "Evaluator (separate package): data split and read thinning, rulers, frozen rules,\nunified score of EVERY candidate after the agent has decided (benchmark 1)", ha="center", va="center", fontsize=7.6)
    ax.annotate("", xy=(6, 18.6), xytext=(6, 10.4), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(7, 14.2, "agent view\n(no hidden data)", fontsize=6.8, va="center")
    ax.annotate("", xy=(52, 10.4), xytext=(52, 18.6), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(53, 14.2, "frozen pool,\ncards, predictions", fontsize=6.8, va="center")
    for i, (kind, label) in enumerate((("code", "deterministic code"), ("gen", "generator: Gemini 4 Carbon"), ("judge", "judge: Claude Opus 5.5"))):
        ax.add_patch(FancyBboxPatch((66 + i * 0, 7.2 - i * 2.6), 2.2, 1.6, boxstyle="round,pad=0.1", fc=colour[kind], ec="#555", lw=0.6))
        ax.text(69.4, 8 - i * 2.6, label, fontsize=7, va="center")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


CSS_REPORT = """
@page { size: Letter; margin: 15mm 15mm 15mm 15mm; @bottom-center { content: counter(page); font-size: 8pt; color: #666; } }
body { font-family: 'DejaVu Serif', serif; font-size: 8.9pt; line-height: 1.33; color: #111; }
h1 { font-size: 14.5pt; margin: 0 0 2pt 0; } h2 { font-size: 10.4pt; margin: 8pt 0 2.5pt 0; } p { margin: 0 0 4pt 0; text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 3pt 0 5pt 0; font-size: 7.3pt; font-family: 'DejaVu Sans', sans-serif; }
th, td { border: 0.4pt solid #aaa; padding: 1.6pt 3pt; text-align: left; vertical-align: top; } th { background: #eef1f4; }
.fig { text-align: center; margin: 3pt 0 1pt 0; } .cap { font-size: 7.4pt; color: #333; margin-bottom: 4pt; } .small { font-size: 7.6pt; color: #333; }
.two { column-count: 2; column-gap: 14pt; } img { max-width: 100%; } .sel { background: #fdecea; } ul { margin: 1pt 0 4pt 13pt; padding: 0; } li { margin-bottom: 1.5pt; }
tr { break-inside: avoid; } .row { display: flex; gap: 12pt; align-items: flex-end; } .row > div { flex: 1; }
.project-links { font-family: 'DejaVu Sans', sans-serif; font-size: 8pt; line-height: 1.45; margin: 5pt 0 7pt; padding: 5pt 7pt; background: #eef4f8; border-left: 2pt solid #1b4f72; break-inside: avoid; }
.project-links a { color: #174a75; text-decoration: underline; }
"""

CSS_SLIDES = """
@page { size: 13.333in 7.5in; margin: 0.5in 0.6in 0.45in 0.6in; @bottom-right { content: counter(page); font-size: 10pt; color: #777; } }
body { font-family: 'DejaVu Sans', sans-serif; font-size: 15pt; line-height: 1.32; color: #111; }
.slide { page-break-after: always; } h1 { font-size: 26pt; margin: 0 0 10pt 0; color: #1b4f72; } h2 { font-size: 17pt; margin: 8pt 0 4pt 0; }
ul { margin: 4pt 0 6pt 22pt; padding: 0; } li { margin-bottom: 6pt; } table { border-collapse: collapse; font-size: 11.5pt; margin-top: 6pt; }
th, td { border: 0.6pt solid #999; padding: 3pt 7pt; text-align: left; } th { background: #eaf2f8; } .sel { background: #fdecea; } tr { break-inside: avoid; }
.cols { display: flex; gap: 22pt; } .col { flex: 1; } img { max-width: 100%; max-height: 4.6in; } .note { font-size: 11pt; color: #444; } .big { font-size: 19pt; }
.q { background: #f1f4f7; padding: 8pt 12pt; border-left: 5pt solid #2e86c1; font-size: 14pt; }
"""


def e(x):
    return html.escape(str(x))


def table(headers, rows, sel=None):
    out = ["<table><tr>" + "".join(f"<th>{e(h)}</th>" for h in headers) + "</tr>"]
    for r in rows:
        cls = " class='sel'" if sel is not None and r[0] == sel else ""
        out.append(f"<tr{cls}>" + "".join(f"<td>{e(x)}</td>" for x in r) + "</tr>")
    return "".join(out) + "</table>"


def img(path, cap=None, width="100%"):
    if not Path(path).exists():
        return ""
    return f"<div class='fig'><img src='file://{path}' style='width:{width}'></div>" + (f"<div class='cap'>{e(cap)}</div>" if cap else "")


def facts():
    m, t = load("task_m"), load("task_t")
    bm, bt = m["bench"], t["bench"]
    sm, st = row(bm, bm["selected_plan"]), row(bt, bt["selected_plan"])
    best_m, best_t = row(bm, bm["best_true_plan"]), row(bt, bt["best_true_plan"])
    infl_m = [r["inflation_full"] for r in bm["candidates"] if r["inflation_full"] is not None]
    infl_t = [r["inflation_full"] for r in bt["candidates"] if r["inflation_full"] is not None]
    infl_mp = [r["inflation_pilot"] for r in bm["candidates"] if r["inflation_pilot"] is not None]
    infl_tp = [r["inflation_pilot"] for r in bt["candidates"] if r["inflation_pilot"] is not None]
    llm = {role: {k: sum(x["costs"]["llm"][role][k] for x in (m, t)) for k in ("answered_calls", "failed_attempts", "thinking_tokens", "input_tokens", "list_price_usd", "wall_seconds")}
           for role in ("generator", "judge")}
    core_hours = sum((v.get(k) or {}).get("cpu_core_hours", 0) or 0 for x in (m, t) for v in x["costs"]["plans"].values() for k in ("pilot", "full"))
    return dict(m=m, t=t, bm=bm, bt=bt, sm=sm, st=st, best_m=best_m, best_t=best_t, llm=llm, core_hours=core_hours, short=short,
                infl_m=(float(np.mean(infl_m)), float(np.max(np.abs(infl_m)))), infl_t=(float(np.mean(infl_t)), float(np.max(np.abs(infl_t)))),
                infl_mp=(float(np.mean(infl_mp)), float(np.max(np.abs(infl_mp)))), infl_tp=(float(np.mean(infl_tp)), float(np.max(np.abs(infl_tp)))))


def short(name, n):
    """A plan name cut at a word boundary."""
    if n is None or len(name) <= n:
        return name
    return name[:n].rsplit(" ", 1)[0] + " ..."


def bench_table_m(bm, n=None):
    rows = [[r["plan_id"], short(r["name"], n), r["tournament_rank"], r["true_rank"], f"{r['self_score_pilot']:.3f}", f"{r['self_score_full']:.3f}", f"{r['true_score']:.3f}",
             f"{r['inflation_full']:+.3f}", f"{r['n_beyond_cell_type']:,}"] for r in sorted(bm["candidates"], key=lambda r: r["true_rank"])]
    return table(["id", "plan", "tourn.", "unified", "self r, pilot", "self r, full", "unified r", "self - unified", "genes beyond cell type"], rows, bm["selected_plan"])


def bench_table_t(bt, n=None):
    rows = [[r["plan_id"], short(r["name"], n), r["tournament_rank"], r["true_rank"], f"{r['self_gain_pilot']:+.3f}", f"{r['self_gain_full']:+.3f}", f"{r['true_gain']:+.3f}",
             f"{r['true_ci'][0]:+.3f} to {r['true_ci'][1]:+.3f}", str(r["true_conclusion"]).replace("_", " "), f"{r['inflation_full']:+.3f}"]
            for r in sorted(bt["candidates"], key=lambda r: r["true_rank"])]
    return table(["id", "plan", "tourn.", "unified", "self gain, pilot", "self gain, full", "unified gain", "95% interval", "conclusion", "self - unified"], rows, bt["selected_plan"])


def main():
    from weasyprint import HTML
    FIG.mkdir(parents=True, exist_ok=True)
    architecture_figure(FIG / "architecture.png")
    f = facts()
    sys.path.insert(0, str(ROOT / "scripts"))
    import project_text
    HTML(string=f"<html><head><meta charset='utf-8'><style>{CSS_REPORT}</style></head><body>{project_text.report(f, table, img, e, bench_table_m, bench_table_t, FIG, ordinal)}</body></html>").write_pdf(OUT / "report.pdf")
    HTML(string=f"<html><head><meta charset='utf-8'><style>{CSS_SLIDES}</style></head><body>{project_text.slides(f, table, img, e, bench_table_m, bench_table_t, FIG, ordinal)}</body></html>").write_pdf(OUT / "presentation.pdf")
    from pypdf import PdfReader
    for name in ("report.pdf", "presentation.pdf"):
        print(name, "pages:", len(PdfReader(OUT / name).pages))


if __name__ == "__main__":
    main()
