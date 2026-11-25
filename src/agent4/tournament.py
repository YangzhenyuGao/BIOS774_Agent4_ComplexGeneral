"""Step 6 of README 6.1: evidence-judged round-robin tournament with Bradley-Terry ranking.

Every unordered pair of plans is judged twice, once in each presentation order. The judge must cite
evidence ids; citations are checked against the two cards. All inputs and verdicts are stored.
"""
from __future__ import annotations

import itertools
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .io import ROOT, now, write_json
from .llm import client

PLAN_FIELDS = ("name", "complexity", "analysis_unit", "inputs", "preprocessing", "model", "fusion", "training", "validation",
               "metrics", "allowed_conclusions", "budget")


def shared_block(question, objective, difficulties, constraints) -> str:
    obj = {k: objective[k] for k in ("task_family", "analysis_unit", "estimand", "target", "allowed_inputs", "forbidden_inputs",
                                     "required_comparators", "primary_metric", "acceptance_criteria", "conclusion_options", "conclusion_scope")}
    return ("SCIENTIFIC QUESTION:\n" + question.strip() + "\n\nCOMPILED OBJECTIVE:\n" + json.dumps(obj, indent=1) +
            "\n\nMEASURED DIFFICULTIES OF THE DATA:\n" + "\n".join(f"- {d['id']} [{d['tag']}, {d['severity']}] {d['evidence']}" for d in difficulties) +
            "\n\nTASK CONSTRAINTS:\n" + json.dumps({k: constraints[k] for k in ("decision_rule", "how_predictions_are_scored", "scope_of_inference", "limits")}, indent=1))


def plan_block(label, plan, card) -> str:
    design = {k: plan.get(k) for k in PLAN_FIELDS if plan.get(k) not in (None, [], {}, "")}
    return (f"=== {label.upper()} PLAN (id {plan['plan_id']}) ===\nDESIGN:\n{json.dumps(design, indent=1)}\n\nSUMMARY:\n{plan.get('summary', '')}\n\n"
            f"EVIDENCE CARD ({card['scale']} run):\n" + "\n".join(f"[{it['id']}] ({it['kind']}) {it['text']}" for it in card["items"]))


def _judge(shared, first, second, cards, tag):
    system = (ROOT / "templates" / "prompts" / "judge_system.txt").read_text()
    prompt = shared + "\n\n" + plan_block("first", first, cards[first["plan_id"]]) + "\n\n" + plan_block("second", second, cards[second["plan_id"]])
    valid_ids = {it["id"] for p in (first, second) for it in cards[p["plan_id"]]["items"]}

    def validate(parsed):
        if not isinstance(parsed, dict) or parsed.get("winner") not in ("first", "second", "tie"):
            raise ValueError("verdict needs winner first | second | tie")
        cited = parsed.get("cited_evidence") or []
        if not cited:
            raise ValueError("verdict cites no evidence")
        if not set(cited) & valid_ids:
            raise ValueError("none of the cited evidence ids exist on the two cards")

    rec = client.call("judge", system, prompt, tag=tag, validate=validate)
    v = rec["json"]
    cited = list(v.get("cited_evidence") or [])
    return dict(first=first["plan_id"], second=second["plan_id"], winner_position=v["winner"],
                winner=None if v["winner"] == "tie" else (first if v["winner"] == "first" else second)["plan_id"],
                cited_evidence=cited, unknown_citations=sorted(set(cited) - valid_ids), reason=v.get("reason", ""),
                validity_concerns=v.get("validity_concerns"), model=rec["model"], usage=rec["usage"], seconds=rec["seconds"],
                attempts=rec["attempts"], cached=rec["cached"], cache_key=rec["key"], system=system, prompt=prompt, raw_reply=rec["text"])


def bradley_terry(ids, wins, n_iter=500, prior=0.5):
    """Bradley-Terry strengths by minorisation-maximisation. `wins[i][j]` = times i beat j (a tie counts 0.5 each).
    A small symmetric prior keeps strengths finite when a plan wins or loses everything."""
    w = np.asarray(wins, dtype=float) + prior * (1 - np.eye(len(ids)))
    s = np.ones(len(ids))
    for _ in range(n_iter):
        total = w + w.T
        denom = (total / (s[:, None] + s[None, :])).sum(1)
        new = w.sum(1) / denom
        new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(new - s)) < 1e-10:
            s = new
            break
        s = new
    return dict(zip(ids, s.tolist()))


def run(question, objective, difficulties, constraints, plans, cards, out_dir, tag, workers=4):
    """plans: list of plan dicts that passed compliance; cards: {plan_id: evidence card dict}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shared = shared_block(question, objective, difficulties, constraints)
    ids = [p["plan_id"] for p in plans]
    by_id = {p["plan_id"]: p for p in plans}
    jobs = [(a, b) for a, b in itertools.permutations(ids, 2)]

    def one(pair):
        a, b = pair
        return _judge(shared, by_id[a], by_id[b], cards, tag=f"{tag}-{a}-vs-{b}")

    with ThreadPoolExecutor(workers) as pool:
        matches = list(pool.map(one, jobs))
    for m in matches:
        write_json(out_dir / "matches" / f"{m['first']}_vs_{m['second']}.json", m)
    idx = {p: i for i, p in enumerate(ids)}
    wins = np.zeros((len(ids), len(ids)))
    for m in matches:
        i, j = idx[m["first"]], idx[m["second"]]
        if m["winner"] is None:
            wins[i, j] += 0.5
            wins[j, i] += 0.5
        else:
            w = idx[m["winner"]]
            wins[w, j if w == i else i] += 1.0
    strength = bradley_terry(ids, wins)
    pair_outcome = {}
    for a, b in itertools.combinations(ids, 2):
        ab = next(m for m in matches if m["first"] == a and m["second"] == b)
        ba = next(m for m in matches if m["first"] == b and m["second"] == a)
        pair_outcome[f"{a}|{b}"] = dict(order_ab=ab["winner"] or "tie", order_ba=ba["winner"] or "tie", consistent=ab["winner"] == ba["winner"])
    points = {p: float(wins[idx[p]].sum()) for p in ids}
    ranking = sorted(ids, key=lambda p: (-strength[p], -points[p], float(by_id[p].get("budget", {}).get("full_minutes") or 0)))
    position_first_wins = sum(m["winner_position"] == "first" for m in matches)
    result = dict(
        finished=now(), plans=ids, ranking=ranking, champion=ranking[0], bradley_terry_strength=strength, points=points,
        wins_matrix=dict(rows=ids, values=wins.tolist()), n_matches=len(matches),
        order_consistency=float(np.mean([v["consistent"] for v in pair_outcome.values()])), pair_outcomes=pair_outcome,
        position_bias=dict(first_wins=position_first_wins, second_wins=sum(m["winner_position"] == "second" for m in matches),
                           ties=sum(m["winner_position"] == "tie" for m in matches)),
        judge_models=sorted({m["model"] for m in matches}),
        cost=dict(calls=len(matches), input_tokens=sum((m["usage"].get("input_tokens") or 0) for m in matches),
                  output_tokens=sum((m["usage"].get("output_tokens") or 0) for m in matches),
                  list_price_usd=sum((m["usage"].get("list_price_usd") or 0) for m in matches),
                  seconds=sum(m["seconds"] for m in matches)),
        unknown_citations=sum(len(m["unknown_citations"]) for m in matches))
    write_json(out_dir / "ranking.json", result)
    return result
