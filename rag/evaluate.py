"""Run every retriever over eval/questions.jsonl and report Recall@1/3/5 and MRR@5,
overall and per question kind. Writes eval/results.md and prints the same table."""
import json
from collections import defaultdict
from pathlib import Path

from .db import connect
from .search import RETRIEVERS

QUESTIONS = Path(__file__).resolve().parent.parent / "eval" / "questions.jsonl"
RESULTS = Path(__file__).resolve().parent.parent / "eval" / "results.md"
KS = (1, 3, 5)


def score_ranking(ranked_ids: list[str], gold: set[str]) -> dict:
    out = {}
    for k in KS:
        out[f"recall@{k}"] = 1.0 if gold & set(ranked_ids[:k]) else 0.0
    rr = 0.0
    for i, d in enumerate(ranked_ids[:5], start=1):
        if d in gold:
            rr = 1.0 / i
            break
    out["mrr@5"] = rr
    return out


def main() -> None:
    questions = [json.loads(l) for l in QUESTIONS.open(encoding="utf-8")]
    per_kind: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    misses: list[str] = []
    with connect() as conn:
        for q in questions:
            gold = set(q["gold"])
            for name, fn in RETRIEVERS.items():
                ranked = [h.doc_id for h in fn(conn, q["question"], 5)]
                s = score_ranking(ranked, gold)
                per_kind[name]["all"].append(s)
                per_kind[name][q["kind"]].append(s)
                if s["recall@5"] == 0.0:
                    misses.append(f"| {name} | {q['qid']} | {q['question']} | {', '.join(gold)} | {', '.join(ranked[:3])} |")

    def avg(rows: list[dict], key: str) -> float:
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    kinds = ["all"] + sorted({q["kind"] for q in questions})
    lines = [f"# Retrieval eval — {len(questions)} questions, {len(RETRIEVERS)} retrievers", ""]
    for kind in kinds:
        n = len(per_kind["vector"][kind])
        lines += [f"## {kind} (n={n})", "", "| retriever | R@1 | R@3 | R@5 | MRR@5 |", "|---|---|---|---|---|"]
        for name in RETRIEVERS:
            rows = per_kind[name][kind]
            lines.append(
                f"| {name} | {avg(rows,'recall@1'):.2f} | {avg(rows,'recall@3'):.2f} | {avg(rows,'recall@5'):.2f} | {avg(rows,'mrr@5'):.2f} |"
            )
        lines.append("")
    lines += ["## Misses (gold not in top 5)", "", "| retriever | qid | question | gold | top-3 returned |", "|---|---|---|---|---|"]
    lines += misses or ["| — | — | none | — | — |"]
    text = "\n".join(lines) + "\n"
    RESULTS.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
