"""Three retrievers over the same table, so they can be compared on the same eval set.

  vector : cosine similarity on pgvector (HNSW index)
  fulltext: Postgres tsvector, query lexemes OR-ed (to_tsquery), ranked by ts_rank_cd
  hybrid : Reciprocal Rank Fusion of the two lists (k=60)

Results are de-duplicated to one hit per document (best chunk wins) because the
eval set is labelled at the document level.
"""
from dataclasses import dataclass

from .db import connect
from .embed import embed_query, to_pgvector


@dataclass
class Hit:
    doc_id: str
    title: str
    content: str
    score: float


def _dedupe(rows) -> list[Hit]:
    seen: set[str] = set()
    out: list[Hit] = []
    for doc_id, title, content, score in rows:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(Hit(doc_id, title, content, float(score)))
    return out


def vector_search(conn, query: str, k: int = 10) -> list[Hit]:
    q = to_pgvector(embed_query(query))
    rows = conn.execute(
        "SELECT doc_id, title, content, 1 - (embedding <=> %s::vector) AS score "
        "FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
        (q, q, k * 3),
    ).fetchall()
    return _dedupe(rows)[:k]


def fulltext_search(conn, query: str, k: int = 10) -> list[Hit]:
    # OR the query lexemes together. websearch_to_tsquery ANDs them, which on a
    # symptom-style query ("fridge light on but warm inside") matches nothing.
    rows = conn.execute(
        "WITH q AS (SELECT to_tsquery('english', string_agg(lexeme, ' | ')) AS tq "
        "           FROM unnest(to_tsvector('english', %s))) "
        "SELECT doc_id, title, content, ts_rank_cd(tsv, q.tq) AS score "
        "FROM chunks, q WHERE q.tq IS NOT NULL AND tsv @@ q.tq "
        "ORDER BY score DESC LIMIT %s",
        (query, k * 3),
    ).fetchall()
    return _dedupe(rows)[:k]


def rrf(lists: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank). Rank-based, so
    the cosine scale and the ts_rank scale never have to be reconciled."""
    fused: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for hits in lists:
        for rank, h in enumerate(hits, start=1):
            fused[h.doc_id] = fused.get(h.doc_id, 0.0) + 1.0 / (k + rank)
            best.setdefault(h.doc_id, h)
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [Hit(d, best[d].title, best[d].content, s) for d, s in ordered]


def hybrid_search(conn, query: str, k: int = 10) -> list[Hit]:
    return rrf([vector_search(conn, query, k), fulltext_search(conn, query, k)])[:k]


RETRIEVERS = {"vector": vector_search, "fulltext": fulltext_search, "hybrid": hybrid_search}


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) or "fridge light on but warm inside"
    with connect() as conn:
        for name, fn in RETRIEVERS.items():
            print(f"\n== {name}: {query!r}")
            for h in fn(conn, query, 5):
                print(f"  {h.score:.4f}  {h.doc_id}  {h.title}")
