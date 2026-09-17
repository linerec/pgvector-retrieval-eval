"""Load corpus/docs.jsonl -> chunk -> embed -> insert into Postgres (pgvector)."""
import json
import sys
import time
from pathlib import Path

from .chunk import chunk_text
from .db import SCHEMA, connect
from .embed import embed_passages, to_pgvector

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "docs.jsonl"


def main() -> None:
    docs = [json.loads(line) for line in CORPUS.open(encoding="utf-8")]
    rows = []
    for d in docs:
        for ix, chunk in enumerate(chunk_text(d["text"])):
            rows.append((d["doc_id"], d["title"], ix, chunk))
    t0 = time.time()
    vectors = embed_passages([f"{title}. {content}" for _, title, _, content in rows])
    t_embed = time.time() - t0

    with connect() as conn:
        conn.execute(SCHEMA)
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (doc_id, title, chunk_ix, content, embedding) VALUES (%s, %s, %s, %s, %s::vector)",
                [(r[0], r[1], r[2], r[3], to_pgvector(v)) for r, v in zip(rows, vectors)],
            )
        n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print(f"ingested {len(docs)} docs -> {n} chunks; embedding took {t_embed:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
