"""One embedding model for both ingest and query. bge-small: 384 dims, runs on CPU via ONNX."""
from functools import lru_cache

from fastembed import TextEmbedding

MODEL = "BAAI/bge-small-en-v1.5"
# bge models are trained with an instruction prefix on the *query* side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    return TextEmbedding(MODEL)


def embed_passages(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _model().embed(texts, batch_size=32)]


def embed_query(text: str) -> list[float]:
    return list(_model().embed([QUERY_PREFIX + text]))[0].tolist()


def to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
