"""Sentence-window chunking. Articles here are short, so most become 1-2 chunks;
the function exists so the pipeline is the same shape it would be on long manuals."""
import re


def chunk_text(text: str, max_chars: int = 500, overlap_sentences: int = 1) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    for s in sentences:
        if current and len(" ".join(current)) + len(s) + 1 > max_chars:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
        current.append(s)
    if current:
        chunks.append(" ".join(current))
    return chunks
