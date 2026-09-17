from rag.chunk import chunk_text
from rag.search import Hit, rrf


def test_chunk_respects_max_chars_and_overlaps():
    text = " ".join(f"Sentence number {i} is here." for i in range(20))
    chunks = chunk_text(text, max_chars=120, overlap_sentences=1)
    assert len(chunks) > 1
    assert all(len(c) <= 120 + 30 for c in chunks)
    # overlap: last sentence of chunk i is first sentence of chunk i+1
    for a, b in zip(chunks, chunks[1:]):
        assert a.split(". ")[-1].rstrip(".") in b


def test_chunk_short_text_is_single_chunk():
    assert chunk_text("One sentence only.") == ["One sentence only."]


def test_rrf_prefers_docs_present_in_both_lists():
    a = [Hit("A", "", "", 0.9), Hit("B", "", "", 0.8), Hit("C", "", "", 0.7)]
    b = [Hit("C", "", "", 3.0), Hit("D", "", "", 2.0)]
    fused = rrf([a, b])
    assert fused[0].doc_id == "C"          # rank 3 + rank 1 beats rank 1 alone
    assert [h.doc_id for h in fused] == ["C", "A", "B", "D"]


def test_rrf_is_deterministic_on_ties():
    a = [Hit("X", "", "", 1.0)]
    b = [Hit("Y", "", "", 1.0)]
    assert [h.doc_id for h in rrf([a, b])] == ["X", "Y"]
