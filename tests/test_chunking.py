import pytest

from src.schemas import DocumentResult, PageText
from src.text_chunker import chunk_document, chunk_page_text, split_sentences


def _doc(pages: list[str]) -> DocumentResult:
    return DocumentResult(
        doc_id="a" * 64,
        doc_name="paper.pdf",
        pages=[PageText(i + 1, t) for i, t in enumerate(pages)],
        page_count=len(pages),
    )


LONG = " ".join(f"Sentence {i} describes gene expression in sample {i}." for i in range(60))


def test_sentence_splitter_keeps_abbreviations():
    s = split_sentences("Smith et al. 2020 used RNA-seq. See Fig. 2 for details. Done.")
    assert s == ["Smith et al. 2020 used RNA-seq.", "See Fig. 2 for details.", "Done."]


def test_sentence_splitter_respects_paragraphs():
    assert split_sentences("Results in Fig.\n\nNext paragraph.") == ["Results in Fig.", "Next paragraph."]


def test_chunks_respect_size_limit():
    chunks = chunk_page_text(LONG, chunk_size=300, overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)


def test_chunks_overlap():
    chunks = chunk_page_text(LONG, chunk_size=300, overlap=120)
    for prev, nxt in zip(chunks, chunks[1:]):
        last_sentence = prev.rsplit(". ", 1)[-1]
        assert last_sentence in nxt  # trailing sentence repeated at the start of the next chunk


def test_zero_overlap_has_no_repeated_sentences():
    chunks = chunk_page_text(LONG, chunk_size=300, overlap=0)
    sentences = [s for c in chunks for s in split_sentences(c)]
    assert len(sentences) == len(set(sentences)) == 60


def test_all_text_is_covered():
    chunks = chunk_page_text(LONG, chunk_size=250, overlap=50)
    joined = " ".join(chunks)
    for i in range(60):
        assert f"sample {i}." in joined


def test_very_long_sentence_is_split():
    sentence = "word " * 400
    chunks = chunk_page_text(sentence.strip(), chunk_size=200, overlap=20)
    assert all(len(c) <= 200 for c in chunks)
    assert sum(c.count("word") for c in chunks) >= 400


def test_invalid_parameters():
    with pytest.raises(ValueError):
        chunk_page_text("text", chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_page_text("text", chunk_size=0, overlap=0)


def test_chunks_never_cross_pages_and_keep_metadata():
    doc = _doc(["Page one text about methods. " * 20, "Page two text about results. " * 20])
    chunks = chunk_document(doc, chunk_size=200, overlap=40)
    assert {c.page_number for c in chunks} == {1, 2}
    for c in chunks:
        assert c.doc_name == "paper.pdf" and c.doc_id == "a" * 64
        if c.page_number == 1:
            assert "results" not in c.text
        else:
            assert "methods" not in c.text
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_empty_pages_produce_no_chunks():
    assert chunk_document(_doc(["", "   "])) == []


def test_skip_references_drops_bibliography_but_keeps_appendix():
    body = "Intro text. " * 30
    doc = _doc(
        [body, body, body,
         "Conclusion text with our final remarks.\n\nReferences\n\n[1] A. Author, Some Journal (2020).",
         "[2] B. Author, Other Journal (2021).",
         "Appendix A: Extra methods\n\nWe used 5-fold cross-validation."]
    )
    kept = " ".join(c.text for c in chunk_document(doc, 200, 20, skip_references=True))
    assert "final remarks" in kept
    assert "Some Journal" not in kept and "Other Journal" not in kept
    assert "5-fold cross-validation" in kept
    everything = " ".join(c.text for c in chunk_document(doc, 200, 20, skip_references=False))
    assert "Some Journal" in everything


def test_skip_references_handles_unheaded_numbered_lists():
    body = "Body text. " * 30
    doc = _doc([body, body, body, "[1] C. Thaiss, Cell 159, 514 (2014). [2] D. Zeevi, Cell (2015)."])
    kept = " ".join(c.text for c in chunk_document(doc, 200, 20, skip_references=True))
    assert "Thaiss" not in kept
