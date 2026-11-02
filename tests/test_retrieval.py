import numpy as np
import pytest

from src.schemas import Chunk
from src.vector_store import VectorStore

TEXTS = {
    "doc-a": [
        "We performed bulk RNA sequencing on mouse liver tissue.",
        "Differential expression was tested with DESeq2.",
        "The study was limited by a small sample size.",
    ],
    "doc-b": [
        "Gut microbiome samples were collected daily from mice.",
        "Metagenomic reads were assembled into genomes.",
        "No circadian oscillation was detected in abundance.",
    ],
}


def _chunks():
    out = []
    for doc_id, texts in TEXTS.items():
        for i, t in enumerate(texts):
            out.append(Chunk(f"{doc_id}-p{i + 1}-c0", doc_id, f"{doc_id}.pdf", i + 1, t, 0))
    return out


@pytest.fixture
def store(fake_embedder):
    chunks = _chunks()
    s = VectorStore(fake_embedder.dim)
    s.add(chunks, fake_embedder.embed_documents([c.text for c in chunks]))
    return s


def test_search_returns_most_similar_first(store, fake_embedder):
    results = store.search(fake_embedder.embed_query("which method tested differential expression DESeq2"), k=3)
    assert results[0].chunk.text.startswith("Differential expression")
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].score >= results[1].score >= results[2].score


def test_metadata_mapping_is_preserved(store, fake_embedder):
    r = store.search(fake_embedder.embed_query("gut microbiome samples collected daily"), k=1)[0]
    assert r.chunk.doc_name == "doc-b.pdf" and r.chunk.page_number == 1


def test_document_filter(store, fake_embedder):
    q = fake_embedder.embed_query("RNA sequencing DESeq2 sample size")
    results = store.search(q, k=3, doc_ids={"doc-b"})
    assert len(results) == 3
    assert all(r.chunk.doc_id == "doc-b" for r in results)


def test_k_larger_than_index(store, fake_embedder):
    assert len(store.search(fake_embedder.embed_query("mice"), k=50)) == 6


def test_empty_store_returns_nothing(fake_embedder):
    assert VectorStore(fake_embedder.dim).search(fake_embedder.embed_query("x"), k=3) == []


def test_remove_document_keeps_mapping_consistent(store, fake_embedder):
    assert store.remove_document("doc-a") == 3
    assert len(store) == 3 and store.index.ntotal == 3
    r = store.search(fake_embedder.embed_query("metagenomic reads assembled genomes"), k=1)[0]
    assert r.chunk.text.startswith("Metagenomic")
    assert store.doc_ids == {"doc-b"}


def test_clear(store):
    store.clear()
    assert len(store) == 0 and store.index.ntotal == 0


def test_add_validates_shapes(fake_embedder):
    s = VectorStore(fake_embedder.dim)
    with pytest.raises(ValueError):
        s.add(_chunks()[:2], np.zeros((3, fake_embedder.dim), dtype="float32"))
    with pytest.raises(ValueError):
        s.add(_chunks()[:1], np.zeros((1, 7), dtype="float32"))


def test_save_and_load_roundtrip(store, fake_embedder, tmp_path):
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    q = fake_embedder.embed_query("circadian oscillation abundance")
    assert [r.chunk.chunk_id for r in loaded.search(q, k=3)] == [r.chunk.chunk_id for r in store.search(q, k=3)]


@pytest.mark.slow
def test_real_embedding_model_retrieves_semantically():
    """Integration test with the real Sentence Transformers model (downloads ~90 MB once)."""
    from src.embeddings import Embedder

    emb = Embedder()
    chunks = _chunks()
    s = VectorStore(emb.dim)
    s.add(chunks, emb.embed_documents([c.text for c in chunks]))
    vectors = emb.embed_documents(["test"])
    assert np.isclose(np.linalg.norm(vectors[0]), 1.0, atol=1e-4)
    top = s.search(emb.embed_query("Which tool was used for the differential gene expression test?"), k=1)[0]
    assert top.chunk.text.startswith("Differential expression")
    top = s.search(emb.embed_query("What were the shortcomings of the study?"), k=1)[0]
    assert "limited" in top.chunk.text
    top = s.search(emb.embed_query("Did bacteria follow a daily rhythm?"), k=1)[0]
    assert top.chunk.doc_id == "doc-b"

    # FAISS must agree with a brute-force NumPy cosine ranking (guards the macOS
    # duplicate-OpenMP workaround in src/__init__.py).
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(500, emb.dim)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    big = VectorStore(emb.dim)
    big.add([Chunk(str(i), "d", "d.pdf", 1, "t", i) for i in range(500)], vecs)
    for q in vecs[:20]:
        expected = np.argsort(-(vecs @ q))[:10]
        got = [int(r.chunk.chunk_id) for r in big.search(q, k=10)]
        assert got == expected.tolist()
