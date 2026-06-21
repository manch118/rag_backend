import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ANTHROPIC_API_KEY"] = "test_key_placeholder"

chroma_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
if os.path.exists(chroma_dir):
    shutil.rmtree(chroma_dir)

from config import settings
print("=== CONFIG VERIFICATION ===")
for attr in ["chunk_size", "chunk_overlap", "embedding_model", "embed_dim", "chroma_path", "rrf_k", "top_k", "generation_model", "eval_judge_model"]:
    print(f"  {attr}: {getattr(settings, attr)}")
assert settings.chunk_size == 800
assert settings.chunk_overlap == 120
assert settings.embed_dim == 384
assert settings.rrf_k == 60
assert settings.top_k == 8
print("PASS")
print()

from models import Document, IngestResponse, Query, RetrievedChunk, FeedbackEntry, StatsResponse, Chunk
print("=== MODELS VERIFICATION ===")
d = Document(text="test", doc_id="id1", source="src", metadata={"key": "val"})
assert d.text == "test" and d.doc_id == "id1"
q = Query(query="hello", top_k=5)
assert q.query == "hello" and q.top_k == 5
rc = RetrievedChunk(chunk_id="c1", text="text", source="src", score=0.9, rank=1)
assert rc.score == 0.9 and rc.rank == 1
ir = IngestResponse(doc_id="d1", chunk_count=3, message="ok")
assert ir.chunk_count == 3
fe = FeedbackEntry(query="q", generated_answer="p", relevance_score=4.0, completeness_score=3.0, specificity_score=5.0, fluency_score=2.0, comment="nice")
assert fe.relevance_score == 4.0
sr = StatsResponse(total_feedback=2, avg_relevance=3.5, avg_completeness=3.0, avg_specificity=4.0, avg_fluency=2.5, avg_overall=3.25, recent_entries=[fe])
assert sr.total_feedback == 2 and sr.avg_overall == 3.25
print("PASS")
print()

import ingestion
from ingestion import token_count, chunk_text, ingest_document, get_all_chunks, _collection
print("=== INGESTION VERIFICATION ===")
tc = token_count("Hello, world!")
assert tc > 0
print(f"token_count('Hello, world!'): {tc}")

long_text = " ".join(["word"] * 2000)
chunks = chunk_text(long_text)
print(f"chunk_count for 2000 words: {len(chunks)}")
assert len(chunks) >= 2
for c in chunks:
    ct = token_count(c)
    assert ct <= settings.chunk_size * 2
print("Chunk size bounds: PASS")

test_doc = """A serene mountain landscape at sunset with snow-capped peaks reflecting in a crystal-clear alpine lake. Pine trees line the shoreline. The sky is painted in shades of orange, purple, and pink.

A winding dirt path leads through a meadow of wildflowers in full bloom. Butterflies dance among the petals while a gentle breeze carries the scent of pine and earth.

In the distance, a small wooden cabin sits nestled among the trees, smoke curling from its stone chimney. The warm glow of lantern light spills from its windows.

Above, stars begin to emerge in the deepening twilight sky. The first bright evening star appears over the mountain ridge, a beacon in the fading light.

The scene is peaceful and idyllic, a perfect moment captured in nature's beauty. The air is cool and crisp, carrying the promise of a clear night ahead."""

result = ingest_document(text=test_doc, doc_id="mountain_test", source="test", metadata={"type": "landscape"})
print(f"ingest result: {result}")
assert result["doc_id"] == "mountain_test"
assert result["chunk_count"] > 0
print(f"Chunks created: {result['chunk_count']}")

all_chunks = get_all_chunks()
print(f"Total chunks in ChromaDB: {len(all_chunks)}")
assert len(all_chunks) == result["chunk_count"]
for cid, doc, meta in all_chunks:
    print(f"  chunk: {cid}, meta: {meta}")
print("PASS")
print()

print("=== BM25 VERIFICATION ===")
assert ingestion._bm25_index is not None
assert len(ingestion._bm25_corpus) == result["chunk_count"]
print(f"BM25 corpus size: {len(ingestion._bm25_corpus)}")
scores = ingestion._bm25_index.get_scores(["mountain", "lake"])
print(f"BM25 scores for ['mountain', 'lake']: {scores}")
assert len(scores) == len(ingestion._bm25_corpus)
assert not any(s is None for s in scores)
print("PASS")
print()

from retrieval import retrieve, dense_search, bm25_search, rrf_merge
print("=== RETRIEVAL VERIFICATION ===")
results = retrieve("mountain lake sunset", top_k=8)
print(f"retrieve('mountain lake sunset', top_k=8): {len(results)} results")
for cid, text, score in results:
    print(f"  {cid}: score={score:.4f}, text_len={len(text)}")
assert len(results) <= 8
assert len(results) > 0
assert results[0][2] > 0

results2 = retrieve("ocean beach waves", top_k=5)
print(f"retrieve('ocean beach waves', top_k=5): {len(results2)} results")
assert len(results2) > 0

dense = dense_search("mountain", top_k=15)
print(f"dense_search('mountain'): {len(dense)} results")
assert len(dense) > 0

bm25 = bm25_search("mountain", top_k=15)
print(f"bm25_search('mountain'): {len(bm25)} results")
assert len(bm25) > 0

merged = rrf_merge(dense, bm25, k=60, top_k=8)
print(f"rrf_merge result count: {len(merged)}")
assert len(merged) <= 8
assert len(merged) > 0
for cid, text, score in merged:
    assert score > 0
print("PASS")
print()

from feedback import post_feedback, get_stats, _store
print("=== FEEDBACK VERIFICATION ===")
_store.clear()
fe1 = FeedbackEntry(query="mountain landscape", generated_answer="a beautiful mountain scene", relevance_score=4.0, completeness_score=3.5, specificity_score=4.5, fluency_score=5.0, comment="Great prompt")
fe2 = FeedbackEntry(query="sunset lake", generated_answer="a sunset over a lake", relevance_score=3.0, completeness_score=2.5, specificity_score=3.0, fluency_score=4.0, comment="Good")

r1 = post_feedback(fe1)
assert r1["status"] == "ok"
r2 = post_feedback(fe2)
assert r2["status"] == "ok"

stats = get_stats()
print(f"total_feedback: {stats.total_feedback}")
print(f"avg_relevance: {stats.avg_relevance}")
print(f"avg_completeness: {stats.avg_completeness}")
print(f"avg_specificity: {stats.avg_specificity}")
print(f"avg_fluency: {stats.avg_fluency}")
print(f"avg_overall: {stats.avg_overall}")
print(f"recent_entries: {len(stats.recent_entries)}")
assert stats.total_feedback == 2
assert stats.avg_relevance == 3.5
assert stats.avg_completeness == 3.0
assert stats.avg_specificity == 3.75
assert stats.avg_fluency == 4.5
assert stats.avg_overall == 3.69
assert len(stats.recent_entries) == 2
print("PASS")
print()

from main import app
print("=== MAIN APP VERIFICATION ===")
assert app.title == "RAG Backend"
routes = [route.path for route in app.routes]
print(f"Routes: {routes}")
for r in ["/ingest", "/retrieve", "/query", "/evaluate", "/feedback", "/stats"]:
    assert r in routes
print("PASS")
print()

from generation import generate
print("=== GENERATION VERIFICATION (no-API paths) ===")
fallback = generate("test query", [])
assert fallback == "I don't have enough information from the provided context to answer this query."
print(f"Empty context fallback: {fallback}")
print("PASS")
print()

print("=== ALL LOCAL TESTS PASSED ===")