import numpy as np
from sentence_transformers import SentenceTransformer
from ingestion import _collection, _bm25_index, _bm25_corpus, _bm25_chunk_ids
from config import settings


_encoder = SentenceTransformer(settings.embedding_model)


def dense_search(query: str, top_k: int = 15) -> list[tuple[str, str, float]]:
    query_embedding = _encoder.encode([query]).tolist()
    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, _collection.count() or 1),
        include=["documents", "distances"]
    )
    items = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            items.append((results["ids"][0][i], results["documents"][0][i], results["distances"][0][i]))
    return items


def bm25_search(query: str, top_k: int = 15) -> list[tuple[str, str, float]]:
    if _bm25_index is None or not _bm25_corpus:
        return []
    tokenized_query = query.split()
    scores = _bm25_index.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    items = []
    for idx in top_indices:
        chunk_id = _bm25_chunk_ids[idx] if idx < len(_bm25_chunk_ids) else f"chunk_{idx}"
        items.append((chunk_id, _bm25_corpus[idx], float(scores[idx])))
    return items


def rrf_merge(
    dense_results: list[tuple[str, str, float]],
    bm25_results: list[tuple[str, str, float]],
    k: int = 60,
    top_k: int = 8
) -> list[tuple[str, str, float]]:
    score_map = {}
    doc_map = {}
    for rank, (cid, text, _) in enumerate(dense_results):
        score_map[cid] = score_map.get(cid, 0.0) + 1.0 / (k + rank + 1)
        doc_map[cid] = text
    for rank, (cid, text, _) in enumerate(bm25_results):
        score_map[cid] = score_map.get(cid, 0.0) + 1.0 / (k + rank + 1)
        doc_map[cid] = text
    sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return [(cid, doc_map[cid], score) for cid, score in sorted_items[:top_k]]


def retrieve(query: str, top_k: int = 8) -> list[tuple[str, str, float]]:
    dense_results = dense_search(query, top_k=max(top_k * 2, 15))
    bm25_results = bm25_search(query, top_k=max(top_k * 2, 15))
    return rrf_merge(dense_results, bm25_results, k=settings.rrf_k, top_k=top_k)