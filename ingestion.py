import uuid
import tiktoken
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from config import settings


_tokenizer = tiktoken.get_encoding("cl100k_base")
_encoder = SentenceTransformer(settings.embedding_model)
_client = chromadb.PersistentClient(path=settings.chroma_path)
_collection = _client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)
_bm25_corpus: list[str] = []
_bm25_chunk_ids: list[str] = []
_bm25_index: BM25Okapi | None = None


def token_count(text: str) -> int:
    return len(_tokenizer.encode(text))


def chunk_text(text: str) -> list[str]:
    chunks = []
    tokens = _tokenizer.encode(text)
    i = 0
    while i < len(tokens):
        end = min(i + settings.chunk_size, len(tokens))
        chunk_tokens = tokens[i:end]
        chunks.append(_tokenizer.decode(chunk_tokens))
        i += settings.chunk_size - settings.chunk_overlap
        if i >= len(tokens):
            break
    return chunks


def ingest_document(text: str, doc_id: str = "", source: str = "", metadata: dict | None = None) -> dict:
    if not doc_id:
        doc_id = str(uuid.uuid4())
    if metadata is None:
        metadata = {}
    chunks = chunk_text(text)
    ids = []
    documents = []
    metadatas = []
    for idx, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}_chunk_{idx}"
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append({
            "doc_id": doc_id,
            "chunk_index": idx,
            "source": source,
            **metadata
        })
    embeddings = _encoder.encode(documents).tolist()
    _collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    rebuild_bm25_corpus()
    return {"doc_id": doc_id, "chunk_count": len(chunks)}


def get_all_chunks() -> list[tuple[str, str, dict]]:
    count = _collection.count()
    if count == 0:
        return []
    results = _collection.get(include=["documents", "metadatas"])
    items = []
    for i in range(len(results["ids"])):
        items.append((results["ids"][i], results["documents"][i], results["metadatas"][i]))
    return items


def rebuild_bm25_corpus():
    global _bm25_corpus, _bm25_chunk_ids, _bm25_index
    items = get_all_chunks()
    _bm25_chunk_ids = [cid for cid, _, _ in items]
    _bm25_corpus = [doc for _, doc, _ in items]
    if _bm25_corpus:
        tokenized = [doc.split() for doc in _bm25_corpus]
        _bm25_index = BM25Okapi(tokenized)
    else:
        _bm25_index = None