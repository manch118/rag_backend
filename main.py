from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config import settings
from models import (
    Document, IngestResponse, Query, QueryResponse,
    RetrievedChunk, EvaluationRequest, EvalResult, EvalScore,
    FeedbackEntry, FeedbackResponse, StatsResponse
)
from ingestion import ingest_document, rebuild_bm25_corpus
from retrieval import retrieve
from generation import generate
from evaluation import evaluate
from feedback import post_feedback, get_stats


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.anthropic_api_key:
        import os
        settings.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    rebuild_bm25_corpus()
    yield


app = FastAPI(title="RAG Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ingest", response_model=IngestResponse)
def api_ingest(doc: Document):
    result = ingest_document(
        text=doc.text,
        doc_id=doc.doc_id,
        source=doc.source,
        metadata=doc.metadata
    )
    return IngestResponse(
        doc_id=result["doc_id"],
        chunk_count=result["chunk_count"],
        message=f"Ingested {result['chunk_count']} chunks"
    )


@app.get("/retrieve", response_model=list[RetrievedChunk])
def api_retrieve(q: str, top_k: int = 8):
    results = retrieve(q, top_k=top_k)
    return [
        RetrievedChunk(
            chunk_id=cid,
            text=text,
            source="",
            score=round(score, 4),
            rank=idx + 1
        )
        for idx, (cid, text, score) in enumerate(results)
    ]


@app.post("/query", response_model=QueryResponse)
def api_query(query_req: Query):
    results = retrieve(query_req.query, top_k=query_req.top_k)
    sources = [
        RetrievedChunk(
            chunk_id=cid,
            text=text,
            source="",
            score=round(score, 4),
            rank=idx + 1
        )
        for idx, (cid, text, score) in enumerate(results)
    ]
    generated_prompt = generate(query_req.query, results)
    return QueryResponse(
        query=query_req.query,
        generated_answer=generated_prompt,
        sources=sources
    )


@app.post("/evaluate", response_model=EvalResult)
def api_evaluate(eval_req: EvaluationRequest):
    result = evaluate(eval_req.query, eval_req.generated_answer, eval_req.context)
    scores = result.get("scores", {})
    reasoning = result.get("reasoning", "")
    eval_scores = [
        EvalScore(dimension=dim, score=val, reasoning=reasoning)
        for dim, val in scores.items()
    ]
    return EvalResult(
        query=eval_req.query,
        scores=eval_scores,
        average_score=result.get("average_score", 0.0)
    )


@app.post("/feedback", response_model=FeedbackResponse)
def api_feedback(entry: FeedbackEntry):
    return post_feedback(entry)


@app.get("/stats", response_model=StatsResponse)
def api_stats():
    return get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.fastapi_port)