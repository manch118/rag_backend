from pydantic import BaseModel, Field
from typing import Optional


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    source: str = ""
    metadata: dict = Field(default_factory=dict)


class Document(BaseModel):
    text: str
    doc_id: str = ""
    source: str = ""
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    chunk_count: int
    message: str


class Query(BaseModel):
    query: str
    top_k: int = 8


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    score: float
    rank: int


class QueryResponse(BaseModel):
    query: str
    generated_answer: str
    sources: list[RetrievedChunk]


class EvaluationRequest(BaseModel):
    query: str
    generated_answer: str
    context: str


class EvalScore(BaseModel):
    dimension: str
    score: float
    reasoning: str


class EvalResult(BaseModel):
    query: str
    scores: list[EvalScore]
    average_score: float


class FeedbackEntry(BaseModel):
    query: str
    generated_answer: str
    relevance_score: float = 0.0
    completeness_score: float = 0.0
    specificity_score: float = 0.0
    fluency_score: float = 0.0
    comment: str = ""


class FeedbackResponse(BaseModel):
    status: str
    message: str


class StatsResponse(BaseModel):
    total_feedback: int
    avg_relevance: float
    avg_completeness: float
    avg_specificity: float
    avg_fluency: float
    avg_overall: float
    recent_entries: list[FeedbackEntry]