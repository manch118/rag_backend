import json
from anthropic import Anthropic
from config import settings


_client = Anthropic(
    api_key=settings.anthropic_api_key,
    base_url=settings.anthropic_base_url
)


JUDGE_SYSTEM_PROMPT = """You are an evaluation judge for RAG answer quality. Score the generated answer on four dimensions, each on a scale of 1 to 5:

1. relevance: How well does the generated answer address the user query?
2. completeness: Does the answer cover all key aspects of the query?
3. specificity: How detailed and specific is the answer (vs. vague or generic)?
4. fluency: How well-written and natural does the answer read?

Return your evaluation as a JSON object with the following structure:
{"scores": {"relevance": <1-5>, "completeness": <1-5>, "specificity": <1-5>, "fluency": <1-5>}, "reasoning": "<brief explanation>"}"""


def evaluate(query: str, generated_answer: str, context: str) -> dict:
    user_message = (
        f"Query:\n{query}\n\n"
        f"Generated answer:\n{generated_answer}\n\n"
        f"Context passages:\n{context}"
    )

    response = _client.messages.create(
        model=settings.eval_judge_model,
        max_tokens=512,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    raw = response.content[0].text
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"scores": {"relevance": 0, "completeness": 0, "specificity": 0, "fluency": 0}, "reasoning": "Failed to parse judge response"}

    scores = result.get("scores", {})
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    result["average_score"] = round(avg, 2)
    return result