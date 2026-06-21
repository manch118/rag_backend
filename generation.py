from anthropic import Anthropic
from config import settings


_client = Anthropic(
    api_key=settings.anthropic_api_key,
    base_url=settings.anthropic_base_url
)


SYSTEM_PROMPT = """You are a helpful RAG assistant. Your task is to generate a grounded answer based on retrieved context passages. Use only the provided context passages to answer the user's query. If the context is insufficient to answer the query, say so clearly.

Analyze the user's query and the provided context. Then produce a concise, accurate answer that directly addresses the query.

Format your response as plain text only, without any additional explanation."""


def generate(query: str, retrieved_chunks: list[tuple[str, str, float]]) -> str:
    if not retrieved_chunks:
        return "I don't have enough information from the provided context to answer this query."

    context = "\n\n".join([
        f"Context passage {i+1}:\n{text}"
        for i, (_, text, _) in enumerate(retrieved_chunks)
    ])

    user_message = f"Query:\n{query}\n\nRetrieved context:\n{context}"

    response = _client.messages.create(
        model=settings.generation_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}]
    )

    return response.content[0].text