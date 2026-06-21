from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    chunk_size: int = 800
    chunk_overlap: int = 120
    embedding_model: str = "all-MiniLM-L6-v2"
    embed_dim: int = 384
    chroma_path: str = "chroma_db/"
    rrf_k: int = 60
    top_k: int = 8
    generation_model: str = "claude-sonnet-4-20250514"
    eval_judge_model: str = "claude-haiku-3-5-20241022"
    fastapi_port: int = 8000
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://openrouter.ai/api"
    base_dir: str = str(Path(__file__).parent)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()