"""
App configuration from environment variables and .env.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Chat gateway (chat_api.py)
    predict_url: str = "http://127.0.0.1:9000/predict"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    # Comma-separated list of allowed CORS origins (e.g. https://app.example.com)
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    # Rate limit for /chat (e.g. "10/minute", "100/hour")
    chat_rate_limit: str = "10/minute"
    # Max allowed prompt length (characters) for /chat and /predict
    max_prompt_length: int = 2000

    # Safety classifier (classification.py)
    model_dir: str = "./fine_tuned_mobilebert_model_colab"
    # Rate limit for /predict (e.g. "30/minute", "200/hour")
    predict_rate_limit: str = "30/minute"


settings = Settings()
