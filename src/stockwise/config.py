from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_mode: Literal["mock", "api"] = Field(default="mock", alias="LLM_MODE")
    llm_base_url: str = Field(default="https://api.openai.com/v1", alias="LLM_BASE_URL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(default="gpt-4.1-mini", alias="LLM_MODEL")
    llm_timeout_seconds: float = Field(default=45.0, alias="LLM_TIMEOUT_SECONDS")

    database_url: str = Field(
        default="sqlite:///./stockwise.db", alias="STOCKWISE_DATABASE_URL"
    )
    checkpoint_db: Path = Field(default=Path("checkpoints.db"), alias="STOCKWISE_CHECKPOINT_DB")
    data_dir: Path = Field(default=Path("data"), alias="STOCKWISE_DATA_DIR")
    artifact_dir: Path = Field(default=Path("artifacts"), alias="STOCKWISE_ARTIFACT_DIR")
    sync_runs: bool = Field(default=False, alias="STOCKWISE_SYNC_RUNS")

    mlflow_enabled: bool = Field(default=False, alias="MLFLOW_ENABLED")
    mlflow_tracking_uri: str = Field(default="./mlruns", alias="MLFLOW_TRACKING_URI")

    max_cache_bytes: int = 4 * 1024**3
    max_upload_bytes: int = 200 * 1024**2
    default_horizon: int = 28
    default_backtest_folds: int = 3
    default_approval_threshold: float = 5000.0

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "uploads",
            self.artifact_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings

