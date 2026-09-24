from __future__ import annotations

from pathlib import Path

import pytest

from stockwise.config import Settings
from stockwise.data import generate_synthetic_retail_data


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        LLM_MODE="mock",
        STOCKWISE_DATABASE_URL=f"sqlite:///{tmp_path / 'stockwise.db'}",
        STOCKWISE_CHECKPOINT_DB=tmp_path / "checkpoints.db",
        STOCKWISE_DATA_DIR=tmp_path / "data",
        STOCKWISE_ARTIFACT_DIR=tmp_path / "artifacts",
        STOCKWISE_SYNC_RUNS=True,
        MLFLOW_ENABLED=False,
    )


@pytest.fixture
def retail_frame():
    return generate_synthetic_retail_data(items=2, stores=1, days=140, seed=7)

