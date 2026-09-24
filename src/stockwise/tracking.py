from __future__ import annotations

from pathlib import Path
from typing import Any

from stockwise.config import Settings


def log_mlflow_run(
    settings: Settings, run_id: str, result: dict[str, Any], artifacts: dict[str, Path]
) -> None:
    if not settings.mlflow_enabled:
        return
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("stockwise")
    evaluation = result.get("evaluation", {})
    with mlflow.start_run(run_name=run_id):
        mlflow.log_params(
            {
                "selected_model": evaluation.get("selected_model", "unknown"),
                "inventory_data": evaluation.get("inventory_data", "simulated"),
                "fallback_used": evaluation.get("agent_fallback_used", False),
            }
        )
        metrics = {
            key: float(value)
            for key, value in evaluation.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if metrics:
            mlflow.log_metrics(metrics)
        for path in artifacts.values():
            mlflow.log_artifact(str(path))

