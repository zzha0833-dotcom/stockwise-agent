from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from stockwise.config import Settings
from stockwise.schemas import DatasetProfile, ExperimentPlan, ModelName

DEFAULT_PLAN = ExperimentPlan(
    models=[
        ModelName.SEASONAL_NAIVE,
        ModelName.CROSTON_SBA,
        ModelName.LIGHTGBM_GLOBAL,
    ],
    backtest_folds=3,
    horizon=28,
    max_iterations=1,
    rationale="Compare a seasonal baseline, an intermittent-demand method, and a global tree model.",
)


class LLMService:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def is_mock(self) -> bool:
        return self.settings.llm_mode == "mock"

    def _client(self) -> OpenAI:
        if not self.settings.llm_api_key:
            raise RuntimeError("LLM_API_KEY is required when LLM_MODE=api")
        return OpenAI(
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url,
            timeout=self.settings.llm_timeout_seconds,
            max_retries=0,
        )

    def _json_completion(self, *, system: str, user: str) -> dict[str, Any]:
        client = self._client()
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.settings.llm_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0,
                }
                try:
                    response = client.chat.completions.create(
                        **kwargs, response_format={"type": "json_object"}
                    )
                except Exception:
                    response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.4 * (2**attempt))
        raise RuntimeError(f"LLM request failed after retries: {last_error}")

    def create_plan(
        self, profile: DatasetProfile, *, horizon: int, experiment_budget: int
    ) -> tuple[ExperimentPlan, bool]:
        if self.is_mock:
            plan = DEFAULT_PLAN.model_copy(
                update={
                    "models": DEFAULT_PLAN.models[:experiment_budget],
                    "horizon": horizon,
                }
            )
            return plan, False
        prompt = {
            "dataset_profile": profile.model_dump(),
            "allowed_models": [model.value for model in ModelName],
            "experiment_budget": experiment_budget,
            "horizon": horizon,
            "rules": [
                "Use only allowed models",
                "Include seasonal_naive as a baseline",
                "Use at most three backtest folds",
                "Use at most one reflection iteration",
            ],
        }
        try:
            payload = self._json_completion(
                system=(
                    "You are a forecasting experiment planner. Return one JSON object matching "
                    "ExperimentPlan with models, backtest_folds, horizon, max_iterations and rationale."
                ),
                user=json.dumps(prompt),
            )
            return ExperimentPlan.model_validate(payload), False
        except (RuntimeError, ValidationError, json.JSONDecodeError):
            plan = DEFAULT_PLAN.model_copy(
                update={"models": DEFAULT_PLAN.models[:experiment_budget], "horizon": horizon}
            )
            return plan, True

    def grounded_explanation(self, evidence: dict[str, Any]) -> tuple[str, bool]:
        fallback = (
            f"The selected model is {evidence['selected_model']} with WMAPE "
            f"{evidence['wmape']:.4f}. The workflow produced {evidence['recommendation_count']} "
            f"replenishment recommendations; {evidence['approval_count']} require human approval. "
            "Inventory inputs are simulated with a fixed seed."
        )
        if self.is_mock:
            return fallback, False
        try:
            payload = self._json_completion(
                system=(
                    "Write a concise operational summary grounded only in the supplied evidence. "
                    "Return JSON with a single 'summary' string. Do not create new numbers."
                ),
                user=json.dumps(evidence),
            )
            summary = str(payload["summary"])
            if not numbers_are_grounded(summary, evidence):
                return fallback, True
            return summary, False
        except Exception:
            return fallback, True


def guard_plan(plan: ExperimentPlan, *, budget: int, horizon: int) -> ExperimentPlan:
    allowed = list(ModelName)
    models: list[ModelName] = []
    for value in plan.models:
        model = ModelName(value)
        if model in allowed and model not in models:
            models.append(model)
    if ModelName.SEASONAL_NAIVE not in models:
        models.insert(0, ModelName.SEASONAL_NAIVE)
    models = models[: max(1, min(budget, len(allowed)))]
    return ExperimentPlan(
        models=models,
        backtest_folds=min(3, max(1, plan.backtest_folds)),
        horizon=horizon,
        max_iterations=min(1, max(0, plan.max_iterations)),
        rationale=plan.rationale[:500],
    )


def numbers_are_grounded(text: str, evidence: dict[str, Any]) -> bool:
    observed = {match for match in re.findall(r"-?\d+(?:\.\d+)?", json.dumps(evidence))}
    claimed = set(re.findall(r"-?\d+(?:\.\d+)?", text))
    return claimed.issubset(observed)

