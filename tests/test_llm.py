from __future__ import annotations

from stockwise.config import Settings
from stockwise.llm import LLMService, guard_plan, numbers_are_grounded
from stockwise.schemas import ExperimentPlan, ModelName


def test_plan_guard_enforces_baseline_budget_and_horizon():
    unsafe = ExperimentPlan(
        models=[ModelName.LIGHTGBM_GLOBAL], backtest_folds=3, horizon=56, max_iterations=1
    )
    guarded = guard_plan(unsafe, budget=2, horizon=28)
    assert guarded.models[0] == ModelName.SEASONAL_NAIVE
    assert len(guarded.models) == 2
    assert guarded.horizon == 28


def test_numeric_grounding_rejects_new_numbers():
    evidence = {"wmape": 0.15, "recommendations": 10}
    assert numbers_are_grounded("WMAPE is 0.15 for 10 recommendations.", evidence)
    assert not numbers_are_grounded("WMAPE is 0.12 for 10 recommendations.", evidence)


def test_api_timeout_uses_validated_deterministic_fallback(monkeypatch, retail_frame):
    from stockwise.data import validate_and_profile

    service = LLMService(
        Settings(LLM_MODE="api", LLM_API_KEY="test-placeholder", LLM_TIMEOUT_SECONDS=0.01)
    )

    def timeout(**_kwargs):
        raise RuntimeError("simulated timeout")

    monkeypatch.setattr(service, "_json_completion", timeout)
    plan, fallback = service.create_plan(
        validate_and_profile(retail_frame), horizon=28, experiment_budget=2
    )

    assert fallback is True
    assert plan.models == [ModelName.SEASONAL_NAIVE, ModelName.CROSTON_SBA]
    assert plan.horizon == 28


def test_mock_mode_never_requires_an_api_key(retail_frame):
    from stockwise.data import validate_and_profile

    service = LLMService(Settings(LLM_MODE="mock", LLM_API_KEY=""))
    plan, fallback = service.create_plan(
        validate_and_profile(retail_frame), horizon=14, experiment_budget=1
    )
    assert fallback is False
    assert plan.models == [ModelName.SEASONAL_NAIVE]
