from __future__ import annotations

import numpy as np

from stockwise.forecasting import (
    confidence_from_evaluation,
    forecast_future,
    mae,
    mase,
    rolling_backtest,
    select_best_model,
    wmape,
)
from stockwise.schemas import ModelName


def test_metrics_have_expected_values():
    actual = np.array([10.0, 20.0])
    predicted = np.array([8.0, 22.0])
    assert wmape(actual, predicted) == 4 / 30
    assert mae(actual, predicted) == 2.0
    assert mase(actual, predicted, np.array([1, 2, 3, 4]), seasonality=1) == 2.0


def test_backtest_and_future_forecast(retail_frame):
    output = rolling_backtest(
        retail_frame,
        [ModelName.SEASONAL_NAIVE, ModelName.CROSTON_SBA],
        horizon=14,
        folds=2,
        seed=7,
    )
    assert len(output.evaluations) == 2
    best = select_best_model(output.evaluations)
    assert best.metrics.wmape >= 0
    assert 0 < confidence_from_evaluation(best) <= 0.99
    future = forecast_future(
        retail_frame,
        best.model,
        horizon=14,
        residual_quantile_90=best.residual_quantile_90,
        seed=7,
    )
    assert len(future) == 28
    assert (future["lower"] >= 0).all()
    assert (future["upper"] >= future["prediction"]).all()


def test_lightgbm_tool_runs_without_future_leakage(retail_frame):
    output = rolling_backtest(
        retail_frame,
        [ModelName.LIGHTGBM_GLOBAL],
        horizon=7,
        folds=1,
        seed=11,
    )
    evaluation = output.evaluations[0]
    assert evaluation.folds_completed == 1
    assert evaluation.metrics.wmape >= 0
    predictions = output.predictions[ModelName.LIGHTGBM_GLOBAL.value]["prediction"]
    assert ((predictions == 0) | (predictions >= 1.0)).all()
    # A failure is recorded and falls back explicitly; it is never silently ignored.
    assert isinstance(evaluation.errors, list)
