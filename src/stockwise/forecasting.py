from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
import polars as pl
from lightgbm import LGBMRegressor
from statsforecast.models import CrostonSBA, SeasonalNaive

from stockwise.schemas import MetricSet, ModelEvaluation, ModelName

KEYS = ["store_id", "item_id"]
MINIMUM_DAILY_UNIT_FORECAST = 1.0


@dataclass
class BacktestOutput:
    evaluations: list[ModelEvaluation]
    predictions: dict[str, pd.DataFrame]
    residuals: dict[str, np.ndarray]


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.abs(y_true).sum())
    if denominator == 0:
        return float(np.abs(y_true - y_pred).mean())
    return float(np.abs(y_true - y_pred).sum() / denominator)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def mase(y_true: np.ndarray, y_pred: np.ndarray, insample: np.ndarray, seasonality: int = 7) -> float:
    if len(insample) <= seasonality:
        scale = float(np.abs(np.diff(insample)).mean()) if len(insample) > 1 else 1.0
    else:
        scale = float(np.abs(insample[seasonality:] - insample[:-seasonality]).mean())
    scale = scale if scale > 1e-9 else 1.0
    return float(np.abs(y_true - y_pred).mean() / scale)


def _to_pandas(frame: pl.DataFrame) -> pd.DataFrame:
    pdf = frame.to_pandas()
    pdf["date"] = pd.to_datetime(pdf["date"])
    pdf["sales"] = pdf["sales"].astype(float)
    pdf["sell_price"] = pdf["sell_price"].astype(float).fillna(0)
    pdf["promo_flag"] = pdf["promo_flag"].astype(bool)
    return pdf.sort_values(KEYS + ["date"]).reset_index(drop=True)


def seasonal_naive_predict(train: pd.DataFrame, future: pd.DataFrame, seasonality: int = 7) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, future_group in future.groupby(KEYS, sort=False):
        history = train[(train.store_id == keys[0]) & (train.item_id == keys[1])][
            "sales"
        ].to_numpy(dtype=float)
        ordered_future = future_group.sort_values("date")
        if len(history) == 0:
            predictions = np.zeros(len(ordered_future))
        else:
            model = SeasonalNaive(season_length=min(seasonality, len(history)))
            predictions = model.forecast(history, h=len(ordered_future))["mean"]
        for prediction, (_, future_row) in zip(
            predictions, ordered_future.iterrows(), strict=True
        ):
            rows.append(
                {
                    "date": future_row["date"],
                    "store_id": keys[0],
                    "item_id": keys[1],
                    "prediction": max(0.0, float(prediction)),
                }
            )
    return pd.DataFrame(rows)


def croston_sba_predict(train: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, future_group in future.groupby(KEYS, sort=False):
        history = train[(train.store_id == keys[0]) & (train.item_id == keys[1])][
            "sales"
        ].to_numpy(dtype=float)
        ordered_future = future_group.sort_values("date")
        if len(history) == 0 or not np.any(history > 0):
            predictions = np.zeros(len(ordered_future))
        else:
            predictions = CrostonSBA().forecast(history, h=len(ordered_future))["mean"]
        for prediction, (_, future_row) in zip(
            predictions, ordered_future.iterrows(), strict=True
        ):
            rows.append(
                {
                    "date": future_row["date"],
                    "store_id": keys[0],
                    "item_id": keys[1],
                    "prediction": max(0.0, float(prediction)),
                }
            )
    return pd.DataFrame(rows)


def _feature_frame(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy().sort_values(KEYS + ["date"])
    grouped = frame.groupby(KEYS, sort=False)["sales"]
    for lag in (1, 7, 14, 28):
        frame[f"lag_{lag}"] = grouped.shift(lag)
    shifted = grouped.shift(1)
    frame["rolling_mean_7"] = shifted.groupby([frame[k] for k in KEYS]).rolling(7).mean().reset_index(level=KEYS, drop=True)
    frame["rolling_mean_28"] = shifted.groupby([frame[k] for k in KEYS]).rolling(28).mean().reset_index(level=KEYS, drop=True)
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["day_of_month"] = frame["date"].dt.day
    frame["month"] = frame["date"].dt.month
    frame["promo_flag"] = frame["promo_flag"].astype(int)
    return frame


FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "sell_price",
    "promo_flag",
    "day_of_week",
    "day_of_month",
    "month",
]


def lightgbm_global_predict(
    train: pd.DataFrame,
    future: pd.DataFrame,
    *,
    seed: int = 42,
    n_estimators: int = 120,
) -> pd.DataFrame:
    feature_history = _feature_frame(train)
    training = feature_history.dropna(subset=FEATURES + ["sales"])
    if len(training) < 40:
        raise ValueError("Insufficient rows for the global LightGBM model")
    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=12,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        verbosity=-1,
        n_jobs=1,
    )
    model.fit(training[FEATURES], training["sales"])

    working = train.copy()
    outputs: list[pd.DataFrame] = []
    for current_date in sorted(pd.to_datetime(future["date"].unique())):
        daily = future[future["date"] == current_date].copy()
        daily["sales"] = np.nan
        working = pd.concat([working, daily[working.columns]], ignore_index=True)
        featured = _feature_frame(working)
        mask = featured["date"] == current_date
        current_features = featured.loc[mask, FEATURES].fillna(0)
        predictions = np.clip(model.predict(current_features), 0, None)
        # Demand is measured in discrete units. Tiny positive forecasts create
        # systematic phantom demand on intermittent series, so apply a fixed,
        # data-independent one-unit deadband rather than tuning on holdout labels.
        predictions = np.where(
            predictions < MINIMUM_DAILY_UNIT_FORECAST,
            0.0,
            predictions,
        )
        featured.loc[mask, "sales"] = predictions
        working = featured[train.columns].copy()
        daily_output = featured.loc[mask, ["date", *KEYS]].copy()
        daily_output["prediction"] = predictions
        outputs.append(daily_output)
    return pd.concat(outputs, ignore_index=True)


def _future_template(train: pd.DataFrame, horizon: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in train.groupby(KEYS, sort=False):
        last = group.sort_values("date").iloc[-1]
        last_date = pd.Timestamp(last["date"])
        for step in range(1, horizon + 1):
            current = last_date + timedelta(days=step)
            rows.append(
                {
                    "date": current,
                    "store_id": keys[0],
                    "item_id": keys[1],
                    "sales": np.nan,
                    "sell_price": float(last["sell_price"]),
                    "promo_flag": False,
                    "event_name": "",
                }
            )
    return pd.DataFrame(rows)


def predict_model(
    model_name: ModelName | str,
    train: pd.DataFrame,
    future: pd.DataFrame,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    model_name = ModelName(model_name)
    if model_name == ModelName.SEASONAL_NAIVE:
        return seasonal_naive_predict(train, future)
    if model_name == ModelName.CROSTON_SBA:
        return croston_sba_predict(train, future)
    if model_name == ModelName.LIGHTGBM_GLOBAL:
        return lightgbm_global_predict(train, future, seed=seed)
    raise ValueError(f"Unsupported model: {model_name}")


def rolling_backtest(
    frame: pl.DataFrame,
    models: Iterable[ModelName | str],
    *,
    horizon: int = 28,
    folds: int = 3,
    seed: int = 42,
) -> BacktestOutput:
    pdf = _to_pandas(frame)
    unique_dates = sorted(pdf["date"].unique())
    minimum = horizon * folds + 35
    if len(unique_dates) < minimum:
        raise ValueError(f"At least {minimum} daily observations are required for backtesting")
    evaluations: list[ModelEvaluation] = []
    all_predictions: dict[str, pd.DataFrame] = {}
    residual_map: dict[str, np.ndarray] = {}

    for model_value in models:
        model_name = ModelName(model_value)
        started = time.perf_counter()
        fold_outputs: list[pd.DataFrame] = []
        errors: list[str] = []
        insample_values: list[np.ndarray] = []
        for fold in range(folds, 0, -1):
            test_start_idx = len(unique_dates) - horizon * fold
            train_end = pd.Timestamp(unique_dates[test_start_idx - 1])
            test_end_idx = min(test_start_idx + horizon, len(unique_dates))
            test_dates = unique_dates[test_start_idx:test_end_idx]
            train = pdf[pdf["date"] <= train_end].copy()
            test = pdf[pdf["date"].isin(test_dates)].copy()
            try:
                predicted = predict_model(model_name, train, test, seed=seed)
            except Exception as exc:  # fallback is intentional and recorded
                errors.append(f"fold {fold}: {type(exc).__name__}: {exc}")
                predicted = seasonal_naive_predict(train, test)
            merged = test.merge(predicted, on=["date", *KEYS], how="inner")
            merged["fold"] = folds - fold + 1
            fold_outputs.append(merged)
            insample_values.append(train["sales"].to_numpy())
        combined = pd.concat(fold_outputs, ignore_index=True)
        y_true = combined["sales"].to_numpy(dtype=float)
        y_pred = combined["prediction"].to_numpy(dtype=float)
        residuals = y_true - y_pred
        scale_sample = np.concatenate(insample_values)
        residual_q90 = float(np.quantile(np.abs(residuals), 0.9)) if len(residuals) else 0.0
        coverage = float(np.mean(np.abs(residuals) <= residual_q90)) if len(residuals) else 0.0
        evaluation = ModelEvaluation(
            model=model_name,
            metrics=MetricSet(
                wmape=round(wmape(y_true, y_pred), 6),
                mae=round(mae(y_true, y_pred), 6),
                mase=round(mase(y_true, y_pred, scale_sample), 6),
                interval_coverage=round(coverage, 6),
            ),
            residual_quantile_90=round(residual_q90, 6),
            folds_completed=folds,
            runtime_seconds=round(time.perf_counter() - started, 4),
            errors=errors,
        )
        evaluations.append(evaluation)
        all_predictions[model_name.value] = combined
        residual_map[model_name.value] = residuals
    return BacktestOutput(evaluations, all_predictions, residual_map)


def select_best_model(evaluations: list[ModelEvaluation]) -> ModelEvaluation:
    if not evaluations:
        raise ValueError("No model evaluations were produced")
    return min(evaluations, key=lambda item: (item.metrics.wmape, item.metrics.mase))


def forecast_future(
    frame: pl.DataFrame,
    model_name: ModelName | str,
    *,
    horizon: int,
    residual_quantile_90: float,
    seed: int = 42,
) -> pd.DataFrame:
    train = _to_pandas(frame)
    future = _future_template(train, horizon)
    predicted = predict_model(model_name, train, future, seed=seed)
    predicted["lower"] = np.clip(predicted["prediction"] - residual_quantile_90, 0, None)
    predicted["upper"] = predicted["prediction"] + residual_quantile_90
    return predicted


def confidence_from_evaluation(evaluation: ModelEvaluation) -> float:
    error_component = max(0.0, 1.0 - min(1.0, evaluation.metrics.wmape))
    coverage = evaluation.metrics.interval_coverage or 0.0
    fallback_penalty = 0.15 if evaluation.errors else 0.0
    return round(max(0.05, min(0.99, 0.7 * error_component + 0.3 * coverage - fallback_penalty)), 4)


def relative_improvement(best: ModelEvaluation, baseline: ModelEvaluation) -> float:
    denominator = baseline.metrics.wmape
    if denominator <= 1e-12:
        return 0.0
    return round((denominator - best.metrics.wmape) / denominator, 6)
