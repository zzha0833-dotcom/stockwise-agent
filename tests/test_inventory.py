from __future__ import annotations

import pandas as pd

from stockwise.inventory import aggregate_inventory_metrics, build_recommendations


def test_inventory_recommendations_are_reproducible():
    dates = pd.date_range("2026-01-01", periods=28)
    forecast = pd.DataFrame(
        {
            "date": dates,
            "store_id": "STORE_01",
            "item_id": "ITEM_001",
            "prediction": 12.0,
            "lower": 10.0,
            "upper": 14.0,
        }
    )
    first = build_recommendations(
        forecast,
        confidence=0.8,
        residual_std=2.0,
        approval_amount_threshold=0,
        seed=42,
    )
    second = build_recommendations(
        forecast,
        confidence=0.8,
        residual_std=2.0,
        approval_amount_threshold=0,
        seed=42,
    )
    assert first == second
    assert first[0].requires_approval is True
    assert first[0].evidence["inventory_source"] == "simulated_with_fixed_seed"
    metrics = aggregate_inventory_metrics(first)
    assert metrics["inventory_data"] == "simulated"


def test_low_confidence_and_anomaly_approval_routes():
    dates = pd.date_range("2026-01-01", periods=28)
    forecast = pd.DataFrame(
        {
            "date": dates,
            "store_id": "STORE_01",
            "item_id": "ITEM_002",
            "prediction": 25.0,
            "lower": 18.0,
            "upper": 32.0,
        }
    )
    recommendation = build_recommendations(
        forecast,
        confidence=0.5,
        residual_std=3.0,
        approval_amount_threshold=1_000_000,
        anomaly_count=2,
        seed=9,
    )[0]

    assert recommendation.safety_stock > 0
    assert recommendation.reorder_point > 0
    assert recommendation.suggested_order_qty >= 0
    assert recommendation.requires_approval is True
    assert any("confidence" in reason.lower() for reason in recommendation.reasons)
