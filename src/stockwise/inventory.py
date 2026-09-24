from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockwise.schemas import ReplenishmentRecommendation, RiskLevel


@dataclass(frozen=True)
class InventoryScenario:
    on_hand: float
    lead_time_days: int
    unit_cost: float
    holding_cost_rate: float
    stockout_penalty_rate: float


def _series_rng(store_id: str, item_id: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha256(f"{store_id}|{item_id}|{seed}".encode()).digest()
    number = int.from_bytes(digest[:8], "big") % (2**32)
    return np.random.default_rng(number)


def simulate_inventory_scenario(
    store_id: str, item_id: str, average_daily_demand: float, *, seed: int = 42
) -> InventoryScenario:
    rng = _series_rng(store_id, item_id, seed)
    lead_time = int(rng.integers(3, 15))
    on_hand_days = float(rng.uniform(3.0, 22.0))
    return InventoryScenario(
        on_hand=round(max(0.0, average_daily_demand * on_hand_days), 2),
        lead_time_days=lead_time,
        unit_cost=round(float(rng.uniform(2.5, 45.0)), 2),
        holding_cost_rate=round(float(rng.uniform(0.01, 0.04)), 4),
        stockout_penalty_rate=round(float(rng.uniform(0.25, 0.7)), 4),
    )


def build_recommendations(
    forecast: pd.DataFrame,
    *,
    confidence: float,
    residual_std: float,
    approval_amount_threshold: float,
    anomaly_count: int = 0,
    seed: int = 42,
) -> list[ReplenishmentRecommendation]:
    recommendations: list[ReplenishmentRecommendation] = []
    z_value = 1.645  # approximately 95% one-sided service level
    for (store_id, item_id), group in forecast.groupby(["store_id", "item_id"], sort=True):
        ordered = group.sort_values("date")
        average_daily = float(ordered["prediction"].mean())
        scenario = simulate_inventory_scenario(store_id, item_id, average_daily, seed=seed)
        lead_time_demand = float(ordered.head(scenario.lead_time_days)["prediction"].sum())
        safety_stock = z_value * max(residual_std, 0.1) * math.sqrt(scenario.lead_time_days)
        reorder_point = lead_time_demand + safety_stock
        horizon_demand = float(ordered["prediction"].sum())
        target_stock = horizon_demand + safety_stock
        order_qty = max(0.0, target_stock - scenario.on_hand)
        order_value = order_qty * scenario.unit_cost
        expected_shortage = max(0.0, lead_time_demand - scenario.on_hand)
        projected_leftover = max(0.0, scenario.on_hand + order_qty - horizon_demand)
        holding_cost = projected_leftover * scenario.unit_cost * scenario.holding_cost_rate
        stockout_cost = expected_shortage * scenario.unit_cost * scenario.stockout_penalty_rate
        fill_rate = 1.0 if lead_time_demand <= 1e-9 else min(1.0, scenario.on_hand / lead_time_demand)

        reasons: list[str] = []
        if scenario.on_hand < lead_time_demand:
            risk = RiskLevel.HIGH
            reasons.append("On-hand inventory is below forecast lead-time demand")
        elif scenario.on_hand < reorder_point:
            risk = RiskLevel.MEDIUM
            reasons.append("On-hand inventory is below the calculated reorder point")
        else:
            risk = RiskLevel.LOW
            reasons.append("On-hand inventory covers the lead-time demand and safety stock")
        requires_approval = False
        if order_value >= approval_amount_threshold:
            requires_approval = True
            reasons.append("Suggested order value exceeds the approval threshold")
        if confidence < 0.65:
            requires_approval = True
            reasons.append("Forecast confidence is below 0.65")
        if anomaly_count > 0 and risk == RiskLevel.HIGH:
            requires_approval = True
            reasons.append("High-risk recommendation coincides with detected data anomalies")

        recommendations.append(
            ReplenishmentRecommendation(
                store_id=store_id,
                item_id=item_id,
                on_hand=round(scenario.on_hand, 2),
                lead_time_days=scenario.lead_time_days,
                forecast_demand=round(horizon_demand, 2),
                safety_stock=round(safety_stock, 2),
                reorder_point=round(reorder_point, 2),
                suggested_order_qty=round(order_qty, 2),
                unit_cost=round(scenario.unit_cost, 2),
                order_value=round(order_value, 2),
                projected_holding_cost=round(holding_cost, 2),
                projected_stockout_cost=round(stockout_cost, 2),
                projected_fill_rate=round(fill_rate, 4),
                risk_level=risk,
                confidence=round(confidence, 4),
                requires_approval=requires_approval,
                approval_status="pending" if requires_approval else "not_required",
                reasons=reasons,
                evidence={
                    "lead_time_demand": round(lead_time_demand, 2),
                    "horizon_demand": round(horizon_demand, 2),
                    "residual_std": round(residual_std, 4),
                    "approval_amount_threshold": round(approval_amount_threshold, 2),
                    "inventory_source": "simulated_with_fixed_seed",
                },
            )
        )
    return recommendations


def aggregate_inventory_metrics(
    recommendations: list[ReplenishmentRecommendation],
) -> dict[str, float | int | str]:
    if not recommendations:
        return {
            "recommendations": 0,
            "projected_fill_rate": 0.0,
            "projected_holding_cost": 0.0,
            "projected_stockout_cost": 0.0,
            "total_order_value": 0.0,
            "approval_rate": 0.0,
            "inventory_data": "simulated",
        }
    return {
        "recommendations": len(recommendations),
        "projected_fill_rate": round(
            float(np.mean([item.projected_fill_rate for item in recommendations])), 4
        ),
        "projected_holding_cost": round(
            sum(item.projected_holding_cost for item in recommendations), 2
        ),
        "projected_stockout_cost": round(
            sum(item.projected_stockout_cost for item in recommendations), 2
        ),
        "total_order_value": round(sum(item.order_value for item in recommendations), 2),
        "approval_rate": round(
            sum(item.requires_approval for item in recommendations) / len(recommendations), 4
        ),
        "inventory_data": "simulated",
    }

