from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelName(StrEnum):
    SEASONAL_NAIVE = "seasonal_naive"
    CROSTON_SBA = "croston_sba"
    LIGHTGBM_GLOBAL = "lightgbm_global"


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DatasetProfile(BaseModel):
    rows: int
    stores: int
    items: int
    series: int
    start_date: str
    end_date: str
    missing_sales: int = 0
    negative_sales: int = 0
    duplicate_keys: int = 0
    zero_demand_ratio: float = 0.0
    anomaly_count: int = 0
    warnings: list[str] = Field(default_factory=list)


class RunRequest(BaseModel):
    dataset_id: str
    stores: list[str] | None = None
    item_limit: int = Field(default=30, ge=1, le=100)
    horizon: int = Field(default=28, ge=7, le=56)
    experiment_budget: int = Field(default=3, ge=1, le=4)
    approval_amount_threshold: float = Field(default=5000.0, ge=0)
    random_seed: int = Field(default=42, ge=0)


class ExperimentPlan(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    models: list[ModelName]
    backtest_folds: int = Field(default=3, ge=1, le=3)
    horizon: int = Field(default=28, ge=7, le=56)
    max_iterations: int = Field(default=1, ge=0, le=1)
    rationale: str = ""

    @field_validator("models")
    @classmethod
    def models_must_be_unique(cls, value: list[ModelName]) -> list[ModelName]:
        if not value:
            raise ValueError("At least one candidate model is required")
        if len(set(value)) != len(value):
            raise ValueError("Candidate models must be unique")
        return value


class MetricSet(BaseModel):
    wmape: float
    mae: float
    mase: float
    interval_coverage: float | None = None


class ModelEvaluation(BaseModel):
    model: ModelName
    metrics: MetricSet
    residual_quantile_90: float
    folds_completed: int
    runtime_seconds: float
    errors: list[str] = Field(default_factory=list)


class ForecastPoint(BaseModel):
    date: str
    store_id: str
    item_id: str
    prediction: float
    lower: float
    upper: float


class ForecastResult(BaseModel):
    selected_model: ModelName
    evaluations: list[ModelEvaluation]
    confidence: float
    points: list[ForecastPoint]


class ReplenishmentRecommendation(BaseModel):
    store_id: str
    item_id: str
    on_hand: float
    lead_time_days: int
    forecast_demand: float
    safety_stock: float
    reorder_point: float
    suggested_order_qty: float
    unit_cost: float
    order_value: float
    projected_holding_cost: float
    projected_stockout_cost: float
    projected_fill_rate: float
    risk_level: RiskLevel
    confidence: float
    requires_approval: bool
    approval_status: Literal["not_required", "pending", "approved", "rejected"]
    reasons: list[str]
    evidence: dict[str, float | int | str]


class ApprovalRequest(BaseModel):
    decision: Literal["approve", "reject"]
    note: str = Field(default="", max_length=500)


class EventView(BaseModel):
    sequence: int
    node: str
    status: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class RunView(BaseModel):
    id: str
    dataset_id: str
    status: RunStatus
    current_node: str
    fallback_used: bool
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class RunResults(BaseModel):
    run: RunView
    profile: DatasetProfile | None = None
    plan: ExperimentPlan | None = None
    forecast: ForecastResult | None = None
    recommendations: list[ReplenishmentRecommendation] = Field(default_factory=list)
    summary: str | None = None
    evaluation: dict[str, float | int | str] = Field(default_factory=dict)

