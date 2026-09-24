from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
import pandas as pd
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from stockwise.config import Settings
from stockwise.data import load_dataset, subset_frame, validate_and_profile
from stockwise.forecasting import (
    confidence_from_evaluation,
    forecast_future,
    relative_improvement,
    rolling_backtest,
    select_best_model,
)
from stockwise.inventory import aggregate_inventory_metrics, build_recommendations
from stockwise.llm import LLMService, guard_plan
from stockwise.schemas import (
    ExperimentPlan,
    ModelEvaluation,
    ModelName,
    ReplenishmentRecommendation,
)
from stockwise.storage import Storage


class AgentState(TypedDict, total=False):
    run_id: str
    dataset_path: str
    request: dict[str, Any]
    profile: dict[str, Any]
    plan: dict[str, Any]
    iteration: int
    continue_experiment: bool
    evaluations: list[dict[str, Any]]
    residual_std: float
    best_model: str
    confidence: float
    history_points: list[dict[str, Any]]
    forecast_points: list[dict[str, Any]]
    recommendations: list[dict[str, Any]]
    approval_required: bool
    approval_decision: str
    summary: str
    evaluation: dict[str, Any]
    fallback_used: bool
    errors: list[str]


class StockWiseAgent:
    def __init__(self, settings: Settings, storage: Storage):
        self.settings = settings
        self.storage = storage
        self.llm = LLMService(settings)
        self.connection = sqlite3.connect(settings.checkpoint_db, check_same_thread=False)
        self.checkpointer = SqliteSaver(self.connection)
        self.graph = self._build_graph()

    def close(self) -> None:
        self.connection.close()

    def _event(
        self,
        state: AgentState,
        node: str,
        status: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        run_id = state["run_id"]
        self.storage.add_event(
            run_id, node=node, status=status, message=message, payload=payload or {}
        )
        self.storage.update_run(run_id, current_node=node)

    def _frame(self, state: AgentState):
        request = state["request"]
        return subset_frame(
            load_dataset(Path(state["dataset_path"])),
            item_limit=int(request["item_limit"]),
            stores=request.get("stores"),
        )

    def validate_input(self, state: AgentState) -> dict[str, Any]:
        frame = self._frame(state)
        profile = validate_and_profile(frame)
        self._event(state, "validate_input", "completed", "Dataset validation completed")
        return {"profile": profile.model_dump(), "errors": []}

    def analyze_data(self, state: AgentState) -> dict[str, Any]:
        profile = state["profile"]
        payload = {
            "zero_demand_ratio": profile["zero_demand_ratio"],
            "anomaly_count": profile["anomaly_count"],
            "warnings": profile["warnings"],
        }
        self._event(
            state, "analyze_data", "completed", "Demand pattern analysis completed", payload
        )
        return {}

    def create_plan(self, state: AgentState) -> dict[str, Any]:
        from stockwise.schemas import DatasetProfile

        request = state["request"]
        plan, fallback = self.llm.create_plan(
            DatasetProfile.model_validate(state["profile"]),
            horizon=int(request["horizon"]),
            experiment_budget=int(request["experiment_budget"]),
        )
        self._event(
            state,
            "create_plan",
            "fallback" if fallback else "completed",
            "Experiment plan created",
            {"models": [str(item) for item in plan.models]},
        )
        return {"plan": plan.model_dump(mode="json"), "fallback_used": fallback, "iteration": 0}

    def guard_plan(self, state: AgentState) -> dict[str, Any]:
        request = state["request"]
        plan = guard_plan(
            ExperimentPlan.model_validate(state["plan"]),
            budget=int(request["experiment_budget"]),
            horizon=int(request["horizon"]),
        )
        self._event(
            state,
            "guard_plan",
            "completed",
            "Experiment plan passed the allow-list guard",
            {"models": [str(item) for item in plan.models]},
        )
        return {"plan": plan.model_dump(mode="json")}

    def run_backtest(self, state: AgentState) -> dict[str, Any]:
        frame = self._frame(state)
        plan = ExperimentPlan.model_validate(state["plan"])
        request = state["request"]
        output = rolling_backtest(
            frame,
            plan.models,
            horizon=plan.horizon,
            folds=plan.backtest_folds,
            seed=int(request["random_seed"]),
        )
        best = select_best_model(output.evaluations)
        residuals = output.residuals[ModelName(best.model).value]
        fallback = state.get("fallback_used", False) or any(
            evaluation.errors for evaluation in output.evaluations
        )
        self._event(
            state,
            "run_backtest",
            "completed",
            "Rolling-origin backtest completed",
            {"best_model": str(best.model), "wmape": best.metrics.wmape},
        )
        return {
            "evaluations": [item.model_dump(mode="json") for item in output.evaluations],
            "best_model": ModelName(best.model).value,
            "residual_std": float(np.std(residuals)) if len(residuals) else 0.0,
            "fallback_used": fallback,
        }

    def reflect(self, state: AgentState) -> dict[str, Any]:
        plan = ExperimentPlan.model_validate(state["plan"])
        evaluations = [ModelEvaluation.model_validate(item) for item in state["evaluations"]]
        current = {ModelName(item) for item in plan.models}
        available = [item for item in ModelName if item not in current]
        request = state["request"]
        should_continue = (
            state.get("iteration", 0) < plan.max_iterations
            and len(plan.models) < int(request["experiment_budget"])
            and bool(available)
            and select_best_model(evaluations).metrics.wmape > 0.25
        )
        if should_continue:
            updated = plan.model_copy(update={"models": [*plan.models, available[0]]})
            self._event(
                state,
                "reflect",
                "continued",
                "One additional allow-listed model was added after reflection",
                {"added_model": available[0].value},
            )
            return {
                "plan": updated.model_dump(mode="json"),
                "iteration": state.get("iteration", 0) + 1,
                "continue_experiment": True,
            }
        self._event(state, "reflect", "completed", "Experiment stopping criteria reached")
        return {"continue_experiment": False}

    @staticmethod
    def reflection_route(state: AgentState) -> Literal["iterate", "forecast"]:
        return "iterate" if state.get("continue_experiment") else "forecast"

    def generate_forecast(self, state: AgentState) -> dict[str, Any]:
        frame = self._frame(state)
        evaluations = [ModelEvaluation.model_validate(item) for item in state["evaluations"]]
        best = select_best_model(evaluations)
        request = state["request"]
        forecast = forecast_future(
            frame,
            best.model,
            horizon=int(request["horizon"]),
            residual_quantile_90=best.residual_quantile_90,
            seed=int(request["random_seed"]),
        )
        confidence = confidence_from_evaluation(best)
        points = [
            {
                "date": str(pd.Timestamp(row.date).date()),
                "store_id": str(row.store_id),
                "item_id": str(row.item_id),
                "prediction": round(float(row.prediction), 4),
                "lower": round(float(row.lower), 4),
                "upper": round(float(row.upper), 4),
            }
            for row in forecast.itertuples(index=False)
        ]
        history = frame.sort(["store_id", "item_id", "date"]).group_by(
            ["store_id", "item_id"], maintain_order=True
        ).tail(90)
        mean = float(history["sales"].mean() or 0)
        std = float(history["sales"].std() or 0)
        anomaly_threshold = mean + 4 * std
        history_points = [
            {
                "date": str(row[0]),
                "store_id": str(row[1]),
                "item_id": str(row[2]),
                "sales": round(float(row[3]), 4),
                "anomaly": bool(float(row[3]) > anomaly_threshold) if std > 0 else False,
            }
            for row in history.select("date", "store_id", "item_id", "sales").iter_rows()
        ]
        self._event(
            state,
            "generate_forecast",
            "completed",
            "Future demand forecast generated",
            {"points": len(points), "confidence": confidence},
        )
        return {
            "history_points": history_points,
            "forecast_points": points,
            "confidence": confidence,
            "best_model": ModelName(best.model).value,
        }

    def inventory_decision(self, state: AgentState) -> dict[str, Any]:
        request = state["request"]
        forecast = pd.DataFrame(state["forecast_points"])
        recommendations = build_recommendations(
            forecast,
            confidence=float(state["confidence"]),
            residual_std=float(state["residual_std"]),
            approval_amount_threshold=float(request["approval_amount_threshold"]),
            anomaly_count=int(state["profile"]["anomaly_count"]),
            seed=int(request["random_seed"]),
        )
        payload = [item.model_dump(mode="json") for item in recommendations]
        self.storage.replace_recommendations(state["run_id"], payload)
        self._event(
            state,
            "inventory_decision",
            "completed",
            "Deterministic replenishment recommendations calculated",
            {"recommendations": len(payload)},
        )
        return {"recommendations": payload}

    def classify_risk(self, state: AgentState) -> dict[str, Any]:
        pending = [item for item in state["recommendations"] if item["requires_approval"]]
        self._event(
            state,
            "classify_risk",
            "completed",
            "Approval routing completed",
            {"approval_required": len(pending)},
        )
        return {"approval_required": bool(pending)}

    def approval_gate(self, state: AgentState) -> dict[str, Any]:
        if not state.get("approval_required"):
            self._event(state, "approval_gate", "skipped", "No recommendation requires approval")
            return {"approval_decision": "not_required"}
        self._event(
            state,
            "approval_gate",
            "waiting",
            "Workflow paused for human approval",
            {
                "pending": sum(
                    bool(item["requires_approval"]) for item in state["recommendations"]
                )
            },
        )
        response = interrupt(
            {
                "run_id": state["run_id"],
                "message": "Approve or reject the pending replenishment recommendations",
            }
        )
        decision = str(response.get("decision", "reject")) if isinstance(response, dict) else "reject"
        updated = []
        for item in state["recommendations"]:
            if item["requires_approval"]:
                item = {**item, "approval_status": "approved" if decision == "approve" else "rejected"}
            updated.append(item)
        self._event(
            state,
            "approval_gate",
            "completed",
            f"Human decision recorded: {decision}",
        )
        return {"approval_decision": decision, "recommendations": updated}

    def generate_report(self, state: AgentState) -> dict[str, Any]:
        evaluations = [ModelEvaluation.model_validate(item) for item in state["evaluations"]]
        best = select_best_model(evaluations)
        baseline = next(
            (item for item in evaluations if ModelName(item.model) == ModelName.SEASONAL_NAIVE),
            evaluations[0],
        )
        recommendations = [
            ReplenishmentRecommendation.model_validate(item) for item in state["recommendations"]
        ]
        inventory_metrics = aggregate_inventory_metrics(recommendations)
        evidence = {
            "selected_model": ModelName(best.model).value,
            "wmape": best.metrics.wmape,
            "recommendation_count": len(recommendations),
            "approval_count": sum(item.requires_approval for item in recommendations),
        }
        summary, fallback = self.llm.grounded_explanation(evidence)
        evaluation = {
            "selected_model": ModelName(best.model).value,
            "wmape": best.metrics.wmape,
            "mae": best.metrics.mae,
            "mase": best.metrics.mase,
            "baseline_relative_improvement": relative_improvement(best, baseline),
            "agent_fallback_used": bool(state.get("fallback_used") or fallback),
            **inventory_metrics,
        }
        self._event(state, "generate_report", "completed", "Grounded report generated")
        return {
            "summary": summary,
            "evaluation": evaluation,
            "fallback_used": bool(state.get("fallback_used") or fallback),
        }

    def persist(self, state: AgentState) -> dict[str, Any]:
        self._event(state, "persist", "completed", "Agent state and results are ready to persist")
        return {}

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("validate_input", self.validate_input)
        builder.add_node("analyze_data", self.analyze_data)
        builder.add_node("create_plan", self.create_plan)
        builder.add_node("guard_plan", self.guard_plan)
        builder.add_node("run_backtest", self.run_backtest)
        builder.add_node("reflect", self.reflect)
        builder.add_node("generate_forecast", self.generate_forecast)
        builder.add_node("inventory_decision", self.inventory_decision)
        builder.add_node("classify_risk", self.classify_risk)
        builder.add_node("approval_gate", self.approval_gate)
        builder.add_node("generate_report", self.generate_report)
        builder.add_node("persist", self.persist)
        builder.add_edge(START, "validate_input")
        builder.add_edge("validate_input", "analyze_data")
        builder.add_edge("analyze_data", "create_plan")
        builder.add_edge("create_plan", "guard_plan")
        builder.add_edge("guard_plan", "run_backtest")
        builder.add_edge("run_backtest", "reflect")
        builder.add_conditional_edges(
            "reflect",
            self.reflection_route,
            {"iterate": "run_backtest", "forecast": "generate_forecast"},
        )
        builder.add_edge("generate_forecast", "inventory_decision")
        builder.add_edge("inventory_decision", "classify_risk")
        builder.add_edge("classify_risk", "approval_gate")
        builder.add_edge("approval_gate", "generate_report")
        builder.add_edge("generate_report", "persist")
        builder.add_edge("persist", END)
        return builder.compile(checkpointer=self.checkpointer)

    @staticmethod
    def _config(run_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": run_id}}

    def start(self, initial_state: AgentState) -> dict[str, Any]:
        return self.graph.invoke(initial_state, config=self._config(initial_state["run_id"]))

    def resume(self, run_id: str, decision: str, note: str = "") -> dict[str, Any]:
        return self.graph.invoke(
            Command(resume={"decision": decision, "note": note}), config=self._config(run_id)
        )

    def snapshot(self, run_id: str) -> dict[str, Any]:
        snapshot = self.graph.get_state(self._config(run_id))
        return dict(snapshot.values)


def state_to_result(state: dict[str, Any]) -> dict[str, Any]:
    evaluations = state.get("evaluations", [])
    forecast = None
    if state.get("best_model") and state.get("forecast_points"):
        forecast = {
            "selected_model": state["best_model"],
            "evaluations": evaluations,
            "confidence": state.get("confidence", 0.0),
            "history": state.get("history_points", []),
            "points": state["forecast_points"],
        }
    return {
        "profile": state.get("profile"),
        "plan": state.get("plan"),
        "forecast": forecast,
        "recommendations": state.get("recommendations", []),
        "summary": state.get("summary"),
        "evaluation": state.get("evaluation", {}),
        "fallback_used": state.get("fallback_used", False),
        "approval_decision": state.get("approval_decision"),
    }


def has_interrupt(result: dict[str, Any]) -> bool:
    return bool(result.get("__interrupt__"))
