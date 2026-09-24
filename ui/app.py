from __future__ import annotations

import os
import time
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API_URL = os.getenv("STOCKWISE_API_URL", "http://127.0.0.1:8000/api/v1")


def api_get(path: str) -> Any:
    response = httpx.get(f"{API_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any]) -> Any:
    response = httpx.post(f"{API_URL}{path}", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="StockWise Agent", page_icon="📦", layout="wide")
st.title("StockWise Retail Forecasting Agent")
st.caption("Agentic demand forecasting, replenishment decisions, and human approval")

try:
    health = api_get("/health")
except Exception as exc:
    st.error(f"API is unavailable at {API_URL}: {exc}")
    st.stop()

if health["llm_mode"] == "mock":
    st.warning("Mock LLM mode is active. Set LLM_MODE=api and provide an API key for a live model.")
st.info("Inventory, lead time, and cost inputs are simulated with a fixed seed.")

datasets = api_get("/datasets")
dataset_lookup = {f"{item['name']} · {item['profile']['series']} series": item for item in datasets}

with st.sidebar:
    st.header("Run configuration")
    selected_label = st.selectbox("Dataset", list(dataset_lookup))
    selected = dataset_lookup[selected_label]
    item_limit = st.slider("Item limit", 1, min(100, selected["profile"]["items"]), min(8, selected["profile"]["items"]))
    threshold = st.number_input("Approval amount threshold", min_value=0.0, value=5000.0, step=500.0)
    if st.button("Start Agent run", type="primary", use_container_width=True):
        run = api_post(
            "/runs",
            {
                "dataset_id": selected["id"],
                "item_limit": item_limit,
                "horizon": 28,
                "experiment_budget": 3,
                "approval_amount_threshold": threshold,
                "random_seed": 42,
            },
        )
        st.session_state.run_id = run["id"]
        st.rerun()

st.subheader("Dataset profile")
profile_cols = st.columns(5)
for column, (label, value) in zip(
    profile_cols,
    [
        ("Rows", selected["profile"]["rows"]),
        ("Stores", selected["profile"]["stores"]),
        ("Items", selected["profile"]["items"]),
        ("Series", selected["profile"]["series"]),
        ("Zero-demand ratio", f"{selected['profile']['zero_demand_ratio']:.1%}"),
    ],
    strict=True,
):
    column.metric(label, value)

run_id = st.session_state.get("run_id")
if not run_id:
    st.info("Start an Agent run from the sidebar to view the workflow.")
    st.stop()

run = api_get(f"/runs/{run_id}")
st.subheader(f"Agent run {run_id[:10]}")
st.write(f"Status: **{run['status']}** · Current node: `{run['current_node']}`")
events = api_get(f"/runs/{run_id}/events")
st.dataframe(
    pd.DataFrame(events)[["sequence", "node", "status", "message"]],
    use_container_width=True,
    hide_index=True,
)

if run["status"] in ("queued", "running"):
    if st.button("Refresh"):
        st.rerun()
    time.sleep(0.2)
    st.stop()

results = api_get(f"/runs/{run_id}/results")

if run["status"] == "awaiting_approval":
    st.warning("One or more recommendations require human approval.")
    left, right = st.columns(2)
    if left.button("Approve pending recommendations", type="primary"):
        api_post(f"/runs/{run_id}/approval", {"decision": "approve", "note": "Approved in UI"})
        st.rerun()
    if right.button("Reject pending recommendations"):
        api_post(f"/runs/{run_id}/approval", {"decision": "reject", "note": "Rejected in UI"})
        st.rerun()

if results.get("summary"):
    st.subheader("Grounded summary")
    st.write(results["summary"])

if results.get("forecast"):
    evaluations = pd.DataFrame(results["forecast"]["evaluations"])
    metrics = pd.json_normalize(evaluations["metrics"])
    model_table = pd.concat([evaluations[["model", "runtime_seconds"]], metrics], axis=1)
    st.subheader("Backtest comparison")
    st.dataframe(model_table, use_container_width=True, hide_index=True)

    points = pd.DataFrame(results["forecast"]["points"])
    points["series"] = points["store_id"] + " / " + points["item_id"]
    selected_series = st.selectbox("Forecast series", sorted(points["series"].unique()))
    chart = points[points["series"] == selected_series]
    history = pd.DataFrame(results["forecast"].get("history", []))
    history["series"] = history["store_id"] + " / " + history["item_id"]
    history = history[history["series"] == selected_series]
    figure = px.line(
        chart, x="date", y=["prediction", "lower", "upper"], title="History and 28-day forecast"
    )
    figure.add_scatter(
        x=history["date"], y=history["sales"], mode="lines", name="history"
    )
    anomalies = history[history["anomaly"]]
    if not anomalies.empty:
        figure.add_scatter(
            x=anomalies["date"], y=anomalies["sales"], mode="markers", name="anomaly"
        )
    st.plotly_chart(figure, use_container_width=True)

recommendations = pd.DataFrame(results.get("recommendations", []))
if not recommendations.empty:
    st.subheader("Replenishment recommendations")
    risk_filter = st.multiselect(
        "Risk level", sorted(recommendations["risk_level"].unique()), default=sorted(recommendations["risk_level"].unique())
    )
    visible = recommendations[recommendations["risk_level"].isin(risk_filter)]
    st.dataframe(
        visible[
            [
                "store_id",
                "item_id",
                "risk_level",
                "on_hand",
                "forecast_demand",
                "suggested_order_qty",
                "order_value",
                "approval_status",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

if run["status"] in ("completed", "rejected"):
    report_cols = st.columns(3)
    report_cols[0].link_button("Download HTML", f"{API_URL}/runs/{run_id}/report?format=html")
    report_cols[1].link_button("Download JSON", f"{API_URL}/runs/{run_id}/report?format=json")
    report_cols[2].link_button("Download CSV", f"{API_URL}/runs/{run_id}/report?format=csv")

