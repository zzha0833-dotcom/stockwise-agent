# Architecture and safety boundaries

## Components

- **Data layer:** canonical CSV validation, deterministic synthetic fixtures, M5 import, Polars subset processing, and Parquet storage.
- **Forecast layer:** three allow-listed tools, rolling-origin evaluation, empirical intervals, and explicit fallback records.
- **Decision layer:** deterministic inventory scenarios and replenishment equations.
- **Agent layer:** LangGraph orchestration, a maximum of one reflection loop, SQLite checkpoints, and approval interrupts.
- **Product layer:** FastAPI, Streamlit, SQLite records, and generated reports.

## Agent state

The checkpointed state contains the dataset path, validated request, profile, structured plan, model evaluations, forecast points, replenishment recommendations, approval decision, grounded summary, and fallback flag. API keys and raw file contents are excluded.

## Trust boundaries

The LLM cannot execute arbitrary Python or SQL. It selects from a model allow-list and returns Pydantic-validated JSON. The plan guard injects Seasonal Naive as the baseline, enforces the user budget, limits backtesting to three folds, and limits reflection to one iteration.

Forecasts, metrics, inventory values, approval rules, and report numbers are produced by deterministic code. Any LLM explanation containing a number absent from the supplied evidence is discarded and replaced with a deterministic summary.

## Human approval

The graph pauses if an order exceeds the configured amount threshold, forecast confidence is below 0.65, or a high-risk recommendation coincides with a detected anomaly. A decision resumes the same LangGraph thread from its SQLite checkpoint.

## Data integrity

- Dates must parse successfully.
- Sales must be numeric and non-negative.
- `(date, store_id, item_id)` rows must be unique.
- Feature lags and rolling statistics use shifted values so the current target is never included.
- Synthetic inventory values are deterministic and explicitly labeled.

