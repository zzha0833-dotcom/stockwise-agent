from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from jinja2 import Template

from stockwise.schemas import ApprovalRequest, RunRequest, RunStatus
from stockwise.service import StockWiseService
from stockwise.storage import decode_json

EVALUATION_TEMPLATE = Template(
    """<!doctype html><html><head><meta charset="utf-8"><title>StockWise evaluation</title>
<style>body{font-family:Arial,sans-serif;max-width:900px;margin:2rem auto;color:#161616}
table{border-collapse:collapse;width:100%}th,td{border:1px solid #d9d9d9;padding:.6rem}
th{background:#203864;color:#fff}</style></head><body>
<h1>StockWise reproducible evaluation</h1><p>Inventory inputs are simulated.</p>
<table>{% for key,value in aggregate.items() %}<tr><th>{{ key }}</th><td>{{ value }}</td></tr>{% endfor %}</table>
<h2>Runs</h2><pre>{{ runs_json }}</pre></body></html>"""
)


def evaluate_dataset(
    service: StockWiseService,
    *,
    dataset_id: str,
    item_limit: int,
    seeds: list[int],
    output_dir: Path,
    dataset_label: str = "dataset",
) -> dict[str, Any]:
    dataset = service.storage.get_dataset(dataset_id)
    if dataset is None:
        raise KeyError(f"Unknown dataset: {dataset_id}")
    profile = decode_json(dataset.profile_json)
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        record = service.submit_run(
            RunRequest(
                dataset_id=dataset_id,
                item_limit=item_limit,
                horizon=28,
                experiment_budget=3,
                approval_amount_threshold=5000,
                random_seed=seed,
            ),
            synchronous=True,
        )
        record = service.storage.get_run(record.id) or record
        if record.status == RunStatus.AWAITING_APPROVAL.value:
            service.approve(
                record.id,
                ApprovalRequest(decision="approve", note="Reproducible evaluation approval"),
                synchronous=True,
            )
            record = service.storage.get_run(record.id) or record
        result = decode_json(record.result_json)
        runs.append(
            {
                "run_id": record.id,
                "status": record.status,
                "fallback_used": record.fallback_used,
                **result.get("evaluation", {}),
            }
        )
    completed = [item for item in runs if item["status"] == RunStatus.COMPLETED.value]
    numeric_keys = [
        "wmape",
        "mae",
        "mase",
        "baseline_relative_improvement",
        "projected_fill_rate",
        "approval_rate",
    ]
    aggregate: dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset_name": dataset.name,
        "dataset_rows": profile.get("rows", 0),
        "dataset_stores": profile.get("stores", 0),
        "dataset_items": profile.get("items", 0),
        "dataset_series": profile.get("series", 0),
        "runs": len(runs),
        "task_completion_rate": round(len(completed) / max(1, len(runs)), 4),
        "fallback_rate": round(sum(item["fallback_used"] for item in runs) / max(1, len(runs)), 4),
        "item_limit": item_limit,
        "inventory_data": "simulated",
    }
    for key in numeric_keys:
        values = [float(item[key]) for item in completed if key in item]
        if values:
            aggregate[f"mean_{key}"] = round(mean(values), 6)
    result = {
        "aggregate": aggregate,
        "dataset_profile": profile,
        "runs": runs,
        "seeds": seeds,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(
        character
        for character in dataset_label.lower()
        if character.isalnum() or character in "-_"
    )
    stem = f"evaluation_{safe_label}_{item_limit}items"
    json_path = output_dir / f"{stem}.json"
    html_path = output_dir / f"{stem}.html"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    html_path.write_text(
        EVALUATION_TEMPLATE.render(
            aggregate=aggregate, runs_json=json.dumps(runs, indent=2)
        ),
        encoding="utf-8",
    )
    return {**result, "json_path": str(json_path), "html_path": str(html_path)}
