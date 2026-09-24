from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from jinja2 import Template

REPORT_TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>StockWise Run {{ run_id }}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem auto; max-width: 1100px; color: #161616; }
    h1, h2 { color: #000; } .muted { color: #666; }
    table { width: 100%; border-collapse: collapse; margin: 1rem 0 2rem; }
    th, td { border: 1px solid #d9d9d9; padding: .55rem; text-align: left; }
    th { background: #203864; color: white; } tr:nth-child(even) { background: #f5f7fb; }
    .high { color: #a00000; font-weight: bold; } .medium { color: #9a5a00; }
    code { background: #f2f2f2; padding: .1rem .25rem; }
  </style>
</head>
<body>
  <h1>StockWise Forecast and Replenishment Report</h1>
  <p class="muted">Run {{ run_id }} · Inventory inputs are simulated with a fixed seed.</p>
  <h2>Grounded summary</h2><p>{{ summary }}</p>
  <h2>Forecast evaluation</h2>
  <table><tr><th>Model</th><th>WMAPE</th><th>MAE</th><th>MASE</th><th>Runtime</th></tr>
  {% for item in evaluations %}<tr><td>{{ item.model }}</td><td>{{ item.metrics.wmape }}</td>
  <td>{{ item.metrics.mae }}</td><td>{{ item.metrics.mase }}</td><td>{{ item.runtime_seconds }} s</td></tr>{% endfor %}</table>
  <h2>Replenishment recommendations</h2>
  <table><tr><th>Store</th><th>Item</th><th>Risk</th><th>On hand</th><th>Forecast</th><th>Order qty</th><th>Order value</th><th>Approval</th></tr>
  {% for item in recommendations %}<tr><td>{{ item.store_id }}</td><td>{{ item.item_id }}</td>
  <td class="{{ item.risk_level }}">{{ item.risk_level }}</td><td>{{ item.on_hand }}</td>
  <td>{{ item.forecast_demand }}</td><td>{{ item.suggested_order_qty }}</td><td>{{ item.order_value }}</td><td>{{ item.approval_status }}</td></tr>{% endfor %}</table>
  <h2>Run metrics</h2><pre>{{ evaluation_json }}</pre>
</body></html>"""
)


def write_report(run_id: str, result: dict[str, Any], artifact_dir: Path) -> dict[str, Path]:
    run_dir = artifact_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "result.json"
    html_path = run_dir / "report.html"
    csv_path = run_dir / "recommendations.csv"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    recommendations = result.get("recommendations", [])
    if recommendations:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(recommendations[0].keys()))
            writer.writeheader()
            for row in recommendations:
                writer.writerow(
                    {
                        key: json.dumps(value) if isinstance(value, (list, dict)) else value
                        for key, value in row.items()
                    }
                )
    else:
        csv_path.write_text("store_id,item_id\n", encoding="utf-8")
    html_path.write_text(
        REPORT_TEMPLATE.render(
            run_id=run_id,
            summary=result.get("summary", ""),
            evaluations=result.get("forecast", {}).get("evaluations", []),
            recommendations=recommendations,
            evaluation_json=json.dumps(result.get("evaluation", {}), indent=2),
        ),
        encoding="utf-8",
    )
    return {"json": json_path, "html": html_path, "csv": csv_path}

