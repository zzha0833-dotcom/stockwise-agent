from __future__ import annotations

import time

from fastapi.testclient import TestClient

from stockwise.api import create_app
from stockwise.schemas import RunRequest, RunStatus


def test_api_health_and_run_lifecycle(settings):
    app = create_app(settings)
    client = TestClient(app)
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["inventory_data"] == "simulated_with_fixed_seed"

    datasets = client.get("/api/v1/datasets").json()
    assert datasets
    response = client.post(
        "/api/v1/runs",
        headers={"X-StockWise-Sync": "true"},
        json={
            "dataset_id": datasets[0]["id"],
            "item_limit": 2,
            "horizon": 14,
            "experiment_budget": 2,
            "approval_amount_threshold": 0,
            "random_seed": 3,
        },
    )
    assert response.status_code == 202
    run = response.json()
    assert run["status"] == "awaiting_approval"
    approval = client.post(
        f"/api/v1/runs/{run['id']}/approval",
        headers={"X-StockWise-Sync": "true"},
        json={"decision": "approve", "note": "API test"},
    )
    assert approval.status_code == 202
    results = client.get(f"/api/v1/runs/{run['id']}/results")
    assert results.status_code == 200
    assert results.json()["run"]["status"] == "completed"
    assert client.get(f"/api/v1/runs/{run['id']}/report").status_code == 200
    assert client.get(f"/api/v1/runs/{run['id']}/report?format=json").status_code == 200
    assert client.get(f"/api/v1/runs/{run['id']}/report?format=csv").status_code == 200


def test_csv_upload_and_validation(settings, retail_frame):
    client = TestClient(create_app(settings))
    response = client.post(
        "/api/v1/datasets?name=Uploaded%20sample",
        files={"file": ("retail.csv", retail_frame.write_csv().encode(), "text/csv")},
    )
    assert response.status_code == 201
    assert response.json()["profile"]["rows"] == retail_frame.height

    invalid = client.post(
        "/api/v1/datasets?name=Invalid",
        files={"file": ("broken.csv", b"date,sales\n2026-01-01,1\n", "text/csv")},
    )
    assert invalid.status_code == 422


def test_two_background_runs_can_progress_concurrently(settings):
    app = create_app(settings)
    service = app.state.service
    dataset = service.ensure_demo_dataset(items=2, stores=1)
    runs = [
        service.submit_run(
            RunRequest(
                dataset_id=dataset.id,
                item_limit=1,
                horizon=14,
                experiment_budget=1,
                approval_amount_threshold=1_000_000,
                random_seed=seed,
            ),
            synchronous=False,
        )
        for seed in (101, 202)
    ]
    deadline = time.monotonic() + 20
    terminal = {
        RunStatus.COMPLETED.value,
        RunStatus.AWAITING_APPROVAL.value,
        RunStatus.REJECTED.value,
        RunStatus.FAILED.value,
    }
    while time.monotonic() < deadline:
        records = [service.storage.get_run(run.id) for run in runs]
        if all(record and record.status in terminal for record in records):
            break
        time.sleep(0.05)

    records = [service.storage.get_run(run.id) for run in runs]
    assert all(record and record.status != RunStatus.FAILED.value for record in records)
    assert all(record and record.status in terminal for record in records)
