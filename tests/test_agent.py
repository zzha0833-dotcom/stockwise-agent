from __future__ import annotations

import json

from stockwise.schemas import ApprovalRequest, RunRequest, RunStatus
from stockwise.service import StockWiseService


def test_end_to_end_agent_pauses_and_resumes(settings):
    service = StockWiseService(settings)
    dataset = service.ensure_demo_dataset(items=2, stores=1)
    run = service.submit_run(
        RunRequest(
            dataset_id=dataset.id,
            item_limit=2,
            horizon=14,
            experiment_budget=2,
            approval_amount_threshold=0,
            random_seed=12,
        ),
        synchronous=True,
    )
    run = service.storage.get_run(run.id)
    assert run is not None
    assert run.status == RunStatus.AWAITING_APPROVAL.value
    service.approve(
        run.id, ApprovalRequest(decision="approve", note="test approval"), synchronous=True
    )
    completed = service.storage.get_run(run.id)
    assert completed is not None
    assert completed.status == RunStatus.COMPLETED.value
    result = json.loads(completed.result_json)
    assert result["forecast"]["points"]
    assert result["recommendations"]
    assert all(
        item["approval_status"] in {"approved", "not_required"}
        for item in result["recommendations"]
    )


def test_agent_failure_is_persisted(settings):
    service = StockWiseService(settings)
    dataset = service.ensure_demo_dataset(items=1, stores=1)
    run = service.submit_run(
        RunRequest(dataset_id=dataset.id, item_limit=1, horizon=56, experiment_budget=1),
        synchronous=True,
    )
    record = service.storage.get_run(run.id)
    assert record is not None
    assert record.status == RunStatus.FAILED.value
    assert record.error


def test_rejected_approval_resumes_and_marks_recommendations(settings):
    service = StockWiseService(settings)
    dataset = service.ensure_demo_dataset(items=1, stores=1)
    run = service.submit_run(
        RunRequest(
            dataset_id=dataset.id,
            item_limit=1,
            horizon=14,
            experiment_budget=1,
            approval_amount_threshold=0,
        ),
        synchronous=True,
    )
    assert service.storage.get_run(run.id).status == RunStatus.AWAITING_APPROVAL.value
    service.approve(
        run.id, ApprovalRequest(decision="reject", note="budget owner rejected"), synchronous=True
    )
    rejected = service.storage.get_run(run.id)
    assert rejected.status == RunStatus.REJECTED.value
    result = json.loads(rejected.result_json)
    assert all(
        item["approval_status"] in {"rejected", "not_required"}
        for item in result["recommendations"]
    )
