from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from stockwise.agent import AgentState, StockWiseAgent, has_interrupt, state_to_result
from stockwise.config import Settings
from stockwise.data import generate_synthetic_retail_data, validate_and_profile, write_dataset
from stockwise.report import write_report
from stockwise.schemas import ApprovalRequest, RunRequest, RunStatus
from stockwise.storage import DatasetRecord, RunRecord, Storage
from stockwise.tracking import log_mlflow_run


class StockWiseService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.ensure_directories()
        self.storage = Storage(settings.database_url)
        self.storage.initialize()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="stockwise")
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _run_lock(self, run_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(run_id, threading.Lock())

    def ensure_demo_dataset(self, *, items: int = 8, stores: int = 2) -> DatasetRecord:
        name = f"Synthetic retail demo {items}x{stores}"
        for record in self.storage.list_datasets():
            if record.name == name:
                return record
        path = self.settings.data_dir / "processed" / f"synthetic_demo_{items}x{stores}.parquet"
        frame = generate_synthetic_retail_data(items=items, stores=stores)
        profile = write_dataset(frame, path)
        return self.storage.register_dataset(name=name, path=path, profile=profile)

    def register_dataset_file(self, *, name: str, path: Path) -> DatasetRecord:
        from stockwise.data import load_dataset

        profile = validate_and_profile(load_dataset(path))
        return self.storage.register_dataset(name=name, path=path, profile=profile)

    def submit_run(self, request: RunRequest, *, synchronous: bool | None = None) -> RunRecord:
        dataset = self.storage.get_dataset(request.dataset_id)
        if dataset is None:
            raise KeyError(f"Unknown dataset: {request.dataset_id}")
        run = self.storage.create_run(request)
        self.storage.add_event(
            run.id, node="queued", status="queued", message="Run accepted by the service"
        )
        should_sync = self.settings.sync_runs if synchronous is None else synchronous
        if should_sync:
            self._execute(run.id)
        else:
            self.executor.submit(self._execute, run.id)
        return self.storage.get_run(run.id) or run

    def _execute(self, run_id: str) -> None:
        lock = self._run_lock(run_id)
        if not lock.acquire(blocking=False):
            return
        agent: StockWiseAgent | None = None
        try:
            run = self.storage.get_run(run_id)
            if run is None:
                return
            dataset = self.storage.get_dataset(run.dataset_id)
            if dataset is None:
                raise KeyError(f"Unknown dataset: {run.dataset_id}")
            request = RunRequest.model_validate_json(run.request_json)
            self.storage.update_run(
                run_id, status=RunStatus.RUNNING.value, current_node="starting", error=None
            )
            agent = StockWiseAgent(self.settings, self.storage)
            result = agent.start(
                AgentState(
                    run_id=run_id,
                    dataset_path=dataset.path,
                    request=request.model_dump(mode="json"),
                    fallback_used=False,
                    errors=[],
                )
            )
            if has_interrupt(result):
                snapshot = agent.snapshot(run_id)
                partial = state_to_result(snapshot)
                self.storage.update_run(
                    run_id,
                    status=RunStatus.AWAITING_APPROVAL.value,
                    current_node="approval_gate",
                    result_json=json.dumps(partial, default=str),
                    fallback_used=bool(partial.get("fallback_used")),
                )
                return
            self._finish(run_id, result)
        except Exception as exc:
            self.storage.add_event(
                run_id,
                node="error",
                status="failed",
                message=f"{type(exc).__name__}: {exc}",
            )
            self.storage.update_run(
                run_id,
                status=RunStatus.FAILED.value,
                current_node="error",
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if agent is not None:
                agent.close()
            lock.release()

    def approve(
        self, run_id: str, approval: ApprovalRequest, *, synchronous: bool | None = None
    ) -> RunRecord:
        run = self.storage.get_run(run_id)
        if run is None:
            raise KeyError(f"Unknown run: {run_id}")
        if run.status != RunStatus.AWAITING_APPROVAL.value:
            raise ValueError("Run is not waiting for approval")
        self.storage.record_approval(run_id, approval.decision, approval.note)
        should_sync = self.settings.sync_runs if synchronous is None else synchronous
        if should_sync:
            self._resume(run_id, approval)
        else:
            self.executor.submit(self._resume, run_id, approval)
        return self.storage.get_run(run_id) or run

    def _resume(self, run_id: str, approval: ApprovalRequest) -> None:
        lock = self._run_lock(run_id)
        with lock:
            agent: StockWiseAgent | None = None
            try:
                self.storage.update_run(
                    run_id, status=RunStatus.RUNNING.value, current_node="approval_resume"
                )
                agent = StockWiseAgent(self.settings, self.storage)
                result = agent.resume(run_id, approval.decision, approval.note)
                self._finish(run_id, result, rejected=approval.decision == "reject")
            except Exception as exc:
                self.storage.update_run(
                    run_id,
                    status=RunStatus.FAILED.value,
                    current_node="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            finally:
                if agent is not None:
                    agent.close()

    def _finish(self, run_id: str, state: dict[str, Any], *, rejected: bool = False) -> None:
        result = state_to_result(state)
        paths = write_report(run_id, result, self.settings.artifact_dir)
        log_mlflow_run(self.settings, run_id, result, paths)
        self.storage.replace_recommendations(run_id, result.get("recommendations", []))
        self.storage.update_run(
            run_id,
            status=(RunStatus.REJECTED.value if rejected else RunStatus.COMPLETED.value),
            current_node="completed",
            result_json=json.dumps(result, default=str),
            fallback_used=bool(result.get("fallback_used")),
            report_path=str(paths["html"].resolve()),
            error=None,
        )
