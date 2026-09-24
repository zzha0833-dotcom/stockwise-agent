from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from stockwise.config import Settings, get_settings
from stockwise.data import load_retail_csv, write_dataset
from stockwise.schemas import ApprovalRequest, EventView, RunRequest, RunStatus, RunView
from stockwise.service import StockWiseService
from stockwise.storage import decode_json


def _run_view(record) -> RunView:
    return RunView(
        id=record.id,
        dataset_id=record.dataset_id,
        status=RunStatus(record.status),
        current_node=record.current_node,
        fallback_used=record.fallback_used,
        error=record.error,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    service = StockWiseService(settings)
    demo = service.ensure_demo_dataset()
    app = FastAPI(
        title="StockWise API",
        version="0.1.0",
        description="Agentic retail demand forecasting and replenishment decisions",
    )
    app.state.service = service
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_credentials=False,
        allow_methods=["*"] ,
        allow_headers=["*"],
    )

    @app.get("/api/v1/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": "0.1.0",
            "llm_mode": settings.llm_mode,
            "inventory_data": "simulated_with_fixed_seed",
            "demo_dataset_id": demo.id,
        }

    @app.get("/api/v1/datasets")
    def list_datasets() -> list[dict[str, Any]]:
        return [
            {
                "id": item.id,
                "name": item.name,
                "profile": decode_json(item.profile_json),
                "created_at": item.created_at.isoformat(),
            }
            for item in service.storage.list_datasets()
        ]

    @app.post("/api/v1/datasets", status_code=201)
    async def upload_dataset(name: str, file: UploadFile = File(...)) -> dict[str, Any]:
        suffix = Path(file.filename or "dataset.csv").suffix.lower()
        if suffix != ".csv":
            raise HTTPException(status_code=400, detail="Only canonical CSV uploads are supported")
        upload_path = settings.data_dir / "uploads" / f"{uuid.uuid4().hex}.csv"
        processed_path = settings.data_dir / "processed" / f"{uuid.uuid4().hex}.parquet"
        size = 0
        try:
            with upload_path.open("wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="Uploaded file is too large")
                    handle.write(chunk)
            frame = load_retail_csv(upload_path)
            profile = write_dataset(frame, processed_path)
            record = service.storage.register_dataset(name=name, path=processed_path, profile=profile)
            return {"id": record.id, "name": record.name, "profile": profile.model_dump()}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            upload_path.unlink(missing_ok=True)

    @app.get("/api/v1/runs")
    def list_runs() -> list[RunView]:
        return [_run_view(item) for item in service.storage.list_runs()]

    @app.post("/api/v1/runs", status_code=202)
    def create_run(
        request: RunRequest, x_stockwise_sync: str | None = Header(default=None)
    ) -> RunView:
        try:
            record = service.submit_run(
                request, synchronous=(x_stockwise_sync or "").lower() == "true" or None
            )
            return _run_view(record)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_id}")
    def get_run(run_id: str) -> RunView:
        record = service.storage.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return _run_view(record)

    @app.get("/api/v1/runs/{run_id}/events")
    def get_events(run_id: str) -> list[EventView]:
        if service.storage.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return [
            EventView(
                sequence=item.sequence,
                node=item.node,
                status=item.status,
                message=item.message,
                payload=decode_json(item.payload_json),
                created_at=item.created_at,
            )
            for item in service.storage.events_for_run(run_id)
        ]

    @app.get("/api/v1/runs/{run_id}/results")
    def get_results(run_id: str) -> JSONResponse:
        record = service.storage.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run not found")
        result = decode_json(record.result_json)
        result["run"] = _run_view(record).model_dump(mode="json")
        if not result.get("recommendations"):
            result["recommendations"] = service.storage.recommendations_for_run(run_id)
        return JSONResponse(result)

    @app.post("/api/v1/runs/{run_id}/approval", status_code=202)
    def approve_run(
        run_id: str,
        approval: ApprovalRequest,
        x_stockwise_sync: str | None = Header(default=None),
    ) -> RunView:
        try:
            record = service.approve(
                run_id,
                approval,
                synchronous=(x_stockwise_sync or "").lower() == "true" or None,
            )
            return _run_view(record)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_id}/report")
    def get_report(
        run_id: str, format: Literal["html", "json", "csv"] = "html"
    ) -> FileResponse:
        record = service.storage.get_run(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if not record.report_path or not Path(record.report_path).exists():
            raise HTTPException(status_code=409, detail="Report is not available yet")
        html_path = Path(record.report_path)
        candidates = {
            "html": (html_path, "text/html"),
            "json": (html_path.with_name("result.json"), "application/json"),
            "csv": (html_path.with_name("recommendations.csv"), "text/csv"),
        }
        path, media_type = candidates[format]
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"{format} artifact is missing")
        return FileResponse(path, media_type=media_type, filename=path.name)

    return app


app = create_app()
