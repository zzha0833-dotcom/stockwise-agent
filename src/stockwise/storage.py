from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlmodel import Field, Session, SQLModel, create_engine, select

from stockwise.schemas import DatasetProfile, RunRequest, RunStatus


def utcnow() -> datetime:
    return datetime.now(UTC)


class DatasetRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    path: str
    profile_json: str
    created_at: datetime = Field(default_factory=utcnow)


class RunRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    dataset_id: str = Field(index=True)
    status: str = Field(default=RunStatus.QUEUED.value, index=True)
    current_node: str = "queued"
    request_json: str
    result_json: str = "{}"
    error: str | None = None
    fallback_used: bool = False
    report_path: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class EventRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    sequence: int
    node: str
    status: str
    message: str
    payload_json: str = "{}"
    created_at: datetime = Field(default_factory=utcnow)


class RecommendationRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    store_id: str
    item_id: str
    requires_approval: bool
    approval_status: str
    payload_json: str


class ApprovalRecord(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    decision: str
    note: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class Storage:
    def __init__(self, database_url: str):
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, connect_args=connect_args)

    def initialize(self) -> None:
        SQLModel.metadata.create_all(self.engine)

    def register_dataset(
        self, *, name: str, path: Path, profile: DatasetProfile, dataset_id: str | None = None
    ) -> DatasetRecord:
        record = DatasetRecord(
            id=dataset_id or uuid.uuid4().hex,
            name=name,
            path=str(path.resolve()),
            profile_json=profile.model_dump_json(),
        )
        with Session(self.engine) as session:
            session.add(record)
            session.commit()
            session.refresh(record)
        return record

    def list_datasets(self) -> list[DatasetRecord]:
        with Session(self.engine) as session:
            return list(session.exec(select(DatasetRecord).order_by(DatasetRecord.created_at.desc())))

    def get_dataset(self, dataset_id: str) -> DatasetRecord | None:
        with Session(self.engine) as session:
            return session.get(DatasetRecord, dataset_id)

    def update_dataset(
        self, dataset_id: str, *, path: Path, profile: DatasetProfile
    ) -> DatasetRecord:
        with Session(self.engine) as session:
            record = session.get(DatasetRecord, dataset_id)
            if record is None:
                raise KeyError(f"Unknown dataset: {dataset_id}")
            record.path = str(path.resolve())
            record.profile_json = profile.model_dump_json()
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def create_run(self, request: RunRequest, run_id: str | None = None) -> RunRecord:
        record = RunRecord(
            id=run_id or uuid.uuid4().hex,
            dataset_id=request.dataset_id,
            request_json=request.model_dump_json(),
        )
        with Session(self.engine) as session:
            session.add(record)
            session.commit()
            session.refresh(record)
        return record

    def get_run(self, run_id: str) -> RunRecord | None:
        with Session(self.engine) as session:
            return session.get(RunRecord, run_id)

    def list_runs(self, limit: int = 50) -> list[RunRecord]:
        with Session(self.engine) as session:
            return list(
                session.exec(select(RunRecord).order_by(RunRecord.created_at.desc()).limit(limit))
            )

    def update_run(self, run_id: str, **values: Any) -> RunRecord:
        with Session(self.engine) as session:
            record = session.get(RunRecord, run_id)
            if record is None:
                raise KeyError(f"Unknown run: {run_id}")
            for key, value in values.items():
                if not hasattr(record, key):
                    raise AttributeError(key)
                setattr(record, key, value)
            record.updated_at = utcnow()
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def add_event(
        self,
        run_id: str,
        *,
        node: str,
        status: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> EventRecord:
        with Session(self.engine) as session:
            existing = list(
                session.exec(
                    select(EventRecord)
                    .where(EventRecord.run_id == run_id)
                    .order_by(EventRecord.sequence.desc())
                    .limit(1)
                )
            )
            sequence = existing[0].sequence + 1 if existing else 1
            event = EventRecord(
                run_id=run_id,
                sequence=sequence,
                node=node,
                status=status,
                message=message,
                payload_json=json.dumps(payload or {}, default=str),
            )
            session.add(event)
            session.commit()
            session.refresh(event)
            return event

    def events_for_run(self, run_id: str) -> list[EventRecord]:
        with Session(self.engine) as session:
            return list(
                session.exec(
                    select(EventRecord)
                    .where(EventRecord.run_id == run_id)
                    .order_by(EventRecord.sequence)
                )
            )

    def replace_recommendations(self, run_id: str, recommendations: list[dict[str, Any]]) -> None:
        with Session(self.engine) as session:
            existing = session.exec(
                select(RecommendationRecord).where(RecommendationRecord.run_id == run_id)
            ).all()
            for record in existing:
                session.delete(record)
            for item in recommendations:
                session.add(
                    RecommendationRecord(
                        run_id=run_id,
                        store_id=str(item["store_id"]),
                        item_id=str(item["item_id"]),
                        requires_approval=bool(item["requires_approval"]),
                        approval_status=str(item["approval_status"]),
                        payload_json=json.dumps(item, default=str),
                    )
                )
            session.commit()

    def recommendations_for_run(self, run_id: str) -> list[dict[str, Any]]:
        with Session(self.engine) as session:
            records = session.exec(
                select(RecommendationRecord).where(RecommendationRecord.run_id == run_id)
            ).all()
            return [json.loads(record.payload_json) for record in records]

    def record_approval(self, run_id: str, decision: str, note: str) -> None:
        with Session(self.engine) as session:
            session.add(ApprovalRecord(run_id=run_id, decision=decision, note=note))
            records = session.exec(
                select(RecommendationRecord).where(
                    RecommendationRecord.run_id == run_id,
                    RecommendationRecord.requires_approval.is_(True),
                )
            ).all()
            for record in records:
                payload = json.loads(record.payload_json)
                payload["approval_status"] = "approved" if decision == "approve" else "rejected"
                record.approval_status = payload["approval_status"]
                record.payload_json = json.dumps(payload)
                session.add(record)
            session.commit()


def decode_json(value: str) -> dict[str, Any]:
    return json.loads(value) if value else {}
