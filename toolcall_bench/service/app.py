"""FastAPI surface for immutable datasets and durable evaluation runs."""

from __future__ import annotations

import hmac
import json
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..schemas import BenchmarkTask, RunReport, SchemaValidationError
from .config import Settings
from .database import (
    DatasetRecord,
    JobRecord,
    RunRecord,
    initialize_database,
    make_session_factory,
    utc_now,
)
from .worker import celery_app, evaluate_job_task, run_job

_METRICS = (
    "tool_selection_accuracy",
    "argument_match",
    "schema_validity",
    "execution_success",
    "final_answer_accuracy",
    "overall_success",
)


def _iso(value: Any) -> str | None:
    return None if value is None else value.isoformat()


def _dataset_dict(dataset: DatasetRecord, *, include_tasks: bool = False) -> dict:
    payload = {
        "id": dataset.id,
        "name": dataset.name,
        "version": dataset.version,
        "description": dataset.description,
        "task_count": dataset.task_count,
        "created_at": _iso(dataset.created_at),
    }
    if include_tasks:
        payload["tasks"] = dataset.tasks
    return payload


def _job_dict(job: JobRecord) -> dict:
    return {
        "id": job.id,
        "status": job.status,
        "dataset_id": job.dataset_id,
        "backend": job.backend,
        "run_id": job.run_id,
        "error": job.error,
        "created_at": _iso(job.created_at),
        "updated_at": _iso(job.updated_at),
    }


def _run_dict(run: RunRecord, *, include_report: bool = False) -> dict:
    payload = {
        "id": run.id,
        "job_id": run.job_id,
        "dataset_id": run.dataset_id,
        "dataset_name": run.dataset_name,
        "dataset_version": run.dataset_version,
        "model": run.model,
        "created_at": _iso(run.created_at),
        "summary": run.summary,
    }
    if include_report:
        payload["report"] = run.report
    return payload


def _validated_tasks(raw: Any, *, max_tasks: int) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        raise HTTPException(422, "tasks must be a non-empty list")
    if len(raw) > max_tasks:
        raise HTTPException(413, f"dataset exceeds {max_tasks} tasks")
    tasks: list[dict] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, start=1):
        try:
            task = BenchmarkTask.from_dict(item)
        except (SchemaValidationError, TypeError, AttributeError) as exc:
            raise HTTPException(422, f"task {index}: {exc}") from exc
        if task.task_id in seen:
            raise HTTPException(422, f"duplicate task_id: {task.task_id}")
        seen.add(task.task_id)
        tasks.append(task.to_dict())
    return tasks


def _field(payload: dict, name: str, max_length: int, *, required: bool = True) -> str:
    value = payload.get(name, "")
    if not isinstance(value, str):
        raise HTTPException(422, f"{name} must be a string")
    value = value.strip()
    if required and not value:
        raise HTTPException(422, f"{name} must not be empty")
    if len(value) > max_length or any(char in value for char in "\r\n\x00"):
        raise HTTPException(422, f"{name} is invalid or too long")
    return value


def _combine_summaries(summaries: list[dict]) -> dict:
    total_tasks = sum(int(summary.get("total_tasks", 0)) for summary in summaries)
    result: dict[str, Any] = {"total_tasks": total_tasks}
    for metric in _METRICS:
        passed = sum(int(summary.get(metric, {}).get("passed", 0)) for summary in summaries)
        total = sum(int(summary.get(metric, {}).get("total", 0)) for summary in summaries)
        result[metric] = {"passed": passed, "total": total, "rate": passed / total if total else 0.0}
    return result


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    factory = make_session_factory(settings.database_url)
    celery_app.conf.task_always_eager = settings.celery_always_eager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        initialize_database(factory)
        yield

    app = FastAPI(
        title="ToolCall Bench Evaluation API",
        version="1.0.0",
        lifespan=lifespan,
        description="Dataset snapshots, asynchronous evaluations, and trace-backed results.",
    )
    app.state.settings = settings
    app.state.session_factory = factory
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key"],
        )

    def get_session():
        with factory() as session:
            yield session

    def require_key(x_api_key: str | None = Header(default=None)):
        if settings.api_key and not hmac.compare_digest(x_api_key or "", settings.api_key):
            raise HTTPException(401, "invalid API key")

    @app.get("/health", include_in_schema=False)
    def liveness() -> dict:
        return {"status": "ok"}

    router = APIRouter(prefix="/api", dependencies=[Depends(require_key)])

    @router.get("/health")
    def health() -> dict:
        return {"status": "ok", "demo_mode": settings.demo_mode}

    @router.get("/datasets")
    def list_datasets(
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
        session: Session = Depends(get_session),
    ) -> dict:
        rows = session.scalars(
            select(DatasetRecord).order_by(DatasetRecord.created_at.desc(), DatasetRecord.id.desc())
            .offset(offset).limit(limit)
        ).all()
        return {"items": [_dataset_dict(row) for row in rows], "limit": limit, "offset": offset}

    def create_dataset(payload: dict, session: Session) -> dict:
        if not isinstance(payload, dict):
            raise HTTPException(422, "dataset body must be an object")
        name = _field(payload, "name", 160)
        version = _field(payload, "version", 80)
        description = _field(payload, "description", 10_000, required=False)
        tasks = _validated_tasks(payload.get("tasks"), max_tasks=settings.max_dataset_tasks)
        row = DatasetRecord(
            id=str(uuid4()), name=name, version=version, description=description,
            tasks=tasks, task_count=len(tasks),
        )
        session.add(row)
        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise HTTPException(409, "dataset name and version already exist") from exc
        return _dataset_dict(row)

    @router.post("/datasets", status_code=201)
    def post_dataset(payload: dict = Body(...), session: Session = Depends(get_session)) -> dict:
        return create_dataset(payload, session)

    @router.post("/datasets/import", status_code=201)
    def import_dataset(payload: dict = Body(...), session: Session = Depends(get_session)) -> dict:
        if not isinstance(payload, dict) or not isinstance(payload.get("jsonl"), str):
            raise HTTPException(422, "jsonl must be a string")
        if len(payload["jsonl"].encode("utf-8")) > 20_000_000:
            raise HTTPException(413, "JSONL upload exceeds 20 MB")
        tasks = []
        for line_number, line in enumerate(payload["jsonl"].splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise HTTPException(422, f"line {line_number}: invalid JSON") from exc
        return create_dataset({**payload, "tasks": tasks}, session)

    @router.get("/datasets/{dataset_id}")
    def get_dataset(dataset_id: str, session: Session = Depends(get_session)) -> dict:
        row = session.get(DatasetRecord, dataset_id)
        if row is None:
            raise HTTPException(404, "dataset not found")
        return _dataset_dict(row, include_tasks=True)

    @router.get("/datasets/{dataset_id}/export")
    def export_dataset(dataset_id: str, session: Session = Depends(get_session)) -> Response:
        row = session.get(DatasetRecord, dataset_id)
        if row is None:
            raise HTTPException(404, "dataset not found")
        content = "\n".join(json.dumps(task, ensure_ascii=False) for task in row.tasks) + "\n"
        return Response(content=content, media_type="application/x-ndjson; charset=utf-8")

    @router.post("/evaluations", status_code=202)
    def submit_evaluation(payload: dict = Body(...), session: Session = Depends(get_session)) -> dict:
        if not isinstance(payload, dict):
            raise HTTPException(422, "evaluation body must be an object")
        dataset_id = payload.get("dataset_id")
        if not isinstance(dataset_id, str) or session.get(DatasetRecord, dataset_id) is None:
            raise HTTPException(404, "dataset not found")
        backend = payload.get("backend")
        if not isinstance(backend, dict) or set(backend) - {"type", "model"}:
            raise HTTPException(422, "backend accepts only type and model")
        backend_type = backend.get("type")
        model = backend.get("model")
        if backend_type not in {"mock", "openai", "local"}:
            raise HTTPException(422, "backend type must be mock, openai, or local")
        if settings.demo_mode and backend_type != "mock":
            raise HTTPException(403, "demo mode permits only the mock backend")
        if model is not None and (not isinstance(model, str) or not model.strip() or len(model) > 160):
            raise HTTPException(422, "model must be a non-empty string of at most 160 characters")
        if backend_type == "local" and not model:
            raise HTTPException(422, "local backend requires a model")
        config = {"type": backend_type}
        if model:
            config["model"] = model.strip()
        job = JobRecord(id=str(uuid4()), dataset_id=dataset_id, backend=config, status="queued")
        session.add(job)
        session.commit()

        if settings.demo_mode:
            run_job(job.id, factory=factory, demo_mode=True)
        else:
            try:
                result = evaluate_job_task.apply_async(args=[job.id], retry=False)
                job.celery_task_id = result.id
                session.commit()
            except Exception as exc:
                session.rollback()
                job = session.get(JobRecord, job.id)
                job.status = "failed"
                job.error = f"queue unavailable: {type(exc).__name__}"
                job.updated_at = utc_now()
                session.commit()
                raise HTTPException(503, {"message": "evaluation queue unavailable", "job_id": job.id}) from exc
        session.refresh(job)
        return _job_dict(job)

    @router.get("/jobs/{job_id}")
    def get_job(job_id: str, session: Session = Depends(get_session)) -> dict:
        job = session.get(JobRecord, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return _job_dict(job)

    @router.get("/runs")
    def list_runs(
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
        session: Session = Depends(get_session),
    ) -> dict:
        rows = session.scalars(
            select(RunRecord).order_by(RunRecord.created_at.desc(), RunRecord.id.desc())
            .offset(offset).limit(limit)
        ).all()
        return {"items": [_run_dict(row) for row in rows], "limit": limit, "offset": offset}

    @router.get("/runs/{run_id}")
    def get_run(run_id: str, session: Session = Depends(get_session)) -> dict:
        run = session.get(RunRecord, run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        return _run_dict(run, include_report=True)

    @router.get("/runs/{run_id}/badcases")
    def get_badcases(run_id: str, session: Session = Depends(get_session)) -> dict:
        from ..badcases import analyze_badcases

        run = session.get(RunRecord, run_id)
        if run is None:
            raise HTTPException(404, "run not found")
        report = analyze_badcases(RunReport.from_dict(run.report)).to_dict()
        return {"items": report["badcases"], **report}

    @router.get("/stats")
    def stats(
        dataset_id: str | None = Query(default=None), session: Session = Depends(get_session)
    ) -> dict:
        runs = session.scalars(select(RunRecord)).all()
        if dataset_id is not None:
            comparison_dataset = session.get(DatasetRecord, dataset_id)
            if comparison_dataset is None:
                raise HTTPException(404, "dataset not found")
        else:
            latest_run = max(runs, key=lambda row: (row.created_at, row.id), default=None)
            comparison_dataset = (
                session.get(DatasetRecord, latest_run.dataset_id) if latest_run else None
            )
        model_runs: dict[str, list[dict]] = defaultdict(list)
        for run in runs:
            if comparison_dataset is not None and run.dataset_id == comparison_dataset.id:
                model_runs[run.model].append(run.summary)
        model_metrics = [
            {"model": model, "runs": len(summaries), "total_tasks": sum(
                int(item.get("total_tasks", 0)) for item in summaries
            ), "summary": _combine_summaries(summaries)}
            for model, summaries in sorted(model_runs.items())
        ]
        return {
            "total_runs": len(runs),
            "total_jobs": session.scalar(select(func.count(JobRecord.id))) or 0,
            "failed_jobs": session.scalar(
                select(func.count(JobRecord.id)).where(JobRecord.status == "failed")
            ) or 0,
            "comparison_dataset": (
                _dataset_dict(comparison_dataset) if comparison_dataset is not None else None
            ),
            "model_metrics": model_metrics,
        }

    @router.get("/comparison")
    def comparison(run_ids: str = Query(...), session: Session = Depends(get_session)) -> dict:
        ids = [item.strip() for item in run_ids.split(",") if item.strip()]
        if not ids or len(ids) > 20:
            raise HTTPException(422, "run_ids must contain 1 to 20 comma-separated IDs")
        rows = session.scalars(select(RunRecord).where(RunRecord.id.in_(ids))).all()
        by_id = {row.id: row for row in rows}
        missing = [item for item in ids if item not in by_id]
        if missing:
            raise HTTPException(404, f"run not found: {missing[0]}")
        if len({row.dataset_id for row in rows}) > 1:
            raise HTTPException(422, "runs must use the same dataset version for comparison")
        return {"items": [_run_dict(by_id[item]) for item in ids]}

    app.include_router(router)
    return app


app = create_app()
