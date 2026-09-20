"""Celery task and in-process evaluation implementation.

The durable job row, not the broker result backend, is the source of truth for
status. The same function powers the explicit mock-only demo mode.
"""

from __future__ import annotations

import json
from datetime import timedelta
from uuid import uuid4

from celery import Celery
from sqlalchemy import and_, or_, update
from sqlalchemy.orm import sessionmaker

from ..agent import MockAgent, OpenAICompatibleAdapter
from ..runner import BenchmarkRunner
from ..schemas import BenchmarkTask
from .config import Settings
from .database import DatasetRecord, JobRecord, RunRecord, make_session_factory, utc_now


def _make_celery() -> Celery:
    settings = Settings.from_env()
    app = Celery("toolcall_bench", broker=settings.redis_url, backend=settings.redis_url)
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_always_eager=settings.celery_always_eager,
    )
    return app


celery_app = _make_celery()


def build_adapter(config: dict, *, demo_mode: bool = False):
    """Construct an adapter from a persisted, credential-free configuration."""

    backend_type = config["type"]
    if backend_type == "mock":
        return MockAgent()
    if demo_mode:
        raise ValueError("demo mode permits only the mock backend")
    if backend_type == "openai":
        return OpenAICompatibleAdapter(model=config.get("model") or None)
    if backend_type == "local":
        # Kept late-bound so installations using only the service/mock path do not
        # import optional model runtimes at API startup.
        from .. import agent

        adapter_class = getattr(agent, "LocalModelAdapter", None)
        if adapter_class is None:
            raise ValueError("local model backend is not installed")
        model = config.get("model")
        if not model:
            raise ValueError("local model backend requires a model")
        return adapter_class(model=model)
    raise ValueError(f"unsupported backend: {backend_type}")


def _safe_error(exc: Exception) -> str:
    # Third-party exception strings often include URLs, headers, or prompt data.
    # Persist only a type label; detailed diagnostics belong in restricted logs.
    return f"evaluation failed ({type(exc).__name__})"


def run_job(
    job_id: str,
    *,
    factory: sessionmaker | None = None,
    demo_mode: bool = False,
) -> str | None:
    """Execute one queued job, persisting a full report atomically on success.

    A conditional database UPDATE atomically claims queued jobs. A stale lease
    may be reclaimed after an hour; its previous owner can no longer commit.
    """

    factory = factory or make_session_factory(Settings.from_env().database_url)
    lease_token = str(uuid4())
    now = utc_now()
    stale_before = now - timedelta(hours=1)
    with factory() as session:
        claimed = session.execute(
            update(JobRecord)
            .where(JobRecord.id == job_id)
            .where(
                or_(
                    JobRecord.status == "queued",
                    and_(JobRecord.status == "running", JobRecord.updated_at < stale_before),
                )
            )
            .values(status="running", lease_token=lease_token, error=None, updated_at=now)
        )
        session.commit()
        job = session.get(JobRecord, job_id)
        if job is None:
            raise ValueError(f"unknown evaluation job: {job_id}")
        if claimed.rowcount != 1:
            return job.run_id if job.status == "completed" else None
        config = dict(job.backend)
        dataset = session.get(DatasetRecord, job.dataset_id)
        if dataset is None:
            job.status = "failed"
            job.error = "dataset snapshot was not found"
            job.lease_token = None
            job.updated_at = utc_now()
            session.commit()
            return None
        task_data = list(dataset.tasks)
        dataset_id = dataset.id
        dataset_name = dataset.name
        dataset_version = dataset.version

    try:
        tasks = [BenchmarkTask.from_dict(item) for item in task_data]
        adapter = build_adapter(config, demo_mode=demo_mode)
        report = BenchmarkRunner(adapter=adapter).run(tasks)
        report_data = json.loads(json.dumps(report.to_dict(), ensure_ascii=False, default=str))
        report_data["benchmark"] = f"{dataset_name}@{dataset_version}"
        report_data["metadata"] = {
            **report_data.get("metadata", {}),
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
        }
        summary = report_data["summary"]
        run_id = str(uuid4())
        with factory() as session:
            committed = session.execute(
                update(JobRecord)
                .where(JobRecord.id == job_id)
                .where(JobRecord.status == "running", JobRecord.lease_token == lease_token)
                .values(
                    status="completed", run_id=run_id, error=None,
                    lease_token=None, updated_at=utc_now(),
                )
            )
            if committed.rowcount != 1:
                session.rollback()
                return None
            run = RunRecord(
                id=run_id,
                job_id=job_id,
                dataset_id=dataset_id,
                dataset_name=dataset_name,
                dataset_version=dataset_version,
                model=report_data["model"],
                report=report_data,
                summary=summary,
            )
            session.add(run)
            session.commit()
        return run_id
    except Exception as exc:
        with factory() as session:
            session.execute(
                update(JobRecord)
                .where(JobRecord.id == job_id)
                .where(JobRecord.status == "running", JobRecord.lease_token == lease_token)
                .values(
                    status="failed", error=_safe_error(exc), lease_token=None,
                    updated_at=utc_now(),
                )
            )
            session.commit()
        return None


@celery_app.task(name="toolcall_bench.evaluate_job", bind=True)
def evaluate_job_task(self, job_id: str) -> str | None:
    return run_job(job_id)
