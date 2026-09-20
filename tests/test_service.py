"""HTTP, persistence, and worker-path tests without external services."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

from fastapi.testclient import TestClient

from toolcall_bench.service.app import create_app
from toolcall_bench.service.config import Settings


def _task(*, expected_answer="Hello!"):
    return {
        "task_id": "hello-1",
        "prompt": "Say hello.",
        "available_tools": ["calculator"],
        "expected_tool": None,
        "expected_arguments": {},
        "expected_answer": expected_answer,
        "answer_match": "exact",
    }


def _client(tmp_path, **overrides):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'service.sqlite').as_posix()}",
        demo_mode=True,
        **overrides,
    )
    return TestClient(create_app(settings))


def _dataset(client, *, version="1.0.0", expected_answer="Hello!"):
    response = client.post(
        "/api/datasets",
        json={"name": "greeting", "version": version, "tasks": [_task(expected_answer=expected_answer)]},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_demo_workflow_persists_dataset_job_report_and_stats(tmp_path):
    with _client(tmp_path) as client:
        dataset = _dataset(client)
        assert dataset["task_count"] == 1
        assert client.get("/api/datasets").json()["items"][0]["id"] == dataset["id"]
        assert client.get(f"/api/datasets/{dataset['id']}").json()["tasks"] == [_task()]
        exported = client.get(f"/api/datasets/{dataset['id']}/export")
        assert exported.status_code == 200
        assert json.loads(exported.text) == _task()

        submitted = client.post(
            "/api/evaluations", json={"dataset_id": dataset["id"], "backend": {"type": "mock"}}
        )
        assert submitted.status_code == 202, submitted.text
        job = submitted.json()
        assert job["status"] == "completed"
        assert client.get(f"/api/jobs/{job['id']}").json()["run_id"] == job["run_id"]

        run = client.get(f"/api/runs/{job['run_id']}").json()
        assert run["dataset_version"] == "1.0.0"
        assert run["summary"]["overall_success"]["rate"] == 1.0
        assert run["report"]["metadata"]["dataset_id"] == dataset["id"]
        assert run["report"]["results"][0]["task"]["task_id"] == "hello-1"
        assert client.get(f"/api/runs/{job['run_id']}/badcases").json()["failed_tasks"] == 0
        assert client.get("/api/runs").json()["items"][0]["id"] == job["run_id"]
        assert client.get(f"/api/comparison?run_ids={job['run_id']}").json()["items"][0]["id"] == job["run_id"]
        stats = client.get("/api/stats").json()
        assert stats["total_runs"] == 1
        assert stats["model_metrics"][0]["summary"]["overall_success"]["passed"] == 1


def test_dataset_is_immutable_and_jsonl_import_validated(tmp_path):
    with _client(tmp_path) as client:
        _dataset(client)
        duplicate = client.post(
            "/api/datasets", json={"name": "greeting", "version": "1.0.0", "tasks": [_task()]}
        )
        assert duplicate.status_code == 409
        imported = client.post(
            "/api/datasets/import",
            json={"name": "greeting", "version": "2.0.0", "jsonl": json.dumps(_task()) + "\n"},
        )
        assert imported.status_code == 201
        assert client.get("/api/datasets").json()["items"][0]["version"] == "2.0.0"
        malformed = client.post(
            "/api/datasets/import",
            json={"name": "bad", "version": "1", "jsonl": "{not-json}\n"},
        )
        assert malformed.status_code == 422
        assert client.post(
            "/api/datasets", json={"name": "bad", "version": "1", "tasks": [_task(), _task()]}
        ).status_code == 422


def test_badcases_and_backend_validation(tmp_path):
    with _client(tmp_path) as client:
        dataset = _dataset(client, expected_answer="Goodbye.")
        assert client.post(
            "/api/evaluations", json={"dataset_id": dataset["id"], "backend": {"type": "openai"}}
        ).status_code == 403
        assert client.post(
            "/api/evaluations",
            json={"dataset_id": dataset["id"], "backend": {"type": "mock", "base_url": "http://127.0.0.1"}},
        ).status_code == 422
        job = client.post(
            "/api/evaluations", json={"dataset_id": dataset["id"], "backend": {"type": "mock"}}
        ).json()
        badcases = client.get(f"/api/runs/{job['run_id']}/badcases").json()
        assert badcases["failed_tasks"] == 1
        assert badcases["items"][0]["primary_category"] == "reasoning_failure"


def test_optional_api_key_and_missing_resources(tmp_path):
    with _client(tmp_path, api_key="test-key") as client:
        assert client.get("/health").status_code == 200
        assert client.get("/api/datasets").status_code == 401
        assert client.get("/api/datasets", headers={"X-API-Key": "wrong"}).status_code == 401
        headers = {"X-API-Key": "test-key"}
        assert client.get("/api/datasets", headers=headers).status_code == 200
        assert client.get("/api/jobs/missing", headers=headers).status_code == 404
        assert client.get("/api/runs/missing", headers=headers).status_code == 404


def test_queue_failure_is_persisted(tmp_path, monkeypatch):
    from toolcall_bench.service import app as service_app_module

    def fail_queue(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(service_app_module.evaluate_job_task, "apply_async", fail_queue)
    settings = Settings(database_url=f"sqlite:///{(tmp_path / 'queue.sqlite').as_posix()}")
    with TestClient(create_app(settings)) as client:
        dataset = _dataset(client)
        response = client.post(
            "/api/evaluations", json={"dataset_id": dataset["id"], "backend": {"type": "mock"}}
        )
        assert response.status_code == 503
        job_id = response.json()["detail"]["job_id"]
        assert client.get(f"/api/jobs/{job_id}").json()["status"] == "failed"
        assert client.get("/api/stats").json()["failed_jobs"] == 1


def test_celery_eager_mode_uses_persisted_job(tmp_path, monkeypatch):
    database_url = f"sqlite:///{(tmp_path / 'eager.sqlite').as_posix()}"
    monkeypatch.setenv("TOOLCALL_DATABASE_URL", database_url)
    settings = Settings(database_url=database_url, celery_always_eager=True)
    with TestClient(create_app(settings)) as client:
        dataset = _dataset(client)
        response = client.post(
            "/api/evaluations", json={"dataset_id": dataset["id"], "backend": {"type": "mock"}}
        )
        assert response.status_code == 202, response.text
        job = response.json()
        assert job["status"] == "completed"
        assert job["run_id"]


def test_concurrent_delivery_does_not_create_duplicate_run(tmp_path, monkeypatch):
    from sqlalchemy import select

    from toolcall_bench.runner import BenchmarkRunner
    from toolcall_bench.service.database import (
        DatasetRecord,
        JobRecord,
        RunRecord,
        initialize_database,
        make_session_factory,
    )
    from toolcall_bench.service.worker import run_job

    factory = make_session_factory(f"sqlite:///{(tmp_path / 'concurrent.sqlite').as_posix()}")
    initialize_database(factory)
    dataset_id = str(uuid4())
    job_id = str(uuid4())
    with factory() as session:
        session.add(DatasetRecord(
            id=dataset_id, name="greeting", version="1", description="",
            tasks=[_task()], task_count=1,
        ))
        session.add(JobRecord(id=job_id, dataset_id=dataset_id, backend={"type": "mock"}, status="queued"))
        session.commit()

    started = Event()
    release = Event()
    real_run = BenchmarkRunner.run

    def delayed_run(self, *args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return real_run(self, *args, **kwargs)

    monkeypatch.setattr(BenchmarkRunner, "run", delayed_run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run_job, job_id, factory=factory, demo_mode=True)
        assert started.wait(timeout=5)
        second = pool.submit(run_job, job_id, factory=factory, demo_mode=True)
        assert second.result(timeout=5) is None
        release.set()
        run_id = first.result(timeout=5)
    assert run_id
    with factory() as session:
        runs = session.scalars(select(RunRecord)).all()
        assert len(runs) == 1
        assert runs[0].id == run_id
        assert session.get(JobRecord, job_id).run_id == run_id
    assert run_job(job_id, factory=factory, demo_mode=True) == run_id
