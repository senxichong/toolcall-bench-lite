"""Database tables for immutable datasets, jobs, and completed run reports."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class DatasetRecord(Base):
    __tablename__ = "benchmark_datasets"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_dataset_name_version"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(80), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    tasks: Mapped[list] = mapped_column(JSON, nullable=False)
    task_count: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class JobRecord(Base):
    __tablename__ = "evaluation_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("benchmark_datasets.id", ondelete="RESTRICT"), nullable=False
    )
    backend: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    error: Mapped[str | None] = mapped_column(Text)
    run_id: Mapped[str | None] = mapped_column(String(36))
    celery_task_id: Mapped[str | None] = mapped_column(String(36))
    lease_token: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
    dataset: Mapped[DatasetRecord] = relationship()


class RunRecord(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("evaluation_jobs.id", ondelete="RESTRICT"),
        nullable=False, unique=True,
    )
    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("benchmark_datasets.id", ondelete="RESTRICT"), nullable=False
    )
    dataset_name: Mapped[str] = mapped_column(String(160), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(80), nullable=False)
    model: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    report: Mapped[dict] = mapped_column(JSON, nullable=False)
    summary: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


def make_session_factory(database_url: str) -> sessionmaker:
    """Return an independent session factory; workers use the same URL as the API."""

    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite:") else {}
    engine = create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def initialize_database(factory: sessionmaker) -> None:
    """Create the initial schema. Existing tables are left intact."""

    Base.metadata.create_all(factory.kw["bind"])
