"""Service configuration; importing this module never opens a network connection."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///./toolcall_bench.db"
    redis_url: str = "redis://localhost:6379/0"
    demo_mode: bool = False
    celery_always_eager: bool = False
    cors_origins: tuple[str, ...] = field(default_factory=tuple)
    api_key: str | None = None
    max_dataset_tasks: int = 10_000

    @classmethod
    def from_env(cls) -> Settings:
        origins = tuple(
            item.strip()
            for item in os.getenv("TOOLCALL_CORS_ORIGINS", "").split(",")
            if item.strip()
        )
        return cls(
            database_url=os.getenv("TOOLCALL_DATABASE_URL", cls.database_url),
            redis_url=os.getenv("TOOLCALL_REDIS_URL", cls.redis_url),
            demo_mode=_env_bool("TOOLCALL_DEMO_MODE"),
            celery_always_eager=_env_bool("TOOLCALL_CELERY_ALWAYS_EAGER"),
            cors_origins=origins,
            api_key=os.getenv("TOOLCALL_API_KEY") or None,
            max_dataset_tasks=int(os.getenv("TOOLCALL_MAX_DATASET_TASKS", "10000")),
        )
