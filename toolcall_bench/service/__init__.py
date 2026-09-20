"""Optional HTTP and worker service for persisted benchmark evaluations."""

from .app import create_app

__all__ = ["create_app"]
