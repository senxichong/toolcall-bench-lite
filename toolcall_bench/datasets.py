"""Immutable, content-verified benchmark dataset versions.

The local registry is useful for CLI and offline evaluation. The service layer may
persist the same canonical snapshots in its database, but it should retain the
content digest as the common provenance key.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schemas import BenchmarkTask, SchemaValidationError
from .tasks import TaskFileError, load_tasks

_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")
_VERSION_PATTERN = re.compile(r"^v[0-9]{6}$")


class DatasetError(ValueError):
    """A versioned dataset cannot be read, verified, or imported."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name) or ".." in name:
        raise DatasetError(
            "dataset name must be 1-64 ASCII letters, digits, '.', '_' or '-', "
            "start with a letter or digit, and not contain '..'"
        )
    return name


def _validate_version(version: str) -> str:
    if not isinstance(version, str) or not _VERSION_PATTERN.fullmatch(version):
        raise DatasetError("dataset version must have the form v000001")
    return version


def _validate_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    value = dict(metadata or {})
    if not all(isinstance(key, str) for key in value):
        raise DatasetError("dataset metadata keys must be strings")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"dataset metadata must contain JSON values: {exc}") from exc
    return value


def _canonical_bytes(tasks: Iterable[BenchmarkTask]) -> bytes:
    """Produce a stable digest independent of JSON key order and whitespace."""

    lines = [
        json.dumps(
            task.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for task in tasks
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


@dataclass(frozen=True, slots=True)
class DatasetVersion:
    """Manifest for one immutable benchmark snapshot."""

    name: str
    version: str
    sha256: str
    task_count: int
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_name(self.name)
        _validate_version(self.version)
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise DatasetError("sha256 must be a lowercase hexadecimal digest")
        if (
            isinstance(self.task_count, bool)
            or not isinstance(self.task_count, int)
            or self.task_count < 1
        ):
            raise DatasetError("task_count must be a positive integer")
        if not isinstance(self.created_at, str) or not self.created_at:
            raise DatasetError("created_at must be a non-empty string")
        _validate_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "version": self.version,
            "sha256": self.sha256,
            "task_count": self.task_count,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetVersion:
        if not isinstance(data, Mapping):
            raise DatasetError("dataset manifest must be a JSON object")
        required = {"name", "version", "sha256", "task_count", "created_at"}
        missing = sorted(required - set(data))
        if missing:
            raise DatasetError(f"dataset manifest missing field(s): {', '.join(missing)}")
        if data.get("schema_version", 1) != 1:
            raise DatasetError("unsupported dataset manifest schema version")
        return cls(
            name=data["name"],
            version=data["version"],
            sha256=data["sha256"],
            task_count=data["task_count"],
            created_at=data["created_at"],
            metadata=_validate_metadata(data.get("metadata", {})),
        )


class DatasetManager:
    """Manage named, immutable local benchmark versions.

    Versions are reserved with exclusive directory creation, then completed with
    atomically replaced files. Readers ignore incomplete directories. An import
    with the same canonical content digest is idempotent.
    """

    def __init__(self, root: str | Path = "datasets") -> None:
        self.root = Path(root).expanduser().resolve()

    def import_file(
        self,
        name: str,
        source: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> DatasetVersion:
        """Import a validated JSONL or JSON-array benchmark file."""

        source_path = Path(source)
        if source_path.suffix.lower() == ".json":
            try:
                payload = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise DatasetError(f"cannot read JSON benchmark {source_path}: {exc}") from exc
            if not isinstance(payload, list):
                raise DatasetError("JSON benchmark must be an array of task objects")
            try:
                tasks = [BenchmarkTask.from_dict(item) for item in payload]
            except (SchemaValidationError, TypeError) as exc:
                raise DatasetError(f"invalid task in JSON benchmark: {exc}") from exc
        else:
            try:
                tasks = load_tasks(source_path)
            except TaskFileError as exc:
                raise DatasetError(str(exc)) from exc
        return self.import_tasks(name, tasks, metadata=metadata)

    def import_tasks(
        self,
        name: str,
        tasks: Iterable[BenchmarkTask],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> DatasetVersion:
        """Store canonical tasks as a new version or return an identical existing one."""

        _validate_name(name)
        task_list = list(tasks)
        if not task_list:
            raise DatasetError("cannot import an empty dataset")
        if not all(isinstance(task, BenchmarkTask) for task in task_list):
            raise DatasetError("all imported records must be BenchmarkTask instances")
        ids = [task.task_id for task in task_list]
        if len(ids) != len(set(ids)):
            raise DatasetError("cannot import duplicate task_id values")
        canonical = _canonical_bytes(task_list)
        digest = hashlib.sha256(canonical).hexdigest()
        for existing in self.list_versions(name):
            if existing.sha256 == digest:
                return existing

        directory = self.root / name
        directory.mkdir(parents=True, exist_ok=True)
        number = 1
        while True:
            version = f"v{number:06d}"
            version_dir = directory / version
            try:
                version_dir.mkdir()
                break
            except FileExistsError:
                number += 1

        manifest = DatasetVersion(
            name=name,
            version=version,
            sha256=digest,
            task_count=len(task_list),
            created_at=_now(),
            metadata=_validate_metadata(metadata),
        )
        try:
            self._atomic_write(version_dir / "tasks.jsonl", canonical)
            self._atomic_write(
                version_dir / "manifest.json",
                (json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n").encode(
                    "utf-8"
                ),
            )
        except OSError:
            # Leave the reserved directory for inspection; it is not visible
            # through list_versions until a complete manifest exists.
            raise
        return manifest

    def list_datasets(self) -> list[str]:
        """List names that have at least one complete version."""

        if not self.root.exists():
            return []
        return sorted(
            path.name
            for path in self.root.iterdir()
            if path.is_dir()
            and _NAME_PATTERN.fullmatch(path.name)
            and self.list_versions(path.name)
        )

    def list_versions(self, name: str) -> list[DatasetVersion]:
        """List complete versions, oldest first, without assuming contiguous IDs."""

        _validate_name(name)
        directory = self.root / name
        if not directory.is_dir():
            return []
        versions: list[DatasetVersion] = []
        for version_dir in sorted(directory.iterdir()):
            if not version_dir.is_dir() or not _VERSION_PATTERN.fullmatch(version_dir.name):
                continue
            manifest_path = version_dir / "manifest.json"
            if not manifest_path.is_file() or not (version_dir / "tasks.jsonl").is_file():
                continue
            manifest = self._read_manifest(manifest_path)
            if manifest.name != name or manifest.version != version_dir.name:
                raise DatasetError(f"manifest identity mismatch: {manifest_path}")
            versions.append(manifest)
        return versions

    def get_version(self, name: str, version: str | None = None) -> DatasetVersion:
        """Get a manifest, using the latest version when omitted."""

        _validate_name(name)
        versions = self.list_versions(name)
        if not versions:
            raise DatasetError(f"dataset has no complete versions: {name}")
        if version is None:
            return versions[-1]
        _validate_version(version)
        for item in versions:
            if item.version == version:
                return item
        raise DatasetError(f"dataset version not found: {name}/{version}")

    def load_version(self, name: str, version: str | None = None) -> list[BenchmarkTask]:
        """Read tasks and verify count and canonical SHA-256 before returning."""

        manifest = self.get_version(name, version)
        path = self.path_for(manifest.name, manifest.version)
        try:
            tasks = load_tasks(path)
        except TaskFileError as exc:
            raise DatasetError(f"corrupt dataset version {name}/{manifest.version}: {exc}") from exc
        observed_digest = hashlib.sha256(_canonical_bytes(tasks)).hexdigest()
        if len(tasks) != manifest.task_count or observed_digest != manifest.sha256:
            raise DatasetError(f"integrity check failed for dataset {name}/{manifest.version}")
        return tasks

    def export_version(
        self,
        name: str,
        destination: str | Path,
        *,
        version: str | None = None,
        format: str = "jsonl",
    ) -> Path:
        """Export a verified version as JSONL or a JSON array."""

        tasks = self.load_version(name, version)
        output = Path(destination)
        if format == "jsonl":
            self._atomic_write(output, _canonical_bytes(tasks))
        elif format == "json":
            output.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_write(
                output,
                (json.dumps([task.to_dict() for task in tasks], ensure_ascii=False, indent=2) + "\n").encode(
                    "utf-8"
                ),
            )
        else:
            raise DatasetError("export format must be 'jsonl' or 'json'")
        return output

    def path_for(self, name: str, version: str | None = None) -> Path:
        """Return the verified snapshot path for a named version."""

        manifest = self.get_version(name, version)
        path = (self.root / manifest.name / manifest.version / "tasks.jsonl").resolve()
        if self.root not in path.parents:
            raise DatasetError("dataset path escapes the configured root")
        return path

    def diff_versions(self, name: str, older: str, newer: str) -> dict[str, Any]:
        """Compare versions by stable task IDs and canonical record contents."""

        left = {task.task_id: task.to_dict() for task in self.load_version(name, older)}
        right = {task.task_id: task.to_dict() for task in self.load_version(name, newer)}
        added = sorted(set(right) - set(left))
        removed = sorted(set(left) - set(right))
        changed = sorted(key for key in left.keys() & right.keys() if left[key] != right[key])
        return {
            "name": name,
            "older": older,
            "newer": newer,
            "added": added,
            "removed": removed,
            "changed": changed,
        }

    @staticmethod
    def _read_manifest(path: Path) -> DatasetVersion:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetError(f"cannot read dataset manifest {path}: {exc}") from exc
        return DatasetVersion.from_dict(payload)

    @staticmethod
    def _atomic_write(destination: Path, payload: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".tmp-", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


__all__ = ["DatasetError", "DatasetManager", "DatasetVersion"]
