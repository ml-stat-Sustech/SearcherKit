from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


CHECKPOINT_VERSION = 1


class CheckpointError(Exception):
    """Checkpoint state cannot be read, written, or interpreted safely."""


class CheckpointCorruptionError(CheckpointError):
    """Checkpoint payload is not valid JSON or has an unexpected shape."""


@dataclass
class CheckpointConfig:
    enabled: bool = True
    dir: str | None = None
    resume: bool = True
    filename: str = "manifest.json"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class CheckpointStore:
    """JSON manifest used to resume long batch runs at sample granularity."""

    def __init__(self, path: str | Path, *, resume: bool = True) -> None:
        self.path = Path(path)
        self.resume = resume
        self._lock = asyncio.Lock()
        self._state = self._empty_state()
        if resume and self.path.exists():
            self._state = self._load()

    @classmethod
    def from_output_dir(
        cls,
        output_dir: str | Path,
        *,
        config: CheckpointConfig | dict[str, Any] | None = None,
    ) -> "CheckpointStore | None":
        checkpoint_config = normalize_checkpoint_config(config)
        if not checkpoint_config.enabled:
            return None
        checkpoint_dir = (
            Path(checkpoint_config.dir)
            if checkpoint_config.dir
            else Path(output_dir) / "checkpoints"
        )
        return cls(
            checkpoint_dir / checkpoint_config.filename,
            resume=checkpoint_config.resume,
        )

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    def sample_state(self, sample_id: str) -> dict[str, Any] | None:
        samples = self._state.get("samples", {})
        if not isinstance(samples, dict):
            return None
        value = samples.get(sample_id)
        return value if isinstance(value, dict) else None

    def is_completed(self, sample_id: str, record_path: str | Path) -> bool:
        sample = self.sample_state(sample_id)
        if sample is None:
            return False
        return sample.get("status") == "completed" and Path(record_path).exists()

    def is_resume_candidate(self, sample_id: str) -> bool:
        sample = self.sample_state(sample_id)
        if sample is None:
            return False
        return sample.get("status") in {"running", "failed"}

    async def start_run(self, *, run_id: str, output_dir: str | Path) -> None:
        async with self._lock:
            runs = self._state.setdefault("runs", [])
            if not isinstance(runs, list):
                raise CheckpointCorruptionError("checkpoint field 'runs' must be a list")
            runs.append(
                {
                    "run_id": run_id,
                    "output_dir": str(output_dir),
                    "started_at": utc_now(),
                }
            )
            self._flush()

    async def mark_pending(
        self,
        *,
        sample_id: str,
        index: int,
        trace_id: str,
        record_path: str | Path,
        prompt: str,
    ) -> None:
        async with self._lock:
            sample = self._ensure_sample(sample_id)
            if sample.get("status") == "completed":
                return
            sample.update(
                {
                    "index": index,
                    "sample_id": sample_id,
                    "trace_id": trace_id,
                    "record_path": str(record_path),
                    "prompt_preview": prompt[:240],
                    "status": "pending",
                    "updated_at": utc_now(),
                }
            )
            sample.setdefault("attempts", 0)
            self._flush()

    async def mark_started(
        self,
        *,
        sample_id: str,
        run_id: str,
        trace_id: str,
        started_at: str,
    ) -> None:
        async with self._lock:
            sample = self._ensure_sample(sample_id)
            attempts = int(sample.get("attempts", 0)) + 1
            sample.update(
                {
                    "run_id": run_id,
                    "trace_id": trace_id,
                    "status": "running",
                    "attempts": attempts,
                    "started_at": started_at,
                    "updated_at": utc_now(),
                    "error": None,
                }
            )
            self._flush()

    async def mark_completed(
        self,
        *,
        sample_id: str,
        stats: dict[str, Any],
        record_path: str | Path,
        ended_at: str,
        elapsed: float,
    ) -> None:
        async with self._lock:
            sample = self._ensure_sample(sample_id)
            sample.update(
                {
                    "status": "completed",
                    "record_path": str(record_path),
                    "ended_at": ended_at,
                    "elapsed": elapsed,
                    "stats": stats,
                    "updated_at": utc_now(),
                    "error": None,
                }
            )
            self._flush()

    async def mark_failed(
        self,
        *,
        sample_id: str,
        error: str,
        ended_at: str,
        elapsed: float,
    ) -> None:
        async with self._lock:
            sample = self._ensure_sample(sample_id)
            sample.update(
                {
                    "status": "failed",
                    "ended_at": ended_at,
                    "elapsed": elapsed,
                    "error": error,
                    "updated_at": utc_now(),
                }
            )
            self._flush()

    def _ensure_sample(self, sample_id: str) -> dict[str, Any]:
        samples = self._state.setdefault("samples", {})
        if not isinstance(samples, dict):
            raise CheckpointCorruptionError("checkpoint field 'samples' must be an object")
        sample = samples.setdefault(sample_id, {})
        if not isinstance(sample, dict):
            raise CheckpointCorruptionError(f"checkpoint sample {sample_id!r} must be an object")
        return sample

    def _empty_state(self) -> dict[str, Any]:
        return {
            "version": CHECKPOINT_VERSION,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "runs": [],
            "samples": {},
        }

    def _load(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CheckpointCorruptionError(f"invalid checkpoint JSON: {self.path}") from exc
        except OSError as exc:
            raise CheckpointError(f"cannot read checkpoint: {self.path}") from exc

        if not isinstance(payload, dict):
            raise CheckpointCorruptionError("checkpoint root must be an object")
        if payload.get("version") != CHECKPOINT_VERSION:
            raise CheckpointCorruptionError(
                f"unsupported checkpoint version: {payload.get('version')!r}"
            )
        if not isinstance(payload.get("samples"), dict):
            raise CheckpointCorruptionError("checkpoint field 'samples' must be an object")
        if not isinstance(payload.get("runs"), list):
            raise CheckpointCorruptionError("checkpoint field 'runs' must be a list")
        return payload

    def _flush(self) -> None:
        self._state["updated_at"] = utc_now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = json.dumps(self._state, ensure_ascii=False, indent=2, default=str)
        try:
            tmp_path.write_text(payload, encoding="utf-8")
            tmp_path.replace(self.path)
        except OSError as exc:
            raise CheckpointError(f"cannot write checkpoint: {self.path}") from exc


def normalize_checkpoint_config(config: CheckpointConfig | dict[str, Any] | None) -> CheckpointConfig:
    if config is None:
        return CheckpointConfig()
    if isinstance(config, CheckpointConfig):
        return config
    if is_dataclass(config):
        return CheckpointConfig(**asdict(config))
    if isinstance(config, dict):
        return CheckpointConfig(**config)
    if hasattr(config, "items"):
        return CheckpointConfig(**dict(config.items()))
    raise CheckpointError(f"unsupported checkpoint config type: {type(config)}")
