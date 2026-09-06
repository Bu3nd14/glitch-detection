"""Asynchronous, append-only session audit records; never part of the DSP decision path."""
from __future__ import annotations

import json
import queue
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

from .contracts import DSPEvent, GemmaAnnotation

AUDIT_SCHEMA_VERSION = "session-audit-v1"


@dataclass(frozen=True)
class AuditStats:
    state: str
    path: str | None
    queued: int
    dropped: int
    errors: int
    last_error: str | None


def validate_log_dir(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("--log-dir must be a relative path contained by the project root")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError("--log-dir must be contained by the project root")
    return resolved


class SessionAuditLogger:
    """Bounded producer queue and one writer thread; producers only use put_nowait."""
    def __init__(self, root: Path, log_dir: str, *, fixture: str, stream_id: str, profile_id: str,
                  gemma_enabled: bool, clock: Callable[[], datetime] = lambda: datetime.now().astimezone(),
                  capacity: int = 512, source_kind: str = "fixture", source_fingerprint: str | None = None) -> None:
        self._lock = threading.Lock()
        self._queue: queue.Queue[dict[str, object] | None] = queue.Queue(maxsize=capacity)
        self._closed = False
        self._dropped = self._errors = 0
        self._last_error: str | None = None
        self.session_id = uuid.uuid4().hex
        self._state = "starting"
        self.path: Path | None = None
        self._file: TextIO | None = None
        self._thread: threading.Thread | None = None
        try:
            directory = validate_log_dir(root, log_dir)
            directory.mkdir(parents=True, exist_ok=True)
            moment = clock()
            stamp = moment.strftime("%Y%m%d-%H%M%S")
            index = 0
            while True:
                suffix = "" if index == 0 else f"-{index}"
                candidate = directory / f"session-{stamp}{suffix}.jsonl"
                try:
                    self._file = candidate.open("x", encoding="utf-8")
                    self.path = candidate
                    break
                except FileExistsError:
                    index += 1
            self._state = "running"
            self._thread = threading.Thread(target=self._run, name="session-audit-writer")
            self._thread.start()
            self._submit({"record_type": "session_started", "schema_version": AUDIT_SCHEMA_VERSION,
                "session_id": self.session_id, "timestamp": moment.isoformat(), "fixture": fixture,
                "stream_id": stream_id, "source_kind": source_kind, "source_fingerprint": source_fingerprint,
                "profile_id": profile_id,
                "app": {"name": "glitch-detection-poc", "version": "0.1.0"},
                "gemma": {"enabled": gemma_enabled, "model": "gemma4:e4b", "local_loopback_only": True}})
        except (OSError, ValueError) as error:
            self._state = "unavailable"
            self._last_error = f"{type(error).__name__}: {error}"

    def _submit(self, record: dict[str, object]) -> None:
        with self._lock:
            if self._closed or self._state != "running":
                self._errors += 1
                return
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                self._dropped += 1

    def dsp_event(self, event: DSPEvent, timestamp: str) -> None:
        self._submit({"record_type": "dsp_event", "schema_version": AUDIT_SCHEMA_VERSION,
            "session_id": self.session_id, "timestamp": timestamp, "correlation_id": event.event_id,
            "event_id": event.event_id, "revision": event.revision, "lifecycle": event.lifecycle,
            "status": event.status, "glitch_type": event.glitch_type, "start_frame": event.start_frame,
            "end_frame": event.end_frame, "emitted_frame": event.emitted_frame, "raw_score": event.raw_score,
            "evidence_ids": list(event.evidence_ids), "detector_ids": list(event.detector_ids),
            "epoch_id": event.epoch_id, "profile_id": event.profile_id})

    def gemma_annotation(self, annotation: GemmaAnnotation, timestamp: str) -> None:
        self._submit({"record_type": "gemma_annotation", "schema_version": AUDIT_SCHEMA_VERSION,
            "session_id": self.session_id, "timestamp": timestamp, "correlation_id": annotation.event_id,
            "event_id": annotation.event_id, "revision": annotation.event_revision,
            "annotation_version": annotation.annotation_version, "annotation_status": annotation.annotation_status,
            "glitch_type_annotation": annotation.glitch_type_annotation, "confidence": annotation.confidence,
            "supporting_evidence_ids": list(annotation.supporting_evidence_ids), "explanation": annotation.explanation,
            "error": annotation.error, "cannot_override_dsp": annotation.cannot_override_dsp,
            "model": annotation.model, "model_metadata": dict(annotation.model_metadata),
            "prompt_schema_version": annotation.prompt_schema_version, "timeout_s": annotation.timeout_s})

    def close(self, reason: str, summary: dict[str, object], timestamp: str) -> None:
        self._submit({"record_type": "session_ended", "schema_version": AUDIT_SCHEMA_VERSION,
            "session_id": self.session_id, "timestamp": timestamp, "reason": reason, "summary": summary})
        with self._lock:
            self._closed = True
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=1)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._errors += 1
                self._last_error = "writer did not stop"
            elif self._state == "running":
                self._state = "closed"

    def stats(self) -> AuditStats:
        with self._lock:
            return AuditStats(self._state, str(self.path) if self.path else None, self._queue.qsize(), self._dropped,
                self._errors, self._last_error)

    def _run(self) -> None:
        assert self._file is not None
        try:
            while True:
                record = self._queue.get()
                if record is None:
                    return
                self._file.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
                self._file.flush()
        except OSError as error:
            with self._lock:
                self._errors += 1
                self._last_error = f"writer: {type(error).__name__}"
        finally:
            self._file.close()
