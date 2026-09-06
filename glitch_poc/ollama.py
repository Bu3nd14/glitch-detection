"""Bounded, structured-only local Ollama annotation integration."""
from __future__ import annotations

import json
import math
import multiprocessing
import queue
import re
import time
from base64 import b64decode
from binascii import Error as Base64Error
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from threading import Condition, Event, Lock, Thread
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .contracts import DSPEvent, GemmaAnnotation

MODEL = "gemma4:e4b"
PROMPT_SCHEMA_VERSION = "gemma-grounded-v2"
ANNOTATION_VERSION = "gemma-annotation-v1"
_MAX_PAYLOAD_BYTES = 32_768
_MAX_RESPONSE_BYTES = 16_384
_MAX_DEPTH = 5
_MAX_ITEMS = 64
_IDENTIFIER_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")
_FEATURE_NAMES = frozenset({"derivative_peak", "context_rms", "near_peak_ratio", "peak_level",
    "correlation_lag_80ms", "rms_ratio", "correlation_lag_500ms", "correlation_lag_1000ms",
    "rms", "baseline_rms", "floor_ratio"})
_EVIDENCE_FIELDS = frozenset({"stream_id", "start_frame", "end_frame", "sample_rate_hz", "window_ms",
                               "features", "profile_id"})
_FEATURES_BY_TYPE = {
    "click": frozenset({"derivative_peak", "context_rms"}),
    "clipping": frozenset({"near_peak_ratio", "peak_level"}),
    "stutter": frozenset({"correlation_lag_80ms", "rms_ratio"}),
    "dropout": frozenset({"rms", "baseline_rms", "rms_ratio", "floor_ratio"}),
    "loop": frozenset({"correlation_lag_500ms", "correlation_lag_1000ms"}),
}
_MAX_GROUNDED_EVIDENCE = 2
_PROCESS_CONTEXT = multiprocessing.get_context("spawn")


@dataclass(frozen=True)
class OllamaConfig:
    endpoint: str = "http://127.0.0.1:11434/api/generate"
    timeout_s: float = 10.0
    queue_capacity: int = 16
    dedupe_capacity: int = 256
    max_retries: int = 0  # POST generation is deliberately not retried: it is not idempotent.


@dataclass(frozen=True)
class GemmaQueueStats:
    capacity: int
    queued: int
    submitted: int
    dropped_backpressure: int
    completed: int
    invalid: int
    errors: int
    cancelled_pending: int
    pending: int
    worker_state: str


@dataclass(frozen=True)
class _AnnotationRequest:
    event: DSPEvent
    evidence: tuple[tuple[str, dict[str, object]], ...]


def validate_local_ollama_url(url: str) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username or parsed.password):
        raise ValueError("Ollama endpoint must use http on localhost, 127.0.0.1, or ::1")


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 160 or not set(value) <= _IDENTIFIER_CHARS:
        raise ValueError(f"invalid {name}")
    if len(value) >= 24 and len(value) % 4 == 0:
        try:
            if b64decode(value, validate=True):
                raise ValueError(f"encoded data is not valid {name}")
        except Base64Error:
            pass
        except ValueError:
            raise
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"invalid {name}")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"invalid {name}")
    return float(value)


def _canonical_event(event: DSPEvent, evidence_ids: tuple[str, ...]) -> dict[str, object]:
    if event.lifecycle != "CLOSED" or event.status not in {"detected", "uncertain"}:
        raise ValueError("only consolidated detected or uncertain events may be annotated")
    return {"event_id": _identifier(event.event_id, "event_id"), "revision": _integer(event.revision, "revision"),
        "lifecycle": event.lifecycle, "status": event.status, "glitch_type": event.glitch_type,
        "start_frame": _integer(event.start_frame, "start_frame"), "end_frame": _integer(event.end_frame, "end_frame"),
        "emitted_frame": _integer(event.emitted_frame, "emitted_frame"), "raw_score": _number(event.raw_score, "raw_score"),
        "detector_ids": [_identifier(item, "detector_id") for item in event.detector_ids],
        "evidence_ids": [_identifier(item, "evidence_id") for item in evidence_ids],
        "profile_id": _identifier(event.profile_id, "profile_id"), "epoch_id": _integer(event.epoch_id, "epoch_id"),
        "context_complete": event.context_complete}


def selected_evidence_ids(event: DSPEvent, evidence: tuple[tuple[str, dict[str, object]], ...]) -> tuple[str, ...]:
    """Newest deterministic evidence windows with features relevant to the final detector type."""
    by_id = dict(evidence)
    if len(by_id) != len(evidence) or len(event.evidence_ids) > _MAX_ITEMS:
        raise ValueError("invalid evidence collection")
    allowed = _FEATURES_BY_TYPE.get(event.glitch_type)
    if allowed is None:
        raise ValueError("unrecognized glitch type")
    for evidence_id in event.evidence_ids:
        source = by_id.get(evidence_id)
        if source is None or set(source) != _EVIDENCE_FIELDS or not isinstance(source["features"], dict):
            raise ValueError("unrecognized evidence structure")
        features = source["features"]
        if len(features) > _MAX_ITEMS or not set(features) <= _FEATURE_NAMES:
            raise ValueError("unrecognized evidence feature")
    selected: list[str] = []
    for evidence_id in reversed(event.evidence_ids):
        source = by_id.get(evidence_id)
        assert source is not None
        selected_features = source["features"]
        assert isinstance(selected_features, dict)
        if set(selected_features) & allowed:
            selected.append(evidence_id)
        if len(selected) == _MAX_GROUNDED_EVIDENCE:
            break
    if not selected:
        raise ValueError("no relevant grounded evidence")
    return tuple(reversed(selected))


def _canonical_evidence(event: DSPEvent, evidence: tuple[tuple[str, dict[str, object]], ...],
                        evidence_ids: tuple[str, ...]) -> list[dict[str, object]]:
    by_id = dict(evidence)
    allowed = _FEATURES_BY_TYPE[event.glitch_type]
    result: list[dict[str, object]] = []
    for evidence_id in evidence_ids:
        source = by_id[evidence_id]
        features = source["features"]
        assert isinstance(features, dict)
        compact = {name: _number(value, name) for name, value in sorted(features.items()) if name in allowed}
        if not compact:
            raise ValueError("no relevant grounded feature")
        result.append({"evidence_id": _identifier(evidence_id, "evidence_id"),
            "values": compact})
    return result


def annotation_payload(event: DSPEvent, evidence: tuple[tuple[str, dict[str, object]], ...]) -> dict[str, object]:
    """Build the only model input: a JSON document of canonical DSP facts."""
    evidence_ids = selected_evidence_ids(event, evidence)
    canonical_event = _canonical_event(event, evidence_ids)
    grounded = {
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "instruction": ("Return minified JSON only: no markdown, reasoning, or extra fields. explanation <=400 chars. "
                        "Describe coherence of provided DSP facts only; cannot modify DSP verdict, event, score, interval, or evidence."),
        "output_schema": {
            "event_id": "string", "annotation_status": "coherent|insufficient_evidence|error",
            "glitch_type_annotation": "string|null", "confidence": "low|medium|high",
            "supporting_evidence_ids": ["input evidence_id"], "explanation": "string",
            "cannot_override_dsp": True,
        },
        "dsp_event": canonical_event,
        "evidence": _canonical_evidence(event, evidence, evidence_ids),
        "profile": {"profile_id": canonical_event["profile_id"]},
        "descriptive_context": {"context_complete": event.context_complete},
    }
    payload: dict[str, object] = {"model": MODEL, "stream": False, "format": "json",
                                   "options": {"temperature": 0, "num_predict": 256}, "keep_alive": "5m",
                                  "prompt": json.dumps(grounded, separators=(",", ":"), sort_keys=True)}
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_PAYLOAD_BYTES:
        raise ValueError("Ollama annotation payload exceeds limit")
    return payload


class OllamaClient:
    def __init__(self, config: OllamaConfig | None = None) -> None:
        config = config or OllamaConfig()
        validate_local_ollama_url(config.endpoint)
        self.config = config

    def annotate(self, event: DSPEvent, evidence: tuple[tuple[str, dict[str, object]], ...]) -> GemmaAnnotation:
        started = time.time()
        try:
            payload = annotation_payload(event, evidence)
            request = Request(self.config.endpoint, data=json.dumps(payload).encode("utf-8"),
                              headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=self.config.timeout_s) as response:
                raw = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                raise ValueError("Ollama response exceeds limit")
            outer = json.loads(raw)
            if not isinstance(outer, dict) or not isinstance(outer.get("response"), str):
                raise TypeError("Ollama response lacks JSON response string")
            parsed = json.loads(outer["response"])
            return self._validated(event, selected_evidence_ids(event, evidence), parsed, outer, started)
        except (URLError, TimeoutError, OSError) as error:
            return self._error(event, started, f"transport: {error}")
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            return self._error(event, started, f"invalid: {error}")

    def _validated(self, event: DSPEvent, allowed_ids: tuple[str, ...], value: object, outer: dict[str, object], started: float) -> GemmaAnnotation:
        required = {"event_id", "annotation_status", "glitch_type_annotation", "confidence",
                    "supporting_evidence_ids", "explanation", "cannot_override_dsp"}
        if not isinstance(value, dict) or set(value) != required:
            return self._error(event, started, "invalid: output schema")
        ids = value["supporting_evidence_ids"]
        if (value["event_id"] != event.event_id or value["annotation_status"] not in
                {"coherent", "insufficient_evidence", "error"} or value["confidence"] not in {"low", "medium", "high"}
                or value["glitch_type_annotation"] not in {None, *({event.glitch_type})}
                or not isinstance(ids, list) or not all(isinstance(item, str) and item in allowed_ids for item in ids)
                or not isinstance(value["explanation"], str) or len(value["explanation"]) > 400
                or value["cannot_override_dsp"] is not True):
            return self._error(event, started, "invalid: ungrounded output")
        metadata = tuple(sorted((key, str(outer[key])) for key in ("model", "created_at") if key in outer))
        return GemmaAnnotation(ANNOTATION_VERSION, event.event_id, event.revision, str(value["annotation_status"]),
            value["glitch_type_annotation"], str(value["confidence"]), tuple(ids), value["explanation"], True,
            started, MODEL, metadata, PROMPT_SCHEMA_VERSION, self.config.timeout_s)

    def _error(self, event: DSPEvent, started: float, error: str) -> GemmaAnnotation:
        return GemmaAnnotation(ANNOTATION_VERSION, event.event_id, event.revision, "error", None, None, (), "", True,
            started, MODEL, (), PROMPT_SCHEMA_VERSION, self.config.timeout_s, error)


def _annotation_executor(client: OllamaClient, receive: Any, send: Any) -> None:
    """Spawned once from the main runtime thread; never forked or spawned from the annotation thread."""
    try:
        while True:
            item = receive.recv()
            if item is None:
                return
            try:
                send.send(("annotation", client.annotate(item.event, item.evidence)))
            except Exception as error:  # noqa: BLE001 - isolate arbitrary client failures in the child.
                send.send(("error", f"client exception: {type(error).__name__}"))
    except EOFError:
        return
    finally:
        receive.close()
        send.close()


class GemmaWorker:
    """Single-concurrency bounded worker; enqueue never waits and never affects DSP."""
    def __init__(self, client: OllamaClient, on_annotation: Callable[[GemmaAnnotation], None],
                 on_abandoned: Callable[[GemmaAnnotation], None] | None = None) -> None:
        self.client, self.on_annotation, self.on_abandoned = client, on_annotation, on_abandoned
        self._queue: queue.Queue[_AnnotationRequest] = queue.Queue(maxsize=client.config.queue_capacity)
        self._stop, self._lock = Event(), Lock()
        self._pending_idle = Condition(self._lock)
        self._dispatch_lock = Lock()
        self._dispatch_idle = Condition(self._dispatch_lock)
        self._accept_callbacks = True
        self._callbacks_inflight = 0
        self._seen: OrderedDict[tuple[str, int, str], None] = OrderedDict()
        self._pending: dict[tuple[str, int, str], _AnnotationRequest] = {}
        self._submitted = self._dropped = self._completed = self._invalid = self._errors = self._cancelled = 0
        self._state = "idle"
        self._thread: Thread | None = None
        self._executor_process: Any | None = None
        self._executor_send: Any | None = None
        self._executor_receive: Any | None = None
        self._executor_start_error: str | None = None

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            with self._dispatch_lock:
                self._accept_callbacks = True
            self._start_executor()
            self._thread = Thread(target=self._run, name="gemma-annotation")
            self._thread.start()

    def submit(self, event: DSPEvent, evidence: tuple[tuple[str, dict[str, object]], ...]) -> bool:
        if event.lifecycle != "CLOSED" or event.status not in {"detected", "uncertain"}: return False
        key = (event.event_id, event.revision, event.lifecycle)
        with self._lock:
            if key in self._seen:
                self._seen.move_to_end(key)
                return False
            try: self._queue.put_nowait(_AnnotationRequest(event, evidence))
            except queue.Full:
                self._dropped += 1; return False
            self._seen[key] = None
            self._pending[key] = _AnnotationRequest(event, evidence)
            if len(self._seen) > self.client.config.dedupe_capacity:
                self._seen.popitem(last=False)
            self._submitted += 1
            return True

    def _run(self) -> None:
        while not self._stop.is_set():
            try: item = self._queue.get(timeout=.05)
            except queue.Empty: continue
            with self._lock: self._state = "analyzing"
            annotation = self._invoke(item)
            if annotation is None:
                self._queue.task_done()
                break
            reserved = self._reserve_callback()
            if reserved:
                delivered = False
                if self._can_invoke_reserved():
                    try:
                        self.on_annotation(annotation)
                        delivered = True
                    except Exception:  # noqa: BLE001 - a UI/log callback must not kill the annotation worker.
                        with self._lock:
                            self._errors += 1
                self._release_callback()
                if delivered:
                    self._complete(item, annotation)
            self._queue.task_done()
        with self._lock:
            self._state = "stopped"

    def _invoke(self, item: _AnnotationRequest) -> GemmaAnnotation | None:
        """Wait on the executor started before Textual owns terminal file descriptors."""
        process = self._executor_process
        receive = self._executor_receive
        send = self._executor_send
        if process is None or receive is None or send is None:
            return self._error(item.event, self._executor_start_error or "client executor unavailable")
        try:
            send.send(item)
            while process.is_alive():
                if self._stop.is_set():
                    return None
                if receive.poll(.02):
                    kind, value = receive.recv()
                    if kind == "annotation" and isinstance(value, GemmaAnnotation):
                        return value
                    return self._error(item.event, str(value))
            if receive.poll():
                kind, value = receive.recv()
                if kind == "annotation" and isinstance(value, GemmaAnnotation): return value
                return self._error(item.event, str(value))
            return self._error(item.event, "client process exited")
        except (BrokenPipeError, EOFError, OSError) as error:
            return self._error(item.event, f"client executor: {_safe_error(error)}")

    def _error(self, event: DSPEvent, error: str) -> GemmaAnnotation:
        return GemmaAnnotation(ANNOTATION_VERSION, event.event_id, event.revision, "error", None, None, (), "", True,
            time.time(), MODEL, (), PROMPT_SCHEMA_VERSION, self.client.config.timeout_s, error)

    def stop(self) -> None:
        self._halt("shutdown_cancel", self.on_abandoned)

    def drain_and_stop(self, timeout_s: float) -> bool:
        """Drain accepted work after normal EOF; synthesize explicit errors when budget expires."""
        with self._pending_idle:
            self._state = "draining"
            drained = self._pending_idle.wait_for(lambda: not self._pending, timeout=max(0.0, timeout_s))
        if drained:
            self._halt("shutdown_complete", None)
            return True
        self._halt("shutdown_drain_timeout", self.on_annotation)
        return False

    def _halt(self, reason: str, terminal_callback: Callable[[GemmaAnnotation], None] | None) -> None:
        self._stop.set()
        with self._dispatch_idle:
            self._accept_callbacks = False
            if not self._dispatch_idle.wait_for(lambda: self._callbacks_inflight == 0, timeout=.5):
                raise RuntimeError("Gemma annotation callback did not stop within bound")
        self._stop_executor()
        cancelled_queue = 0
        while True:
            try:
                self._queue.get_nowait(); self._queue.task_done(); cancelled_queue += 1
            except queue.Empty: break
        if self._thread is not None: self._thread.join(timeout=.5)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Gemma worker did not stop within bound")
            self._state = "stopped"
            pending = list(self._pending.values())
            self._pending.clear()
            self._completed += len(pending)
            self._errors += len(pending)
            self._cancelled += max(cancelled_queue, len(pending))
            self._pending_idle.notify_all()
        if terminal_callback is not None:
            for item in pending:
                try:
                    terminal_callback(self._error(item.event, reason))
                except Exception:  # noqa: BLE001 - shutdown outcomes must not escape to DSP/UI.
                    with self._lock:
                        self._errors += 1

    @property
    def worker_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def process_start_method(self) -> str:
        return _PROCESS_CONTEXT.get_start_method()

    def _reserve_callback(self) -> bool:
        with self._dispatch_lock:
            if not self._accept_callbacks:
                return False
            self._callbacks_inflight += 1
            return True

    def _release_callback(self) -> None:
        with self._dispatch_idle:
            self._callbacks_inflight -= 1
            if self._callbacks_inflight == 0:
                self._dispatch_idle.notify_all()

    def _complete(self, item: _AnnotationRequest, annotation: GemmaAnnotation) -> None:
        key = (item.event.event_id, item.event.revision, item.event.lifecycle)
        with self._pending_idle:
            self._pending.pop(key, None)
            self._completed += 1
            if annotation.annotation_status == "error":
                self._errors += 1
                self._invalid += annotation.error is not None and annotation.error.startswith("invalid:")
            self._state = "idle"
            if not self._pending:
                self._pending_idle.notify_all()

    def _can_invoke_reserved(self) -> bool:
        with self._dispatch_lock:
            return self._accept_callbacks

    def _start_executor(self) -> None:
        self._executor_start_error = None
        child_receive, parent_send = _PROCESS_CONTEXT.Pipe(duplex=False)
        parent_receive, child_send = _PROCESS_CONTEXT.Pipe(duplex=False)
        process = _PROCESS_CONTEXT.Process(target=_annotation_executor,
            args=(self.client, child_receive, child_send), name="gemma-ollama-executor")
        try:
            process.start()
        except (OSError, TypeError, ValueError) as error:
            child_receive.close(); parent_send.close(); parent_receive.close(); child_send.close()
            self._executor_start_error = f"client executor start: {_safe_error(error)}"
            return
        child_receive.close()
        child_send.close()
        self._executor_process, self._executor_send, self._executor_receive = process, parent_send, parent_receive

    def _stop_executor(self) -> None:
        process = self._executor_process
        if process is None:
            return
        if process.is_alive():
            process.terminate()
        process.join(timeout=.1)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=.1)
        if process.is_alive():
            raise RuntimeError("Gemma child executor did not terminate")
        if self._executor_send is not None:
            self._executor_send.close()
        if self._executor_receive is not None:
            self._executor_receive.close()
        self._executor_process = self._executor_send = self._executor_receive = None

    def stats(self) -> GemmaQueueStats:
        with self._lock:
            return GemmaQueueStats(self.client.config.queue_capacity, self._queue.qsize(), self._submitted, self._dropped,
                self._completed, self._invalid, self._errors, self._cancelled, len(self._pending), self._state)


def _safe_error(error: BaseException) -> str:
    """Useful process-start diagnostics without exposing paths, URIs, payloads, or arbitrary long text."""
    detail = " ".join(str(error).split())
    detail = re.sub(r"(?:[A-Za-z][A-Za-z0-9+.-]*://|/)[^\s]+", "[redacted]", detail)
    detail = re.sub(r"(?i)(audio|pcm|waveform|spectrogram|base64|secret)[^\s]*", "[redacted]", detail)
    return f"{type(error).__name__}: {detail[:160] or 'no detail'}"
