from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from threading import Event, Lock, Thread
from time import monotonic, sleep

import numpy as np

from .audit import AuditStats, SessionAuditLogger
from .calibration import DRAIN_OVERHEAD_S, default_drain_timeout_seconds
from .contracts import DSPEvent, GemmaAnnotation, PCMBlock
from .dsp import DSPProcessor
from .ollama import GemmaQueueStats, GemmaWorker
from .ring import PCMBlockRing, RingStats


class WallClockPacer:
    def __init__(self, sample_rate_hz: int, *, clock: Callable[[], float] = monotonic,
                 sleeper: Callable[[float], None] = sleep, stop_event: Event | None = None) -> None:
        self.sample_rate_hz, self.clock, self.sleeper = sample_rate_hz, clock, sleeper
        self.origin: float | None = None
        self._stop_event = stop_event

    def set_stop_event(self, stop_event: Event) -> None:
        self._stop_event = stop_event

    def pace(self, block: PCMBlock) -> float | None:
        if self.origin is None:
            self.origin = self.clock() - block.start_frame / self.sample_rate_hz
        delay = self.origin + block.start_frame / self.sample_rate_hz - self.clock()
        if delay > 0:
            if self._stop_event is not None:
                if self._stop_event.wait(delay):
                    return None
            else:
                self.sleeper(delay)
        return max(delay, 0.0)


@dataclass(frozen=True)
class RuntimeSnapshot:
    position_frames: int
    rms: float
    peak: float
    waveform: tuple[float, ...]
    ring: RingStats
    ffmpeg_state: str
    ffmpeg_error: str | None
    underruns: int
    dsp_events: tuple[DSPEvent, ...]
    dsp_evidence: tuple[tuple[str, dict[str, object]], ...]
    dsp_audit: tuple[dict[str, int | str], ...]
    gemma_annotations: tuple[GemmaAnnotation, ...]
    gemma_queue: GemmaQueueStats | None
    audit_log: AuditStats | None


class SliceRuntime:
    """Paced PCM consumer with authoritative no-reference DSP in this worker thread."""
    def __init__(self, ring: PCMBlockRing, producer: object, *, waveform_points: int = 64,
                 pacer: WallClockPacer | None = None, gemma_worker: GemmaWorker | None = None,
                 source_label: str = "source unavailable") -> None:
        self.ring, self.producer, self.waveform_points = ring, producer, waveform_points
        self.source_label = source_label
        self._stop = Event()
        self.pacer = pacer or WallClockPacer(48_000, stop_event=self._stop)
        self.pacer.set_stop_event(self._stop)
        self.position_frames = self.underruns = 0
        self._waveform = np.zeros(waveform_points, dtype=np.float32)
        self._rms = self._peak = 0.0
        self.dsp = DSPProcessor()
        self._events: list[DSPEvent] = []
        self._annotations: list[GemmaAnnotation] = []
        self.gemma_worker = gemma_worker
        self.audit_log: SessionAuditLogger | None = None
        self._snapshot_lock = Lock()
        self._consumer_thread: Thread | None = None
        self.last_gemma_drain: dict[str, object] | None = None

    def start(self) -> None:
        if self._consumer_thread is not None and self._consumer_thread.is_alive():
            return
        self._stop.clear()
        if self.gemma_worker is not None and self.gemma_worker.stats().worker_state != "stopped":
            self.gemma_worker.start()
        self._consumer_thread = Thread(target=self._consume, name="paced-pcm-consumer", daemon=True)
        self._consumer_thread.start()

    def _consume(self) -> None:
        while not self._stop.is_set():
            if not self.step():
                self._stop.wait(0.005)

    def stop(self, *, graceful_gemma_drain: bool = False, gemma_drain_timeout_s: float | None = None) -> None:
        self._stop.set()
        if self._consumer_thread is not None:
            self._consumer_thread.join(timeout=0.2)
        with self._snapshot_lock:
            records = self.dsp.finish()
            self._events.extend(records)
        self._submit_annotations(records)
        self._audit_events(records)
        if self.gemma_worker is not None:
            if graceful_gemma_drain:
                stats = self.gemma_worker.stats()
                per_request_timeout = self.gemma_worker.client.config.timeout_s
                overhead = DRAIN_OVERHEAD_S if stats.pending else 0.0
                effective = gemma_drain_timeout_s if gemma_drain_timeout_s is not None else default_drain_timeout_seconds(
                    per_request_timeout, pending_requests=stats.pending, overhead_s=overhead)
                drained = self.gemma_worker.drain_and_stop(effective)
                final = self.gemma_worker.stats()
                self.last_gemma_drain = {"requested_budget_s": gemma_drain_timeout_s,
                                         "override_budget_s": gemma_drain_timeout_s,
                                         "per_request_timeout_s": per_request_timeout, "overhead_s": overhead,
                                         "effective_budget_s": effective, "accepted_pending_initial": stats.pending,
                                         "accepted_pending_final": final.pending, "pending_at_start": stats.pending,
                                         "submitted": final.submitted,
                                         "completed": final.completed, "errors": final.errors,
                                         "cancelled_pending": final.cancelled_pending, "pending": final.pending,
                                         "drained": drained, "reason": "shutdown_complete" if drained else "shutdown_drain_timeout"}
            else:
                self.gemma_worker.stop()
                final = self.gemma_worker.stats()
                self.last_gemma_drain = {"requested_budget_s": None, "override_budget_s": None,
                                         "per_request_timeout_s": self.gemma_worker.client.config.timeout_s, "overhead_s": 0.0,
                                         "effective_budget_s": 0.0, "accepted_pending_initial": final.pending,
                                         "accepted_pending_final": final.pending,
                                         "pending_at_start": final.pending, "submitted": final.submitted,
                                         "completed": final.completed, "errors": final.errors,
                                         "cancelled_pending": final.cancelled_pending, "pending": final.pending,
                                         "drained": False, "reason": "shutdown_cancel"}

    def gemma_drain_budget(self, requested_budget_s: float | None = None) -> float:
        if self.gemma_worker is None:
            return 0.0
        if requested_budget_s is not None:
            return requested_budget_s
        return default_drain_timeout_seconds(self.gemma_worker.client.config.timeout_s,
                                             pending_requests=self.gemma_worker.stats().pending)

    @property
    def consumer_alive(self) -> bool:
        return self._consumer_thread is not None and self._consumer_thread.is_alive()

    def step(self) -> bool:
        block = self.ring.pop()
        if block is None:
            if getattr(self.producer, "state", "") == "running":
                self.underruns += 1
            return False
        if self.pacer.pace(block) is None:
            return False
        mono = block.samples.mean(axis=1)
        rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
        peak = float(np.max(np.abs(mono)))
        indices = np.linspace(0, len(mono) - 1, self.waveform_points, dtype=int)
        with self._snapshot_lock:
            records = self.dsp.process(block)
            self._events.extend(records)
            if getattr(self.producer, "state", "") == "eof" and self.ring.stats().fill == 0:
                final_records = self.dsp.finish()
                self._events.extend(final_records)
                records.extend(final_records)
            self._rms, self._peak = rms, peak
            self._waveform[:] = mono[indices]
            self.position_frames = block.start_frame + block.frame_count
        self._submit_annotations(records)
        self._audit_events(records)
        return True

    def _audit_events(self, records: list[DSPEvent]) -> None:
        if self.audit_log is not None:
            timestamp = datetime.now().astimezone().isoformat()
            for record in records:
                self.audit_log.dsp_event(record, timestamp)

    def _submit_annotations(self, records: list[DSPEvent]) -> None:
        if self.gemma_worker is not None:
            # Copy deterministic evidence after the DSP transaction; queue insertion is non-blocking.
            evidence = tuple((key, value.to_dict()) for key, value in self.dsp.evidence.items())
            for record in records:
                self.gemma_worker.submit(record, evidence)

    def add_annotation(self, annotation: GemmaAnnotation) -> None:
        """Worker callback: annotations are separate and cannot mutate DSP records."""
        with self._snapshot_lock:
            self._annotations.append(annotation)
        if self.audit_log is not None:
            self.audit_log.gemma_annotation(annotation, datetime.now().astimezone().isoformat())

    def add_abandoned_annotation(self, annotation: GemmaAnnotation) -> None:
        """Immediate-shutdown outcome: audit it without reopening the normal worker callback gate."""
        self.add_annotation(annotation)

    def drain_paced(self, maximum_blocks: int = 4) -> int:
        """Advance audio between UI snapshots without exposing FFmpeg to the UI."""
        consumed = 0
        while consumed < maximum_blocks and self.step():
            consumed += 1
        return consumed

    def snapshot(self) -> RuntimeSnapshot:
        with self._snapshot_lock:
            evidence = tuple((key, value.to_dict()) for key, value in sorted(self.dsp.evidence.items()))
            return RuntimeSnapshot(self.position_frames, self._rms, self._peak,
                tuple(float(value) for value in self._waveform), self.ring.stats(),
                getattr(self.producer, "state", "unknown"), getattr(self.producer, "error", None), self.underruns,
                tuple(self._events), evidence, tuple(self.dsp.audit), tuple(self._annotations),
                self.gemma_worker.stats() if self.gemma_worker else None,
                self.audit_log.stats() if self.audit_log else None)
