"""Incremental, deterministic no-reference DSP for the constrained poc-d2-v2 POC."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from .contracts import DSPCandidate, DSPEvent, FeatureWindow, PCMBlock


@dataclass(frozen=True)
class DSPProfile:
    profile_id: str = "poc-d2-v2"
    hop_ms: int = 10
    merge_ms: int = 50
    cooldown_ms: int = 100
    click_derivative: float = 0.50
    clip_level: float = 0.44
    clip_ratio: float = 0.08
    repeat_correlation: float = 0.995
    loop_correlation: float = 0.998


@dataclass
class _Open:
    event_id: str
    kind: str
    start: int
    end: int
    raw_score: float
    detectors: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    revision: int = 1
    status: str = "detected"
    context_complete: bool = True
    exit_hops: int = 0
    close_at: int | None = None
    phase: str = "ACTIVE"


class EventBuilder:
    """Append-only lifecycle builder with per-class hysteresis and no cross-epoch state."""
    _priority: ClassVar = {"clipping": 4, "dropout": 3, "stutter": 2, "loop": 1, "click": 0}

    def __init__(self, sample_rate_hz: int, profile: DSPProfile, epoch_id: int = 0) -> None:
        self.sample_rate_hz, self.profile, self.epoch_id = sample_rate_hz, profile, epoch_id
        self._open: dict[str, _Open] = {}
        self._cooldown_until: dict[str, int] = {}
        self._observed: set[str] = set()
        self._sequence = 0

    def observe(self, candidate: DSPCandidate, emitted_frame: int) -> list[DSPEvent]:
        kind = candidate.candidate_type
        current = self._open.get(kind)
        if current is not None and candidate.start_frame > current.end + self._frames(self.profile.merge_ms):
            # ``observe`` may be called sparsely in tests or after scheduler delay: frame ranges,
            # not call cadence, define the merge contract.
            current.close_at = current.end + self._frames(self.profile.merge_ms)
            current.phase = "MERGE_WAIT"
            closed = self._close(kind, current, emitted_frame, "merge_gap_elapsed")
            return closed + self.observe(candidate, emitted_frame)
        if (current is not None and current.phase == "MERGE_WAIT" and current.close_at is not None
                and candidate.start_frame > current.close_at):
            closed = self._close(kind, current, emitted_frame, "merge_gap_elapsed")
            return closed + self.observe(candidate, emitted_frame)
        if current is None:
            if candidate.raw_score < 1.0 or candidate.start_frame < self._cooldown_until.get(kind, 0):
                return []
            event_id = f"{candidate.profile_id}:e{self.epoch_id}:{self._sequence + 1:05d}"
            conflicts = [item for item in self._open.values()
                         if candidate.start_frame < item.end and candidate.end_frame > item.start]
            if any(self._priority[item.kind] > self._priority[kind] for item in conflicts):
                return []
            superseded: list[DSPEvent] = []
            for item in conflicts:
                if self._priority[item.kind] < self._priority[kind]:
                    item.revision += 1
                    superseded.append(self._record(item, "SUPERSEDED", emitted_frame, close_reason="priority_conflict",
                                                   superseded_by=event_id))
                    del self._open[item.kind]
            self._observed.add(kind)
            self._sequence += 1
            current = _Open(
                event_id, kind,
                candidate.start_frame, candidate.end_frame, candidate.raw_score,
                [candidate.detector_name], [candidate.evidence_id], status=self._initial_status(kind),
                context_complete=kind not in {"dropout", "loop"},
            )
            self._open[kind] = current
            return superseded + [self._record(current, "OPENED", emitted_frame)]
        if candidate.raw_score < .50:
            return []
        self._observed.add(kind)
        if candidate.raw_score >= 0.75:
            current.end = max(current.end, candidate.end_frame)
            current.raw_score = max(current.raw_score, candidate.raw_score)
            self._append_unique(current.detectors, candidate.detector_name)
            self._append_unique(current.evidence, candidate.evidence_id)
            current.exit_hops, current.close_at = 0, None
            current.phase = "ACTIVE"
            current.revision += 1
            return [self._record(current, "UPDATED", emitted_frame)]
        return []

    def advance(self, emitted_frame: int) -> list[DSPEvent]:
        records: list[DSPEvent] = []
        for kind, current in list(self._open.items()):
            if kind not in self._observed:
                current.exit_hops += 1
                required = 1 if kind == "click" else 2
                if current.exit_hops >= required and current.close_at is None:
                    current.close_at = emitted_frame + self._frames(self.profile.merge_ms)
                    current.phase = "MERGE_WAIT"
            if current.close_at is not None and emitted_frame >= current.close_at:
                records.extend(self._close(kind, current, emitted_frame, "signal_exit"))
        self._observed.clear()
        return records

    def set_status(self, kind: str, status: str, emitted_frame: int, *, context_complete: bool,
                   force: bool = False) -> list[DSPEvent]:
        current = self._open.get(kind)
        if current is None:
            return []
        if not force and current.status == status and current.context_complete == context_complete:
            return []
        current.status, current.context_complete = status, context_complete
        current.revision += 1
        return [self._record(current, "UPDATED", emitted_frame)]

    def reset(self, emitted_frame: int, reason: str) -> list[DSPEvent]:
        records: list[DSPEvent] = []
        for kind in sorted(self._open, key=lambda value: -self._priority[value]):
            current = self._open[kind]
            current.revision += 1
            records.append(self._record(current, "CANCELLED", emitted_frame, context_complete=False,
                                        close_reason=reason))
        self._open.clear()
        self._cooldown_until.clear()
        self._observed.clear()
        return records

    def flush(self, emitted_frame: int) -> list[DSPEvent]:
        records: list[DSPEvent] = []
        for kind in sorted(self._open, key=lambda value: -self._priority[value]):
            current = self._open[kind]
            current.revision += 1
            records.append(self._record(current, "CLOSED", emitted_frame, close_reason="eos"))
        self._open.clear()
        return records

    def _close(self, kind: str, current: _Open, emitted_frame: int, reason: str) -> list[DSPEvent]:
        current.revision += 1
        record = self._record(current, "CLOSED", emitted_frame, close_reason=reason)
        # Cooldown starts at the actual merge deadline, not at a delayed next candidate.
        closed_frame = current.close_at if current.close_at is not None else emitted_frame
        self._cooldown_until[kind] = closed_frame + self._frames(self.profile.cooldown_ms)
        del self._open[kind]
        return [record]

    def _record(self, current: _Open, lifecycle: str, emitted: int, *, context_complete: bool | None = None,
                close_reason: str | None = None, superseded_by: str | None = None) -> DSPEvent:
        return DSPEvent(current.event_id, current.revision, lifecycle, current.status, current.kind,
                        current.start, current.end, emitted, current.raw_score, tuple(current.detectors),
                        tuple(current.evidence), self.profile.profile_id, self.epoch_id,
                        current.context_complete if context_complete is None else context_complete, close_reason,
                        superseded_by)

    def _frames(self, milliseconds: int) -> int:
        return self.sample_rate_hz * milliseconds // 1000

    @staticmethod
    def _append_unique(items: list[str], item: str) -> None:
        if item not in items:
            items.append(item)

    @staticmethod
    def _initial_status(kind: str) -> str:
        return "uncertain" if kind in {"dropout", "loop"} else "detected"


class DSPProcessor:
    """Consumer-thread DSP; chunk boundaries are normalized into deterministic 10 ms hops."""
    def __init__(self, sample_rate_hz: int = 48_000, profile: DSPProfile | None = None) -> None:
        self.sample_rate_hz, self.profile = sample_rate_hz, profile or DSPProfile()
        self.hop_frames = sample_rate_hz * self.profile.hop_ms // 1000
        self.epoch_id = 0
        self.builder = EventBuilder(sample_rate_hz, self.profile, self.epoch_id)
        self.evidence: dict[str, FeatureWindow] = {}
        self.audit: list[dict[str, int | str]] = []
        self._history: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self._pending: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self._pending_start = 0
        self._input_end: int | None = None
        self._sequence: int | None = None
        self._stream_id = "observed"
        self._active_rms: deque[float] = deque(maxlen=50)
        self._low_start: int | None = None
        self._low_hops = self._descent_hops = 0
        self._low_floor = 1.0
        self._dropout_post: list[float] | None = None
        self._dropout_baseline = 0.0
        self._previous_q: float | None = None
        self._status_records: list[DSPEvent] = []

    def process(self, block: PCMBlock) -> list[DSPEvent]:
        if block.sample_rate_hz != self.sample_rate_hz:
            raise ValueError("PCMBlock sample rate does not match DSPProcessor")
        self._stream_id = block.stream_id
        reason = self._gap_reason(block)
        records: list[DSPEvent] = []
        if reason is not None:
            records.extend(self._reset(block.start_frame, reason, block.missing_frames))
        self._sequence, self._input_end = block.sequence, block.start_frame + block.frame_count
        if not len(self._pending):
            self._pending_start = block.start_frame
        self._pending = np.concatenate((self._pending, block.samples.mean(axis=1, dtype=np.float32)))
        while len(self._pending) >= self.hop_frames:
            hop, self._pending = self._pending[:self.hop_frames], self._pending[self.hop_frames:]
            hop_start, self._pending_start = self._pending_start, self._pending_start + self.hop_frames
            records.extend(self._process_hop(hop, hop_start))
        return records

    def finish(self) -> list[DSPEvent]:
        frame = self._pending_start + len(self._pending)
        records: list[DSPEvent] = []
        if self._low_start is not None:
            records.extend(self.builder.set_status("dropout", "uncertain", frame, context_complete=False,
                                                   force=True))
        records.extend(self.builder.flush(frame))
        return records

    def _gap_reason(self, block: PCMBlock) -> str | None:
        if block.discontinuity or block.missing_frames:
            return "discontinuity"
        if self._input_end is not None and block.start_frame != self._input_end:
            return "frame_gap"
        if self._sequence is not None and block.sequence != self._sequence + 1:
            return "sequence_gap"
        return None

    def _reset(self, frame: int, reason: str, missing_frames: int) -> list[DSPEvent]:
        records = self.builder.reset(frame, reason)
        self.epoch_id += 1
        self.builder = EventBuilder(self.sample_rate_hz, self.profile, self.epoch_id)
        self._history = np.empty(0, dtype=np.float32)
        self._pending = np.empty(0, dtype=np.float32)
        self._active_rms.clear()
        self._low_start = self._dropout_post = None
        self._low_hops = self._descent_hops = 0
        self._previous_q = None
        self.audit.append({"kind": "dsp_reset", "frame": frame, "missing_frames": missing_frames,
                           "reason": reason, "epoch_id": self.epoch_id})
        return records

    def _process_hop(self, hop: NDArray[np.float32], start: int) -> list[DSPEvent]:
        end = start + len(hop)
        data = np.concatenate((self._history, hop))
        candidates = self._click(data, start, end) + self._clipping(hop, start, end)
        candidates += self._stutter(data, start, end) + self._loop(data, start, end)
        self._status_records = []
        candidates += self._dropout(hop, start, end)
        records: list[DSPEvent] = list(self._status_records)
        for candidate in candidates:
            records.extend(self.builder.observe(candidate, end))
        records.extend(self.builder.advance(end))
        self._history = data[-int(2.1 * self.sample_rate_hz):].copy()
        return records

    def _candidate(self, detector: str, kind: str, start: int, end: int, score: float, threshold: float,
                   window_ms: int, values: dict[str, float]) -> DSPCandidate:
        evidence_id = f"{self.profile.profile_id}:e{self.epoch_id}:{detector}:{start}:{end}"
        self.evidence[evidence_id] = FeatureWindow(self._stream_id, start, end, self.sample_rate_hz, window_ms,
            tuple(sorted((key, round(value, 8)) for key, value in values.items())), self.profile.profile_id)
        return DSPCandidate(detector, kind, start, end, score, threshold, evidence_id, self.profile.profile_id)

    def _click(self, data: NDArray[np.float32], start: int, end: int) -> list[DSPCandidate]:
        derivative = np.abs(np.diff(data))
        if not len(derivative):
            return []
        # Restrict to this hop: history may contain an earlier, stronger click.
        hop_derivative = derivative[-self.hop_frames:]
        peak = float(np.max(hop_derivative))
        if peak < self.profile.click_derivative * 0.75:
            return []
        position = int(np.argmax(hop_derivative))
        click_frame = start + position + 1
        context = data[max(0, len(data) - self.hop_frames + position - 480):len(data) - self.hop_frames + position]
        context_rms = float(np.sqrt(np.mean(context.astype(np.float64) ** 2))) if len(context) else 0.0
        if context_rms > 0.22:
            return []
        return [self._candidate("click", "click", click_frame, min(end, click_frame + 240),
            peak / self.profile.click_derivative, self.profile.click_derivative, 5,
            {"derivative_peak": peak, "context_rms": context_rms})]

    def _clipping(self, hop: NDArray[np.float32], start: int, end: int) -> list[DSPCandidate]:
        ratio = float(np.mean(np.abs(hop) >= self.profile.clip_level))
        if ratio < self.profile.clip_ratio * 0.75:
            return []
        return [self._candidate("flat_top", "clipping", start, end, ratio / self.profile.clip_ratio,
            self.profile.clip_ratio, 10, {"near_peak_ratio": ratio, "peak_level": float(np.max(np.abs(hop)))})]

    def _stutter(self, data: NDArray[np.float32], start: int, end: int) -> list[DSPCandidate]:
        window, lag = self.sample_rate_hz // 50, self.sample_rate_hz * 80 // 1000
        if len(data) < window + lag:
            return []
        current, prior = data[-window:], data[-window - lag:-lag]
        if min(float(np.std(current)), float(np.std(prior))) < 1e-4:
            return []
        corr = float(np.corrcoef(current, prior)[0, 1])
        ratio = float(np.sqrt(np.mean(current.astype(np.float64) ** 2)) / max(np.sqrt(np.mean(prior.astype(np.float64) ** 2)), 1e-6))
        if corr < self.profile.repeat_correlation * 0.75 or not 0.80 <= ratio <= 1.25:
            return []
        return [self._candidate("block_repeat", "stutter", start, end, corr / self.profile.repeat_correlation,
            self.profile.repeat_correlation, 20, {"correlation_lag_80ms": corr, "rms_ratio": ratio})]

    def _loop(self, data: NDArray[np.float32], start: int, end: int) -> list[DSPCandidate]:
        one = self.sample_rate_hz
        if len(data) < 2 * one:
            return []
        recent, prior = data[-one:], data[-2 * one:-one]
        half_recent, half_prior = recent[-one // 2:], recent[-one:-one // 2]
        if min(float(np.std(recent)), float(np.std(prior)), float(np.std(half_recent)), float(np.std(half_prior))) < 1e-4:
            return []
        corr1000, corr500 = float(np.corrcoef(recent, prior)[0, 1]), float(np.corrcoef(half_recent, half_prior)[0, 1])
        if min(corr500, corr1000) < self.profile.loop_correlation * 0.75:
            return []
        return [self._candidate("loop", "loop", start, end, min(corr500, corr1000) / self.profile.loop_correlation,
            self.profile.loop_correlation, 1000, {"correlation_lag_500ms": corr500, "correlation_lag_1000ms": corr1000})]

    def _dropout(self, hop: NDArray[np.float32], start: int, end: int) -> list[DSPCandidate]:
        rms = float(np.sqrt(np.mean(hop.astype(np.float64) ** 2)))
        baseline = float(np.median(self._active_rms)) if self._active_rms else rms
        q = rms / max(baseline, 1e-6)
        output: list[DSPCandidate] = []
        if self._dropout_post is not None:
            self._dropout_post.append(rms)
            output.append(self._candidate("dropout", "dropout", self._low_start or start, end, .75, .75, 100,
                {"rms": rms, "baseline_rms": self._dropout_baseline, "rms_ratio": q}))
            if len(self._dropout_post) >= 10:
                post = np.array(self._dropout_post)
                ratio = float(np.median(post) / max(self._dropout_baseline, 1e-6))
                cv = float(np.std(post) / max(np.mean(post), 1e-6))
                detected = self._low_floor <= .05 and .5 <= ratio <= 2 and cv <= .20 and self._descent_hops <= 2
                self._status_records.extend(self.builder.set_status(
                    "dropout", "detected" if detected else "uncertain", end, context_complete=True))
                self._dropout_post = None
                self._low_start = None
            return output
        if baseline > .015 and q <= .18:
            if self._low_start is None:
                self._low_start, self._low_hops, self._low_floor = start, 0, q
            self._low_hops += 1
            self._low_floor = min(self._low_floor, q)
            if self._low_hops >= 10 and self._descent_hops < 3:
                output.append(self._candidate("dropout", "dropout", self._low_start, end, 1 / max(q, .01), .18, 100,
                    {"rms": rms, "baseline_rms": baseline, "rms_ratio": q, "floor_ratio": self._low_floor}))
        else:
            if self._low_start is not None and self._low_hops >= 10:
                self._dropout_post, self._dropout_baseline = [], baseline
                return output
            if q < .9 and self._previous_q is not None and q < self._previous_q * .90:
                self._descent_hops += 1
            elif q >= .9:
                self._descent_hops = 0
            self._low_start = None
            self._low_hops = 0
            self._active_rms.append(rms)
        self._previous_q = q
        return output
