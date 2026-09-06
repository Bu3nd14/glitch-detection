from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PCMBlock:
    """One contiguous, normalized PCM interval. Frame ranges are half-open."""

    stream_id: str
    sequence: int
    start_frame: int
    frame_count: int
    sample_rate_hz: int
    channels: int
    monotonic_timestamp_s: float
    discontinuity: bool
    missing_frames: int
    profile_id: str
    samples: NDArray[np.float32]

    @classmethod
    def create(
        cls, stream_id: str, sequence: int, start_frame: int, samples: NDArray[np.float32], *,
        sample_rate_hz: int = 48_000, profile_id: str = "poc-d2-v2",
        discontinuity: bool = False, missing_frames: int = 0,
    ) -> PCMBlock:
        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError("PCMBlock samples must be shaped (frames, 2)")
        return cls(stream_id, sequence, start_frame, len(samples), sample_rate_hz, 2,
                    time.monotonic(), discontinuity, missing_frames, profile_id, samples)


@dataclass(frozen=True)
class FeatureWindow:
    """Deterministic, serializable features for a half-open PCM frame interval."""

    stream_id: str
    start_frame: int
    end_frame: int
    sample_rate_hz: int
    window_ms: int
    features: tuple[tuple[str, float], ...]
    profile_id: str

    @property
    def start_seconds(self) -> float:
        return self.start_frame / self.sample_rate_hz

    @property
    def end_seconds(self) -> float:
        return self.end_frame / self.sample_rate_hz

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["features"] = dict(self.features)
        return result


@dataclass(frozen=True)
class DSPCandidate:
    detector_name: str
    candidate_type: str
    start_frame: int
    end_frame: int
    raw_score: float
    threshold: float
    evidence_id: str
    profile_id: str = "poc-d2-v2"


@dataclass(frozen=True)
class DSPEvent:
    """Append-only authoritative lifecycle record; score is not a probability."""

    event_id: str
    revision: int
    lifecycle: str
    status: str
    glitch_type: str
    start_frame: int
    end_frame: int
    emitted_frame: int
    raw_score: float
    detector_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    profile_id: str
    epoch_id: int
    context_complete: bool
    close_reason: str | None = None
    superseded_by: str | None = None

    @property
    def score(self) -> float:
        """Compatibility alias for display only; never interpret as probability."""
        return self.raw_score

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GemmaAnnotation:
    """Non-authoritative, versioned record kept separately from DSP lifecycle records."""

    annotation_version: str
    event_id: str
    event_revision: int
    annotation_status: str
    glitch_type_annotation: str | None
    confidence: str | None
    supporting_evidence_ids: tuple[str, ...]
    explanation: str
    cannot_override_dsp: bool
    created_at_unix_s: float
    model: str
    model_metadata: tuple[tuple[str, str], ...]
    prompt_schema_version: str
    timeout_s: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["supporting_evidence_ids"] = list(self.supporting_evidence_ids)
        result["model_metadata"] = dict(self.model_metadata)
        return result
