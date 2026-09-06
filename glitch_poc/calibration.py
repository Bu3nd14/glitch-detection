"""Pure timeout-calibration policy for the optional local Gemma annotator."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


def proposed_timeout_seconds(valid_elapsed_seconds: Iterable[float], *, minimum_s: int = 5,
                             maximum_s: int = 60, safety_factor: float = 1.5) -> int | None:
    """p95 (or max for fewer than 20 samples), safety margin, integer-second clamp."""
    values = sorted(value for value in valid_elapsed_seconds if value >= 0 and math.isfinite(value))
    if not values:
        return None
    if len(values) < 20:
        percentile = values[-1]
    else:
        percentile = values[math.ceil(.95 * len(values)) - 1]
    return max(minimum_s, min(maximum_s, math.ceil(percentile * safety_factor)))


def default_drain_timeout_seconds(annotation_timeout_s: float, *, fixture_requests: int = 4,
                                  overhead_s: int = 3) -> float:
    """Serial fixture drain budget: four requests plus bounded process/queue overhead."""
    return max(15.0, fixture_requests * annotation_timeout_s + overhead_s)


def proposal_for_four_classes(samples: Iterable[Mapping[str, object]], expected_types: frozenset[str]) -> int | None:
    """Return a proposal only when each expected class has at least one valid bounded response."""
    grouped: dict[str, list[float]] = {kind: [] for kind in expected_types}
    for sample in samples:
        kind, elapsed = sample.get("glitch_type"), sample.get("elapsed_s")
        if sample.get("response_valid_grounded") is True and isinstance(kind, str) and kind in grouped and isinstance(elapsed, (int, float)):
            grouped[kind].append(float(elapsed))
    if any(not values for values in grouped.values()):
        return None
    return proposed_timeout_seconds(value for values in grouped.values() for value in values)
