"""Lifecycle-aware, reproducible fixture metrics for ``poc-d2-v2``."""
from __future__ import annotations

import json
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

from glitch_poc.contracts import DSPEvent, PCMBlock
from glitch_poc.dsp import DSPProcessor

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "fixtures" / "audio"
BLOCK_FRAMES = 1024
SAMPLE_RATE = 48_000


def blocks(path: Path, block_frames: int = BLOCK_FRAMES) -> list[PCMBlock]:
    with wave.open(str(path), "rb") as source:
        pcm = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").reshape(-1, 2)
    samples = pcm.astype(np.float32) / 32768.0
    return [PCMBlock.create("fixture", sequence, frame, samples[frame:frame + block_frames])
            for sequence, frame in enumerate(range(0, len(samples), block_frames))]


def analyze(path: Path) -> list[DSPEvent]:
    processor, records = DSPProcessor(), []
    for block in blocks(path):
        records.extend(processor.process(block))
    records.extend(processor.finish())
    return records


def canonicalize(records: list[DSPEvent]) -> list[DSPEvent]:
    """One final record per lifecycle; cancelled/superseded records never score as alerts."""
    latest: dict[str, DSPEvent] = {}
    for record in records:
        if record.lifecycle in {"CANCELLED", "SUPERSEDED"}:
            latest.pop(record.event_id, None)
        else:
            latest[record.event_id] = record
    return [record for record in latest.values() if record.lifecycle not in {"CANCELLED", "SUPERSEDED"}]


def _iou(left: DSPEvent, right: tuple[str, int, int]) -> float:
    _, start, end = right
    intersection = max(0, min(left.end_frame, end) - max(left.start_frame, start))
    union = max(left.end_frame, end) - min(left.start_frame, start)
    return intersection / union if union else 0.0


def matches(event: DSPEvent, truth: tuple[str, int, int]) -> bool:
    kind, start, end = truth
    if event.glitch_type != kind:
        return False
    if kind == "click":
        return abs(event.start_frame - start) <= 240 and event.end_frame - event.start_frame <= 960
    overlap = max(0, min(event.end_frame, end) - max(event.start_frame, start))
    return overlap >= 960 and _iou(event, truth) >= .5


def percentile(values: list[float], amount: float) -> float | None:
    return float(np.percentile(np.array(values), amount)) if values else None


def latency(records: list[DSPEvent], truths: list[tuple[str, int, int]]) -> dict[str, float | None]:
    canonical = {record.event_id: record for record in canonicalize(records)}
    by_id: dict[str, list[DSPEvent]] = defaultdict(list)
    for record in records:
        if record.event_id in canonical:
            by_id[record.event_id].append(record)
    first, decisions, finals = [], [], []
    for truth in truths:
        sequence = next((items for event_id, items in by_id.items() if matches(canonical[event_id], truth)), None)
        if sequence is None:
            continue
        start = truth[1]
        opened = next((item for item in sequence if item.lifecycle == "OPENED"), None)
        decision = next((item for item in sequence if item.status == "detected"), None)
        closed = next((item for item in reversed(sequence) if item.lifecycle == "CLOSED"), None)
        if opened:
            first.append((opened.emitted_frame - start) * 1000 / SAMPLE_RATE)
        if decision:
            decisions.append((decision.emitted_frame - start) * 1000 / SAMPLE_RATE)
        if closed:
            finals.append((closed.emitted_frame - start) * 1000 / SAMPLE_RATE)
    return {stage: percentile(values, percentile_value) for stage, values, percentile_value in
            (("first_alert_p50", first, 50), ("first_alert_p95", first, 95), ("first_alert_p99", first, 99),
             ("decision_p50", decisions, 50), ("decision_p95", decisions, 95), ("decision_p99", decisions, 99),
             ("finalization_p50", finals, 50), ("finalization_p95", finals, 95), ("finalization_p99", finals, 99))}


def guarded_exposure(frame_count: int, truth: list[tuple[str, int, int]]) -> int:
    intervals = sorted((max(0, start - 2400), min(frame_count, end + 2400)) for _, start, end in truth)
    covered = 0
    last = 0
    for start, end in intervals:
        covered += max(0, end - max(last, start))
        last = max(last, end)
    return frame_count - covered


def main() -> None:
    manifest = json.loads((ASSETS / "poc_ground_truth.json").read_text())
    truth = [(fault["class"], fault["interval"]["start"], fault["interval"]["end"])
             for fault in manifest["faults"]]
    corrupted_records, clean_records = analyze(ASSETS / "poc_corrupted.wav"), analyze(ASSETS / "poc_clean.wav")
    observed = canonicalize(corrupted_records)
    detected = [event for event in observed if event.status == "detected"]
    uncertain = [event for event in observed if event.status == "uncertain"]
    unmatched = list(detected)
    matched: list[tuple[str, int, int]] = []
    for item in truth:
        hit = next((event for event in unmatched if matches(event, item)), None)
        if hit is not None:
            unmatched.remove(hit)
            matched.append(item)
    fn = [item for item in truth if item not in matched]
    clean_seconds = manifest["format"]["frames"] / SAMPLE_RATE
    unannotated_seconds = guarded_exposure(manifest["format"]["frames"], truth) / SAMPLE_RATE
    exposure_seconds = clean_seconds + unannotated_seconds
    false_alarms = len(unmatched) + len(canonicalize(clean_records))
    report = {
        "profile_id": "poc-d2-v2", "sample_count": {"clean": len(clean_records), "corrupted": len(corrupted_records),
        "canonical_detected": len(detected), "canonical_uncertain": len(uncertain)},
        "matching": {"tp": len(matched), "fp": len(unmatched), "fn": len(fn),
                     "uncertain": len(uncertain), "precision": len(matched) / len(detected) if detected else 1.0,
                     "recall": len(matched) / len(truth)},
        "latency_ms": latency(corrupted_records, truth),
        "latency_by_class_ms": {item[0]: latency(corrupted_records, [item]) for item in truth},
        "false_alarm": {"count": false_alarms, "exposure_seconds": exposure_seconds,
                        "per_hour": false_alarms / exposure_seconds * 3600 if exposure_seconds else None,
                        "insufficient_corpus": True},
        "canonical_events": [event.to_dict() for event in observed],
        "uncertain_events": [event.to_dict() for event in uncertain],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
