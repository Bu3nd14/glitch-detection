from __future__ import annotations

import json
import sys
import unittest
import wave
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glitch_poc.contracts import DSPCandidate, PCMBlock
from glitch_poc.dsp import DSPProcessor, DSPProfile, EventBuilder
from tools.dsp_metrics import canonicalize, latency, matches


def fixture_blocks(name: str, block_frames: int = 1024) -> list[PCMBlock]:
    with wave.open(str(ROOT / "fixtures" / "audio" / name), "rb") as source:
        raw = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").reshape(-1, 2)
    pcm = raw.astype(np.float32) / 32768.0
    return [PCMBlock.create("fixture", index, start, pcm[start:start + block_frames], profile_id="poc-d2-v2")
            for index, start in enumerate(range(0, len(pcm), block_frames))]


def analyze(name: str, block_frames: int = 1024) -> tuple[DSPProcessor, list[object]]:
    processor, records = DSPProcessor(), []
    for block in fixture_blocks(name, block_frames):
        records.extend(processor.process(block))
    records.extend(processor.finish())
    return processor, records


class EventBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = EventBuilder(1_000, DSPProfile(profile_id="poc-d2-v2", hop_ms=10))

    @staticmethod
    def candidate(score: float, evidence: str = "e", detector: str = "d", *, kind: str = "stutter",
                  start: int = 100, end: int = 110) -> DSPCandidate:
        return DSPCandidate(detector, kind, start, end, score, 1.0, evidence, "poc-d2-v2")

    def test_open_update_close_and_ordered_unique_evidence(self) -> None:
        opened = self.builder.observe(self.candidate(1.2, "a", "one"), 110)
        updated = self.builder.observe(self.candidate(.8, "a", "two"), 120)
        self.builder.advance(120)
        self.builder.advance(130); self.builder.advance(140)
        closed = self.builder.advance(190)
        self.assertEqual([record.lifecycle for record in opened + updated + closed], ["OPENED", "UPDATED", "CLOSED"])
        self.assertEqual(updated[0].detector_ids, ("one", "two"))
        self.assertEqual(updated[0].evidence_ids, ("a",))
        self.assertEqual(opened[0].event_id, closed[0].event_id)

    def test_hysteresis_requires_two_exit_hops(self) -> None:
        self.builder.observe(self.candidate(1.1), 110)
        self.builder.advance(110)
        self.builder.observe(self.candidate(.4), 120)
        self.assertEqual(self.builder.advance(120), [])
        self.builder.observe(self.candidate(.4), 130)
        self.builder.advance(130)
        self.assertEqual(self.builder.advance(180)[0].lifecycle, "CLOSED")

    def test_reset_cancels_pending_and_resets_cooldown_epoch(self) -> None:
        first = self.builder.observe(self.candidate(1.1), 110)[0]
        cancelled = self.builder.reset(115, "frame_gap")[0]
        replacement = EventBuilder(1_000, DSPProfile(profile_id="poc-d2-v2"), epoch_id=1)
        second = replacement.observe(self.candidate(1.1), 120)[0]
        self.assertEqual((cancelled.lifecycle, cancelled.close_reason, cancelled.context_complete),
                         ("CANCELLED", "frame_gap", False))
        self.assertNotEqual(first.event_id, second.event_id)

    def test_merge_wait_resumes_only_inside_gap_and_opens_after_cooldown(self) -> None:
        opened = self.builder.observe(self.candidate(1.1, start=10, end=20), 20)[0]
        self.builder.advance(30); self.builder.advance(40)  # MERGE_WAIT until frame 90.
        within = self.builder.observe(self.candidate(1.1, "within", start=60, end=70), 70)
        self.assertEqual(within[0].event_id, opened.event_id)

        builder = EventBuilder(1_000, DSPProfile(profile_id="poc-d2-v2", hop_ms=10))
        first = builder.observe(self.candidate(1.1, start=10, end=20), 20)[0]
        far = builder.observe(self.candidate(1.1, start=200, end=210), 210)
        self.assertEqual([record.lifecycle for record in far], ["CLOSED", "OPENED"])
        self.assertEqual(far[0].end_frame, 20)
        self.assertNotEqual(first.event_id, far[1].event_id)

        builder = EventBuilder(1_000, DSPProfile(profile_id="poc-d2-v2", hop_ms=10))
        first = builder.observe(self.candidate(1.1, start=10, end=20), 20)[0]
        builder.advance(30); builder.advance(40); builder.advance(50)
        cooldown = builder.observe(self.candidate(1.1, start=120, end=130), 130)
        self.assertEqual([record.lifecycle for record in cooldown], ["CLOSED"])
        after = builder.observe(self.candidate(1.1, start=220, end=230), 230)
        self.assertEqual([record.lifecycle for record in after], ["OPENED"])
        self.assertNotEqual(first.event_id, after[0].event_id)

    def test_priority_conflict_is_deterministic_and_candidate_default_is_v2(self) -> None:
        click = self.builder.observe(self.candidate(1.1, kind="click", start=100, end=140), 140)[0]
        clipping = self.builder.observe(self.candidate(1.1, kind="clipping", start=110, end=150), 150)
        self.assertEqual([record.lifecycle for record in clipping], ["SUPERSEDED", "OPENED"])
        self.assertEqual(clipping[0].event_id, click.event_id)
        default = DSPCandidate("d", "click", 300, 310, 1.1, 1.0, "default")
        event = self.builder.observe(default, 310)[0]
        self.assertTrue(event.event_id.startswith("poc-d2-v2:"))


class ProcessorTests(unittest.TestCase):
    def test_feature_immutable_serializable_and_lifecycle_has_required_fields(self) -> None:
        processor, records = analyze("poc_corrupted.wav")
        feature = next(iter(processor.evidence.values()))
        self.assertEqual(json.loads(json.dumps(feature.to_dict()))["profile_id"], "poc-d2-v2")
        with self.assertRaises(FrozenInstanceError):
            feature.window_ms = 1  # type: ignore[misc]
        opened = next(record for record in records if record.lifecycle == "OPENED")
        self.assertGreaterEqual(opened.revision, 1)
        self.assertEqual(opened.emitted_frame % 480, 0)

    def test_discontinuity_cancels_open_lifecycle_and_increments_epoch(self) -> None:
        processor = DSPProcessor()
        click = np.zeros((480, 2), dtype=np.float32); click[10] = .6
        processor.process(PCMBlock.create("x", 0, 0, click))
        result = processor.process(PCMBlock.create("x", 2, 960, np.zeros((480, 2), dtype=np.float32),
            discontinuity=True, missing_frames=480))
        self.assertEqual(result[0].lifecycle, "CANCELLED")
        self.assertEqual((result[0].epoch_id, processor.epoch_id), (0, 1))
        self.assertEqual(processor.audit[-1]["reason"], "discontinuity")

    def test_hard_mute_is_uncertain_at_eos_and_slow_fade_is_clean(self) -> None:
        processor = DSPProcessor(sample_rate_hz=1_000)
        data = np.concatenate((np.full(600, .1, dtype=np.float32), np.zeros(150, dtype=np.float32)))
        records = []
        for sequence, start in enumerate(range(0, len(data), 10)):
            stereo = np.column_stack((data[start:start + 10], data[start:start + 10]))
            records.extend(processor.process(PCMBlock.create("x", sequence, start, stereo, sample_rate_hz=1_000)))
        records.extend(processor.finish())
        self.assertTrue(any(record.glitch_type == "dropout" and record.status == "uncertain" for record in records))
        dropout = [record for record in records if record.glitch_type == "dropout"]
        self.assertEqual([record.lifecycle for record in dropout[-2:]], ["UPDATED", "CLOSED"])
        self.assertLess(dropout[-2].revision, dropout[-1].revision)

        fade_processor = DSPProcessor(sample_rate_hz=1_000)
        fade = np.concatenate((np.full(600, .1, dtype=np.float32), np.linspace(.1, 0, 40, dtype=np.float32),
                               np.zeros(150, dtype=np.float32)))
        fade_records = []
        for sequence, start in enumerate(range(0, len(fade), 10)):
            stereo = np.column_stack((fade[start:start + 10], fade[start:start + 10]))
            fade_records.extend(fade_processor.process(PCMBlock.create("x", sequence, start, stereo, sample_rate_hz=1_000)))
        fade_records.extend(fade_processor.finish())
        self.assertFalse(any(record.glitch_type == "dropout" for record in fade_records))

    def test_block_size_invariance_and_first_alert_targets(self) -> None:
        outputs = []
        for size in (480, 512, 1024, 2048):
            _, records = analyze("poc_corrupted.wav", size)
            opened = [record for record in records if record.lifecycle == "OPENED"]
            outputs.append({record.glitch_type: record for record in opened})
        for kind in ("click", "clipping", "stutter"):
            starts = [result[kind].start_frame for result in outputs]
            self.assertLessEqual(max(starts) - min(starts), 480)
            target = 3_600 if kind == "stutter" else 4_800
            self.assertLessEqual(max(result[kind].emitted_frame - result[kind].start_frame for result in outputs), target)


class FixtureAndMetricTests(unittest.TestCase):
    def test_fixture_detected_lifecycles_match_truth_and_clean_has_no_positive(self) -> None:
        _, records = analyze("poc_corrupted.wav")
        events = canonicalize(records)
        manifest = json.loads((ROOT / "fixtures" / "audio" / "poc_ground_truth.json").read_text())
        for fault in manifest["faults"]:
            truth = (fault["class"], fault["interval"]["start"], fault["interval"]["end"])
            self.assertTrue(any(event.status == "detected" and matches(event, truth) for event in events), truth)
        _, clean = analyze("poc_clean.wav")
        self.assertFalse(any(event.status == "detected" for event in canonicalize(clean)))

    def test_metrics_matching_rejects_duplicate_and_class_mismatch(self) -> None:
        _, records = analyze("poc_corrupted.wav")
        events = canonicalize(records)
        click = next(event for event in events if event.glitch_type == "click")
        truth = ("click", 108000, 108096)
        self.assertTrue(matches(click, truth))
        self.assertFalse(matches(click, ("clipping", 108000, 108096)))
        self.assertEqual(len([event for event in events if matches(event, truth)]), 1)

    def test_cancelled_lifecycle_is_not_canonical_or_latency_sample(self) -> None:
        builder = EventBuilder(48_000, DSPProfile())
        candidate = DSPCandidate("d", "click", 108000, 108240, 1.1, 1.0, "e")
        opened = builder.observe(candidate, 108480)[0]
        cancelled = builder.reset(108960, "frame_gap")[0]
        records = [opened, cancelled]
        self.assertEqual(canonicalize(records), [])
        samples = latency(records, [("click", 108000, 108096)])
        self.assertTrue(all(value is None for value in samples.values()))


if __name__ == "__main__":
    unittest.main()
