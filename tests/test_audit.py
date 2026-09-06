from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glitch_poc import cli
from glitch_poc.audit import SessionAuditLogger, validate_log_dir
from glitch_poc.contracts import DSPEvent, GemmaAnnotation, PCMBlock
from glitch_poc.ring import PCMBlockRing
from glitch_poc.runtime import SliceRuntime, WallClockPacer


def event() -> DSPEvent:
    return DSPEvent("event-1", 2, "CLOSED", "detected", "click", 10, 20, 20, 1.2,
                    ("click",), ("evidence-1",), "poc-d2-v2", 0, True)


class AuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = lambda: datetime(2026, 9, 6, 12, 34, 56, tzinfo=timezone.utc)  # noqa: UP017
        self.directory = f".work/audit-test-logs-{os.getpid()}"

    def test_filename_collision_jsonl_correlation_and_forbidden_fields(self) -> None:
        first = SessionAuditLogger(ROOT, self.directory, fixture="corrupted", stream_id="corrupted",
            profile_id="poc-d2-v2", gemma_enabled=True, clock=self.clock)
        second = SessionAuditLogger(ROOT, self.directory, fixture="clean", stream_id="clean",
            profile_id="poc-d2-v2", gemma_enabled=False, clock=self.clock)
        self.assertNotEqual(first.path, second.path)
        assert first.path is not None
        self.assertEqual(first.path.name, "session-20260906-123456.jsonl")
        self.assertEqual(second.path.name, "session-20260906-123456-1.jsonl")
        first.dsp_event(event(), "2026-09-06T12:34:57+00:00")
        annotation = GemmaAnnotation("v1", "event-1", 2, "coherent", "click", "high", ("evidence-1",),
            "DSP feature supports click.", True, 0.0, "gemma4:e4b", (), "p1", 3.0)
        first.gemma_annotation(annotation, "2026-09-06T12:34:58+00:00")
        first.close("test", {"queue": "ok"}, "2026-09-06T12:34:59+00:00")
        second.close("test", {}, "2026-09-06T12:34:59+00:00")
        records = [json.loads(line) for line in first.path.read_text().splitlines()]
        self.assertEqual([item["record_type"] for item in records],
                         ["session_started", "dsp_event", "gemma_annotation", "session_ended"])
        self.assertEqual(records[1]["correlation_id"], records[2]["correlation_id"])
        forbidden = {"audio", "pcm", "waveform", "spectrogram", "base64", "path", "uri", "secret", "prompt"}
        def scan(value: object) -> None:
            if isinstance(value, dict):
                self.assertFalse({key.lower() for key in value} & forbidden)
                for item in value.values(): scan(item)
            elif isinstance(value, list):
                for item in value: scan(item)
        scan(records)

    def test_containment_and_unavailable_logger_do_not_raise(self) -> None:
        for value in ("../logs", "/tmp/logs"):
            with self.assertRaises(ValueError):
                validate_log_dir(ROOT, value)
        logger = SessionAuditLogger(ROOT, "../blocked", fixture="clean", stream_id="clean",
            profile_id="poc-d2-v2", gemma_enabled=False, clock=self.clock)
        logger.dsp_event(event(), "2026-09-06T12:34:57+00:00")
        logger.close("test", {}, "2026-09-06T12:34:58+00:00")
        self.assertEqual(logger.stats().state, "unavailable")

    def test_unavailable_logger_does_not_block_dsp(self) -> None:
        logger = SessionAuditLogger(ROOT, "../blocked", fixture="clean", stream_id="clean",
            profile_id="poc-d2-v2", gemma_enabled=False, clock=self.clock)
        ring = PCMBlockRing(2, 8)
        ring.push(PCMBlock.create("clean", 0, 0, np.zeros((8, 2), dtype=np.float32)))
        producer = type("Producer", (), {"state": "eof", "error": None})()
        runtime = SliceRuntime(ring, producer, pacer=WallClockPacer(48_000, clock=lambda: 0.0, sleeper=lambda _: None))
        runtime.audit_log = logger
        self.assertTrue(runtime.step())
        self.assertEqual(logger.stats().state, "unavailable")

    def test_cli_finalizes_clean_session_without_gemma(self) -> None:
        directory = f".work/audit-cli-{os.getpid()}"
        class Producer:
            state = "eof"
            error = None
            def __init__(self, *args: object, **kwargs: object) -> None: pass
            def start(self) -> bool: return True
            def stop(self) -> None: pass
        with patch("glitch_poc.cli.FFmpegProducer", Producer):
            self.assertEqual(cli.main(["clean", "--no-ui", "--log-dir", directory]), 0)
        files = sorted((ROOT / directory).glob("session-*.jsonl"))
        records = [json.loads(line) for line in files[-1].read_text().splitlines()]
        self.assertEqual(records[0]["record_type"], "session_started")
        self.assertEqual(records[-1]["record_type"], "session_ended")
        self.assertFalse(records[0]["gemma"]["enabled"])
        self.assertNotIn("path", json.dumps(records).lower())

    def test_cli_external_file_uses_private_source_identity_in_audit(self) -> None:
        directory = f".work/audit-external-{os.getpid()}"
        external_dir = ROOT / ".work" / f"external audio {os.getpid()}"
        external_dir.mkdir(parents=True, exist_ok=True)
        external = external_dir / "private song.wav"
        shutil.copyfile(ROOT / "fixtures" / "audio" / "poc_clean.wav", external)
        class Producer:
            state = "eof"
            error = None
            def __init__(self, *args: object, **kwargs: object) -> None: pass
            def start(self) -> bool: return True
            def stop(self) -> None: pass
        try:
            spec = cli.resolve_source(ROOT, str(external))
            self.assertEqual(spec.kind, "external_file")
            self.assertNotIn(external.name, spec.stream_id)
            with patch("glitch_poc.cli.FFmpegProducer", Producer):
                self.assertEqual(cli.main([str(external), "--no-ui", "--log-dir", directory]), 0)
            records = [json.loads(line) for line in max((ROOT / directory).glob("session-*.jsonl")).read_text().splitlines()]
            started = records[0]
            self.assertEqual(started["source_kind"], "external_file")
            self.assertEqual(started["source_fingerprint"], spec.fingerprint)
            self.assertEqual(started["stream_id"], spec.stream_id)
            rendered = json.dumps(records)
            self.assertNotIn(str(external), rendered)
            self.assertNotIn(external.name, rendered)
            self.assertNotIn("path", rendered.lower())
        finally:
            external.unlink()
            external_dir.rmdir()

    def test_cli_source_validation_rejects_missing_directory_and_unreadable(self) -> None:
        self.assertEqual(cli.resolve_source(ROOT, "rock-corrupted").path.name, "poc_rock_v1_corrupted.wav")
        self.assertEqual(cli.resolve_source(ROOT, "harvard-corrupted").path.name, "poc_harvard_v1_corrupted.wav")
        with self.assertRaisesRegex(ValueError, "does not exist"):
            cli.resolve_source(ROOT, str(ROOT / ".work" / "missing-audio.wav"))
        with self.assertRaisesRegex(ValueError, "regular file"):
            cli.resolve_source(ROOT, str(ROOT / ".work"))
        readable = ROOT / "fixtures" / "audio" / "poc_clean.wav"
        with patch.object(Path, "open", side_effect=PermissionError), self.assertRaisesRegex(ValueError, "not readable"):
            cli.resolve_source(ROOT, str(readable))

    def test_session_end_keeps_gemma_outcome_summary(self) -> None:
        logger = SessionAuditLogger(ROOT, f".work/audit-summary-{os.getpid()}", fixture="corrupted",
            stream_id="corrupted", profile_id="poc-d2-v2", gemma_enabled=True, clock=self.clock)
        logger.close("headless_exit", {"gemma_queue": {"submitted": 4, "completed": 4, "errors": 2,
            "cancelled_pending": 0}, "dsp_event_count": 4}, "2026-09-06T12:34:59+00:00")
        assert logger.path is not None
        end = json.loads(logger.path.read_text().splitlines()[-1])
        self.assertEqual(end["summary"]["gemma_queue"],
                         {"submitted": 4, "completed": 4, "errors": 2, "cancelled_pending": 0})
