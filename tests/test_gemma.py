from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glitch_poc.contracts import DSPEvent, GemmaAnnotation
from glitch_poc.ollama import GemmaWorker, OllamaClient, OllamaConfig, annotation_payload
from tests.test_dsp import analyze


def event(revision: int = 1, status: str = "uncertain", lifecycle: str = "CLOSED") -> DSPEvent:
    return DSPEvent("event-1", revision, lifecycle, status, "loop", 0, 480, 480, 1.2,
                    ("loop",), ("evidence-1",), "poc-d2-v2", 0, False)


def evidence(features: dict[str, object] | None = None) -> tuple[tuple[str, dict[str, object]], ...]:
    return (("evidence-1", {"stream_id": "observed", "start_frame": 0, "end_frame": 480,
            "sample_rate_hz": 48_000, "window_ms": 100, "features": features or {"correlation_lag_500ms": .999},
            "profile_id": "poc-d2-v2"}),)


class SpawnSuccessClient:
    """Module-level test transport: importable and picklable by multiprocessing spawn."""
    config = OllamaConfig(queue_capacity=8, timeout_s=.1)

    def annotate(self, item: DSPEvent, values: object) -> GemmaAnnotation:
        return GemmaAnnotation("test", item.event_id, item.revision, "coherent", item.glitch_type, "low",
            (), "mocked", True, time.time(), "test", (), "test", .1)


class SpawnSlowClient:
    config = OllamaConfig(queue_capacity=1, timeout_s=.1)

    def annotate(self, item: DSPEvent, values: object) -> GemmaAnnotation:
        time.sleep(.05)
        return OllamaClient(self.config)._error(item, time.time(), "offline")


class SpawnBlockedClient:
    config = OllamaConfig(queue_capacity=2, timeout_s=.1)

    def annotate(self, item: DSPEvent, values: object) -> GemmaAnnotation:
        time.sleep(10)
        return OllamaClient(self.config)._error(item, time.time(), "offline")


class SpawnVerySlowClient:
    config = OllamaConfig(queue_capacity=4, timeout_s=.1)

    def annotate(self, item: DSPEvent, values: object) -> GemmaAnnotation:
        time.sleep(10)
        return OllamaClient(self.config)._error(item, time.time(), "offline")


class GemmaTests(unittest.TestCase):
    def test_payload_is_structured_and_guard_rejects_audio_and_paths(self) -> None:
        payload = annotation_payload(event(), evidence({"correlation_lag_500ms": .99}))
        prompt = json.loads(payload["prompt"])
        self.assertEqual(prompt["dsp_event"]["event_id"], "event-1")
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            annotation_payload(event(), evidence({"Path": "/secret.wav"}))
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            annotation_payload(event(), (("evidence-1", {"features": {"rms": {"Audio": "x"}}}),))
        with self.assertRaisesRegex(ValueError, "encoded"):
            annotation_payload(DSPEvent("YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4", 1, "CLOSED", "uncertain", "loop", 0, 1,
                1, 1.0, ("loop",), ("evidence-1",), "poc-d2-v2", 0, False), evidence())

    def test_invalid_json_and_ungrounded_output_become_errors(self) -> None:
        client = OllamaClient(OllamaConfig(timeout_s=.01))
        class Response:
            def __init__(self, data: bytes) -> None: self.data = data
            def __enter__(self) -> Response:  # noqa: PYI034 - test suite also runs on Python 3.9 here.
                return self
            def __exit__(self, *args: object) -> None: pass
            def read(self, _limit: int) -> bytes: return self.data
        with patch("glitch_poc.ollama.urlopen", return_value=Response(b'{"response":"not json"}')):
            self.assertEqual(client.annotate(event(), ()).annotation_status, "error")
        bad = {"event_id": "other", "annotation_status": "coherent", "glitch_type_annotation": "loop",
               "confidence": "high", "supporting_evidence_ids": [], "explanation": "x", "cannot_override_dsp": True}
        with patch("glitch_poc.ollama.urlopen", return_value=Response(json.dumps({"response": json.dumps(bad)}).encode())):
            self.assertEqual(client.annotate(event(), ()).annotation_status, "error")

    def test_http_body_is_the_whitelisted_structure(self) -> None:
        class Response:
            def __enter__(self) -> Response: return self  # noqa: PYI034
            def __exit__(self, *args: object) -> None: pass
            def read(self, _limit: int) -> bytes:
                reply = {"event_id": "event-1", "annotation_status": "coherent", "glitch_type_annotation": "loop",
                    "confidence": "low", "supporting_evidence_ids": ["evidence-1"], "explanation": "grounded",
                    "cannot_override_dsp": True}
                return json.dumps({"response": json.dumps(reply)}).encode()
        def guarded_open(request: object, timeout: float) -> Response:
            body = json.loads(request.data)
            self.assertEqual(set(body), {"model", "stream", "format", "options", "keep_alive", "prompt"})
            self.assertEqual(body["options"], {"temperature": 0, "num_predict": 256})
            prompt = json.loads(body["prompt"])
            self.assertEqual(set(prompt), {"prompt_schema_version", "instruction", "output_schema", "dsp_event",
                                           "evidence", "profile", "descriptive_context"})
            self.assertEqual(set(prompt["dsp_event"]), {"event_id", "revision", "lifecycle", "status", "glitch_type",
                "start_frame", "end_frame", "emitted_frame", "raw_score", "detector_ids", "evidence_ids", "profile_id",
                "epoch_id", "context_complete"})
            self.assertEqual(set(prompt["evidence"][0]), {"evidence_id", "values"})
            self.assertEqual(set(prompt["evidence"][0]["values"]), {"correlation_lag_500ms"})
            return Response()
        with patch("glitch_poc.ollama.urlopen", side_effect=guarded_open):
            self.assertEqual(OllamaClient().annotate(event(), evidence()).annotation_status, "coherent")

    def test_oversized_model_response_is_invalid(self) -> None:
        class Response:
            def __enter__(self) -> Response: return self  # noqa: PYI034
            def __exit__(self, *args: object) -> None: pass
            def read(self, _limit: int) -> bytes: return b"x" * 20_000
        with patch("glitch_poc.ollama.urlopen", return_value=Response()):
            result = OllamaClient().annotate(event(), evidence())
        self.assertEqual(result.annotation_status, "error")
        self.assertIn("exceeds limit", result.error or "")

    def test_queue_is_bounded_deduplicated_and_never_changes_event(self) -> None:
        annotations = []
        worker = GemmaWorker(SpawnSlowClient(), annotations.append)  # type: ignore[arg-type]
        self.assertTrue(worker.submit(event(), evidence()))
        self.assertFalse(worker.submit(event(), evidence()))
        self.assertFalse(worker.submit(event(2), evidence()))
        self.assertFalse(worker.submit(event(status="detected"), evidence()))
        self.assertEqual(worker.stats().dropped_backpressure, 1)
        worker.start()
        deadline = time.monotonic() + 3
        while not annotations and time.monotonic() < deadline:
            time.sleep(.01)
        worker.stop()
        self.assertEqual(event().status, "uncertain")
        self.assertEqual(len(annotations), 1)

    def test_shutdown_terminates_blocked_client_and_cancels_pending(self) -> None:
        annotations = []
        worker = GemmaWorker(SpawnBlockedClient(), annotations.append)  # type: ignore[arg-type]
        worker.start()
        self.assertTrue(worker.submit(event(), evidence()))
        self.assertTrue(worker.submit(event(2), evidence()))
        time.sleep(.05)
        started = time.monotonic(); worker.stop()
        self.assertLess(time.monotonic() - started, .6)
        self.assertFalse(worker.worker_alive)
        self.assertEqual(annotations, [])
        self.assertGreaterEqual(worker.stats().cancelled_pending, 1)

    def test_spawn_context_completes_mock_annotation_without_fork(self) -> None:
        annotations: list[GemmaAnnotation] = []
        worker = GemmaWorker(SpawnSuccessClient(), annotations.append)  # type: ignore[arg-type]
        self.assertEqual(worker.process_start_method, "spawn")
        worker.start()
        self.assertTrue(worker.submit(event(), evidence()))
        deadline = time.monotonic() + 3
        while not annotations and time.monotonic() < deadline:
            time.sleep(.01)
        worker.stop()
        self.assertFalse(worker.worker_alive)
        self.assertEqual([item.explanation for item in annotations], ["mocked"])

    def test_stop_gate_prevents_callback_when_stop_wins_before_dispatch(self) -> None:
        reserved = threading.Event()
        release = threading.Event()
        received: list[GemmaAnnotation] = []
        class GatedWorker(GemmaWorker):
            def _reserve_callback(self) -> bool:
                result = super()._reserve_callback()
                reserved.set()
                release.wait(1)
                return result
        worker = GatedWorker(SpawnSuccessClient(), received.append)  # type: ignore[arg-type]
        worker.start()
        self.assertTrue(worker.submit(event(), evidence()))
        self.assertTrue(reserved.wait(3))
        stopped = threading.Event()
        stopper = threading.Thread(target=lambda: (worker.stop(), stopped.set()))
        stopper.start()
        time.sleep(.05)
        self.assertFalse(stopped.is_set())
        release.set()
        stopper.join(1)
        self.assertTrue(stopped.is_set())
        self.assertEqual(received, [])

    def test_graceful_drain_delivers_four_serial_outcomes(self) -> None:
        annotations: list[GemmaAnnotation] = []
        worker = GemmaWorker(SpawnSuccessClient(), annotations.append)  # type: ignore[arg-type]
        worker.start()
        for revision in range(1, 5):
            self.assertTrue(worker.submit(event(revision), evidence()))
        self.assertTrue(worker.drain_and_stop(10))
        self.assertFalse(worker.worker_alive)
        self.assertEqual(len(annotations), 4)
        stats = worker.stats()
        self.assertEqual((stats.submitted, stats.completed, stats.pending), (4, 4, 0))

    def test_graceful_timeout_delivers_explicit_outcomes_for_every_pending_request(self) -> None:
        annotations: list[GemmaAnnotation] = []
        worker = GemmaWorker(SpawnVerySlowClient(), annotations.append)  # type: ignore[arg-type]
        worker.start()
        for revision in range(1, 5):
            self.assertTrue(worker.submit(event(revision), evidence()))
        self.assertFalse(worker.drain_and_stop(.1))
        self.assertEqual(len(annotations), 4)
        self.assertTrue(all(item.error == "shutdown_drain_timeout" for item in annotations))
        self.assertEqual(worker.stats().pending, 0)

    def test_immediate_cancel_reports_outcomes_via_abandoned_handler(self) -> None:
        abandoned: list[GemmaAnnotation] = []
        worker = GemmaWorker(SpawnVerySlowClient(), lambda _: None, abandoned.append)  # type: ignore[arg-type]
        worker.start()
        self.assertTrue(worker.submit(event(), evidence()))
        self.assertTrue(worker.submit(event(2), evidence()))
        worker.stop()
        self.assertEqual(len(abandoned), 2)
        self.assertTrue(all(item.error == "shutdown_cancel" for item in abandoned))

    def test_dedupe_lru_is_bounded(self) -> None:
        class Client:
            config = OllamaConfig(queue_capacity=4, dedupe_capacity=2, timeout_s=.01)
            def annotate(self, item: DSPEvent, values: object) -> object: return None
        worker = GemmaWorker(Client(), lambda _: None)  # type: ignore[arg-type]
        self.assertTrue(worker.submit(event(1), evidence()))
        self.assertTrue(worker.submit(event(2), evidence()))
        self.assertTrue(worker.submit(event(3), evidence()))
        self.assertTrue(worker.submit(event(1), evidence()), "oldest dedupe key must be evicted")
        self.assertLessEqual(len(worker._seen), 2)
        worker.stop()

    def test_only_closed_detected_or_uncertain_are_enqueued(self) -> None:
        class Client:
            config = OllamaConfig(queue_capacity=8, timeout_s=.01)
            def annotate(self, item: DSPEvent, values: object) -> object: return None
        worker = GemmaWorker(Client(), lambda _: None)  # type: ignore[arg-type]
        self.assertTrue(worker.submit(event(status="detected"), evidence()))
        self.assertTrue(worker.submit(event(2, "uncertain"), evidence()))
        self.assertFalse(worker.submit(event(3, "detected", "OPENED"), evidence()))
        self.assertFalse(worker.submit(event(4, "detected", "UPDATED"), evidence()))
        self.assertFalse(worker.submit(event(5, "detected", "CANCELLED"), evidence()))
        self.assertFalse(worker.submit(event(6, "uncertain", "SUPERSEDED"), evidence()))
        self.assertEqual(worker.stats().queued, 2)
        worker.stop()

    def test_corrupted_fixture_final_events_produce_four_grounded_requests(self) -> None:
        processor, records = analyze("poc_corrupted.wav")
        final = [record for record in records if record.lifecycle == "CLOSED" and record.status in {"detected", "uncertain"}]
        evidence_values = tuple((key, value.to_dict()) for key, value in processor.evidence.items())
        captured = [annotation_payload(record, evidence_values) for record in final]
        self.assertEqual({json.loads(item["prompt"])["dsp_event"]["glitch_type"] for item in captured},
                         {"click", "dropout", "stutter", "clipping"})
        self.assertEqual(len(captured), 4)
        for payload in captured:
            grounded = json.loads(payload["prompt"])
            self.assertLessEqual(len(grounded["evidence"]), 2)
            self.assertLess(len(json.dumps(payload).encode()), 32_768)

    def test_transport_failure_is_nonfatal_and_loopback_is_required(self) -> None:
        with self.assertRaises(ValueError): OllamaClient(OllamaConfig(endpoint="http://remote.invalid/"))
        client = OllamaClient(OllamaConfig(timeout_s=.01))
        with patch("glitch_poc.ollama.urlopen", side_effect=TimeoutError("slow")):
            self.assertEqual(client.annotate(event(), ()).annotation_status, "error")
