from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep

from .audit import SessionAuditLogger, validate_log_dir
from .calibration import default_drain_timeout_seconds
from .ingest import FFmpegProducer
from .ollama import GemmaWorker, OllamaClient, OllamaConfig
from .ring import PCMBlockRing
from .runtime import SliceRuntime
from .tui import GlitchTui


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="DSP-authoritative glitch POC with optional grounded Gemma annotations.")
    result.add_argument("fixture", choices=("clean", "corrupted"))
    result.add_argument("--no-ui", action="store_true", help="Print decimated snapshots for headless use.")
    result.add_argument("--gemma-annotations", action="store_true", help="Opt in to local grounded annotations for consolidated DSP events.")
    result.add_argument("--gemma-drain-timeout", type=float,
                        default=default_drain_timeout_seconds(OllamaConfig().timeout_s),
                        help="Seconds to drain accepted Gemma annotations after natural headless EOF.")
    result.add_argument("--log-dir", default="logs", help="Project-relative directory for session JSONL logs.")
    result.add_argument("--duration", type=float, help="Stop after this many seconds (useful in CI).")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.gemma_drain_timeout < 0:
        parser().error("--gemma-drain-timeout must be non-negative")
    root = Path(__file__).resolve().parents[1]
    try:
        validate_log_dir(root, args.log_dir)
    except ValueError as error:
        parser().error(str(error))
    fixture = root / "fixtures" / "audio" / f"poc_{args.fixture}.wav"
    audit = SessionAuditLogger(root, args.log_dir, fixture=args.fixture, stream_id=args.fixture,
                               profile_id="poc-d2-v2", gemma_enabled=args.gemma_annotations)
    ring = PCMBlockRing(capacity=128, block_frames=1024)
    producer = FFmpegProducer(str(fixture), ring, stream_id=args.fixture)
    if not producer.start():
        audit.close("ffmpeg_start_failed", {"ffmpeg": producer.state, "error": producer.error or ""},
                    datetime.now().astimezone().isoformat())
        print(json.dumps({"ffmpeg": producer.state, "error": producer.error}))
        return 2
    runtime = SliceRuntime(ring, producer)
    runtime.audit_log = audit
    if args.gemma_annotations:
        # Worker starts after the runtime exists so its callback can only append separate records.
        runtime.gemma_worker = GemmaWorker(OllamaClient(), runtime.add_annotation, runtime.add_abandoned_annotation)
    runtime.start()
    if not args.no_ui:
        try:
            GlitchTui(runtime, gemma_drain_timeout_s=args.gemma_drain_timeout).run()
        finally:
            runtime.stop()
            producer.stop()
            audit.close("ui_exit", _audit_summary(runtime), datetime.now().astimezone().isoformat())
        return 0
    started = monotonic()
    next_snapshot = started
    natural_eof = False
    try:
        while producer.state in {"idle", "running"} or ring.stats().fill:
            if monotonic() >= next_snapshot:
                snap = runtime.snapshot()
                print(json.dumps({"position_frames": snap.position_frames, "rms": snap.rms,
                                   "peak": snap.peak, "ffmpeg": snap.ffmpeg_state,
                                  "buffer_fill": snap.ring.fill,
                                   "events": [event.to_dict() for event in snap.dsp_events],
                                    "evidence": dict(snap.dsp_evidence), "dsp_audit": snap.dsp_audit,
                                    "gemma_annotations": [item.to_dict() for item in snap.gemma_annotations],
                                    "gemma_queue": None if snap.gemma_queue is None else snap.gemma_queue.__dict__,
                                    "audit_log": None if snap.audit_log is None else snap.audit_log.__dict__}))
                next_snapshot += 1 / 12
            sleep(0.005)
            if args.duration is not None and monotonic() - started >= args.duration:
                break
        natural_eof = producer.state == "eof" and ring.stats().fill == 0
    finally:
        runtime.stop(graceful_gemma_drain=natural_eof, gemma_drain_timeout_s=args.gemma_drain_timeout)
        producer.stop()
        audit.close("headless_exit", _audit_summary(runtime), datetime.now().astimezone().isoformat())
    return 0


def _audit_summary(runtime: SliceRuntime) -> dict[str, object]:
    snap = runtime.snapshot()
    log = snap.audit_log
    return {"dsp_event_count": len(snap.dsp_events), "gemma_annotation_count": len(snap.gemma_annotations),
            "gemma_queue": None if snap.gemma_queue is None else snap.gemma_queue.__dict__,
            "log": None if log is None else {"state": log.state, "queued": log.queued,
                                                "dropped": log.dropped, "errors": log.errors,
                                                "last_error": log.last_error}}


if __name__ == "__main__":
    raise SystemExit(main())
