from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep

from .audit import SessionAuditLogger, validate_log_dir
from .ingest import FFmpegProducer
from .ollama import GemmaWorker, OllamaClient
from .ring import PCMBlockRing
from .runtime import SliceRuntime
from .tui import GlitchTui

FIXTURE_ALIASES = {
    "clean": "poc_clean.wav",
    "corrupted": "poc_corrupted.wav",
    "rock-clean": "poc_rock_v1_clean.wav",
    "rock-corrupted": "poc_rock_v1_corrupted.wav",
    "harvard-clean": "poc_harvard_v1_clean.wav",
    "harvard-corrupted": "poc_harvard_v1_corrupted.wav",
}


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    kind: str
    label: str
    fingerprint: str
    stream_id: str


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="DSP-authoritative glitch POC with optional grounded Gemma annotations.")
    result.add_argument("source", help="Fixture alias (clean, corrupted, rock-clean, rock-corrupted, harvard-clean, harvard-corrupted) or local audio file.")
    result.add_argument("--no-ui", action="store_true", help="Print decimated snapshots for headless use.")
    result.add_argument("--gemma-annotations", action="store_true", help="Opt in to local grounded annotations for consolidated DSP events.")
    result.add_argument("--gemma-drain-timeout", type=float,
                        help="Explicit total graceful-drain budget cap/override in seconds; default auto-scales pending work.")
    result.add_argument("--log-dir", default="logs", help="Project-relative directory for session JSONL logs.")
    result.add_argument("--duration", type=float, help="Stop after this many seconds (useful in CI).")
    return result


def resolve_source(root: Path, value: str) -> SourceSpec:
    """Resolve a local source without preserving its path in runtime-facing metadata."""
    if value in FIXTURE_ALIASES:
        path = root / "fixtures" / "audio" / FIXTURE_ALIASES[value]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        return SourceSpec(path, "fixture", f"fixture {value}", digest, f"fixture-{value}-{digest}")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise ValueError("audio source does not exist")
    if not path.is_file():
        raise ValueError("audio source must be a regular file")
    try:
        with path.open("rb"):
            pass
    except OSError:
        raise ValueError("audio source is not readable") from None
    stat = path.stat()
    digest = hashlib.sha256(f"{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:12]
    return SourceSpec(path, "external_file", f"external file {digest}", digest, f"external-file-{digest}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.gemma_drain_timeout is not None and args.gemma_drain_timeout < 0:
        parser().error("--gemma-drain-timeout must be non-negative")
    root = Path(__file__).resolve().parents[1]
    try:
        validate_log_dir(root, args.log_dir)
    except ValueError as error:
        parser().error(str(error))
    try:
        source = resolve_source(root, args.source)
    except ValueError as error:
        parser().error(str(error))
    audit = SessionAuditLogger(root, args.log_dir, fixture=source.label, stream_id=source.stream_id,
                                profile_id="poc-d2-v2", gemma_enabled=args.gemma_annotations,
                                source_kind=source.kind, source_fingerprint=source.fingerprint)
    ring = PCMBlockRing(capacity=128, block_frames=1024)
    producer = FFmpegProducer(str(source.path), ring, stream_id=source.stream_id)
    if not producer.start():
        audit.close("ffmpeg_start_failed", {"ffmpeg": producer.state, "error": producer.error or ""},
                    datetime.now().astimezone().isoformat())
        print(json.dumps({"ffmpeg": producer.state, "error": producer.error}))
        return 2
    runtime = SliceRuntime(ring, producer, source_label=source.label)
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
            "gemma_drain": runtime.last_gemma_drain,
            "log": None if log is None else {"state": log.state, "queued": log.queued,
                                                "dropped": log.dropped, "errors": log.errors,
                                                "last_error": log.last_error}}


if __name__ == "__main__":
    raise SystemExit(main())
