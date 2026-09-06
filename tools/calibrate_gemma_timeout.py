"""Offline structured-only calibration across four real DSP event payloads; audio never leaves this process."""
from __future__ import annotations

import json
import sys
import wave
from datetime import datetime
from pathlib import Path
from time import monotonic

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glitch_poc.calibration import default_drain_timeout_seconds, proposal_for_four_classes
from glitch_poc.contracts import DSPEvent, PCMBlock
from glitch_poc.dsp import DSPProcessor
from glitch_poc.ollama import MODEL, OllamaClient, OllamaConfig, annotation_payload

EXPECTED_TYPES = frozenset({"click", "dropout", "stutter", "clipping"})


def fixture_payloads() -> list[tuple[DSPEvent, tuple[tuple[str, dict[str, object]], ...]]]:
    """Run local DSP over the fixture; only resulting canonical facts are ever passed to Ollama."""
    fixture = ROOT / "fixtures" / "audio" / "poc_corrupted.wav"
    with wave.open(str(fixture), "rb") as source:
        pcm = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").reshape(-1, 2).astype(np.float32) / 32768
    processor, records = DSPProcessor(), []
    for sequence, start in enumerate(range(0, len(pcm), 1024)):
        records.extend(processor.process(PCMBlock.create("calibration", sequence, start, pcm[start:start + 1024])))
    records.extend(processor.finish())
    evidence = tuple((key, value.to_dict()) for key, value in processor.evidence.items())
    final = [record for record in records if record.lifecycle == "CLOSED" and record.glitch_type in EXPECTED_TYPES]
    if {record.glitch_type for record in final} != EXPECTED_TYPES:
        raise RuntimeError("fixture did not produce all four canonical final types")
    return [(record, evidence) for record in sorted(final, key=lambda item: item.glitch_type)]


def run_probe(client: OllamaClient, phase: str, event: DSPEvent,
              evidence: tuple[tuple[str, dict[str, object]], ...]) -> dict[str, object]:
    payload_bytes = len(json.dumps(annotation_payload(event, evidence), separators=(",", ":")).encode())
    started = monotonic()
    annotation = client.annotate(event, evidence)
    elapsed = monotonic() - started
    valid = annotation.annotation_status in {"coherent", "insufficient_evidence"}
    return {"phase": phase, "glitch_type": event.glitch_type, "timestamp": datetime.now().astimezone().isoformat(),
            "payload_bytes": payload_bytes, "elapsed_s": elapsed, "success": valid,
            "response_valid_grounded": valid, "error_class": None if annotation.error is None else annotation.error.split(":", 1)[0],
            "annotation_status": annotation.annotation_status, "model": annotation.model, "timeout_s": client.config.timeout_s}


def main() -> int:
    # 60 s is probe-only; runtime defaults change only after all classes validate.
    client = OllamaClient(OllamaConfig(timeout_s=60.0))
    payloads = fixture_payloads()
    samples = [run_probe(client, "cold", event, evidence) for event, evidence in payloads]
    samples.extend(run_probe(client, "warm", event, evidence) for event, evidence in payloads)
    timeout = proposal_for_four_classes(samples, EXPECTED_TYPES)
    output = ROOT / ".work" / "gemma-timeout-calibration.json"
    output.parent.mkdir(exist_ok=True)
    record = {"schema_version": "gemma-timeout-calibration-v2", "model": MODEL,
              "generated_at": datetime.now().astimezone().isoformat(), "samples": samples,
              "expected_types": sorted(EXPECTED_TYPES), "valid_sample_count": sum(item["success"] for item in samples),
              "proposed_timeout_s": timeout, "drain_timeout_s": None if timeout is None else default_drain_timeout_seconds(timeout),
              "verified_all_classes": timeout is not None,
              "policy": {"percentile": "max_if_n_lt_20_else_p95", "safety_factor": 1.5,
                         "rounding": "ceil_seconds", "minimum_s": 5, "maximum_s": 60}}
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if timeout is None:
        print(f"not verified for all four classes; diagnostics written to {output}", file=sys.stderr)
        return 1
    print(json.dumps({"proposed_timeout_s": timeout, "drain_timeout_s": record["drain_timeout_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
