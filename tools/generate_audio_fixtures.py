"""Genera fixture WAV deterministiche source/observed per la POC.

Il modulo usa esclusivamente la libreria standard: la POC futura potrà usare
NumPy/SciPy, ma la generazione del corpus non deve dipendere da esse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import wave
from collections.abc import Iterable
from pathlib import Path

SAMPLE_RATE = 48_000
CHANNELS = 2
SAMPLE_WIDTH = 2
DURATION_SECONDS = 10.0
FRAME_COUNT = int(SAMPLE_RATE * DURATION_SECONDS)
INT16_MAX = 32_767

PAD_CHORDS = ([48, 55, 60], [45, 52, 57], [41, 48, 53], [43, 50, 55], [48, 55, 60], [48, 55, 60])
MELODY_NOTES = [64, 67, 72, 67, 64, 69, 72, 69, 65, 69, 72, 69, 67, 71, 74, 71, 64, 67, 72, 67]

CLICK = (108_000, 108_096)
DROPOUT = (204_000, 218_400)
STUTTER_SOURCE = (293_760, 297_600)
STUTTER = (297_600, 320_640)
CLIPPING = (393_600, 422_400)


def midi_hz(note: int) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def raised_cosine(frame: int, length: int, edge: int) -> float:
    """Envelope unitario con rampe half-cosine, inclusivo degli estremi."""
    if frame < edge:
        return 0.5 - 0.5 * math.cos(math.pi * frame / edge)
    remaining = length - 1 - frame
    if remaining < edge:
        return 0.5 - 0.5 * math.cos(math.pi * remaining / edge)
    return 1.0


def global_fade(frame: int) -> float:
    return raised_cosine(frame, FRAME_COUNT, int(0.080 * SAMPLE_RATE))


def _tone(frame: int, note: int) -> float:
    phase = 2.0 * math.pi * midi_hz(note) * frame / SAMPLE_RATE
    return math.sin(phase) + 0.2 * math.sin(2.0 * phase)


def make_clean_float() -> list[tuple[float, float]]:
    """Crea il mix stereo non normalizzato, con sintesi interamente deterministica."""
    samples: list[tuple[float, float]] = []
    pad_length = 2 * SAMPLE_RATE
    melody_length = SAMPLE_RATE // 2
    pad_edge = int(0.030 * SAMPLE_RATE)
    melody_edge = int(0.015 * SAMPLE_RATE)
    for frame in range(FRAME_COUNT):
        pad_index = frame // pad_length
        # La durata vincolante è 10 s: dei sei accordi specificati, i primi
        # cinque occupano i cinque segmenti completi disponibili.
        chord = PAD_CHORDS[pad_index]
        local_pad = frame % pad_length
        pad = sum(_tone(local_pad, note) for note in chord) / len(chord)
        pad *= raised_cosine(local_pad, pad_length, pad_edge)

        melody_index = frame // melody_length
        local_melody = frame % melody_length
        melody = _tone(local_melody, MELODY_NOTES[melody_index])
        melody *= raised_cosine(local_melody, melody_length, melody_edge)
        # Pan lineare, alternato a ogni nota; il pad resta al centro.
        pan = -0.2 if melody_index % 2 == 0 else 0.2
        left = 0.65 * pad + 0.35 * melody * (1.0 - pan)
        right = 0.65 * pad + 0.35 * melody * (1.0 + pan)
        fade = global_fade(frame)
        samples.append((left * fade, right * fade))

    peak = max(abs(value) for frame in samples for value in frame)
    scale = 0.30 / peak
    return [(left * scale, right * scale) for left, right in samples]


def quantize(samples: Iterable[tuple[float, float]]) -> list[tuple[int, int]]:
    return [
        (max(-32_768, min(INT16_MAX, round(left * INT16_MAX))), max(-32_768, min(INT16_MAX, round(right * INT16_MAX))))
        for left, right in samples
    ]


def make_corrupted(clean: list[tuple[int, int]]) -> list[tuple[int, int]]:
    corrupted = clean.copy()
    for k, frame in enumerate(range(*CLICK)):
        pulse = 0.38 * math.exp(-k / 10.0) * math.cos(math.pi * k)
        pulse += 0.12 * math.exp(-k / 36.0) * math.cos(2.0 * math.pi * 180.0 * k / SAMPLE_RATE)
        value = max(-0.50, min(0.50, pulse))
        quantized = max(-32_768, min(INT16_MAX, round(value * INT16_MAX)))
        corrupted[frame] = (quantized, quantized)

    start, end = DROPOUT
    fade_frames = 240
    for frame in range(start, start + fade_frames):
        factor = (start + fade_frames - frame) / fade_frames
        left, right = clean[frame]
        corrupted[frame] = (round(left * factor), round(right * factor))
    for frame in range(start + fade_frames, end - fade_frames):
        corrupted[frame] = (0, 0)
    for frame in range(end - fade_frames, end):
        factor = (frame - (end - fade_frames) + 1) / fade_frames
        left, right = clean[frame]
        corrupted[frame] = (round(left * factor), round(right * factor))

    block = clean[STUTTER_SOURCE[0] : STUTTER_SOURCE[1]]
    for repetition in range(6):
        destination = STUTTER[0] + repetition * len(block)
        corrupted[destination : destination + len(block)] = block

    for frame in range(*CLIPPING):
        left, right = clean[frame]
        corrupted[frame] = (
            round(max(-0.45, min(0.45, 6.0 * left / INT16_MAX)) * INT16_MAX),
            round(max(-0.45, min(0.45, 6.0 * right / INT16_MAX)) * INT16_MAX),
        )
    return corrupted


def write_wav(path: Path, samples: list[tuple[int, int]]) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(CHANNELS)
        output.setsampwidth(SAMPLE_WIDTH)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(struct.pack(f"<{len(samples) * CHANNELS}h", *(sample for frame in samples for sample in frame)))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event(kind: str, start: int, end: int, **details: object) -> dict[str, object]:
    result: dict[str, object] = {
        "class": kind,
        "interval": {"unit": "frames", "semantics": "half-open", "start": start, "end": end},
        "seconds": {"start": start / SAMPLE_RATE, "end": end / SAMPLE_RATE},
    }
    result.update(details)
    return result


def build_manifest(clean_path: Path, corrupted_path: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "purpose": "Deterministic clean/reference and corrupted/observed WAV pair for the POC.",
        "format": {"container": "WAV", "codec": "PCM s16le", "sample_rate_hz": SAMPLE_RATE, "channels": CHANNELS, "frames": FRAME_COUNT, "duration_seconds": DURATION_SECONDS},
        "files": {
            "clean": {"path": clean_path.name, "sha256": sha256(clean_path)},
            "corrupted": {"path": corrupted_path.name, "sha256": sha256(corrupted_path)},
        },
        "faults": [
            event("click", *CLICK, replacement="both channels", formula="0.38*exp(-k/10)*cos(pi*k)+0.12*exp(-k/36)*cos(2*pi*180*k/48000)"),
            event("dropout", *DROPOUT, fade_out_frames=240, zero_interval={"start": 204_240, "end": 218_160}, fade_in_frames=240),
            event("stutter", *STUTTER, source_interval={"start": STUTTER_SOURCE[0], "end": STUTTER_SOURCE[1]}, repetitions=6),
            event("clipping", *CLIPPING, transform="clip(6*x_clean, -0.45, +0.45)"),
        ],
    }


def generate(output_directory: Path) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    clean_path = output_directory / "poc_clean.wav"
    corrupted_path = output_directory / "poc_corrupted.wav"
    manifest_path = output_directory / "poc_ground_truth.json"
    clean = quantize(make_clean_float())
    write_wav(clean_path, clean)
    write_wav(corrupted_path, make_corrupted(clean))
    manifest_path.write_text(json.dumps(build_manifest(clean_path, corrupted_path), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"clean": clean_path, "corrupted": corrupted_path, "manifest": manifest_path}


def main() -> None:
    default = Path(__file__).resolve().parents[1] / "fixtures" / "audio"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=default)
    args = parser.parse_args()
    for name, path in generate(args.output_directory).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
