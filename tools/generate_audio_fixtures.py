"""Genera fixture WAV deterministiche source/observed per la POC.

Il corpus v1 usa la libreria standard; la sintesi rock-v1 usa NumPy locale per
operazioni float64 vettoriali, senza asset, rete o audio esterno.
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

import numpy as np

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

ROCK_DURATION_SECONDS = 16.0
ROCK_FRAME_COUNT = int(SAMPLE_RATE * ROCK_DURATION_SECONDS)
ROCK_SEED = 1_380_926_283
ROCK_CLICK = (111_216, 111_264)
ROCK_DROPOUT = (255_840, 267_360)
ROCK_DROPOUT_ZERO = (256_080, 267_120)
ROCK_STUTTER_SOURCE = (430_080, 433_920)
ROCK_STUTTER = (437_760, 460_800)
ROCK_CLIPPING = (584_640, 610_560)

HARVARD_SAMPLE_RATE = 44_100
HARVARD_FRAMES = 809_508
HARVARD_SOURCE_SHA256 = "971b4163670445c415c6b0fb6813c38093409ecac2f6b4d429ae3574d24ad470"
HARVARD_CLICK = (132_300, 132_344)
HARVARD_DROPOUT = (88_200, 98_784)
HARVARD_DROPOUT_ZERO = (88_421, 98_563)
HARVARD_STUTTER_SOURCE = (502_740, 506_268)
HARVARD_STUTTER = (506_268, 527_436)
HARVARD_CLIPPING = (593_145, 610_785)


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


def write_wav_array(path: Path, samples: np.ndarray) -> None:
    """Write already-quantized little-endian stereo samples without a second quantization."""
    with wave.open(str(path), "wb") as output:
        output.setnchannels(CHANNELS)
        output.setsampwidth(SAMPLE_WIDTH)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(np.asarray(samples, dtype="<i2").tobytes())


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


def _xorshift32_noise(length: int, seed: int = ROCK_SEED) -> np.ndarray:
    """Stable local PRNG: xorshift32, mapped to [-1, 1) as float64."""
    values = np.empty(length, dtype=np.float64)
    state = seed & 0xFFFF_FFFF
    for index in range(length):
        state ^= (state << 13) & 0xFFFF_FFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFF_FFFF
        state &= 0xFFFF_FFFF
        values[index] = state / 2**31 - 1.0
    return values


def _rock_clean_float() -> np.ndarray:
    """Original deterministic 120 BPM rock-like stereo synthesis in float64."""
    frames = np.arange(ROCK_FRAME_COUNT, dtype=np.float64)
    seconds = frames / SAMPLE_RATE
    beat = np.mod(frames, 24_000)
    bar = (frames // 96_000).astype(np.int64)
    variation = 0.82 + 0.06 * np.mod(bar, 4)
    noise = _xorshift32_noise(ROCK_FRAME_COUNT)

    kick_attack = np.minimum(1.0, beat / 240.0)
    kick = 0.16 * kick_attack * np.exp(-beat / 4_200.0) * np.sin(2 * math.pi * (62 - beat / 900.0) * beat / SAMPLE_RATE)
    snare_phase = np.abs(beat - 12_000)
    snare = 0.055 * np.minimum(1.0, snare_phase / 240.0) * np.exp(-snare_phase / 3_000.0) * noise
    hat_phase = np.mod(frames, 6_000)
    hats = 0.016 * np.minimum(1.0, hat_phase / 180.0) * np.exp(-hat_phase / 1_300.0) * noise

    # A sixteenth-note fill is deliberately musical control material, not a fault.
    fill_phase = np.mod(frames - 168_000, 3_000)
    fill_mask = (frames >= 168_000) & (frames < 191_040)
    fill = np.where(fill_mask, 0.030 * np.minimum(1.0, fill_phase / 180.0) * np.exp(-fill_phase / 900.0) * noise, 0.0)

    bass_note = np.array([55.0, 49.0, 41.2, 46.25])[np.mod(bar, 4)]
    bass = 0.095 * np.sin(2 * math.pi * bass_note * seconds) + 0.018 * np.sin(4 * math.pi * bass_note * seconds)
    chord_root = np.array([110.0, 98.0, 82.4, 92.5])[np.mod(bar, 4)]
    guitar = sum(0.022 * np.sin(2 * math.pi * chord_root * ratio * seconds)
                 for ratio in (1.0, 1.25, 1.5))
    guitar *= 0.75 + 0.25 * np.sin(2 * math.pi * 2.0 * seconds) ** 2

    mono = variation * (kick + snare + hats + fill + bass)
    left = mono + guitar * 0.85
    right = mono + guitar * 1.15
    mix = np.column_stack((left, right))

    # Clean controls: stop/fade, tremolo, soft limiter and a documented hard mute.
    stop_out = np.clip((seconds - 6.0) / 0.020, 0.0, 1.0)
    stop_in = np.clip((seconds - 6.30) / 0.020, 0.0, 1.0)
    stop_gain = np.where(seconds < 6.02, 1.0 - .55 * stop_out,
                         np.where(seconds < 6.30, .45, .45 + .55 * stop_in))
    mix *= stop_gain[:, None]
    tremolo = 0.72 + 0.28 * (0.5 + 0.5 * np.sin(2 * math.pi * 6 * seconds))
    mix[(seconds >= 7.0) & (seconds < 8.0)] *= tremolo[(seconds >= 7.0) & (seconds < 8.0), None]
    limiter = (seconds >= 13.4) & (seconds < 14.2)
    mix[limiter] = 0.30 * np.tanh(mix[limiter] / 0.30)
    mute = (seconds >= 14.6) & (seconds < 14.82)
    mute_phase = seconds[mute] - 14.6
    mute_gain = np.where(mute_phase < .005, 1 - mute_phase / .005,
                         np.where(mute_phase >= .215, (mute_phase - .215) / .005, 0.0))
    mix[mute] *= mute_gain[:, None]
    peak = float(np.max(np.abs(mix)))
    return mix * (0.28 / peak)


def _rock_quantize(clean: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(clean * INT16_MAX), -32_768, INT16_MAX).astype(np.int16)


def _rock_corrupted(clean: np.ndarray) -> np.ndarray:
    corrupted = clean.copy()
    for offset, frame in enumerate(range(*ROCK_CLICK)):
        pulse = 0.62 * math.exp(-offset / 9.0) * math.cos(math.pi * offset)
        value = int(np.clip(round(pulse * INT16_MAX), -32_768, INT16_MAX))
        corrupted[frame] = (value, value)
    start, end = ROCK_DROPOUT
    edge = 240
    for frame in range(start, start + edge):
        corrupted[frame] = np.rint(clean[frame] * ((start + edge - frame) / edge)).astype(np.int16)
    corrupted[ROCK_DROPOUT_ZERO[0]:ROCK_DROPOUT_ZERO[1]] = 0
    for frame in range(end - edge, end):
        corrupted[frame] = np.rint(clean[frame] * ((frame - (end - edge) + 1) / edge)).astype(np.int16)
    source = clean[ROCK_STUTTER_SOURCE[0]:ROCK_STUTTER_SOURCE[1]]
    for destination in range(ROCK_STUTTER[0], ROCK_STUTTER[1], len(source)):
        corrupted[destination:destination + len(source)] = source
    edge = 960
    plateau = 0.46
    for frame in range(*ROCK_CLIPPING):
        factor = min(1.0, (frame - ROCK_CLIPPING[0]) / edge, (ROCK_CLIPPING[1] - 1 - frame) / edge)
        transformed = np.clip(8.0 * clean[frame].astype(np.float64) / INT16_MAX, -plateau, plateau)
        value = clean[frame].astype(np.float64) / INT16_MAX
        corrupted[frame] = np.rint((value + factor * (transformed - value)) * INT16_MAX).astype(np.int16)
    return corrupted


def _rock_manifest(clean_path: Path, corrupted_path: Path, clean: np.ndarray) -> dict[str, object]:
    derivative = float(np.max(np.abs(np.diff(clean.astype(np.float64) / INT16_MAX, axis=0))))
    lag = 24_000
    correlation = float(np.corrcoef(clean[:-lag, 0], clean[lag:, 0])[0, 1])
    return {
        "schema_version": 2,
        "profile_id": "poc-d2-v2",
        "generator": {"name": "poc-rock-v1", "seed": ROCK_SEED, "prng": "xorshift32", "quantization": "single float64-to-s16le rounding"},
        "format": {"container": "WAV", "codec": "PCM s16le", "sample_rate_hz": SAMPLE_RATE, "channels": CHANNELS, "frames": ROCK_FRAME_COUNT, "duration_seconds": ROCK_DURATION_SECONDS},
        "files": {"clean": {"path": clean_path.name, "sha256": sha256(clean_path)}, "corrupted": {"path": corrupted_path.name, "sha256": sha256(corrupted_path)}},
        "faults": [
            event("click", *ROCK_CLICK, replacement="both channels", formula="0.62*exp(-k/9)*cos(pi*k)"),
            event("dropout", *ROCK_DROPOUT, fade_out_frames=240, zero_interval={"start": ROCK_DROPOUT_ZERO[0], "end": ROCK_DROPOUT_ZERO[1]}, fade_in_frames=240,
                  expected_status="uncertain", allowed_statuses=["uncertain"],
                  rationale="No-reference RMS recovery is intentionally ambiguous in this musical context."),
            event("stutter", *ROCK_STUTTER, source_interval={"start": ROCK_STUTTER_SOURCE[0], "end": ROCK_STUTTER_SOURCE[1]}, repetitions=6),
            event("clipping", *ROCK_CLIPPING, plateau=0.46, ramp_frames=960, transform="ramp(20ms, clip(8*x_clean, -0.46, +0.46))"),
        ],
        "clean_controls": [
            {"name": "drum_fill", "interval": [168_000, 191_040]}, {"name": "stop_fade", "interval": [288_000, 303_360]},
            {"name": "tremolo", "interval": [336_000, 384_000]}, {"name": "soft_limiter", "interval": [643_200, 681_600]},
            {"name": "hard_mute", "interval": [700_800, 711_360], "expected_status": "uncertain",
             "rationale": "Intentional no-reference ambiguity control; it is not an operational false alarm."},
        ],
        "clean_invariants": {"peak_s16": int(np.max(np.abs(clean))), "max_derivative_float": derivative, "correlation_lag_500ms_left": correlation, "detected_target_events_expected": 0},
        "identifiability_limits": ["One synthetic song cannot estimate operational false-alarm rates.",
                                    "The corrupted dropout is profile-dependent and not reliably identifiable as detected.",
                                    "The hard mute control intentionally produces an uncertain dropout."],
    }


def generate_rock(output_directory: Path) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    clean_path = output_directory / "poc_rock_v1_clean.wav"
    corrupted_path = output_directory / "poc_rock_v1_corrupted.wav"
    manifest_path = output_directory / "poc_rock_v1_ground_truth.json"
    clean = _rock_quantize(_rock_clean_float())
    write_wav_array(clean_path, clean)
    write_wav_array(corrupted_path, _rock_corrupted(clean))
    manifest_path.write_text(json.dumps(_rock_manifest(clean_path, corrupted_path, clean), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"clean": clean_path, "corrupted": corrupted_path, "manifest": manifest_path}


def _riff_data_chunk(raw: bytes) -> tuple[int, int]:
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("harvard.wav is not a RIFF/WAVE file")
    offset = 12
    while offset + 8 <= len(raw):
        kind, size = raw[offset:offset + 4], struct.unpack_from("<I", raw, offset + 4)[0]
        data_offset = offset + 8
        if kind == b"data":
            return data_offset, size
        offset = data_offset + size + size % 2
    raise ValueError("harvard.wav has no data chunk")


def _harvard_runtime(frame: int) -> int:
    return round(frame * SAMPLE_RATE / HARVARD_SAMPLE_RATE)


def _harvard_event(kind: str, interval: tuple[int, int], **details: object) -> dict[str, object]:
    result = event(kind, *interval, **details)
    result["source_sample_rate_hz"] = HARVARD_SAMPLE_RATE
    result["runtime_interval"] = {"unit": "frames", "sample_rate_hz": SAMPLE_RATE,
                                  "semantics": "half-open", "start": _harvard_runtime(interval[0]),
                                  "end": _harvard_runtime(interval[1])}
    return result


def _harvard_corrupted(raw: bytes) -> bytes:
    data_offset, data_size = _riff_data_chunk(raw)
    if data_size != HARVARD_FRAMES * CHANNELS * SAMPLE_WIDTH:
        raise ValueError("harvard.wav PCM data size does not match the approved source")
    pcm = np.frombuffer(raw, dtype="<i2", count=HARVARD_FRAMES * CHANNELS, offset=data_offset).reshape(-1, 2).copy()
    clean = pcm.copy()
    for offset, frame in enumerate(range(*HARVARD_CLICK)):
        value = int(np.clip(np.rint(.80 * math.exp(-offset / 8) * math.cos(math.pi * offset) * INT16_MAX), -32_768, INT16_MAX))
        pcm[frame] = (value, value)
    edge = 221
    for frame in range(HARVARD_DROPOUT[0], HARVARD_DROPOUT[0] + edge):
        pcm[frame] = np.rint(clean[frame] * ((HARVARD_DROPOUT[0] + edge - frame) / edge)).astype(np.int16)
    pcm[HARVARD_DROPOUT_ZERO[0]:HARVARD_DROPOUT_ZERO[1]] = 0
    for frame in range(HARVARD_DROPOUT[1] - edge, HARVARD_DROPOUT[1]):
        pcm[frame] = np.rint(clean[frame] * ((frame - (HARVARD_DROPOUT[1] - edge) + 1) / edge)).astype(np.int16)
    source = clean[HARVARD_STUTTER_SOURCE[0]:HARVARD_STUTTER_SOURCE[1]]
    for destination in range(HARVARD_STUTTER[0], HARVARD_STUTTER[1], len(source)):
        pcm[destination:destination + len(source)] = source
    for frame in range(*HARVARD_CLIPPING):
        ramp = min(1.0, (frame - HARVARD_CLIPPING[0]) / 882, (HARVARD_CLIPPING[1] - 1 - frame) / 882)
        original = clean[frame].astype(np.float64) / INT16_MAX
        clipped = np.clip(8.0 * original, -.46, .46)
        pcm[frame] = np.rint((original + ramp * (clipped - original)) * INT16_MAX).astype(np.int16)
    result = bytearray(raw)
    result[data_offset:data_offset + data_size] = pcm.astype("<i2", copy=False).tobytes()
    return bytes(result)


def _harvard_manifest(source: Path, clean: Path, corrupted: Path) -> dict[str, object]:
    controls = [
        ("plosive", (63_000, 64_680), "clean"), ("sibilant", (213_885, 216_090), "clean"),
        ("long_pause", (169_344, 188_160), "clean"), ("ambiguous_natural_pause", (348_751, 353_713), "uncertain"),
        ("repeated_word", (0, 0), "unverified_non_scored"),
    ]
    return {
        "schema_version": 3, "profile_id": "poc-d2-v2",
        "source": {"path": source.name, "sha256": HARVARD_SOURCE_SHA256, "sample_rate_hz": HARVARD_SAMPLE_RATE,
                   "frames": HARVARD_FRAMES, "runtime_sample_rate_hz": SAMPLE_RATE},
        "format": {"container": "WAV", "codec": "PCM s16le", "sample_rate_hz": HARVARD_SAMPLE_RATE,
                   "channels": CHANNELS, "frames": HARVARD_FRAMES, "duration_seconds": HARVARD_FRAMES / HARVARD_SAMPLE_RATE},
        "files": {"clean": {"path": clean.name, "sha256": sha256(clean)}, "corrupted": {"path": corrupted.name, "sha256": sha256(corrupted)}},
        "faults": [
            _harvard_event("click", HARVARD_CLICK, expected_status="detected", formula="0.80*exp(-k/8)*cos(pi*k)"),
            _harvard_event("dropout", HARVARD_DROPOUT, expected_status="detected", fade_out_frames=221,
                           zero_interval={"start": HARVARD_DROPOUT_ZERO[0], "end": HARVARD_DROPOUT_ZERO[1]}, fade_in_frames=221),
            _harvard_event("stutter", HARVARD_STUTTER, expected_status="detected", source_interval={"start": HARVARD_STUTTER_SOURCE[0], "end": HARVARD_STUTTER_SOURCE[1]}, repetitions=6),
            _harvard_event("clipping", HARVARD_CLIPPING, expected_status="detected", plateau=.46, ramp_frames=882),
        ],
        "controls": [{"name": name, "source_interval": list(interval), "runtime_interval": [_harvard_runtime(interval[0]), _harvard_runtime(interval[1])], "expected_status": status} for name, interval, status in controls],
        "baseline_events": [
            {"class": "clipping", "status": "detected", "runtime_interval": [261_120, 263_520]},
            {"class": "clipping", "status": "detected", "runtime_interval": [511_200, 513_600]},
            {"class": "dropout", "status": "uncertain", "runtime_interval": [379_200, 390_240]},
        ],
        "identifiability_limits": ["Baseline speech events are declared, not injected faults.", "The DSP runtime does not use the clean/reference waveform.", "One source clip is insufficient for operational rates."],
    }


def generate_harvard(output_directory: Path) -> dict[str, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    source = output_directory / "harvard.wav"
    if not source.exists():
        canonical = Path(__file__).resolve().parents[1] / "fixtures" / "audio"
        source = canonical / "harvard.wav"
        if not source.exists():
            source = canonical / "poc_harvard_v1_clean.wav"
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HARVARD_SOURCE_SHA256:
        raise ValueError("harvard.wav hash does not match the approved source")
    canonical_source = Path(__file__).resolve().parents[1] / "fixtures" / "audio" / "harvard.wav"
    if output_directory.resolve() == canonical_source.parent.resolve() and not canonical_source.exists():
        canonical_source.write_bytes(raw)
        source = canonical_source
    clean, corrupted = output_directory / "poc_harvard_v1_clean.wav", output_directory / "poc_harvard_v1_corrupted.wav"
    manifest = output_directory / "poc_harvard_v1_ground_truth.json"
    clean.write_bytes(raw)
    corrupted.write_bytes(_harvard_corrupted(raw))
    manifest.write_text(json.dumps(_harvard_manifest(source, clean, corrupted), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"clean": clean, "corrupted": corrupted, "manifest": manifest}


def main() -> None:
    default = Path(__file__).resolve().parents[1] / "fixtures" / "audio"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=default)
    args = parser.parse_args()
    generated = {**generate(args.output_directory), **{f"rock_{name}": path for name, path in generate_rock(args.output_directory).items()}, **{f"harvard_{name}": path for name, path in generate_harvard(args.output_directory).items()}}
    for name, path in generated.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
