# Audio Glitch Detection POC

This repository contains a local, deterministic proof of concept for audio glitch detection.
It streams a fixture WAV through persistent FFmpeg, turns it into normalized stereo PCM at 48 kHz,
processes 10 ms DSP hops, and writes append-only audit records. The runtime is for inspection,
calibration, and local validation, not for repair or production deployment.

It currently runs on one of two fixtures: `fixtures/audio/poc_clean.wav` or `fixtures/audio/poc_corrupted.wav`.
The main entrypoint is `glitch-poc` (`glitch_poc.cli:main`). The active profile id is `poc-d2-v2`.

**Out of scope**: microphone capture, cloud inference, automatic repair, and live source/reference comparison in the UI.

## Grounded Deterministic RAG

The code uses a grounded deterministic pattern: DSP decides, Gemma annotates.
`OllamaClient` only accepts consolidated `CLOSED` DSP events with status `detected` or `uncertain`.
The model payload is canonical DSP facts plus selected evidence windows, not raw audio.
The runtime tests also reject audio-like, path-like, and base64-like payloads.

In production there is no raw-audio model path. Raw-audio negative control lives only in offline validation and tests,
and it is not exposed by the CLI.

## Architecture

```text
fixtures/audio/poc_*.wav
        |
        v
FFmpeg decode/resample (48 kHz, stereo, f32le)
        |
        v
PCMBlockRing  --->  SliceRuntime pacing  --->  DSPProcessor
        |                                  |        |
        |                                  |        +--> FeatureWindow evidence
        |                                  |        +--> EventBuilder -> DSPEvent
        |                                  |
        |                                  +--> GemmaWorker / OllamaClient
        |
        +--> SessionAuditLogger -> logs/session-*.jsonl
```

| Layer | Responsibility | Technology | Input | Output | Authoritative |
|---|---|---|---|---|---|
| FFmpeg ingest | Decode fixture audio and pace PCM production | FFmpeg subprocess, `pcm_blocks`, `FFmpegProducer` | Fixture WAV | `PCMBlock` stream, FFmpeg state/error | No |
| Ring buffer | Bounded FIFO with overwrite accounting | `PCMBlockRing` | `PCMBlock` | Buffered PCM + `RingStats` | No |
| DSP detector bank | Build features and candidate glitches | NumPy in `DSPProcessor` | `PCMBlock` | `DSPCandidate`, `FeatureWindow`, interim records | Yes |
| Event builder | Merge, hysteresis, cooldown, priority conflicts | `EventBuilder` inside `glitch_poc.dsp` | `DSPCandidate` + emitted frame | Authoritative `DSPEvent` lifecycle records | Yes |
| Gemma queue/worker | Local, bounded annotation worker | `GemmaWorker`, `OllamaClient`, `spawn` child process | Closed `detected`/`uncertain` events + evidence | `GemmaAnnotation` or `error` | No |
| Audit/log | Append-only session log | `SessionAuditLogger` | Session, DSP, Gemma records | JSONL under `logs/` | No |

Note: the current FFmpeg ingest only decodes/resamples. It does not use `astats`, `silencedetect`, or `ebur128` in the runtime.

## DSP Detectors

The raw score is a ratio-like severity value, not a probability.

| Code detector name | `candidate_type` | What it catches | Features and thresholds used in code | Initial status |
|---|---|---|---|---|
| `click` | `click` | Impulsive discontinuity | 5 ms window; derivative peak; `click_derivative=0.50`; context RMS must stay at or below `0.22` | `detected` |
| `flat_top` | `clipping` | Saturated samples / flat top | 10 ms window; `clip_level=0.44`; `clip_ratio=0.08`; evidence uses near-peak ratio and peak level | `detected` |
| `block_repeat` | `stutter` | Repeated block with similar energy | 20 ms window; autocorrelation around 80 ms lag; `repeat_correlation=0.995`; RMS ratio must stay in `0.80..1.25` | `detected` |
| `loop` | `loop` | Longer periodic loop | 1000 ms window; autocorrelation at 500 ms and 1000 ms lags; `loop_correlation=0.998` | `uncertain` |
| `dropout` | `dropout` | Low-RMS section with post-context check | 100 ms window; low-RMS ratio `q <= 0.18`; floor ratio `<= 0.05`; post-context ratio `0.5..2`; CV `<= 0.20`; descent hops `<= 2` | `uncertain` |

When candidates overlap, priority is `clipping > dropout > stutter > loop > click`.
The builder can supersede lower-priority open events.

## Gemma e4b

Gemma runs locally through Ollama at `http://127.0.0.1:11434/api/generate`.
The worker is single-concurrency, bounded to 16 queued requests, and uses `spawn`, not `fork`.
There are no POST retries. Default timeout is 10 s, and shutdown drain is bounded.
The request payload is minified JSON with `temperature=0`, `num_predict=256`, and `keep_alive=5m`.

| Aspect | Code reality |
|---|---|
| Input | Canonical DSP JSON for a closed `detected` or `uncertain` event, plus at most 2 selected evidence windows |
| Output | `annotation_status` = `coherent`, `insufficient_evidence`, or `error`; `cannot_override_dsp=True` |
| Failure handling | Invalid JSON, transport failure, timeout, or schema mismatch become `error`; DSP keeps running |
| Negative control | Raw audio is not a production input; it is only a validation concept and test coverage check |

The allowed evidence features are type-specific: click uses `derivative_peak` and `context_rms`; clipping uses `near_peak_ratio` and `peak_level`; stutter uses `correlation_lag_80ms` and `rms_ratio`; dropout uses `rms`, `baseline_rms`, `rms_ratio`, and `floor_ratio`; loop uses `correlation_lag_500ms` and `correlation_lag_1000ms`.

## How To Run

Requirements: Python 3.12+, FFmpeg on `PATH`, and Ollama only if you want Gemma annotations.
Install the project with:

```sh
python3.12 -m pip install -e .
```

Run the POC with the console script or module entrypoint:

```sh
glitch-poc clean --no-ui --duration 1
glitch-poc corrupted --no-ui --duration 10.5
glitch-poc corrupted --gemma-annotations
```

Equivalent module form:

```sh
python3.12 -m glitch_poc.cli corrupted --gemma-annotations
```

Available CLI arguments:

| Argument | Meaning |
|---|---|
| `fixture` | Required positional value: `clean` or `corrupted` |
| `--no-ui` | Print JSON snapshots instead of opening the TUI |
| `--gemma-annotations` | Enable the local Gemma worker |
| `--gemma-drain-timeout` | Bounded post-EOF drain budget, 43 s with current defaults |
| `--log-dir` | Project-relative log directory, default `logs` |
| `--duration` | Stop after N seconds |

The TUI needs at least a 100x30 terminal.
It shows four panels: `OBSERVED PCM`, `RUNTIME HEALTH`, `DSP DETECTION`, and `GEMMA DETAIL`/`GEMMA SUMMARY`.

Keys:

| Key | Action |
|---|---|
| `q` | Cancel pending Gemma work and quit immediately |
| `d` | Drain Gemma boundedly after EOF, then quit |
| `[` / `]` | Scroll the Gemma panel |
| `Ctrl+S` | Toggle Gemma detail vs summary |

In headless mode, the app prints one JSON snapshot roughly 12 times per second.
The snapshot contains `position_frames`, `rms`, `peak`, `ffmpeg`, `buffer_fill`, `events`, `evidence`, `dsp_audit`, `gemma_annotations`, `gemma_queue`, and `audit_log`.

## JSONL Audit

Logs are written to `logs/session-YYYYMMDD-HHMMSS[-N].jsonl` by `SessionAuditLogger`.
The schema version is `session-audit-v1`.

| Record type | Main fields |
|---|---|
| `session_started` | `session_id`, `timestamp`, `fixture`, `stream_id`, `profile_id`, `app`, `gemma` |
| `dsp_event` | `event_id`, `revision`, `lifecycle`, `status`, `glitch_type`, `start_frame`, `end_frame`, `emitted_frame`, `raw_score`, `detector_ids`, `evidence_ids`, `epoch_id`, `profile_id` |
| `gemma_annotation` | `event_id`, `revision`, `annotation_status`, `glitch_type_annotation`, `confidence`, `supporting_evidence_ids`, `explanation`, `error`, `cannot_override_dsp`, `model`, `model_metadata`, `prompt_schema_version`, `timeout_s` |
| `session_ended` | `reason`, `summary` |

The audit log intentionally excludes audio, PCM, waveform, spectrogram, prompt bodies, secrets, paths, and URIs.

## Known Limits

Validation currently covers the synthetic fixture pair and the current `poc-d2-v2` thresholds.
The detector bank is deterministic, but its thresholds are tuned to this corpus and not proven on real-world material.
The current UI shows observed PCM only; it does not display a side-by-side source/reference waveform.
Codec-artifact detection is not a dedicated runtime detector yet.

`tools/reference_diff_detector.py` is an offline calibration helper only. It compares clean vs corrupted files and is not part of the runtime path.

## Deviations From The Original Design

| Design expectation | Code reality |
|---|---|
| FFmpeg telemetry filters (`astats`, `silencedetect`, `ebur128`) | Not implemented in runtime; FFmpeg is used only as a persistent decoder/resampler |
| Source vs observed TUI | Not implemented; the current UI shows observed PCM, runtime health, DSP, and Gemma panes |
| Optional spectrogram input to Gemma | Not implemented; Gemma receives structured DSP facts only |
| Dedicated codec-artifact detector | Not implemented; only click, clipping, dropout, stutter, and loop are present |
| Uppercase `D` quit binding | The code binds lower-case `d` for bounded drain and quit |
| Model digest / rich model metadata in logs | Session logs keep model name and timestamp metadata, not a digest contract |
| User-facing raw-audio negative-control mode | Not present; raw audio is only excluded by tests and validation helpers |

If you need the implementation-level truth, treat the code and tests as authoritative.
