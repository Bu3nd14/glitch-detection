# Audio Glitch Detection POC

This repository contains a local, deterministic proof of concept for audio glitch detection.
It streams a fixture WAV through persistent FFmpeg, turns it into normalized stereo PCM at 48 kHz,
processes 10 ms DSP hops, and writes append-only audit records. The runtime is for inspection,
calibration, and local validation, not for repair or production deployment.

It runs on the original, rock-like, and Harvard fixture pairs, or on a readable local audio file supported by FFmpeg.
The main entrypoint is `glitch-poc` (`glitch_poc.cli:main`). The active profile id is `poc-d2-v2`.

`tools/generate_audio_fixtures.py` also regenerates the local deterministic 16-second
rock-like pair `poc_rock_v1_clean.wav` / `poc_rock_v1_corrupted.wav` and its v2
manifest. It uses float64 synthesis, xorshift32 seed `1380926283`, and a single
PCM s16le quantization. The CLI exposes it as `rock-clean` and `rock-corrupted`.
Its musical controls (fill, stop fade, tremolo, limiter,
hard mute) are negative controls, and the hard mute may yield a documented
`uncertain` dropout. The corrupted rock dropout is also intentionally
`uncertain`: this fixture tests no-reference ambiguity, not four-of-four
`detected` outcomes. The DSP has no clean/reference input at runtime. No
`poc-d2-v2` thresholds are changed for it. Run `python -m tools.dsp_metrics`
for v1, rock-v1, and an explicitly `insufficient_corpus` aggregate; it reports
detected-only recall separately from expected and unexpected uncertainty.

`poc-harvard-v1` preserves the approved 44.1 kHz `harvard.wav` byte-for-byte as
its clean fixture and injects four faults only in the PCM data chunk. The CLI
aliases are `harvard-clean` and `harvard-corrupted`; tests and metrics use the
real FFmpeg 44.1 kHz-to-48 kHz runtime decode path. Declared speech baseline
clipping/dropout and controls are reported separately from injected faults, so
Harvard naive counts are not scored false alarms.

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
glitch-poc rock-corrupted --no-ui --duration 1
glitch-poc harvard-corrupted --no-ui --duration 1
glitch-poc "/absolute/path/with spaces/song.flac" --no-ui --duration 10
glitch-poc "/absolute/path/with spaces/song.mp3" --gemma-annotations
```

Equivalent module form:

```sh
python3.12 -m glitch_poc.cli corrupted --gemma-annotations
```

### Expected Fixture Runs

Run the original corrupted fixture with:

```sh
glitch-poc corrupted --gemma-annotations
```

Expected final DSP events: `click`, `dropout`, `stutter`, and `clipping`, all
`detected`. With Ollama available, press `d` after EOF to drain the Gemma queue;
the four final events receive separate grounded annotations and are correlated in
the session JSONL log.

Run the percussive rock fixture with:

```sh
glitch-poc rock-corrupted --gemma-annotations
```

Expected final DSP events: `click`, `stutter`, and `clipping` as `detected`, plus
the injected `dropout` as `uncertain`. A fifth `dropout uncertain` is expected from
the intentional hard-mute musical control; it is not a fifth corruption. Press `d`
after EOF to drain Gemma and write final annotation outcomes to the session log.

Run the Harvard speech fixture with:

```sh
glitch-poc harvard-corrupted --gemma-annotations
```

Expected injected faults: `click`, `dropout`, `stutter`, and `clipping`, all
`detected`. The unmodified speech reference also contains two documented baseline
`clipping detected` events and one natural-pause `dropout uncertain`; these are not
injected faults and are excluded from the scored Harvard metrics. Gemma therefore
receives seven final events in this run. Press `d` after EOF to drain all accepted
annotations; without an explicit cap, the total drain budget is calculated from the
number still pending and is recorded in the session JSONL summary.

Available CLI arguments:

| Argument | Meaning |
|---|---|
| `source` | Required fixture alias (`clean`, `corrupted`, `rock-clean`, `rock-corrupted`, `harvard-clean`, `harvard-corrupted`) or readable local audio file supported by FFmpeg |
| `--no-ui` | Print JSON snapshots instead of opening the TUI |
| `--gemma-annotations` | Enable the local Gemma worker |
| `--gemma-drain-timeout` | Explicit total post-EOF drain cap/override; otherwise auto-calculated from pending work |
| `--log-dir` | Project-relative log directory, default `logs` |
| `--duration` | Stop after N seconds |

External files are decoded by the persistent FFmpeg process but never copied or
written beside their source. Audit records and Gemma payloads contain only a
privacy-safe source kind/fingerprint and stream ID, never a source path, URI, or
filename; JSONL logs remain under project-relative `logs/` by default.

At natural EOF, without `--gemma-drain-timeout`, Gemma's total graceful-drain
budget is computed as `pending accepted requests × per-request timeout + 3 s`.
The per-request timeout remains 10 s; the budget is therefore 43 s for four
pending requests and 73 s for seven. An explicit `--gemma-drain-timeout` is a
user cap/override of that total budget. `D` uses the same calculation and shows
the pending count/budget; `q` still cancels immediately. Session audit summaries
record requested/effective budget and final queue outcomes.

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
