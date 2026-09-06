# TUI POC Plan for Audio Glitch Detection

## Goal

Quickly demonstrate that the system can detect and explain corruptions introduced during a file-based audio stream played in real time, without making FFmpeg, the TUI, or Ollama blocking dependencies in the DSP path.

The POC must be technically measurable and polished enough to make streams, faults, detection, and Gemma analysis immediately understandable.

## Duration

- Four core development days.
- An optional fifth day as buffer.
- Feature freeze at the end of day three.
- The buffer day introduces no new features.

## Included Scope

- WAV, FLAC, and MP3 file input.
- Wall-clock-paced playback.
- Separate `source` and `observed` PCM streams.
- Ring buffer and event pre/post-roll.
- Deterministic fault injection with seed.
- Click, clipping, dropout, block repeat, and simulated starvation.
- Reference-based comparison between source and observed.
- Pragmatic no-reference DSP detectors.
- Event builder with merge, hysteresis, and cooldown.
- Asynchronous send to local `gemma4:e4b` through Ollama.
- Structured feature vector, preliminary DSP classification, confidence, and evidence ID provided to Gemma; optional compact spectrogram.
- Validated JSON output separate from the DSP verdict.
- Four-quadrant TUI, timeline, and runtime metrics.
- Audit and replay through JSONL.

## Out of Scope

- Microphone, loopback, and hardware acquisition.
- Universal or production-ready thresholds.
- Automatic glitch correction.
- Exhaustive codec artifact support.
- Installer, distribution, and full persistent storage.
- Cloud inference.

## TUI Layout

### Source vs Observed

- Decimated and overlapped waveform.
- Reference-based difference.
- RMS, peak, and file position.
- Buffer fill and clock drift.

### Runtime Health

- Late callbacks and starvation.
- Queue depth and dropped blocks.
- FFmpeg and Ollama status.
- CPU, memory, and thermal pressure.

### DSP Detection

- `clean`, `uncertain`, or `detected` state.
- Glitch type, score, and evidence.
- For periodic events, periodicity (autocorrelation lag) and RMS delta shown side by side, to distinguish an energy-preserving block repeat from a loop that amplifies energy.
- Active detectors and event timestamp.
- Source/observed difference.

### Gemma Annotation

- `idle`, `queued`, `analyzing`, `validated`, `invalid`, or `timeout` state.
- Feature vector, evidence ID, and optional spectrogram analyzed.
- Explanation, evidence ID, and latency.
- Clear indicator when the analysis refers to a previous event.

The quadrant renders the annotation schema fields (type, confidence, evidence ID, explanation, latency) in a human-readable form on screen, not raw JSON. The full JSON with extended schema (`event_id`, `model`, `model_digest`, `ollama_version`, `temperature`, `validation_status`) remains accessible only through replay/export (`R`/`E` commands), not line-by-line in the quadrant.

The bottom bar contains the timeline and commands: `Space` pauses the UI, `I` injects a fault, `R` opens the DSP evidence replay, `E` exports, and `Q` quits.

## Stack

- Python 3.12 or later.
- Textual for TUI, layout, and async workers.
- Persistent FFmpeg 9.0.1 for decode and resample to PCM `f32le`.
- NumPy and SciPy for buffers, features, comparison, and detectors.
- httpx for the Ollama async client.
- Pydantic for structured events and output.
- JSONL for audit and replay.

## Runtime Architecture

```text
Source file
      |
      v
Persistent FFmpeg -> source PCM -> paced player
                                     |
                                     v
                               Fault injector
                                     |
                                     v
                               Observed PCM
                                /     |     \
                               v      v      v
                          DSP worker  TUI   evidence store
                               |               |
                               v               v
                         Event builder -> bounded Ollama queue
                               |               |
                               v               v
                          Event sink      Gemma worker
                               |               |
                               +-------+-------+
                                       v
                                  JSONL audit
```

The TUI reads decimated snapshots at 10-15 FPS. It does not read directly from the processes and does not enter the audio path. Ollama uses concurrency one, timeout, and a circuit breaker; detection continues when the model is unavailable.

## Delivery Plan

### Day 1: Vertical Slice

- Create scaffold, configuration, and event contracts.
- Wire FFmpeg, paced PCM, and ring buffer.
- Show waveform and pipeline state in the 2x2 TUI.
- Run an end-to-end Ollama spike with structured feature vector and optional compact spectrogram.

Output: a file streams in real time and is visible in the TUI; the structured payload accepted by Ollama is verified and documented. Raw audio, if used at all, is only executed as a separate negative control.

### Day 2: Fault And Detection

- Implement deterministic fault injection.
- Introduce click, clipping, dropout, and block repeat.
- Implement source/observed comparison and initial detectors.
- Aggregate windows into events: start from the RMS of the difference on 5 ms windows with a 50 ms merge gap (starting values from a preliminary manual validation on a test file pair, not final thresholds) and verify that the fragmentation seen with sample-by-sample comparison is not reproduced.
- For block repeat, compute both autocorrelation (periodicity) and RMS delta against the expected value: an energy-preserving block repeat and a loop that amplifies energy are distinct patterns and must not be confused in the same detector.
- Validate the no-reference detectors (without the `clean` file) on the same corpus, using the reference-based comparison as the oracle: verify that they localize the same events within the declared tolerance, since in production the clean reference is not available. Starting point: `reference_diff_detector.py` (windowed RMS + merge gap, classification by RMS ratio/clipping/autocorrelation).

Output: a demo generates known faults and shows them with timestamps and evidence; the no-reference detectors are validated against the reference-based oracle on the same corpus.

### Day 3: Gemma And Audit

- Serialize feature vector, preliminary DSP classification, confidence, and evidence ID for `uncertain` events.
- Optionally generate a compact low-resolution spectrogram as complementary evidence.
- Implement bounded queue, timeout, and Ollama state.
- Validate JSON output and evidence ID.
- Record session and events in JSONL.
- Allow replay of the last event and its DSP evidence in the TUI.

Output: Gemma receives structured features without blocking DSP and every response is validated or discarded. The annotation never modifies the DSP event.

### Day 4: Polish And Benchmark

- Refine visual hierarchy, meters, badges, and timeline.
- Freeze a small repeatable demo corpus.
- Measure p50, p95, and p99 latencies.
- Measure expected, missed, and false alerts.
- Prepare README and a single demo command.

Output: a presentable POC accompanied by a feasibility report and observed limits.

## Go / No-Go Criteria

- The full pipeline runs for 30 minutes without deadlock or unbounded queue growth.
- Turning off Ollama does not interrupt playback, DSP, or logging.
- All faults in the demo corpus are localized within the declared tolerance for each detector.
- DSP p95 latency stays under 100 ms for events that do not require future context.
- The TUI maintains at least 10 FPS without compromising audio deadlines.
- Gemma receives and analyzes the structured feature vector and optional spectrogram; response and latency are logged and reviewable.
- The same seed produces the same faults and the same DSP events.
- The no-reference detectors (without the `clean` file) identify, within the declared tolerance, the same events found by the reference-based comparison on the same corpus: detection must work even when no clean reference exists in production.

## Main Risks

- Ollama structured payload: run the spike on day one, verify serialization and JSON schema, then freeze the data contract.
- Legitimate transients: use a controlled corpus and reference-based comparison in the POC.
- Expensive rendering: send only decimated snapshots to the TUI.
- Thermal contention: limit Ollama to one concurrent request and run a soak test.
- Scope creep: no new features after day three.

## OpenCode Agent Workflow

1. `dsp-designer` freezes features, windows, faults, and acceptance criteria.
2. `coder`, with Terra, implements a vertical slice and writes tests.
3. `tester`, with Luna, verifies independently and emits `PASS`, `PASS WITH RISKS`, or `FAIL`.

## Definition Of Done

A single command starts a file-based real-time demo, introduces deterministic faults, shows source and observed, emits DSP events, sends structured features to Gemma, records JSONL, and produces enough metrics to decide whether to proceed with the project.
