# Audio glitch detection POC

Day 3: persistent FFmpeg decodes PCM `f32le` 48 kHz stereo into a bounded ring;
the paced consumer runs the authoritative no-reference DSP detector
`poc-d2-v2`. DSP is the only source of event ID, lifecycle, status, type, intervals,
score, and evidence. Gemma is an optional local annotator: it cannot modify or
suppress any DSP event.

Requires Python 3.12+, FFmpeg in `PATH`, and the project dependencies:

```sh
.work/venv/bin/python -m pip install -e .
.work/venv/bin/python -m glitch_poc.cli clean --no-ui --duration 1
.work/venv/bin/python -m glitch_poc.cli corrupted --no-ui --duration 10.5
.work/venv/bin/python -m glitch_poc.cli corrupted --no-ui --gemma-annotations
```

The TUI is 2x2 and requires at least a 100x30 terminal; `q` exits. It does not use
or require an audio device: the stream still advances on wall clock.
For smoke/headless runs (useful in CI) use:

```sh
glitch-poc clean --no-ui --duration 1
python3.12 -m unittest discover -s tests -v
```

Headless snapshots include append-only lifecycle records (`OPENED`,
`UPDATED`, `CLOSED`, `CANCELLED`, `SUPERSEDED`) with stable `event_id`,
revision, emitted epoch, ordered detector/evidence IDs, and state. `OPENED`
alerts are emitted on a 10 ms hop, without waiting for merge/finalization;
the score is a raw ratio/severity **not probabilistic**. The profile uses
5/20 ms windows (click and flat-top), 100 ms (dropout), 20/100 ms (block repeat),
500/1000 ms (uncertain loop), 50 ms merge, and 100 ms cooldown. The clipping threshold is `abs(float PCM) >= 0.44`
with at least 8% of samples: it is calibrated exclusively on the fixture
`clip(6*x, +-0.45)`, and is not a universal PCM16 threshold.

Reproducible verification (cache/artifacts under `.work/`):

```sh
PYTHONPYCACHEPREFIX="$PWD/.work/pycache" .work/venv/bin/python -m unittest discover -s tests -v
PYTHONPYCACHEPREFIX="$PWD/.work/pycache" .work/venv/bin/python -m tools.dsp_metrics > .work/dsp-metrics.json
```

POC limits: the detectors are heuristics calibrated only on the synthetic corpus.
Dropout remains `uncertain` without recovery/post-context; it does not universally
separate semantic silence from failure. Loop is always `uncertain` and does not
enter recall without positive ground truth. Discontinuity, frame/sequence
gap, and overflow cancel open lifecycles, reset baseline/history/cooldown,
increment the epoch, and produce auditable records: no merge crosses a
gap. The report marks `insufficient_corpus=true`: a 10 s fixture per class
is not enough to claim an operational false-alarm rate.

## Gemma grounded (opt-in)

Gemma is disabled by default. `--gemma-annotations` starts a dedicated worker with
concurrency **1**, a bounded queue (16 requests), a calibrated 10 s timeout, and no
POST retries; a full queue drops only the annotation and increments telemetry. The
endpoint is loopback HTTP only (`127.0.0.1`, `localhost`, `::1`) and the model is
`gemma4:e4b`. No `.env`, keys, or cloud services are read.
An isolated executor uses the multiprocessing `spawn` method (never `fork`): it is
started before the TUI and the annotation thread, then processes requests in
series. This avoids creating processes after macOS terminal initialization and
remains terminable during shutdown.

Start local Ollama first (with `gemma4:e4b` available), then use exactly:

```sh
.work/venv/bin/python -m glitch_poc.cli corrupted --gemma-annotations
```

In the TUI, wait for the fixture DSP to finish (about 10.5 s), then the four
serial annotations take about another 25-30 s on the calibrated model.
The Gemma panel keeps the status header and renders all recent annotations
in a scrollable history with wrapped explanations, errors, and evidence. Use the
mouse wheel (or the panel's native focus) or `[` and `]` to scroll;
`Ctrl+S` toggles `GEMMA DETAIL`, the full scrollable list with evidence and
explanations/errors, and `GEMMA SUMMARY`, the table of DSP `CLOSED`
events linked to Gemma annotations, including disabled, pending, or error events.
The footer shows the same commands. `q` exits immediately and cancels pending work
with an auditable outcome; `D` instead performs bounded drain and then exits.

The worker receives only consolidated DSP `CLOSED` records with `detected` or
`uncertain` state, deduplicated by `(event_id, revision, lifecycle)`: it does not
annotate `OPENED`, `UPDATED`, `clean`, `CANCELLED`, or `SUPERSEDED`. In the
`corrupted` fixture, after lifecycle closure (by 10.5 s all candidates should be
closed), click, dropout, stutter, and clipping are annotation candidates.
Annotation is a descriptive consistency assessment and does not verify or
authoritatively change the DSP verdict.

In headless mode, only after natural EOF does the runtime drain already accepted
annotations for up to 43 s (`--gemma-drain-timeout 43`): one response, timeout, or
error is logged for each candidate. On timeout, the log contains an explicit
`shutdown_drain_timeout` error for each remaining request. TUI exit with `q`, Ctrl-C,
interrupted duration, and FFmpeg failure instead use immediate bounded cancellation;
cancelled outcomes remain auditable. The TUI does not start drain automatically at
EOF: it does so only on shutdown, so it stays responsive.
The reproducible offline calibration is `python tools/calibrate_gemma_timeout.py`:
it runs the DSP locally on the fixture only to recover the four real final events,
but sends Ollama only their compact whitelisted JSON payloads (never audio or
PCM). It records cold and warm timings for click, clipping, dropout, and stutter in
`.work/gemma-timeout-calibration.json`. The proposal uses max with fewer than 20 samples
(otherwise p95), margin 1.5, second-level rounding, and a 5-60 s clamp; the runtime is
updated only if all four classes produce valid grounded output.
Prompt schema v2 limits input to two recent evidence items and the relevant features
per type; it sets `temperature=0`, `num_predict=256`, minified JSON, and explanation <=400 chars.
The latest v2 run measured 4.307-6.431 s across 8 valid responses: 10 s timeout and
43 s serial drain.
The input is structured JSON with canonical event, profile, feature/evidence ID, and
limited descriptive context; it contains no PCM, audio, waveform, spectrogram,
base64, path, or URI. Timeout, unavailable model, and invalid JSON produce a separate
`GemmaAnnotation` with `error` state, without stopping DSP, logging, or the TUI.

Example of a valid fabricated response:

```json
{"event_id":"poc-d2-v2:e0:00001","annotation_status":"coherent","glitch_type_annotation":"loop","confidence":"medium","supporting_evidence_ids":["poc-d2-v2:e0:loop:48000:48480"],"explanation":"The provided correlations support the loop.","cannot_override_dsp":true}
```

The TUI keeps the 2x2 layout for 100x30 terminals: the Gemma panel shows
queue/worker, `DSP verdict`, and `Gemma annotation` for all final events. If Ollama
or the model is unavailable, the annotation records `error`; DSP, the TUI, and logging
continue without degrading the verdict.

## Session Audit

Every TUI or headless start creates an append-only JSONL file in `logs/`, for example
`logs/session-20260906-123456.jsonl` (a suffix avoids collisions). You can
choose a relative directory contained in the project: `--log-dir logs/demo`.
The first/last records are `session_started`/`session_ended`; every DSP lifecycle is
a `dsp_event` record and every Gemma response is a `gemma_annotation` record, linked
by `session_id`, `event_id`, and `correlation_id`. The log contains no audio, PCM,
waveform, audio paths, secrets, or full prompts. A writer error is shown in
Runtime Health and does not block DSP or the TUI.

The canonical fixtures are in `fixtures/audio/`; to regenerate them:

```sh
python3 tools/generate_audio_fixtures.py
```
