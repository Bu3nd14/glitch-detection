# Architecture Document
## Real-Time Audio Glitch Detection System with Deterministic Grounding and Constrained Local Analysis (Gemma e4b on Ollama)

**Version:** 1.2
**Date:** September 2026
**Status:** Proposed design, pre-implementation

**Changelog 1.2**: section 6.4 was added with the results of an initial empirical validation (reference-based comparison on a clean/corrupted file pair), confirming the need for windowed analysis with merge/hysteresis (not sample-by-sample comparison) and the feasibility of autocorrelation as a discriminator for block-repeat/loop. Sections 8.2 and 8.5 were revised: Gemma input is limited to the feature vector, raw audio is allowed only as a negative-control test, based on empirical evidence from 5 September 2026.

---

## 1. Executive Summary

This document describes the architecture of a system for real-time audio glitch detection (click, pop, dropout, clipping, stutter, codec artifacts), based on the **clear separation between deterministic detection and optional semantic analysis**. The FFmpeg/DSP pipeline produces operational events; Gemma e4b, running locally through Ollama, intervenes only asynchronously on a narrow subset of candidates for triage and explanation.

The guiding principle is: the model must never "search" for the glitch in the entire raw audio stream, nor must it ever create, remove, or modify a real-time alert. Its role is exclusively to explain or triage genuinely ambiguous cases that a classic Digital Signal Processing (DSP) pipeline has already isolated and characterized. Timeout, unavailability, or invalid Ollama output must therefore not interrupt detection.

This architectural choice is driven by three system constraints:

1. **Latency**: even a small model such as e4b cannot be invoked on every window of a continuous audio stream without introducing latency incompatible with real-time use.
2. **Compute cost**: inference, even on edge hardware, has a non-trivial cost per call when multiplied across the full volume of a continuous stream.
3. **Reliability**: a small model, if exposed to unfiltered input, tends to have a higher false-positive/false-negative rate than a large model; constraining it to work only on concrete, pre-characterized DSP evidence drastically reduces this risk.

### 1.1 Target local environment

As of this document, the target system is a MacBook Air with Apple M5 (10 core) and 24 GB of unified memory. Ollama 0.33.3 is installed and the `gemma4:e4b` model is already available locally (about 9.6 GB). The local manifest indicates Gemma 4 architecture with 8B parameters, Q4_K_M quantization, 131072 context length, and `completion`, `vision`, `audio`, `tools`, and `thinking` capabilities. These capabilities belong to the model in general, but the production pipeline does not use raw audio input: it uses only the structured feature vector and optionally the spectrogram, as defined in Section 8.2. FFmpeg 9.0.1 is installed via Homebrew and includes the `astats`, `silencedetect`, and `ebur128` filters. The environment makes local inference concurrent with DSP plausible, but does not replace a benchmark of latency, memory, and thermal impact under sustained load.

---

## 2. Goals and Non-Goals

### 2.1 Goals

- Detect in real time (or nearly so, with bounded latency) the following audio glitch types:
  - Click and pop (brief impulsive discontinuities)
  - Dropout (signal loss/anomalous silence)
  - Clipping (signal saturation)
  - Stutter/anomalous repetitions
  - Codec/compression artifacts (blocking, pre-echo, ringing)
- Minimize the computational load on the language model, reserving it only for ambiguous cases
- Provide structured, traceable, and auditable output: every classification must be explicitly tied to the DSP features that motivated it
- Ensure that the model does not add a positive annotation without sufficient evidence, explicitly declaring `insufficient_evidence` when appropriate
- Keep the entire DSP pipeline (Layer 0) fully deterministic and inspectable, without ML inference dependencies in this phase
- Keep audio, features, and inference on the local machine: Ollama must be reached exclusively through loopback and no cloud API is required

### 2.2 Non-goals (for this version)

- No automatic glitch correction/repair system is targeted (de-click, interpolation) - this document covers only the *detection and classification* phase, not *repair*
- No "agentic" architecture with self-correction loops or dynamic DSP threshold retuning is targeted in this iteration (see Section 12, future extensions)
- No large cloud model is assumed: the design is constrained to a small model such as Gemma e4b, intended for edge/local execution

---

## 3. Core Architectural Principle (analogy with textual RAG)

The architecture described here is a direct generalization of a "hardened" RAG (Retrieval-Augmented Generation) pattern, where the language model does not retrieve or decide data priority, but receives only material already filtered, ranked, and structured by a deterministic backend, with the explicit constraint of not asserting anything that is not supported by that material.

The table below makes the conceptual mapping between the two domains explicit:

| Concept in textual RAG | Equivalent in glitch detection |
|---|---|
| Chunking of source text | Segmentation of the audio stream into time windows (overlapping) |
| Keyword retrieval (Kiwix) | Extraction of coarse DSP features (ffmpeg: astats, silencedetect, ebur128) |
| Ranking by source priority | Ranking by weight/reliability of the different DSP detectors |
| Fine analysis after retrieval | Fine DSP analysis (scipy/librosa): predictive error, zero-crossing, spectral flatness |
| Only relevant chunks passed to the model | Only selected DSP events passed asynchronously to the model |
| "Cannot include without reference" constraint | "Do not annotate without valid evidence ID" constraint |
| Query understanding as a separate step from pure retrieval | Feature extraction as a separate step from semantic classification |

This mapping is not merely aesthetic: it implies the same risk categories apply in both domains. In particular, the "garbage in, garbage out" risk - if the retrieval/feature extraction layer is tuned poorly, the model never sees the relevant cases, or is flooded with false positives, regardless of how well its final judgment is constrained.

---

## 4. Layered Architecture Overview

```text
┌──────────────────────────────────────────────────────────────────┐
│ LAYER 0a - Persistent ingest (FFmpeg)                            │
│  Audio stream -> decode/resample -> normalized PCM + telemetry    │
└──────────────────────────────────────────────────────────────────┘
                              │ continuous PCM stream
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ LAYER 0b - Feature stream and detector bank (scipy / NumPy)      │
│  Multi-scale detector for click, clipping, dropout, stutter, and  │
│  codec artifact -> deterministic score and evidence               │
└──────────────────────────────────────────────────────────────────┘
                              │ DSP candidates
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ LAYER 1 - Event builder and policy engine                         │
│  Window merge, hysteresis, cooldown, calibration, and decision    │
│  detected / uncertain / clean                                     │
└──────────────────────────────────────────────────────────────────┘
                 │ immediate alert            │ bounded async copy
                 ▼                            ▼
┌──────────────────────────────┐  ┌─────────────────────────────────┐
│ LAYER 3 - Event sink         │  │ LAYER 2 - Local Gemma/Ollama   │
│ Deterministic output and audit│  │ Optional triage/explanation    │
└──────────────────────────────┘  └─────────────────────────────────┘
                 │                            │ validated annotation
                 └──────────────┬─────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│ EVENT STORE - Event, feature, detector version, and LLM annotation │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Layer 0a - Persistent Ingest and Telemetry (FFmpeg)

### 5.1 Role in the system

This layer acquires, decodes, and normalizes the stream into a stable PCM format. FFmpeg runs as a persistent per-stream process, avoiding per-window startup and repeated parsing of intermediate files. It is not the only detection gate: an overly restrictive pre-filter would produce irreversible false negatives, especially for very short clicks and stutter.

### 5.2 Technical components

- **Uniform decode and resampling**: FFmpeg normalizes input format, sample rate, and channels and sends PCM (`f32le` or `s16le`) through a pipe to the pipeline ring buffer. Sample index and monotonic clock are the primary time source.
- **`astats` filter**: computed over a sliding window, it provides RMS, peak, DC offset, and crest factor. These values are compared against configurable thresholds to identify clipping candidates (peak near 0 dBFS), DC offset anomalies, and abrupt crest-factor changes (an indirect indicator of impulsiveness, hence possible clicks).
- **`silencedetect` filter**: detects segments below a configurable volume threshold for a configurable minimum duration - a dropout candidate if silence is not expected in context (for example, it is not a natural pause in speech/music).
- **`ebur128` filter**: provides integrated loudness and true peak according to the EBU R128 standard, useful to identify clipping more rigorously than `astats` peak level alone.

### 5.3 Layer output

For each time window, the layer produces:
- A set of scalar features (RMS, peak, DC offset, crest factor, loudness, true peak)
- One or more coarse boolean flags (e.g. `peak_exceeds_threshold`, `silence_detected`, `dc_offset_anomaly`)

FFmpeg flags are telemetry and candidate signals, not a requirement to access Layer 0b. Low-cost incremental DSP features must remain active on the continuous PCM; only expensive analyses and the model are applied to candidates.

### 5.4 Known limits

ffmpeg, through these filters, is not designed to precisely discriminate a true impulsive click from a legitimate musical transient (for example, a percussive attack), nor to distinguish stutter from an intentional musical repetition. For this reason its output is explicitly treated as "coarse" and never as the final classification.

---

## 6. Layer 0b - Feature Stream and Detector Bank (NumPy / SciPy)

### 6.1 Role in the system

This layer reads PCM from a ring buffer and maintains incremental multi-scale features. Cheap detectors run continuously; more expensive analyses are activated only by candidates. The per-detector separation avoids forcing a few-sample click, dropout, stutter, and codec artifacts into the same window size.

### 6.2 Applied techniques

- **Predictive/Autoregressive (AR) error for click detection**: a predictive linear model is built on the context window (previous samples), the next expected sample is predicted, and the result is compared with the actual sample. A prediction error above a configurable threshold is a strong indicator of impulsive discontinuity (click/pop). This is the base technique used in professional audio restoration tools (for example, declick algorithms in iZotope RX or Adobe Audition Click/Pop Eliminator), here reimplemented without proprietary dependencies.
- **Zero-crossing rate (ZCR)**: useful for identifying abrupt changes in the tonal/noisy content of the signal, often correlated with codec artifacts or interruptions.
- **Spectral flatness and spectral centroid**: help distinguish a broadband impulsive noise (typical of a click) from legitimate tonal content, reducing false positives from musical transients.
- **Sliding-window RMS envelope analysis**: confirms dropout candidates identified by `silencedetect`, checking that the "hole" in the signal is not consistent with the envelope expected from the immediately preceding/following context.
- **Run-length and plateau for clipping**: distinguishes truly saturated samples from merely high true peak. Sample clipping and inter-sample peak are different evidence and must not be treated as synonyms.
- **Autocorrelation/fingerprint for stutter**: compares blocks over a longer context to detect almost identical repetitions; ZCR and spectral features alone are not sufficient.

### 6.3 Layer output

For each window, the layer maintains the necessary cheap features. When a detector crosses its activation threshold it produces an **enriched DSP candidate**, with available FFmpeg features, fine features, candidate type, raw score, and deterministic evidence IDs.

Windows without candidates do not generate events. Expensive analyses are stopped when the suspicion is not confirmed and no ordinary window reaches the model.

### 6.4 Preliminary empirical validation

Before implementation, a manual test was run on a pair of 10-second files (48 kHz, stereo, sample-accurate aligned): one clean file and the same file with 4 injected glitches at known timestamps. The reference-based comparison (sample-by-sample difference between the two files) produced two results relevant to the design:

- **Sample-level comparison is unusable as-is**: with a simple absolute-difference threshold, the 4 real glitches were fragmented into 658 separate "regions", because of residual sample-by-sample noise even inside the same event. This empirically confirms why Layer 1 (merge, hysteresis, cooldown) is not optional: without it, even a perfect reference-based comparison produces an unusable number of alerts.
- **Window aggregation with merge solves the problem**: by computing the RMS of the difference over 5 ms windows and merging above-threshold windows with a 50 ms merge tolerance, the 658 micro-regions were correctly reduced to 4 macro-events, exactly matching the 4 injected glitches. These values (5 ms/50 ms) are an empirically validated starting point for Layer 1 calibration, not a final threshold.

On the same 4 events, a comparison between RMS, peak, and autocorrelation enabled type discrimination consistent with the taxonomy in Section 2.1:

| Event | Duration | Dominant feature | Assigned type |
|---|---|---|---|
| 1 | ~5 ms | Local peak more than doubled vs. clean, no periodicity | Click |
| 2 | 300 ms | RMS dropped to ~3% of the clean value in the same window | Dropout |
| 3 | 480 ms | Autocorrelation ~= 1.00 at ~80 ms lag, RMS comparable to clean | Block repeat (energy preserved) |
| 4 | 600 ms | Autocorrelation ~= 0.97 at ~15 ms lag, RMS more than tripled vs. clean | High-energy periodic loop ("buzz") |

Events 3 and 4 are both periodic (high autocorrelation) but with opposite energy behavior: the first preserves the original signal energy (repetition of an existing block), the second significantly amplifies it (probable micro-loop generating a continuously intrusive artifact). This suggests **not treating "stutter" as a single category in the detector bank**, but always computing periodicity (autocorrelation) and energy delta (dirty RMS / expected RMS) together, to distinguish a silent block-repeat from a loop that produces a much more perceptually invasive artifact.

A final methodological note: this test was reference-based (it requires the clean file). In production, on the live stream, Layer 0b does not have access to a clean reference and must rely only on no-reference detectors (AR error, ZCR, spectral flatness, run-length for clipping, autocorrelation). The practical value of paired clean/corrupted corpora is therefore not as a runtime detection enablement, but as a **calibration oracle**: use the reference-based comparison on the test corpus to determine the real timestamps and types, and verify that the no-reference detectors, applied only to the `corrupted` file, identify the same events within the declared tolerance. This verification has not yet been done and remains a necessary step before the no-reference detectors can be considered calibrated.

Reference implementation of the results above: `reference_diff_detector.py`.

---

## 7. Layer 1 - Event Builder, Policy, and Async Queue

### 7.1 Event construction and decision

Confirmed overlapping windows are merged into events through hysteresis, merge gap, and cooldown, avoiding duplicate alerts for the same glitch. Each event receives a priority computed as a function of:
- The detector type that generated the suspicion (for example, an AR-confirmed click may have a different priority from a dropout confirmed by the RMS envelope)
- The numerical DSP confidence computed in Layer 0b
- Any configurable weights assigned in advance to different glitch types, based on how critical they are for the specific use case (for example, in a broadcast context, clipping may take priority over stutter)

This is the direct equivalent of "ranking by source priority" in the original textual RAG system: an explicit and deterministic weighting logic, never delegated to the model.

The operational decision is three-state (`detected`, `uncertain`, `clean`) and is made here, deterministically. The confidence shown to users must be calibrated per detector on a validation set; a raw score must not be presented as a probability.

### 7.2 Queue to Ollama

Only a copy of `uncertain` or explicitly selected events is inserted into a bounded queue toward Layer 2. The queue applies priority, deadline, and deduplication. In case of saturation, items destined for the model can be dropped with explicit logging; the DSP event and its alert are never dropped. Batching is allowed only if the benchmark shows it does not violate the annotation deadline.

---

## 8. Layer 2 - Optional Local Analysis (Gemma e4b on Ollama)

### 8.1 Role in the system

This is the only point in the pipeline where a language model intervenes. Its job is not to "search" for a glitch or emit the operational verdict, but to add a triage annotation or explanation to events already isolated by DSP. The authoritative result remains that of Layer 1.

### 8.2 Model input

For each candidate event, the model receives as its only primary input the **structured feature vector** produced by Layer 0b. The vector includes:
- Predictive/AR error
- Zero-crossing rate
- Spectral flatness and spectral centroid
- Available coarse FFmpeg features (`astats`, `silencedetect`, `ebur128`)
- Preliminary DSP classification with score and confidence
- Evidence ID, minimal time context, and metadata required to interpret the features correctly

Optionally, a compact low-resolution spectrogram may be provided as complementary visual evidence. The spectrogram does not replace the feature vector and does not become an authoritative source.

### 8.2.1 Note on raw audio input

The raw audio clip is not part of Gemma's operational input and must not be sent as a component of the production pipeline. It is allowed only in a separate validation test, explicitly labeled as a **negative control**, whose purpose is to measure and document the model's inability to directly detect glitches from audio.

This choice comes from the empirical test run on 5 September 2026: `gemma4:e4b`, through Ollama, did not detect audible synthetic glitches with known timestamps when given either the full corrupted WAV or short clips isolated around the events. Even a direct comparison between reference and observed clips produced unreliable classifications and positions. The result excludes raw audio as an operational input regardless of prompt or clip length.

### 8.3 Grounding constraint

The model operates under an explicit constraint: **it cannot add a positive annotation if the provided features do not sufficiently support the decision**. In that case it returns `insufficient_evidence`. The constraint is structural: closed JSON schema, allowed-value enums, `evidence_id` selectable only from the provided list, and post-hoc validation. Low temperature and a restrictive prompt reduce variability but are not an anti-hallucination guarantee.

### 8.4 Expected output format

Every model response must be structured (for example JSON), including at least:
- `event_id`: DSP event identifier
- `glitch_type`: classified type, or `insufficient_evidence`
- `confidence`: numerical or categorical value
- `supporting_features`: explicit list of the provided vector features that justify the classification (direct equivalent of `[source: chunk_id]` in textual RAG)

The model response is stored as a separate `llm_annotation`. It cannot overwrite `detected`, DSP type, timestamp, severity, score, or measured features.

### 8.5 Why a small model is adequate in this layer

Because the task is limited to triage, disambiguation, and explanation on already characterized events, Gemma e4b remains a plausible interpreter of the structured feature vector. Its value depends on the ability to reason about the relationships between DSP features, preliminary classification, confidence, and evidence ID, not on the ability to "listen": the empirical test of 5 September 2026 excluded raw audio as a reliable source for glitch detection.

The model's contribution must be demonstrated with an A/B test against an equivalent deterministic policy. If it does not improve the disambiguation of `uncertain` cases or the readability of the report, it can be removed without changing detection.

### 8.6 Local deployment with Ollama

- Ollama runs on the same Mac and is reached through HTTP API on loopback; the service must not be exposed on the LAN.
- The initially configured model is `gemma4:e4b`. The model name, digest, Ollama version, template, and generation parameters are logged with every annotation. The local manifest sets `temperature=1`, so the application must override it with a low, validated value to reduce variability.
- The pipeline uses an asynchronous client with timeout, cancellation, limited retry, and circuit breaker. No audio thread or callback waits for Ollama.
- The model is kept warm (`keep_alive`) during a test session to avoid cold starts; parallelism and context size must be limited to avoid stealing memory and bandwidth from DSP.
- The process must monitor p50/p95/p99 latency, queue depth, timeouts, invalid outputs, memory, and thermal pressure. On a fanless MacBook Air, prolonged tests are required to detect throttling.
- Pipeline startup checks Ollama health, but a negative result disables only LLM annotations and does not stop detection.

---

## 9. Layer 3 - Structured Output and Logging

Every DSP event is logged immediately with:
- Wall-clock timestamp, monotonic clock, initial/final sample index, and event ID
- Type, state, and severity determined by the policy engine
- Raw score, calibrated probability when available, and evidence ID
- DSP features, detector version, threshold profile, and optional reference to pre/post-roll audio
- Ollama queue state and indication of any missed deadlines

If available, Gemma's annotation is added later with model/digest, Ollama version, generation parameters, validated output, and cited evidence ID. It remains a separate record and does not modify the DSP event.

This layer makes the system **auditable**: it is possible to reconstruct the deterministic chain that produced each event and, separately, verify what material was supplied to the model and what annotation it returned.

---

## 10. Latency and Throughput Considerations

- Layer 0a uses a single persistent FFmpeg process per stream. Creating a process per window or writing temporary files is not compatible with the real-time path.
- Layer 0b continuously runs only cheap incremental features; expensive analyses operate on candidates. The budget must be verified on continuous PCM, not inferred a priori.
- Layer 2 is the most latency-variable component. Because it is asynchronous and non-authoritative, it does not belong to the DSP alert deadline, but has a separate deadline for report enrichment.
- The acquisition callback must be limited to timestamping and copying into the preallocated ring buffer. FFmpeg, DSP, Ollama, and logging are isolated through bounded queues and explicit backpressure.
- Exact sizing (window size, overlap, thresholds, batch size) must be calibrated empirically on the specific use case and available hardware, and is not defined a priori in this document.

---

## 11. Failure Modes and Risks

| Risk | Layer involved | Mitigation |
|---|---|---|
| FFmpeg telemetry is too permissive and triggers too many expensive analyses | Layer 0a/0b | Tuning on a reference dataset and separate budget for expensive triggers |
| DSP detector thresholds are too strict and miss real glitches | Layer 0b | Periodic validation against labeled datasets and synthetic glitches with known parameters |
| Fine analysis (Layer 0b) does not distinguish legitimate musical transients from real clicks | Layer 0b | Combine multiple features (AR error + spectral flatness), not a single indicator |
| Model annotates without sufficient evidence, bypassing the prompt prose constraint | Layer 2 | Closed schema and evidence IDs allowed only if present in the event; invalid output discarded |
| Inference saturation during bursts of candidate windows (for example, very degraded audio) | Layer 1/2 | Priority batching, possible controlled drop of lower-priority windows with explicit drop logging |
| Ollama unavailable, timeout, or model evicted from memory | Layer 2 | Circuit breaker and automatic fallback to the already emitted DSP event; no blocking of the real-time path |
| CPU/GPU/memory contention between Gemma and DSP on the same Mac | Layer 0b/2 | Bounded queue, limited Ollama concurrency, load benchmark, and latency/thermal-pressure monitoring |
| Repeated FFmpeg startup or fragile log parsing | Layer 0a | Persistent process, supervision, explicit PCM format, and telemetry parser tested for the installed version |
| A single FFmpeg pre-filter misses short glitches | Layer 0a/0b | Cheap DSP features always on and end-to-end recall evaluation per glitch type |

---

## 12. Future Extensions (out of scope for this version)

- **Repair layer**: once the glitch is classified, a later module (not covered here) could apply correction techniques (interpolation, resynthesis) specific to the glitch type.
- **Agentic loop with threshold auto-tuning**: a natural extension, inspired by "agentic graph" patterns, would introduce a "critic" node that periodically evaluates the false-positive/false-negative rate of Layer 0a/0b and proposes dynamic threshold adjustments instead of manually configured fixed thresholds. However, this would introduce a level of adaptability and less predictability than the current design, and should be evaluated only if manual tuning proves insufficient.
- **Larger models for particularly ambiguous cases**: a possible second escalation level, in which cases that even Gemma e4b classifies as "insufficient evidence" are forwarded (offline, non-real-time) to a larger model for a deeper analysis.

---

## 13. Summary

The system described applies to the audio domain the principle of **minimizing the model's decision autonomy and maximizing deterministic, inspectable work**. FFmpeg is the persistent ingest, decode, and normalization process; the DSP detector bank operates at multiple scales and produces authoritative events; Gemma e4b, running locally on Ollama, adds only validated asynchronous annotations. In this way privacy and local operation are preserved and a model error can never turn into an operational false alert.
