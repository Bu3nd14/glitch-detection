# UI Mockup - Demo A/B: Gemma Negative Control vs Grounded Pipeline

Layout for the demo TUI. It shows the difference between an offline
negative control, where Gemma listens to raw audio, and the operational pipeline
with deterministic detection and grounded annotation.

```text
┌─ SOURCE AUDIO ──────────────────────────────────────────────┐
│ ▂▃▅█▂▁▁▁▃▅▇█▆▄▂▁▁▂▃▅ [ASCII waveform/plot]                  │
│ ground truth: glitch injected at t=2.250s (synthetic click) │
└─────────────────────────────────────────────────────────────┘

┌─ A) OFFLINE NEGATIVE CONTROL ┐ ┌─ B) GROUNDED PIPELINE ──────┐
│ Input: raw audio (WAV)       │ │ DSP feature vector (Layer 0b)│
│ Prompt: "are there glitches?"│ │ window_id: w_2250           │
│                              │ │ AR_error: 0.87 (threshold 0.6)│
│ Control output:              │ │ ZCR: 0.42                    │
│ > "The audio sounds clean"   │ │ spectral_flatness: 0.91      │
│                              │ │ peak: -0.3 dBFS             │
│ [OFFLINE TEST ONLY]          │ │                               │
│ [NOT PRODUCTION PIPELINE]    │ │ Authoritative DSP verdict:   │
│                              │ │ > detected: true             │
│                              │ │ > glitch_type: click         │
│                              │ │ > timestamp: 2.250s          │
│                              │ │ > evidence: [AR_error,       │
│                              │ │   spectral_flatness]         │
│                              │ │                               │
│                              │ │ Grounded Gemma annotation:   │
│                              │ │ > coherent / insufficient    │
│                              │ │ > supporting_features:       │
│                              │ │   [AR_error, spectral_flatness]│
└──────────────────────────────┘ └──────────────────────────────┘

┌─ SCOREBOARD ───────────────────────────────────────────────┐
│ Ground truth glitch:  t=2.250s  present                    │
│ A) Raw audio offline: MISSED                               │
│ B) DSP authoritative: DETECTED  t=2.250s                   │
│ Grounded Gemma:       coherent / insufficient annotation   │
│ latency A: offline | DSP: 0.3s | LLM: 0.4s async           │
└─────────────────────────────────────────────────────────────┘
```

## Notes on the Panel Contents

1. **Source panel**: audio with glitches artificially injected at known
   timestamps. It provides the objective ground truth for the demo.

2. **Panel A (negative control)**: raw audio passed to Gemma in a separate
   offline test. It shows the empirical false negative; it is not part of the
   production path and does not affect DSP alerts or metrics.

3. **Panel B (grounded)**: shows the authoritative DSP verdict first and the
   features that support it. Gemma receives the feature vector, evidence ID, and
   optional spectrogram, then adds only a structured annotation:
   it does not create or modify the event.

4. **Final scoreboard**: compares ground truth, negative control, and DSP
   detection. Detection metrics are attributed exclusively to DSP; Gemma latency
   and state are async enrichment metrics.
