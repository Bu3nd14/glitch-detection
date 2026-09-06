# Mockup UI — Demo A/B: Controllo Negativo Gemma vs Pipeline Grounded

Layout per la TUI di dimostrazione. Mostra la differenza tra un controllo
negativo offline, in cui Gemma ascolta audio grezzo, e la pipeline operativa
con detection deterministica e annotazione grounded.

```text
┌─ AUDIO SORGENTE ───────────────────────────────────────────┐
│ ▂▃▅█▂▁▁▁▃▅▇█▆▄▂▁▁▂▃▅ [waveform ASCII/plot]                  │
│ ground truth: glitch iniettato a t=2.250s (click sintetico) │
└─────────────────────────────────────────────────────────────┘

┌─ A) CONTROLLO NEGATIVO OFFLINE ┐ ┌─ B) PIPELINE GROUNDED ───────┐
│ Input: audio grezzo (WAV)      │ │ DSP feature vector (Layer 0b)│
│ Prompt: "ci sono glitch?"      │ │ window_id: w_2250            │
│                                 │ │ AR_error: 0.87 (soglia 0.6)  │
│ Output controllo:               │ │ ZCR: 0.42                    │
│ > "L'audio sembra pulito"      │ │ spectral_flatness: 0.91      │
│                                 │ │ peak: -0.3 dBFS              │
│ [SOLO TEST OFFLINE]             │ │                               │
│ [NON PIPELINE PRODUZIONE]       │ │ Verdetto DSP autoritativo:   │
│                                 │ │ > detected: true              │
│                                 │ │ > glitch_type: click          │
│                                 │ │ > timestamp: 2.250s           │
│                                 │ │ > evidence: [AR_error,        │
│                                 │ │   spectral_flatness]          │
│                                 │ │                               │
│                                 │ │ Gemma annotation grounded:    │
│                                 │ │ > coherent / insufficient     │
│                                 │ │ > supporting_features:        │
│                                 │ │   [AR_error, spectral_flatness]│
└─────────────────────────────────┘ └───────────────────────────────┘

┌─ SCOREBOARD ───────────────────────────────────────────────┐
│ Ground truth glitch:  t=2.250s  presente                   │
│ A) Audio grezzo offline: MISSED                             │
│ B) DSP autoritativo:  DETECTED  t=2.250s                   │
│ Gemma grounded:       annotazione coerente / insufficiente  │
│ latenza A: offline | DSP: 0.3s | LLM: 0.4s async            │
└─────────────────────────────────────────────────────────────┘
```

## Note sui contenuti dei riquadri

1. **Riquadro sorgente**: audio con glitch iniettati artificialmente a
   timestamp noti. Fornisce il ground truth oggettivo della demo.

2. **Riquadro A (controllo negativo)**: audio grezzo passato a Gemma in una
   prova offline separata. Mostra il falso negativo empirico; non è parte del
   percorso di produzione e non influisce su alert o metriche DSP.

3. **Riquadro B (grounded)**: mostra prima il verdetto autoritativo del DSP e
   le feature che lo supportano. Gemma riceve feature vector, evidence ID ed
   eventuale spettrogramma, quindi aggiunge solo un'annotazione strutturata:
   non crea né modifica l'evento.

4. **Scoreboard finale**: confronta ground truth, controllo negativo e
   detection DSP. Le metriche di detection sono attribuite esclusivamente al
   DSP; latenza e stato di Gemma sono metriche di arricchimento asincrono.
