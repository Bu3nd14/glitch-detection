"""
Script di riferimento: confronto reference-based tra un file audio pulito e
uno con glitch, usato per la validazione empirica descritta in
architettura.md (Sezione 6.4).

Uso previsto: oracolo di calibrazione per i detector no-reference del
Layer 0b, NON componente della pipeline di produzione (in produzione non
esiste un file "clean" di riferimento sullo stream live).

Dipendenze: numpy (nessuna dipendenza da scipy/librosa in questa versione
minima; l'architettura prevede di sostituire/affiancare queste euristiche
con AR error, ZCR e spectral flatness veri).
"""

import wave

import numpy as np


def load_wav(path):
    """Carica un WAV PCM16 e ritorna (sample_rate, array Nx canali int16)."""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        n = wf.getnframes()
        raw = wf.readframes(n)
        data = np.frombuffer(raw, dtype=np.int16).reshape(-1, ch)
    return sr, data


def find_macro_events(clean, dirty, sr, window_ms=5, merge_gap_ms=50, rms_threshold=300):
    """
    Confronta clean e dirty (stesso numero di campioni, sample-accurate) e
    ritorna una lista di macro-eventi (start_sample, end_sample), evitando
    la frammentazione di un confronto campione-per-campione grezzo.

    window_ms: dimensione della finestra di analisi RMS (default 5ms,
        valore di partenza validato empiricamente, non definitivo).
    merge_gap_ms: finestre sopra soglia separate da meno di questo gap
        vengono unite nello stesso evento (default 50ms).
    rms_threshold: soglia sull'RMS della differenza per finestra.
    """
    c = clean.astype(np.float64).mean(axis=1)
    d = dirty.astype(np.float64).mean(axis=1)
    diff = np.abs(c - d)

    win = max(1, int(window_ms / 1000 * sr))
    n_win = len(diff) // win

    diff_rms = np.array([
        np.sqrt(np.mean(diff[i * win:(i + 1) * win] ** 2)) for i in range(n_win)
    ])

    mask = diff_rms > rms_threshold
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []

    gap_tol = max(1, int(merge_gap_ms / window_ms))
    events = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i - prev > gap_tol:
            events.append((start * win, (prev + 1) * win))
            start = i
        prev = i
    events.append((start * win, (prev + 1) * win))
    return events


def classify_event(clean, dirty, sr, start, end):
    """
    Classificazione euristica di un macro-evento già individuato.
    Ritorna un dict con tipo, feature di supporto e valori grezzi —
    pensato per essere sostituito/affiancato dai detector veri del
    Layer 0b (AR error, run-length clipping, ecc.), non come
    classificatore definitivo.
    """
    seg_clean = clean[start:end].astype(np.int32)
    seg_dirty = dirty[start:end].astype(np.int32)

    c_mono = seg_clean.mean(axis=1).astype(np.float64)
    d_mono = seg_dirty.mean(axis=1).astype(np.float64)

    clean_rms = np.sqrt(np.mean(c_mono ** 2))
    dirty_rms = np.sqrt(np.mean(d_mono ** 2))
    clean_peak = np.max(np.abs(c_mono))
    dirty_peak = np.max(np.abs(d_mono))

    # Clipping: percentuale di campioni realmente saturi (non solo peak alto)
    clip_pct = float(np.mean(np.abs(seg_dirty) >= 32000) * 100)

    # Autocorrelazione per periodicità (block repeat / loop)
    d_centered = d_mono - d_mono.mean()
    best_lag, best_corr = 0, 0.0
    max_lag = min(4000, len(d_centered) // 2)
    for lag in range(50, max_lag, 10):
        a, b = d_centered[:-lag], d_centered[lag:]
        if a.std() < 1e-6 or b.std() < 1e-6:
            continue
        corr = float(np.corrcoef(a, b)[0, 1])
        if corr > best_corr:
            best_corr, best_lag = corr, lag

    dur_ms = (end - start) / sr * 1000

    if clip_pct > 0.5:
        glitch_type = "clipping"
    elif dirty_rms < clean_rms * 0.2:
        glitch_type = "dropout"
    elif best_corr > 0.9:
        # Periodico: distinguere block-repeat (energia preservata) da loop
        # ad alta energia ("buzz") tramite il rapporto RMS.
        if dirty_rms > clean_rms * 1.5:
            glitch_type = "loop_high_energy"
        else:
            glitch_type = "block_repeat"
    elif dur_ms < 10 and dirty_peak > clean_peak * 1.5:
        glitch_type = "click"
    else:
        glitch_type = "uncertain"

    return {
        "start_sample": start,
        "end_sample": end,
        "t_start_s": start / sr,
        "t_end_s": end / sr,
        "duration_ms": dur_ms,
        "glitch_type": glitch_type,
        "clean_rms": clean_rms,
        "dirty_rms": dirty_rms,
        "clean_peak": clean_peak,
        "dirty_peak": dirty_peak,
        "clip_pct": clip_pct,
        "autocorr_best_lag_samples": best_lag,
        "autocorr_best_corr": round(best_corr, 3),
    }


def main(clean_path, dirty_path):
    sr_c, clean = load_wav(clean_path)
    sr_d, dirty = load_wav(dirty_path)
    assert sr_c == sr_d, "sample rate diversi tra i due file"
    assert clean.shape == dirty.shape, "i due file non sono sample-accurate allineati"

    events = find_macro_events(clean, dirty, sr_c)
    print(f"Trovati {len(events)} macro-eventi\n")

    for start, end in events:
        result = classify_event(clean, dirty, sr_c, start, end)
        print(
            f"t={result['t_start_s']:.4f}-{result['t_end_s']:.4f}s "
            f"dur={result['duration_ms']:.2f}ms -> {result['glitch_type']}\n"
            f"  clean_rms={result['clean_rms']:.0f} dirty_rms={result['dirty_rms']:.0f} "
            f"clean_peak={result['clean_peak']:.0f} dirty_peak={result['dirty_peak']:.0f}\n"
            f"  clip_pct={result['clip_pct']:.3f}% "
            f"autocorr_lag={result['autocorr_best_lag_samples']}samples "
            f"corr={result['autocorr_best_corr']}\n"
        )


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Uso: python reference_diff_detector.py <clean.wav> <dirty.wav>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
