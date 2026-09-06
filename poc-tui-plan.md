# Piano POC TUI per Audio Glitch Detection

## Obiettivo

Dimostrare rapidamente che il sistema può rilevare e spiegare corruzioni introdotte durante uno stream audio file-based riprodotto in tempo reale, senza rendere FFmpeg, la TUI o Ollama dipendenze bloccanti del percorso DSP.

La POC deve essere tecnicamente misurabile e abbastanza curata da rendere immediatamente comprensibili stream, fault, detection e analisi di Gemma.

## Durata

- Quattro giorni di sviluppo core.
- Un quinto giorno opzionale come buffer.
- Feature freeze alla fine del terzo giorno.
- Il giorno di buffer non introduce nuove funzionalità.

## Scope Incluso

- Input WAV, FLAC e MP3 da file.
- Riproduzione paced rispetto al wall clock.
- Stream PCM `source` e `observed` separati.
- Ring buffer e pre/post-roll degli eventi.
- Fault injection deterministica con seed.
- Click, clipping, dropout, block repeat e starvation simulata.
- Confronto reference-based tra source e observed.
- Detector DSP pragmatici no-reference.
- Event builder con merge, hysteresis e cooldown.
- Invio asincrono a `gemma4:e4b` tramite Ollama locale.
- Feature vector strutturato, classificazione DSP preliminare, confidence ed evidence ID forniti a Gemma; spettrogramma compatto opzionale.
- Output JSON validato e separato dal verdetto DSP.
- TUI a quattro quadranti, timeline e metriche runtime.
- Audit e replay tramite JSONL.

## Fuori Scope

- Microfono, loopback e acquisizione hardware.
- Threshold universali o già pronti per la produzione.
- Correzione automatica dei glitch.
- Supporto esaustivo degli artefatti codec.
- Installer, distribuzione e storage persistente completo.
- Inferenza cloud.

## Layout TUI

### Source vs Observed

- Waveform decimate e sovrapposte.
- Differenza reference-based.
- RMS, peak e posizione nel file.
- Buffer fill e clock drift.

### Runtime Health

- Callback in ritardo e starvation.
- Profondità delle code e blocchi scartati.
- Stato FFmpeg e Ollama.
- CPU, memoria e pressione termica.

### DSP Detection

- Stato `clean`, `uncertain` o `detected`.
- Tipo di glitch, score ed evidenze.
- Per gli eventi periodici, periodicità (lag di autocorrelazione) e delta RMS affiancati, per distinguere un block repeat a energia preservata da un loop che amplifica l'energia.
- Detector attivi e timestamp dell'evento.
- Differenza source/observed.

### Gemma Annotation

- Stato `idle`, `queued`, `analyzing`, `validated`, `invalid` o `timeout`.
- Feature vector, evidence ID e spettrogramma opzionale analizzati.
- Spiegazione, evidence ID e latenza.
- Indicatore evidente quando l'analisi è relativa a un evento precedente.

Il quadrante rende i campi dello schema di annotazione (tipo, confidence, evidence ID, spiegazione, latenza) in forma leggibile a schermo, non JSON grezzo. Il JSON completo con schema esteso (`event_id`, `model`, `model_digest`, `ollama_version`, `temperature`, `validation_status`) resta accessibile solo tramite replay/export (comandi `R`/`E`), non è mostrato riga per riga nel quadrante.

La fascia inferiore contiene timeline e comandi: `Space` pausa la UI, `I` inietta un fault, `R` apre il replay delle evidenze DSP, `E` esporta e `Q` termina.

## Stack

- Python 3.12 o successivo.
- Textual per TUI, layout e worker asincroni.
- FFmpeg 9.0.1 persistente per decode e resample verso PCM `f32le`.
- NumPy e SciPy per buffer, feature, confronto e detector.
- httpx per il client asincrono Ollama.
- Pydantic per eventi e output strutturati.
- JSONL per audit e replay.

## Architettura Runtime

```text
File sorgente
      |
      v
FFmpeg persistente -> PCM source -> player paced
                                      |
                                      v
                              Fault injector
                                      |
                                      v
                               PCM observed
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

La TUI legge snapshot decimati a 10-15 FPS. Non legge direttamente dai processi e non entra nel percorso audio. Ollama usa concorrenza uno, timeout e circuit breaker; la detection continua quando il modello non è disponibile.

## Piano Di Consegna

### Giorno 1: Vertical Slice

- Creare scaffold, configurazione e contratti evento.
- Collegare FFmpeg, PCM paced e ring buffer.
- Mostrare waveform e stato pipeline nella TUI 2x2.
- Eseguire uno spike end-to-end di Ollama con feature vector strutturato ed eventuale spettrogramma compatto.

Uscita: un file scorre in tempo reale ed è visibile nella TUI; il payload strutturato accettato da Ollama è verificato e documentato. L'eventuale audio grezzo è eseguito soltanto come controllo negativo separato.

### Giorno 2: Fault E Detection

- Implementare fault injector deterministico.
- Introdurre click, clipping, dropout e block repeat.
- Implementare confronto source/observed e detector iniziali.
- Aggregare finestre in eventi: partire da RMS della differenza su finestre di 5 ms con merge gap di 50 ms (valori di partenza da una validazione manuale preliminare su una coppia di file di test, non soglie finali) e verificare che non si riproduca la frammentazione osservata con un confronto campione per campione.
- Per block repeat, calcolare sia autocorrelazione (periodicità) sia delta RMS rispetto all'atteso: un block repeat a energia preservata e un loop che amplifica l'energia sono pattern distinti e non vanno confusi nello stesso detector.
- Validare i detector no-reference (senza il file `clean`) sullo stesso corpus, usando il confronto reference-based come oracolo: verificare che localizzino gli stessi eventi entro la tolleranza dichiarata, poiché in produzione il riferimento pulito non è disponibile. Punto di partenza: `reference_diff_detector.py` (RMS a finestra + merge gap, classificazione per RMS ratio/clipping/autocorrelazione).

Uscita: una demo genera fault noti e li mostra con timestamp ed evidenze; i detector no-reference sono validati contro l'oracolo reference-based sullo stesso corpus.

### Giorno 3: Gemma E Audit

- Serializzare feature vector, classificazione DSP preliminare, confidence ed evidence ID per gli eventi `uncertain`.
- Generare opzionalmente uno spettrogramma compatto a bassa risoluzione come evidenza complementare.
- Implementare coda bounded, timeout e stato Ollama.
- Validare output JSON ed evidence ID.
- Registrare sessione ed eventi in JSONL.
- Consentire il replay dell'ultimo evento e delle sue evidenze DSP nella TUI.

Uscita: Gemma riceve feature strutturate senza bloccare il DSP e ogni risposta è validata o scartata. L'annotazione non modifica mai l'evento DSP.

### Giorno 4: Polish E Benchmark

- Rifinire gerarchia visiva, meter, badge e timeline.
- Congelare un piccolo corpus dimostrativo ripetibile.
- Misurare latenze p50, p95 e p99.
- Misurare eventi attesi, mancati e falsi alert.
- Preparare README e comando demo unico.

Uscita: POC presentabile accompagnata da un report di fattibilità e dai limiti osservati.

## Criteri Go / No-Go

- La pipeline completa funziona per 30 minuti senza deadlock o crescita non limitata delle code.
- Spegnere Ollama non interrompe playback, DSP o logging.
- Tutti i fault del corpus demo sono localizzati entro la tolleranza dichiarata per ciascun detector.
- La latenza DSP p95 resta sotto 100 ms per gli eventi che non richiedono contesto futuro.
- La TUI mantiene almeno 10 FPS senza compromettere le deadline audio.
- Gemma riceve e analizza il feature vector strutturato ed eventuale spettrogramma; risposta e latenza sono registrate e valutabili.
- Lo stesso seed produce gli stessi fault e gli stessi eventi DSP.
- I detector no-reference (senza il file `clean`) individuano, entro la tolleranza dichiarata, gli stessi eventi trovati dal confronto reference-based sullo stesso corpus: la detection deve funzionare anche quando in produzione non esiste un riferimento pulito.

## Rischi Principali

- Payload strutturato Ollama: eseguire lo spike il primo giorno, verificare serializzazione e schema JSON, quindi congelare il contratto dati.
- Transienti legittimi: usare corpus controllato e confronto reference-based nella POC.
- Rendering costoso: inviare alla TUI soltanto snapshot decimati.
- Contesa termica: limitare Ollama a una richiesta concorrente ed eseguire un soak test.
- Scope creep: nessuna nuova feature dopo il terzo giorno.

## Workflow Degli Agenti OpenCode

1. `dsp-designer` congela feature, finestre, fault e criteri di accettazione.
2. `coder`, con Terra, implementa una vertical slice e scrive i test.
3. `tester`, con Luna, verifica indipendentemente ed emette `PASS`, `PASS CON RISCHI` o `FAIL`.

## Definition Of Done

Un solo comando avvia una demo file-based real-time, introduce fault deterministici, mostra source e observed, emette eventi DSP, invia feature strutturate a Gemma, registra JSONL e produce metriche sufficienti per decidere se procedere con il progetto.
