# Audio glitch detection POC

Giorno 3: FFmpeg persistente decodifica PCM `f32le` 48 kHz stereo in un ring
bounded; il consumer paced esegue il detector DSP no-reference autorevole
`poc-d2-v2`. Il DSP è l'unica fonte di event ID, lifecycle, status, tipo, intervalli,
score ed evidence. Gemma è un annotatore locale opzionale: non può modificare o
sopprimere alcun evento DSP.

Richiede Python 3.12+, FFmpeg nel `PATH`, e le dipendenze del progetto:

```sh
.work/venv/bin/python -m pip install -e .
.work/venv/bin/python -m glitch_poc.cli clean --no-ui --duration 1
.work/venv/bin/python -m glitch_poc.cli corrupted --no-ui --duration 10.5
.work/venv/bin/python -m glitch_poc.cli corrupted --no-ui --gemma-annotations
```

La TUI è 2x2 e prevede almeno un terminale 100x30; `q` esce. Non usa né
richiede un dispositivo audio: il flusso avanza comunque al wall clock.
Per smoke/headless (utile in CI) usare:

```sh
glitch-poc clean --no-ui --duration 1
python3.12 -m unittest discover -s tests -v
```

Gli snapshot headless includono record lifecycle append-only (`OPENED`,
`UPDATED`, `CLOSED`, `CANCELLED`, `SUPERSEDED`) con `event_id` stabile,
revision, epoch, frame emesso, detector/evidence IDs ordinati e stato. Gli
alert `OPENED` sono emessi a hop di 10 ms, senza attendere merge/finalizzazione;
lo score è un rapporto/severità grezzo **non probabilistico**. Il profilo usa
finestre 5/20 ms (click e flat-top), 100 ms (dropout), 20/100 ms (block repeat),
500/1000 ms (loop uncertain), merge 50 ms e cooldown 100 ms. La soglia clipping è `abs(float PCM) >= 0.44`
con almeno l'8% dei campioni: è calibrata esclusivamente sulla fixture
`clip(6*x, ±0.45)`, non è una soglia PCM16 universale.

Verifica riproducibile (cache/artefatti sotto `.work/`):

```sh
PYTHONPYCACHEPREFIX="$PWD/.work/pycache" .work/venv/bin/python -m unittest discover -s tests -v
PYTHONPYCACHEPREFIX="$PWD/.work/pycache" .work/venv/bin/python -m tools.dsp_metrics > .work/dsp-metrics.json
```

Limiti POC: i detector sono euristiche calibrate solo sul corpus sintetico. Il
dropout resta `uncertain` senza recovery/post-contesto; non separa
universalmente silenzio semantico e guasto. Loop è sempre `uncertain` e non
entra nella recall senza ground truth positivo. Discontinuity, frame/sequence
gap e overflow cancellano lifecycle aperti, resettano baseline/history/cooldown,
incrementano l'epoch e producono record auditabili: nessun merge attraversa un
gap. Il report marca `insufficient_corpus=true`: una fixture da 10 s per classe
non consente di dichiarare un tasso operativo di falsi allarmi.

## Gemma grounded (opt-in)

Gemma è disabilitata per default. `--gemma-annotations` avvia un worker dedicato a
concorrenza **1**, con coda bounded (16 richieste), timeout calibrato di 10 s e nessun retry
POST; una coda piena scarta solo l'annotazione e incrementa la telemetria. L'endpoint
è esclusivamente HTTP loopback (`127.0.0.1`, `localhost`, `::1`) e il modello è
`gemma4:e4b`. Non vengono letti `.env`, chiavi o servizi cloud.
Un executor isolato usa il metodo multiprocessing `spawn` (mai `fork`): viene
avviato prima della TUI e del thread di annotazione, poi processa le richieste in
serie. Questo evita di creare processi dopo l'inizializzazione del terminale macOS
e resta terminabile durante lo shutdown.

Avvia prima Ollama locale (con `gemma4:e4b` disponibile), poi usa esattamente:

```sh
.work/venv/bin/python -m glitch_poc.cli corrupted --gemma-annotations
```

Nella TUI attendere la chiusura DSP della fixture (circa 10.5 s), quindi le quattro
annotazioni seriali richiedono indicativamente altri 25--30 s sul modello calibrato.
Il riquadro Gemma conserva la testata di stato e rende tutte le annotazioni recenti
in una cronologia scrollabile con wrap di spiegazioni, errori ed evidence. Usare la
rotella del mouse (o il focus nativo del pannello) oppure `[` e `]` per scorrere;
`Ctrl+S` alterna `GEMMA DETAIL`, la lista completa scrollabile con evidenze e
spiegazioni/errori, e `GEMMA SUMMARY`, la tabella degli eventi DSP `CLOSED`
correlati alle annotazioni Gemma, inclusi gli eventi disabled, pending o error. Il
Footer riporta gli stessi comandi. `q` esce subito e cancella il lavoro pendente con
outcome auditabile; `D` esegue invece drain bounded e poi esce.

Il worker riceve soltanto record DSP consolidati `CLOSED` con stato `detected` o
`uncertain`, deduplicati per `(event_id, revision, lifecycle)`: non annota `OPENED`,
`UPDATED`, `clean`, `CANCELLED` o `SUPERSEDED`. Nella fixture `corrupted`, dopo la
chiusura dei lifecycle (a 10.5 s tutti i candidati dovrebbero essere chiusi), click,
dropout, stutter e clipping sono candidati all'annotazione. L'annotazione è una
valutazione descrittiva di coerenza e non verifica né modifica autoritativamente il
verdetto DSP.

In headless, soltanto dopo EOF naturale il runtime drena le annotazioni già accettate
per un massimo di 43 s (`--gemma-drain-timeout 43`): una risposta, timeout o errore è
registrato per ciascun candidato. Alla scadenza il log contiene un errore esplicito
`shutdown_drain_timeout` per ogni richiesta residua. Uscita TUI con `q`, Ctrl-C,
durata interrotta e failure FFmpeg usano invece cancellazione immediata bounded; gli
outcome cancellati restano auditabili. La TUI non avvia il drain automaticamente a
EOF: lo esegue soltanto alla sua chiusura, così resta reattiva.
La calibrazione offline riproducibile è `python tools/calibrate_gemma_timeout.py`:
esegue localmente il DSP sulla fixture solo per ricavare i quattro eventi finali reali,
ma invia a Ollama esclusivamente i loro payload JSON whitelist compatti (mai audio o
PCM). Registra cold e warm per click, clipping, dropout e stutter in
`.work/gemma-timeout-calibration.json`. La proposta usa max con meno di 20 campioni
(altrimenti p95), margine 1.5, arrotondamento al secondo e clamp 5--60 s; il runtime è
aggiornato solo se tutte le quattro classi producono output grounded valido. Schema
prompt v2 limita a due evidence recenti e alle feature pertinenti per tipo; imposta
`temperature=0`, `num_predict=256`, JSON minificato ed explanation <=400 caratteri.
L'ultima prova v2 ha misurato 4.307--6.431 s su 8 risposte valide: timeout 10 s e
drain seriale 43 s.
L'input è JSON strutturato con evento canonico, profilo, feature/evidence ID e contesto
descrittivo limitato; non contiene PCM, audio, waveform, spettrogrammi, base64, path o
URI. Timeout, modello indisponibile e JSON invalido producono una `GemmaAnnotation`
separata con stato `error`, senza fermare DSP, logging o TUI.

Esempio inventato di risposta valida:

```json
{"event_id":"poc-d2-v2:e0:00001","annotation_status":"coherent","glitch_type_annotation":"loop","confidence":"medium","supporting_evidence_ids":["poc-d2-v2:e0:loop:48000:48480"],"explanation":"Le correlazioni fornite supportano il loop.","cannot_override_dsp":true}
```

La TUI conserva il layout 2x2 per terminali 100x30: il pannello Gemma visualizza
coda/worker, `DSP verdict` e `Gemma annotation` per tutti gli eventi finali. Se Ollama
o il modello non è disponibile, l'annotazione registra `error`; DSP, TUI e logging
continuano senza degradare il verdetto.

## Audit di sessione

Ogni avvio TUI o headless crea un JSONL append-only in `logs/`, ad esempio
`logs/session-20260906-123456.jsonl` (un suffisso evita collisioni). È possibile
scegliere una directory relativa e contenuta nel progetto: `--log-dir logs/demo`.
Il primo/ultimo record sono `session_started`/`session_ended`; ogni lifecycle DSP è
un record `dsp_event` e ogni risposta Gemma un record `gemma_annotation`, collegati
da `session_id`, `event_id` e `correlation_id`. Il log non contiene audio, PCM,
waveform, path audio, segreti o prompt integrali. Un errore del writer è mostrato in
Runtime Health e non blocca DSP o TUI.

Le fixture canoniche sono in `fixtures/audio/`; per rigenerarle:

```sh
python3 tools/generate_audio_fixtures.py
```
