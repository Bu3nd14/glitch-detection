# Documento di Architettura
## Sistema di Real-Time Audio Glitch Detection con Grounding Deterministico e Analisi Locale Vincolata (Gemma e4b su Ollama)

**Versione:** 1.2
**Data:** Settembre 2026
**Stato:** Design proposto, pre-implementazione

**Changelog 1.2**: aggiunta la sezione 6.4 con i risultati di una prima validazione empirica (confronto reference-based su una coppia di file clean/corrupted), che conferma la necessità di un'analisi a finestra con merge/hysteresis (non del confronto campione per campione) e la fattibilità dell'autocorrelazione come discriminante per block-repeat/loop. Sezioni 8.2 e 8.5 riviste: input a Gemma limitato al feature vector, audio grezzo ammesso solo come test di controllo negativo, in base a evidenza empirica del 5 settembre 2026.

---

## 1. Executive Summary

Questo documento descrive l'architettura di un sistema per il rilevamento in tempo reale di glitch audio (click, pop, dropout, clipping, stutter, artefatti di codec), basato sulla **separazione netta tra detection deterministica e analisi semantica opzionale**. La pipeline FFmpeg/DSP produce gli eventi operativi; Gemma e4b, eseguito localmente tramite Ollama, interviene solo in modo asincrono su un sottoinsieme ristretto di candidati per triage e spiegazione.

Il principio guida è: il modello non deve mai "cercare" il glitch nell'intero stream audio grezzo, né deve mai creare, eliminare o modificare un alert real-time. Il suo ruolo è esclusivamente spiegare o sottoporre a triage casi genuinamente ambigui che una pipeline di Digital Signal Processing (DSP) classico ha già isolato e caratterizzato. Timeout, indisponibilità o output invalido di Ollama non devono quindi interrompere la detection.

Questa scelta architetturale è motivata da tre vincoli di sistema:

1. **Latenza**: un modello, anche piccolo come e4b, non può essere invocato su ogni finestra di un flusso audio continuo senza introdurre latenza incompatibile con un uso real-time.
2. **Costo computazionale**: l'inferenza, anche su hardware edge, ha un costo per chiamata non trascurabile se moltiplicato per l'intero volume di uno stream continuo.
3. **Affidabilità**: un modello di piccole dimensioni, se esposto a input non filtrato, tende a un tasso di falsi positivi/negativi più alto rispetto a un modello grande; vincolarlo a lavorare solo su evidenza DSP concreta e già pre-caratterizzata riduce drasticamente questo rischio.

### 1.1 Ambiente locale di destinazione

Alla data di questo documento, il sistema target è un MacBook Air con Apple M5 (10 core) e 24 GB di memoria unificata. Ollama 0.33.3 è installato e il modello `gemma4:e4b` è già disponibile localmente (circa 9,6 GB). Il manifest locale indica architettura Gemma 4 da 8B parametri, quantizzazione Q4_K_M, context length 131072 e capacità `completion`, `vision`, `audio`, `tools` e `thinking`. Queste capacità appartengono al modello in generale, ma la pipeline di produzione non usa l'input audio grezzo: usa esclusivamente il feature vector strutturato ed eventualmente lo spettrogramma, come definito nella Sezione 8.2. FFmpeg 9.0.1 è installato tramite Homebrew e include i filtri `astats`, `silencedetect` ed `ebur128`. L'ambiente rende plausibile l'inferenza locale concorrente al DSP, ma non sostituisce un benchmark di latenza, memoria e impatto termico sotto carico prolungato.

---

## 2. Obiettivi e Non-Obiettivi

### 2.1 Obiettivi

- Rilevare in tempo reale (o quasi, con latenza contenuta) i seguenti tipi di glitch audio:
  - Click e pop (discontinuità impulsive di breve durata)
  - Dropout (perdita di segnale/silenzi anomali)
  - Clipping (saturazione del segnale)
  - Stutter/ripetizioni anomale
  - Artefatti di codec/compressione (blocking, pre-echo, ringing)
- Minimizzare il carico computazionale sul modello linguistico, riservandolo solo a casi ambigui
- Fornire un output strutturato, tracciabile e auditabile: ogni classificazione deve essere ricondotta esplicitamente alle feature DSP che l'hanno motivata
- Garantire che il modello non aggiunga un'annotazione positiva senza evidenza sufficiente, dichiarando esplicitamente `insufficient_evidence` quando è il caso
- Mantenere l'intero pipeline DSP (Layer 0) completamente deterministico e ispezionabile, senza dipendenze da inferenza ML in questa fase
- Mantenere audio, feature e inferenza sul computer locale: Ollama deve essere raggiunto esclusivamente tramite loopback e non è richiesta alcuna API cloud

### 2.2 Non-Obiettivi (per questa versione)

- Non si punta a un sistema di correzione/riparazione automatica del glitch (de-click, interpolazione) — questo documento copre solo la fase di *detection e classificazione*, non il *repair*
- Non si punta, in questa iterazione, a un'architettura "agentic" con loop di auto-correzione o re-tuning dinamico delle soglie DSP (vedi Sezione 12, estensioni future)
- Non si assume un modello grande in cloud: il design è vincolato a un modello piccolo tipo Gemma e4b, pensato per esecuzione edge/locale

---

## 3. Principio Architetturale di Fondo (analogia con RAG testuale)

L'architettura qui descritta è una generalizzazione diretta di un pattern RAG (Retrieval-Augmented Generation) "hardened", in cui il modello linguistico non recupera né decide la priorità dei dati, ma riceve solo materiale già filtrato, ranked e strutturato da un backend deterministico, con il vincolo esplicito di non affermare nulla che non sia supportato da quel materiale.

La tabella seguente esplicita la corrispondenza concettuale tra i due domini:

| Concetto nel RAG testuale | Equivalente nel glitch detection |
|---|---|
| Chunking del testo sorgente | Segmentazione dello stream audio in finestre temporali (overlapping) |
| Retrieval per keyword (Kiwix) | Estrazione di feature DSP grossolane (ffmpeg: astats, silencedetect, ebur128) |
| Ranking per priorità delle fonti | Ranking per peso/affidabilità dei diversi detector DSP |
| Analisi fine post-retrieval | Analisi DSP fine (scipy/librosa): predictive error, zero-crossing, spectral flatness |
| Solo chunk rilevanti passati al modello | Solo eventi DSP selezionati passati asincronicamente al modello |
| Vincolo "cannot include without reference" | Vincolo "non annotare senza evidence ID validi" |
| Query understanding come step separato dal retrieval puro | Feature extraction come step separato dalla classificazione semantica |

Questa corrispondenza non è solo estetica: implica che le stesse categorie di rischio si applicano in entrambi i domini. In particolare, il rischio di "garbage in, garbage out" — se il layer di retrieval/estrazione feature è tarato male, il modello non vede mai i casi rilevanti, oppure viene sommerso di falsi positivi, indipendentemente da quanto è ben vincolato il suo giudizio finale.

---

## 4. Panoramica dell'Architettura a Livelli

```
┌──────────────────────────────────────────────────────────────────┐
│ LAYER 0a — Ingest persistente (FFmpeg)                            │
│  Stream audio → decode/resample → PCM normalizzato + telemetria    │
└──────────────────────────────────────────────────────────────────┘
                              │ stream PCM continuo
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ LAYER 0b — Feature stream e detector bank (scipy / NumPy)         │
│  Detector multi-scala per click, clipping, dropout, stutter e      │
│  codec artifact → score ed evidenze deterministiche                │
└──────────────────────────────────────────────────────────────────┘
                              │ candidati DSP
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ LAYER 1 — Event builder e policy engine                            │
│  Merge finestre, hysteresis, cooldown, calibrazione e decisione     │
│  detected / uncertain / clean                                      │
└──────────────────────────────────────────────────────────────────┘
                 │ alert immediato             │ copia asincrona bounded
                 ▼                             ▼
┌──────────────────────────────┐  ┌─────────────────────────────────┐
│ LAYER 3 — Event sink         │  │ LAYER 2 — Gemma locale/Ollama  │
│ Output e audit deterministici│  │ Triage/spiegazione opzionale   │
└──────────────────────────────┘  └─────────────────────────────────┘
                 │                             │ annotazione validata
                 └──────────────┬──────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│ EVENT STORE — Evento, feature, versione detector e annotazione LLM │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Layer 0a — Ingest Persistente e Telemetria (FFmpeg)

### 5.1 Ruolo nel sistema

Questo layer acquisisce, decodifica e normalizza lo stream in un formato PCM stabile. FFmpeg viene eseguito come processo persistente per stream, evitando startup per finestra e parsing ripetuto di file intermedi. Non è l'unico gate della detection: un pre-filtro troppo restrittivo produrrebbe falsi negativi irreversibili, soprattutto per click molto brevi e stutter.

### 5.2 Componenti tecnici

- **Decode e resampling uniforme**: FFmpeg normalizza formato, sample rate e canali in ingresso e invia PCM (`f32le` o `s16le`) tramite pipe al ring buffer della pipeline. Sample index e clock monotono sono la sorgente temporale primaria.
- **Filtro `astats`**: calcolato per finestra scorrevole, fornisce RMS, peak, DC offset, crest factor. Questi valori vengono confrontati contro soglie configurabili per identificare candidati a clipping (peak vicino a 0dBFS), anomalie di offset DC, variazioni brusche di crest factor (indicatore indiretto di impulsività, quindi possibile click).
- **Filtro `silencedetect`**: rileva segmenti sotto una soglia di volume configurabile per una durata minima configurabile — candidato a dropout se il silenzio non è atteso nel contesto (es. non è una pausa naturale del parlato/musica).
- **Filtro `ebur128`**: fornisce loudness integrata e true peak secondo lo standard EBU R128, utile per identificare clipping più rigorosamente rispetto al solo peak level di `astats`.

### 5.3 Output del layer

Per ogni finestra temporale, il layer produce:
- Un set di feature scalari (RMS, peak, DC offset, crest factor, loudness, true peak)
- Uno o più flag booleani grossolani (es. `peak_exceeds_threshold`, `silence_detected`, `dc_offset_anomaly`)

I flag FFmpeg sono telemetria e segnali candidati, non un requisito necessario per accedere al Layer 0b. Le feature DSP incrementali economiche devono restare attive sul PCM continuo; solo le analisi costose e il modello vengono applicati ai candidati.

### 5.4 Limiti noti

ffmpeg, tramite questi filtri, non è progettato per discriminare con precisione tra un vero click impulsivo e un transiente musicale legittimo (es. un attacco percussivo), né per distinguere uno stutter da una ripetizione musicale intenzionale. Per questo motivo il suo output è trattato esplicitamente come "grossolano" e mai come classificazione finale.

---

## 6. Layer 0b — Feature Stream e Detector Bank (NumPy / SciPy)

### 6.1 Ruolo nel sistema

Questo layer legge il PCM da un ring buffer e mantiene feature incrementali a più scale temporali. I detector economici operano continuamente; analisi più costose vengono attivate solo dai candidati. La separazione per detector evita di forzare click di pochi campioni, dropout, stutter e artefatti di codec nella stessa dimensione di finestra.

### 6.2 Tecniche applicate

- **Predictive/Autoregressive (AR) error per click detection**: si costruisce un modello lineare predittivo sulla finestra di contesto (campioni precedenti), si predice il campione successivo atteso, e si confronta con il campione reale. Un errore di predizione che supera una soglia configurabile è un forte indicatore di discontinuità impulsiva (click/pop). Questa è la tecnica di base impiegata in strumenti professionali di restauro audio (es. algoritmi di declick in iZotope RX o Adobe Audition Click/Pop Eliminator), qui reimplementata senza dipendenze da software proprietario.
- **Zero-crossing rate (ZCR)**: utile per identificare cambiamenti bruschi nel contenuto tonale/rumoroso del segnale, spesso correlati ad artefatti di codec o interruzioni.
- **Spectral flatness e spectral centroid**: aiutano a distinguere un rumore impulsivo a banda larga (tipico di un click) da un contenuto tonale legittimo, riducendo i falsi positivi provenienti da transienti musicali.
- **Analisi dell'inviluppo RMS a finestra scorrevole**: per confermare i candidati a dropout identificati da `silencedetect`, verificando che il "buco" nel segnale non sia coerente con l'inviluppo atteso dal contesto immediatamente precedente/successivo.
- **Run-length e plateau per clipping**: distingue campioni realmente saturati da un semplice true peak elevato. Sample clipping e inter-sample peak sono evidenze diverse e non vanno trattate come sinonimi.
- **Autocorrelazione/fingerprint per stutter**: confronta blocchi su un contesto più lungo per rilevare ripetizioni quasi identiche; ZCR e feature spettrali da sole non sono sufficienti.

### 6.3 Output del layer

Per ogni finestra, il layer mantiene le feature economiche necessarie. Quando un detector supera la soglia di attivazione produce un **candidato DSP arricchito**, con feature FFmpeg disponibili, feature fini, tipo candidato, score grezzo ed evidence ID deterministici.

Le finestre senza candidati non generano eventi. Le analisi costose vengono fermate quando il sospetto non è confermato e nessuna finestra ordinaria raggiunge il modello.

### 6.4 Validazione empirica preliminare

Prima dell'implementazione, è stato condotto un test manuale su una coppia di file di 10 secondi (48kHz, stereo, sample-accurate allineati): un file pulito e uno stesso file con 4 glitch iniettati a timestamp noti. Il confronto reference-based (differenza campione per campione tra i due file) ha prodotto due risultati rilevanti per il design:

- **Il confronto a livello di singolo campione è inutilizzabile as-is**: con una soglia semplice sulla differenza assoluta, i 4 glitch reali sono stati frammentati in 658 "regioni" separate, per via del rumore residuo campione-per-campione anche dentro lo stesso evento. Questo conferma empiricamente perché il Layer 1 (merge, hysteresis, cooldown) non è opzionale: senza di esso, anche un confronto reference-based perfetto produce un numero di alert inutilizzabile.
- **Un'aggregazione a finestra con merge risolve il problema**: calcolando l'RMS della differenza su finestre di 5 ms e unendo le finestre sopra soglia con una tolleranza di merge di 50 ms, le 658 micro-regioni si sono ridotte correttamente a 4 macro-eventi, corrispondenti esattamente ai 4 glitch iniettati. Questi valori (5 ms/50 ms) sono un punto di partenza empiricamente validato per la calibrazione del Layer 1, non una soglia finale.

Sugli stessi 4 eventi, un confronto tra RMS, picco e autocorrelazione ha permesso una discriminazione per tipo coerente con la tassonomia della Sezione 2.1:

| Evento | Durata | Feature dominante | Tipo assegnato |
|---|---|---|---|
| 1 | ~5 ms | Picco locale più che raddoppiato rispetto al clean, nessuna periodicità | Click |
| 2 | 300 ms | RMS crollato a ~3% del valore clean nella stessa finestra | Dropout |
| 3 | 480 ms | Autocorrelazione ≈1.00 a lag ~80 ms, RMS comparabile al clean | Block repeat (energia preservata) |
| 4 | 600 ms | Autocorrelazione ≈0.97 a lag ~15 ms, RMS più che triplicato rispetto al clean | Loop periodico ad alta energia ("buzz") |

Gli eventi 3 e 4 sono entrambi periodici (autocorrelazione alta) ma con comportamento energetico opposto: il primo preserva l'energia del segnale originale (ripetizione di un blocco esistente), il secondo la amplifica sensibilmente (probabile micro-loop che genera un artefatto continuo). Questo suggerisce di **non trattare "stutter" come una categoria unica nel detector bank**, ma di calcolare sempre periodicità (autocorrelazione) e delta energetico (RMS dirty/RMS atteso) insieme, per distinguere un block-repeat "silenzioso" da un loop che produce un artefatto percettivamente molto più invasivo.

Un'ultima nota metodologica: questo test è stato reference-based (richiede il file pulito). In produzione, sullo stream live, il Layer 0b non ha accesso a un riferimento pulito e deve affidarsi ai soli detector no-reference (AR error, ZCR, spectral flatness, run-length per clipping, autocorrelazione). Il valore pratico di corpora clean/corrupted appaiati non è quindi enable per la detection a runtime, ma come **oracolo di calibrazione**: usare il confronto reference-based sul corpus di test per determinare timestamp e tipo "vero", e verificare che i detector no-reference, applicati al solo file `corrupted`, individuino gli stessi eventi entro la tolleranza dichiarata. Questa verifica non è ancora stata fatta e resta un passo necessario prima di considerare i detector no-reference calibrati.

Implementazione di riferimento dei risultati riportati sopra: `reference_diff_detector.py`.

---

## 7. Layer 1 — Event Builder, Policy e Coda Asincrona

### 7.1 Costruzione e decisione dell'evento

Le finestre sovrapposte confermate vengono unite in eventi mediante hysteresis, merge gap e cooldown, evitando alert duplicati per lo stesso glitch. Ogni evento riceve una priorità calcolata come funzione di:
- Il tipo di detector che ha generato il sospetto (es. un click AR-confermato può avere priorità diversa da un dropout confermato da inviluppo RMS)
- La confidence DSP numerica calcolata al Layer 0b
- Eventuali pesi configurabili assegnati a priori ai diversi tipi di glitch, in base a quanto sono critici per il caso d'uso specifico (es. in un contesto broadcast, il clipping può essere prioritario rispetto a uno stutter)

Questo è l'equivalente diretto del "ranking per priorità delle fonti" nel sistema RAG testuale di origine: una logica di pesatura esplicita e deterministica, mai delegata al modello.

La decisione operativa è a tre stati (`detected`, `uncertain`, `clean`) ed è presa qui, deterministicamente. La confidence destinata agli utenti deve essere calibrata per detector su un validation set; uno score grezzo non deve essere presentato come probabilità.

### 7.2 Coda verso Ollama

Solo una copia degli eventi `uncertain` o esplicitamente selezionati viene inserita in una coda bounded verso il Layer 2. La coda applica priorità, deadline e deduplicazione. In caso di saturazione, gli elementi destinati al modello possono essere scartati con logging esplicito; l'evento DSP e il relativo alert non vengono mai scartati. Il batching è ammesso soltanto se il benchmark dimostra che non viola la deadline dell'annotazione.

---

## 8. Layer 2 — Analisi Locale Opzionale (Gemma e4b su Ollama)

### 8.1 Ruolo nel sistema

Questo è l'unico punto della pipeline in cui interviene un modello linguistico. Il suo compito non è "cercare" un glitch né emettere il verdetto operativo, ma aggiungere un'annotazione di triage o una spiegazione a eventi che il DSP ha già isolato. Il risultato autoritativo rimane quello del Layer 1.

### 8.2 Input al modello

Il modello riceve, per ogni evento candidato, come unico input primario il **feature vector strutturato** prodotto dal Layer 0b. Il vettore comprende:
- Errore predittivo/AR
- Zero-crossing rate
- Spectral flatness e spectral centroid
- Feature FFmpeg grossolane disponibili (`astats`, `silencedetect`, `ebur128`)
- Classificazione DSP preliminare con score e confidence
- Evidence ID, contesto temporale minimo e metadati necessari a interpretare correttamente le feature

Opzionalmente può essere fornito uno spettrogramma compatto a bassa risoluzione come evidenza visiva complementare. Lo spettrogramma non sostituisce il feature vector e non diventa una sorgente autoritativa.

### 8.2.1 Nota sull'input audio grezzo

Il clip audio grezzo non fa parte dell'input operativo di Gemma e non deve essere inviato come componente della pipeline di produzione. È ammesso esclusivamente in un test di validazione separato, esplicitamente etichettato come **controllo negativo**, il cui scopo è misurare e documentare l'incapacità del modello di rilevare direttamente i glitch dall'audio.

Questa scelta deriva dal test empirico eseguito il 5 settembre 2026: `gemma4:e4b`, tramite Ollama, non ha rilevato glitch sintetici udibili con timestamp noto né ricevendo il WAV corrotto completo, né ricevendo clip brevi isolati attorno agli eventi. Anche il confronto diretto tra clip reference e observed ha prodotto classificazioni e posizioni inaffidabili. Il risultato esclude l'audio grezzo come input operativo indipendentemente dal prompt o dalla durata del clip.

### 8.3 Vincolo di grounding

Il modello opera sotto un vincolo esplicito: **non può aggiungere un'annotazione positiva se le feature fornite non supportano sufficientemente la decisione**. In tal caso restituisce `insufficient_evidence`. Il vincolo è strutturale: schema JSON chiuso, enum dei valori ammessi, `evidence_id` selezionabili soltanto dall'elenco fornito e validazione post-hoc. Temperatura bassa e prompt restrittivo riducono la variabilità ma non costituiscono una garanzia anti-allucinazione.

### 8.4 Formato di output atteso

Ogni risposta del modello deve essere strutturata (es. JSON), includendo almeno:
- `event_id`: identificatore dell'evento DSP
- `glitch_type`: tipo classificato, oppure `insufficient_evidence`
- `confidence`: valore numerico o categorico
- `supporting_features`: elenco esplicito delle feature del vettore fornito che motivano la classificazione (equivalente diretto del `[source: chunk_id]` nel sistema RAG testuale)

La risposta del modello viene memorizzata come `llm_annotation` separata. Non può sovrascrivere `detected`, tipo DSP, timestamp, severità, score o feature misurate.

### 8.5 Perché un modello piccolo è adeguato in questo layer

Poiché il compito è ristretto a triage, disambiguazione e spiegazione su eventi già caratterizzati, Gemma e4b rimane plausibile come interprete del feature vector strutturato. Il suo valore dipende dalla capacità di ragionare sulle relazioni tra feature DSP, classificazione preliminare, confidence ed evidence ID, non dalla capacità di "ascoltare": il test empirico del 5 settembre 2026 ha escluso l'audio grezzo come sorgente affidabile per la detection dei glitch.

Il contributo del modello deve essere dimostrato con un test A/B rispetto a una policy deterministica equivalente. Se non migliora la disambiguazione dei casi `uncertain` o la leggibilità del report, può essere rimosso senza modificare la detection.

### 8.6 Deployment locale con Ollama

- Ollama viene eseguito sullo stesso Mac e raggiunto tramite API HTTP su loopback; il servizio non deve essere esposto sulla LAN.
- Il modello configurato inizialmente è `gemma4:e4b`. Nome, digest, versione Ollama, template e parametri di generazione vengono registrati con ogni annotazione. Il manifest locale imposta `temperature=1`, quindi l'applicazione deve sovrascriverla con un valore basso e validato per ridurre la variabilità.
- La pipeline usa un client asincrono con timeout, cancellazione, retry limitato e circuit breaker. Nessun thread o callback audio attende Ollama.
- Il modello viene mantenuto caldo (`keep_alive`) durante una sessione di test per evitare cold start; parallelismo e context size vanno limitati per non sottrarre memoria e bandwidth al DSP.
- Il processo deve monitorare latenza p50/p95/p99, profondità della coda, timeout, output invalidi, memoria e pressione termica. Su MacBook Air, privo di ventola, sono necessari test prolungati per rilevare throttling.
- L'avvio della pipeline verifica la salute di Ollama, ma un esito negativo disabilita soltanto le annotazioni LLM e non la detection.

---

## 9. Layer 3 — Output Strutturato e Logging

Ogni evento DSP viene loggato immediatamente con:
- Timestamp wall-clock, clock monotono, sample index iniziale/finale ed event ID
- Tipo, stato e severità determinati dal policy engine
- Score grezzo, probabilità calibrata quando disponibile ed evidence ID
- Feature DSP, versione dei detector, profilo di soglie e riferimento opzionale al pre/post-roll audio
- Stato della coda Ollama e indicazione di eventuali deadline perse

Se disponibile, l'annotazione di Gemma viene aggiunta successivamente con modello/digest, versione Ollama, parametri di generazione, output validato ed evidence ID citati. Rimane un record separato e non modifica l'evento DSP.

Questo layer rende il sistema **auditabile**: è possibile ricostruire la catena deterministica che ha prodotto ogni evento e, separatamente, verificare quale materiale sia stato fornito al modello e quale annotazione abbia restituito.

---

## 10. Considerazioni su Latenza e Throughput

- Il Layer 0a usa un unico processo FFmpeg persistente per stream. Creare un processo per finestra o scrivere file temporanei non è compatibile con il percorso real-time.
- Il Layer 0b esegue continuamente soltanto feature incrementali economiche; le analisi costose operano sui candidati. Il budget deve essere verificato sul PCM continuo, non dedotto a priori.
- Il Layer 2 è il componente a latenza più variabile. Essendo asincrono e non autoritativo, non rientra nella deadline dell'alert DSP ma ha una deadline separata per l'arricchimento del report.
- Il callback di acquisizione deve limitarsi a timestamp e copia nel ring buffer preallocato. FFmpeg, DSP, Ollama e logging sono isolati mediante code bounded e backpressure esplicita.
- Il dimensionamento esatto (dimensione delle finestre, overlap, soglie, dimensione dei batch) va calibrato empiricamente sul caso d'uso specifico e sull'hardware disponibile, e non è definito a priori in questo documento.

---

## 11. Modalità di Fallimento e Rischi

| Rischio | Layer coinvolto | Mitigazione |
|---|---|---|
| Telemetria FFmpeg troppo permissiva genera troppe analisi costose | Layer 0a/0b | Tuning su dataset di riferimento e budget separato per gli attivatori costosi |
| Soglie dei detector DSP troppo restrittive perdono glitch reali | Layer 0b | Validazione periodica contro dataset etichettati e glitch sintetici con parametri noti |
| Analisi fine (Layer 0b) non discrimina bene transienti musicali legittimi da click reali | Layer 0b | Combinazione di più feature (AR error + spectral flatness), non un singolo indicatore |
| Modello annota senza evidenza sufficiente, aggirando il vincolo di prosa nel prompt | Layer 2 | Schema chiuso ed evidence ID ammessi soltanto se presenti nell'evento; output invalido scartato |
| Saturazione dell'inferenza in caso di burst di finestre candidate (es. audio molto degradato) | Layer 1/2 | Batching con priorità, eventuale drop controllato delle finestre a priorità più bassa con logging esplicito del drop |
| Ollama non disponibile, timeout o modello scaricato dalla memoria | Layer 2 | Circuit breaker e fallback automatico all'evento DSP già emesso; nessun blocco del percorso real-time |
| Contesa CPU/GPU/memoria tra Gemma e DSP sullo stesso Mac | Layer 0b/2 | Coda bounded, concorrenza Ollama limitata, benchmark sotto carico e monitoraggio di latenza/pressione termica |
| Avvio ripetuto di FFmpeg o parsing fragile dei log | Layer 0a | Processo persistente, supervisione, formato PCM esplicito e parser di telemetria testato per la versione installata |
| Un unico pre-filtro FFmpeg perde glitch brevi | Layer 0a/0b | Feature DSP economiche sempre attive e valutazione del recall end-to-end per tipo di glitch |

---

## 12. Estensioni Future (fuori scope per questa versione)

- **Layer di repair**: una volta classificato il glitch, un modulo successivo (non trattato qui) potrebbe applicare tecniche di correzione (interpolazione, resynthesis) specifiche per tipo di glitch.
- **Loop agentic con auto-tuning delle soglie**: un'estensione naturale, ispirata ai pattern "agentic graph", introdurrebbe un nodo "critic" che valuta periodicamente il tasso di falsi positivi/negativi del Layer 0a/0b e propone aggiustamenti dinamici delle soglie, invece di soglie fisse configurate manualmente. Questo introdurrebbe però un livello di adattività e minor prevedibilità rispetto al design attuale, e andrebbe valutato solo se il tuning manuale si rivelasse insufficiente.
- **Modelli di dimensioni maggiori per casi particolarmente ambigui**: un possibile secondo livello di escalation, in cui i casi che anche Gemma e4b classifica come "insufficient evidence" vengono inoltrati (in modalità non real-time, offline) a un modello più grande per un'analisi più approfondita.

---

## 13. Sintesi

Il sistema descritto applica a un dominio audio il principio di **minimizzare l'autonomia decisionale del modello e massimizzare il lavoro deterministico e ispezionabile**. FFmpeg è il processo persistente di ingest, decode e normalizzazione; il detector bank DSP opera a più scale e produce gli eventi autoritativi; Gemma e4b, eseguito localmente su Ollama, aggiunge soltanto annotazioni asincrone validate. In questo modo privacy e funzionamento locale sono preservati e un errore del modello non può trasformarsi in un falso alert operativo.
