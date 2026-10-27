# Tesi: metodi di explainability per modelli di previsione finanziaria. Stato di avanzamento

**Domanda di ricerca.** I modelli di previsione dei titoli azionari hanno un segnale debole (IC ≈ 0.0x), e questa instabilità si trasmette ai portafogli top-k. Come guidano le feature in input le previsioni e le decisioni di allocazione? L'instabilità dipende dai **dati** o dall'**architettura del modello**?

---

## Lavoro completato (milestone M1 e M2 del piano)

### 1. Ambiente e letteratura (Step 0)
- Pacchetto Python `xaifin`, con SHAP, Captum e LIME installati e testati.
- Una nota di mezza pagina per ogni modello (target, input, normalizzazione, loss, se è cross-sectional) e una ricerca bibliografica sull'XAI. Il capitolo 2 è in bozza.

### 2. Gruppi di modelli e universi (Step 1)
- **Gruppo A, feature Alpha158:** MASTER, MATCC, FactorVAE.
- **Gruppo B, feature Alpha360:** HIST, DiscoverPLF, FinFormer.
- **Universi:** DJI, NASDAQ 100 ed EURO STOXX 50 (SX5E).
- **Analisi dei dati:** copertura, distribuzione delle label, feature ridondanti (20 duplicati esatti in Alpha158) e volatilità di mercato.
- **Problemi di qualità dei dati individuati:**
  - La composizione storica di SX5E parte da settembre 2020, quindi SX5E non ha l'anno di test 2020.
  - Alcuni ticker corrispondono all'azienda sbagliata.
  - Alcuni prezzi sono errati. Un filtro ora li scarta.

### 3. Framework unificato (Step 2)
- I sei modelli di FinBench sono stati portati dietro un'unica interfaccia comune.
- Un test di equivalenza con gli script originali di FinBench è superato per tutti i modelli.
- Due problemi di FinBench individuati e documentati:
  - La valutazione sul test di FactorVAE usa informazioni future (leakage).
  - Alcune run di FinBench non avevano un seed fissato.
  - La mia versione di FactorVAE è senza leakage.

### 4. Addestramento a finestre mobili (Step 3)
- 252 run: 6 modelli × 3 universi × anni di test 2020–2024 × seed 0, 5, 42. Il protocollo è quello di FinBench: 4 anni di training, 1 di validazione e 1 di test. Le run hanno richiesto circa 88 ore di GPU, senza errori.
- **Segnale:**
  - Il segnale è debole: RankIC circa 0.01–0.05.
  - Il rumore dovuto al seed è circa metà delle differenze tra i modelli.
  - Le regole di selezione spesso tengono un'epoca con IC di validazione ≤ 0.

### 5. Portafoglio di riferimento (Step 4)
- 54 portafogli long-only top-k: top-5 su DJI, top-10 sugli altri due universi, ribilanciati ogni settimana.
- Il backtest coincide esattamente con il codice di valutazione di FinBench (stessi titoli in portafoglio, rendimenti uguali entro 1e-16).
- **Risultato chiave sull'instabilità:** due seed dello stesso modello hanno in comune solo il 17–46% del portafoglio.
- Ho calcolato anche la distanza di ogni titolo dalla soglia del top-k (il "margine"), che verrà riutilizzata nell'analisi controfattuale.

### 6. Confronto con il paper FinBench (KDD '26)

Errore di regressione (MSE) con orizzonte L=5, su 18 coppie modello/universo:

| Risultato | Coppie | Quali |
|---|---|---|
| Uguale (entro circa il 5%) | 10 | HIST e DiscoverPLF ovunque; la maggior parte dei risultati su NASDAQ |
| Migliore | 3 | MASTER su DJI (−21%) e SX5E (−8%); FactorVAE su DJI (−8.5%), pur senza il leakage di FinBench |
| Peggiore | 5 | FinFormer (da +49% a +82%, probabilmente per la scala delle sue previsioni; la qualità del ranking è simile); MATCC su DJI (+32%) e SX5E (+10%) |

- I rendimenti degli indici coincidono (DJI 0.087 contro 0.08 di FinBench), il che conferma dati e backtest.
- **Da completare:** il confronto esatto sui portafogli (top-5, media sui 3 seed) è già scritto come script ma non ancora eseguito. La tabella di regressione per la tesi è già generata.

### 7. Applicazione (Step 11, in parte completato)
- App Streamlit con le pagine Home, Data, Models e Portfolio.

---

## Prossimi passi (dal piano della tesi)

| Milestone | Step | Contenuto |
|---|---|---|
| Chiusura M2 | – | Eseguire il confronto dei portafogli top-5 con FinBench, spiegare le differenze di FinFormer e MATCC, scrivere la sezione "Riproduzione di FinBench" |
| **M3: Spiegazioni** | 5 | Attribuzioni (Integrated Gradients, SHAP, LIME, permutation importance, attention) nelle date di ribilanciamento; ranking per feature, lag e famiglia |
| | 6 | Valutazione delle spiegazioni: faithfulness (deletion/insertion), stabilità, sanity check con pesi randomizzati, accordo tra metodi → scelta dei metodi affidabili |
| **M4: Analisi** | 7 | Confronto tra modelli: accordo tra i loro ranking delle feature e scomposizione della varianza (seed vs modello vs anno vs universo) → **diagnosi dati vs modello** |
| | 8 | Regimi di volatilità (COVID 2020, mercato ribassista 2022): come cambia nel tempo l'importanza delle feature e come cambia l'IC per regime |
| M5: Estensioni | 9 (opzionale) | Stabilità controfattuale del portafoglio: margine dalla soglia del top-k confrontato con il rumore del seed, e la minima modifica dell'input che fa entrare o uscire un titolo |
| | 10 | Test di perturbazione dei dati: rumore sui prezzi, rimozione di una famiglia di feature con nuovo addestramento, spostamenti temporali |
| M6 | 11–12 | Completare l'applicazione e scrivere la tesi |

---

## Domande per il relatore
1. **Portata del confronto con FinBench:** basta confrontare solo l'orizzonte L=5, o devo addestrare anche L=1 e L=20? Ogni orizzonte in più costa circa 90 ore di GPU.
2. **Universi:** bastano DJI, NASDAQ 100 e SX5E, o devo aggiungere S&P 500 o STOXX 600?
3. **Priorità:** il piano passa direttamente all'explainability (Step 5). Devo prima capire perché FinFormer e MATCC vanno peggio che in FinBench, o basta documentare le differenze?
4. **Metodi di explainability:** con quale iniziare, un metodo basato sui gradienti (Integrated Gradients) o SHAP? I pesi di attention vanno considerati come spiegazioni?
5. **Parti opzionali:** l'analisi controfattuale del portafoglio (Step 9) serve, visto che i risultati sull'instabilità tra seed vanno già in quella direzione?
6. **Scrittura:** quando devo iniziare i capitoli di metodologia e risultati? C'è una lunghezza o una scadenza di riferimento?
