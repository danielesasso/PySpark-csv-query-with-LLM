# Analisi Dinamica delle Competenze Aziendali con PySpark e LLM

![alt text](photo/background_svg.svg)![alt text](photo/background_svg.svg)![alt text](photo/background_svg.svg)![alt text](photo/background_svg.svg)![alt text](photo/background_svg.svg)![alt text](photo/background_svg.svg)

## 1. Titolo e Descrizione del Progetto

Questo progetto utilizza **PySpark** e il **Language Model di OpenAI (GPT)** per analizzare e filtrare un dataset tassonomico di competenze aziendali.  
Attraverso il parsing delle query degli utenti, il sistema identifica e pesa le skill rilevanti in modo dinamico, considerando l'importanza attribuita a ciascuna competenza.  
Le competenze vengono rappresentate in forma binaria (0 o 1), permettendo di determinare rapidamente quali utenti possiedono le skill richieste.

I risultati dell'analisi sono visualizzati tramite:
- Grafici che mostrano il numero di utenti idonei per le competenze selezionate.
- Tabelle interattive con "heatmap" per evidenziare in modo immediato la distribuzione delle skill.  

Questo strumento si rivolge principalmente alla gestione delle risorse umane e alla pianificazione strategica delle competenze all'interno delle organizzazioni.

---

## 2. Motivazione

Il progetto nasce dalla necessità di migliorare i processi di gestione delle risorse umane nelle aziende, facilitando l'identificazione delle competenze esistenti e la mappatura delle skill richieste per specifici ruoli o progetti.  
Tradizionalmente, l'analisi delle competenze richiede un notevole impegno manuale e una valutazione soggettiva.  
Utilizzando PySpark per i filtraggi di feature e l'intelligenza semantica di GPT, questo progetto:
- Automatizza l'analisi dei dati per ridurre il tempo e gli errori tipici delle procedure manuali.
- Fornisce una visione oggettiva e quantificabile delle capacità aziendali, essenziale per una pianificazione strategica delle risorse umane.
  

Il codice è strutturato in un notebook Python, che consente una facile manipolazione e visualizzazione dei dati.

---

## 3. Installazione e Configurazione

### 3.1 Prerequisiti generali
- **API di OpenAI GPT**: È necessaria una chiave API valida per utilizzare il modello GPT per l'analisi semantica della query (non necessaria nella demo di colab **demo_Colab_PySpark_LLM.ipynb**.
- **File CSV Tassonomico**:  
  Questi file `.csv` strutturano un elenco dettagliato di competenze aziendali, organizzate secondo una tassonomia ben definita. Ogni elemento è diviso in due parti: un "tema" e una "feature specifica", separati da un trattino. Ad esempio, in "tools-laser", "tools" rappresenta il tema, mentre "laser" è la feature specifica. I file includono anche una colonna che inizia con "ID-" per identificare univocamente gli utenti, oltre ad altre informazioni utili come i ruoli, facilitando così l'associazione delle competenze ai singoli individui. Questa organizzazione permette di navigare e interpretare facilmente il set di dati in base alle necessità specifiche di analisi delle competenze.
  <figure>
    <img src="photo/bar_themed.png" width="80%">
    <br> <figcaption>Tassonomia</figcaption>
  </figure>
  

  _**Nota:** I file presenti nel repository contengono dati e nomi sintetici, non appartenenti a persone reali._

### 3.1 Colab_PySpark_LLM.ipynb e demo_Colab_PySpark_LLM.ipynb
*![Open in Colab](photo/opencolab.png)* <br>
Aprire il codice su Colab tramite il tasto apposito una volta selezionato il file, assicurarsi di scaricare almeno un file .csv tassonomico presente nel github (come populated_matrix.csv) e seguire tutte le istruzioni all'interno del colab per poi eseguire ogni cella in ordine.
<figure>
  <img src="photo/upload.png" alt="upload" width="30%">
</figure> <br>
Una volta dentro colab, selezionare l'icona della cartella e successivamente l'icona di upload (la prima a sinistra rispetto alle cartelle) e caricare il dataset .csv.

### 3.2 mplgraph.py Prerequisiti aggiuntivi
- **Python3.13.2 installato**: Assicurati di avere Python 3.13.2 installato, per controllare la versione installata inserire nel terminale:

```bash
python3 --version

```
Verrà restituito in output la versione di python attuale.

#### 3.2.1 Configurazione dell'Ambiente

È consigliato utilizzare un ambiente virtuale Python per gestire le dipendenze.  
Esegui i seguenti comandi per creare e attivare l'ambiente:

```bash
python3 -m venv myenv
source myenv/bin/activate #Su Windows `myenv\Scripts\activate`

```

#### 3.2.2 Installazione delle Dipendenze

Dopo aver attivato l'ambiente virtuale, installa le dipendenze necessarie eseguendo:

```bash
pip3 install -r requirements.txt
```

#### 3.2.4 Verifica dell'installazione (mplgraph.py)
Per verificare che tutto sia stato configurato correttamente, una volta entrato all'interno modifica le seguenti righe:
  - riga 29: inserisci la key di openai
  - riga 40: modifica il nome e/o il percorso del file `.csv` (è consigliato mettere il csv nella stessa directory per motivi pratici)

 Successivamente entra nella venv (secondo comando nella subsection 3.2.1) e esegui da terminale:

```bash
python3 mpl_graph.py
```


## 4. Funzionalità Principali

- **Caricamento e Analisi del Dataset**: Il codice sfrutta PySpark per leggere e analizzare il file CSV. In particolare, estrae i "blocchi tematici", ovvero insiemi di colonne raggruppate in base a un prefisso comune (separato da un trattino) che rappresenta una categoria o ambito di competenze (ad esempio "tech_skills-"). Il sistema quindi conta le occorrenze di ciascun blocco per fornire una panoramica strutturata delle diverse aree di competenza presenti nel dataset.


- **Parsing delle Query Utente con GPT**: Utilizzando l'API di OpenAI, il sistema interpreta le richieste degli utenti per identificare quali blocchi o feature siano rilevanti e assegna un peso (da 0 a 1) a ciascuna competenza.

- **Filtraggio e Visualizzazione dei Dati**:
  - Seleziona dinamicamente le colonne in base ai pesi assegnati.
  - Calcola un punteggio pesato per ogni utente e li classifica in base ad essi.
  - Visualizza i risultati tramite una tabella interattiva e grafici per facilitare l'interpretazione dei dati.

- **Generazione di un SVG Interattivo**:  
  Il sistema crea un file SVG interattivo che permette agli utenti di esplorare i dati in maniera intuitiva: cliccando sui singoli pixel, è possibile visualizzare dettagli informativi relativi agli utenti, rendendo la navigazione e l’analisi di grandi moli di dati più immediata ed efficace.

## 5. Contribuire
Alcune aree di potenziale contributo includono:

- **Miglioramenti al Codice**: Miglioramento del parsing e ponderazione con GPT, refactoring del codice per migliorarne la leggibilità e le performance.
- **Nuove Funzionalità**: Integrazione di ulteriori formati di dati, supporto per nuove visualizzazioni grafiche o sviluppo di una GUI per un'interazione più intuitiva.
- **Benchmarking e Sperimentazioni**: Valutazione di differenti modelli LLM oltre OpenAI per migliorarne la precisione del parsing delle query.

**Per contribuire:**

1. Fork del repository.
2. Crea un nuovo branch.
3. Apporta le modifiche e invia una pull request.

---

## 6. Licenza

Questo progetto è rilasciato sotto la Licenza MIT.  
Per ulteriori dettagli, vedi il file `LICENSE` incluso nel repository.

---
#### 7. Esempi di Output

Di seguito vengono riportati alcuni esempi visivi dei risultati ottenuti:

- **Heatmap delle Competenze**:
  *![Heatmap con utenti](photo/heatmap.png)*

- **Tabella Interattiva**:
  *![SVG image](photo/interactive_pixels.svg)*
    <figure>
      <img src="photo/interative_pixel_onclick.png" alt="onclick" width="30%">
      <br><figcaption>In caso di click di un pixel</figcaption>
    </figure>

- **Barplot della Frequenza delle Skill**:
   *![Barplot](photo/barplot.png)*
 
- **Mplcursor Linear Segmented Colormap**
  *![Mplcursor](photo/mplgraph.png)*
     <figure>
      <img src="photo/onclickmpl.png" alt="mplonclick" width="70%">
      <br> <figcaption>In caso di click di un pixel</figcaption> <br>
    </figure>

    
    <figure>
      <img src="photo/hover.png" alt="hover" width="30%">
      <br> <figcaption>In caso il mouse fosse sopra una cella</figcaption><br>
    </figure>



