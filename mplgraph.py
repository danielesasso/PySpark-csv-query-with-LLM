from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import *

import pandas as pd
import numpy as np

from collections import Counter
from langchain_community.document_loaders import PySparkDataFrameLoader
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain

import os
import json

import openai
from functools import reduce

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import mplcursors
from matplotlib.colors import LinearSegmentedColormap


#INSERISCI LA CHIAVE
openai.api_key = "" 

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("Il mio primo progetto Spark") \
    .getOrCreate()

#CAMBIA 'populated_matrix.csv' con il csv tassonomico cvhe vuoi usare (leggi la section 3.1 per il tipo di .csv richiesto)
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("populated_matrix.csv")

#prende il nome delle colonne
columns = df.columns
#Estrarre il blocco da ogni colonna
thematic_blocks = [col.split('-', 1)[0] for col in columns]  #prendi la parte prima del primo '-'

#Contare le ricorrenze di ogni blocco
counts = Counter(thematic_blocks)

#Costruisci un dizionario di blocchi {nome_blocco: [colonna1, colonna2, ...]}
blocks_dict = {}
for col in columns:
    block_name = col.split('-', 1)[0]
    if block_name not in blocks_dict:
        blocks_dict[block_name] = []
    blocks_dict[block_name].append(col)

def parse_user_query_strict(query, blocks_dictionary):
    """
    Usa GPT per analizzare la query dell'utente e identificare 
    quali blocchi e/o feature potrebbero essere rilevanti,
    assegnando anche un peso (0-1) a ciascuna feature/blocco.

    Restituisce un dizionario con la struttura:
    {
      "relevant_blocks": [
        {
          "block_name": "NomeBlocco",
          "weight": 0.8
        }
      ],
      "relevant_features": [
        {
          "name": "NomeColonna",
          "weight": 1.0
        }
      ]
    }
    """

    # Prompt per GPT: chiediamo di essere molto selettivi,
    # aggiungendo una richiesta di weight da 0 a 1.
    prompt = f"""
    Devi analizzare la query di un utente su un dataset di skill,
    DIVISO IN BLOCCHI, ognuno con diverse colonne (feature).
    Ecco la struttura (blocks_dictionary) in JSON:
    {json.dumps(blocks_dictionary, indent=2)}

    L'utente fa la seguente richiesta: "{query}"

    *SII MOLTO SELETTIVO*:
    - Se la colonna o il blocco non è strettamente correlato alla richiesta, NON includerlo.
    - Se ritieni che l'utente voglia TUTTO il blocco, allora aggiungi un oggetto in "relevant_blocks"
      con {{"block_name": "NomeBlocco", "weight": <valore 0-1>}}.
    - Se invece l'utente è interessato a UNA o POCHI argomenti all'interno di un blocco,
      indica le feature (colonne) specifiche in "relevant_features", 
      con {{"name": "NomeColonna", "weight": <valore 0-1>}}.
    - Non aggiungere nulla che non sia chiaramente correlato alla query.
    - Se una skill è più importante, assegnale un peso più alto (vicino a 1).
      Se è rilevante ma secondaria, peso intermedio (0.3-0.7).
    - Restituisci SOLO un JSON **valido** con questa struttura:
    {{
      "relevant_blocks": [
        {{
          "block_name": "NomeBlocco",
          "weight": 0.8
        }}
      ],
      "relevant_features": [
        {{
          "name": "NomeColonna",
          "weight": 1.0
        }}
      ]
    }}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sei un assistente che analizza la query dell'utente "
                        "per trovare i blocchi e/o le feature più rilevanti, "
                        "assegnando pesi (0-1) in modo selettivo."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0
        )
        content = response["choices"][0]["message"]["content"]
        parsed_json = json.loads(content)  
        return parsed_json
    except (json.JSONDecodeError, KeyError):
        return {
            "relevant_blocks": [],
            "relevant_features": []
        }
    except Exception as e:
        print("Errore inaspettato nella chiamata a GPT:", e)
        return {
            "relevant_blocks": [],
            "relevant_features": []
        }



def find_relevant_columns(parsed_dict, blocks_dictionary):
    """
    Dati i blocchi/feature con relativi pesi (da 0 a 1),
    costruisce un dizionario {colonna: peso} da selezionare.

    Logica invariata nella selezione (molto severa), con la differenza
    che i relevant_blocks e relevant_features sono liste di oggetti:
      { "block_name": "...", "weight": ... }
      { "name": "...",        "weight": ... }
    
    - Se GPT include un blocco in 'relevant_blocks' ma specifica anche alcune feature
      di quel blocco in 'relevant_features', prendiamo SOLO quelle feature.
    - Se GPT include un blocco (weight=W) e non specifica feature particolari,
      includiamo TUTTO il blocco con peso=W.
    - Se GPT specifica feature di un blocco che non compare in 'relevant_blocks',
      includiamo solo quelle feature con i loro pesi.
    """

    relevant_blocks = parsed_dict.get("relevant_blocks", [])
    relevant_feats = parsed_dict.get("relevant_features", [])

    #Costruiamo un dict: block_name -> weight
    block_weights = {}
    for b in relevant_blocks:
        bname = b.get("block_name")
        w = b.get("weight", 0.5)  # default 0.5 se non presente
        if bname and bname in blocks_dictionary:
            block_weights[bname] = w

    #Costruiamo un dict: feature_name -> weight
    feature_weights = {}
    for f in relevant_feats:
        fname = f.get("name")
        w = f.get("weight", 0.5)
        if fname:
            feature_weights[fname] = w

    #Iniziamo a costruire un dict finale col -> peso
    final_col_weights = {}

    # 1) Se un blocco è elencato, ma GPT ha specificato feature di quel blocco
    #    usiamo solo quelle feature con i loro pesi.
    #    altrimenti, includiamo tutto il blocco con lo stesso peso di blocco.
    for block_name, wblock in block_weights.items():
        # Verifichiamo se ci sono feature di questo blocco
        block_specific_feats = []
        for feat_name, wfeat in feature_weights.items():
            
            # verifichiamo se feat_name appartiene al blocco in blocks_dictionary
            cols_in_block = blocks_dictionary.get(block_name, [])
            if feat_name in cols_in_block:
                block_specific_feats.append((feat_name, wfeat))

        if len(block_specific_feats) > 0:
            # Abbiamo feature specifiche quindi aggiungiamo solo quelle
            for (colname, colweight) in block_specific_feats:
                final_col_weights[colname] = colweight
        else:
            # Nessuna feature specifica, allora includi tutto il blocco
            for col in blocks_dictionary.get(block_name, []):
                final_col_weights[col] = wblock

    # 2) Gestione di feature non citate in relevant_blocks (o citate fuori blocco)
    #       se GPT ha messo per qualche feature (f.name, f.weight) e
    #       il block non è in block_weights, la aggiungiamo lo stesso
    for feat_name, wfeat in feature_weights.items():
        # se non l'abbiamo già inserita
        if feat_name not in final_col_weights:
            final_col_weights[feat_name] = wfeat

    # Stampa a video le feature trovate e il relativo peso
    print("Feature (o colonne) trovate da GPT con peso:")
    for col, w in final_col_weights.items():
        print(f"  - {col} -> {w}")

    return final_col_weights


user_query = input("Benvenuto!\nPuoi chiedermi di identificare persone o gruppi basati su abilità specifiche, conoscenze linguistiche, o altri criteri presenti nel dataset.\nPer esempio, potresti chiedere: 'Chi sa parlare inglese?' oppure 'Chi ha le competenze necessarie per costruire una casa?'.\nChe tipo di utenti stava cercando? ")

#Analisi della query con GPT
parsed_result = parse_user_query_strict(user_query, blocks_dict)

#Ottengo un dizionario {colonna: peso} invece di una semplice lista
col_weights_dict = find_relevant_columns(parsed_result, blocks_dict)

#Se non c'è alcuna colonna, GPT non ha trovato corrispondenze e si ferma
if len(col_weights_dict) == 0:
    # Mostra solo ID
    id_block_columns = blocks_dict.get("ID", [])
    df.select(id_block_columns).show(truncate=False)
    print("Nessuna feature trovata da GPT.")
    import sys
    sys.exit(0)

else:
    #Aggiungiamo all'inizio la presenza di tutte le colonne del blocco "ID" 
    id_block_columns = blocks_dict.get("ID", [])
    final_columns_with_weights = {}

    #Inserisci ID con peso speciale (nullo)
    for col_id in id_block_columns:
        if col_id in df.columns:
            final_columns_with_weights[col_id] = None 

    #Aggiungiamo le colonne GPT trovate (col->weight) 
    for c, w in col_weights_dict.items():
        if c in df.columns:
            final_columns_with_weights[c] = w

    #Ordina le colonne:
    #    - prima ID in ordine in cui compaiono
    #    - poi le skill 
    id_cols = list(final_columns_with_weights.keys())[:len(id_block_columns)]  #i primi len(id_block_columns) sono ID
    skill_cols_and_weights = [(c, w) for c, w in final_columns_with_weights.items() if w is not None]
    skill_cols_and_weights.sort(key=lambda x: x[1], reverse=True)

    #costruiamo la lista finale di colonne
    final_columns = id_cols + [x[0] for x in skill_cols_and_weights]

    print("Colonne finali da selezionare (ID + skill ordinate per peso):", final_columns)


    #Filtro con Spark: prendi gli utenti che hanno almeno una skill = 1 tra quelle scelte
    skill_only_columns = [x[0] for x in skill_cols_and_weights]
    if skill_only_columns:
        #Crea condizioni NOR solo se ci sono skill
        filter_conditions = [F.col(c) == 1 for c in skill_only_columns]
        combined_filter = reduce(Column.__or__, filter_conditions)
        df_filtered = df.filter(combined_filter).select(final_columns)
    else:
        #Se non ci sono skill, seleziona solo ID senza filtrare
        df_filtered = df.select(final_columns)

def convert_spark_to_pandas(spark_df):
    columns = spark_df.columns
    data = []
    for row in spark_df.collect():
        data.append(row.asDict())
    return pd.DataFrame(data, columns=columns)

pdf = convert_spark_to_pandas(df_filtered)

# 2) Calcolo punteggio pesato
weighted_scores = []
for idx, row in pdf.iterrows():
    score = 0.0
    for col in skill_only_columns:
        skill_value = row[col]  # 0 o 1
        weight = col_weights_dict.get(col, 0)
        score += skill_value * weight
    weighted_scores.append(score)

pdf['weighted_score'] = weighted_scores

# 3) Ordiniamo i dati in base al punteggio pesato
pdf_sorted = pdf.sort_values('weighted_score', ascending=False)


print("Scegli un'opzione per la visualizzazione del grafico:")
print("0) Stampare il grafico con tutti gli utenti")
print("1) Mostrare meno utenti")
choice = input("Digita il numero dell'opzione scelta (0-1): ")

if choice in ["1"]:
    # Riduci il numero di utenti
    total_users = len(pdf_sorted)
    print(f"Numero totale di utenti disponibili: {total_users}")
    n_users = int(input(f"Quanti utenti vuoi visualizzare? (1 - {total_users}): "))
    pdf_sorted = pdf_sorted.head(n_users)

# Creiamo una mappa "peso → frazione" per usare un unico colore per colonne con lo stesso peso
unique_weights = sorted({col_weights_dict.get(c, 0) for c in skill_only_columns})
if len(unique_weights) > 1:
    min_w, max_w = unique_weights[0], unique_weights[-1]
    weight_to_fraction = {
        w: (w - min_w) / (max_w - min_w)
        for w in unique_weights
    }
else:
    # Se esiste un solo peso (o nessuno), assegniamo una frazione fissa
    weight_to_fraction = {w: 0.5 for w in unique_weights}

#PREPARAZIONE GRAFICO
col_weights = col_weights_dict
sorted_skills = skill_only_columns 
user_ids = pdf_sorted.filter(regex='^ID-').iloc[:, 0].values
weighted_scores = pdf_sorted['weighted_score'].values
weighted_matrix = pdf_sorted[sorted_skills].mul([col_weights[s] for s in sorted_skills]).values


# -------------------------------------------
#    BLOCCO DEL GRAFICO CON MPLCURSORS
# -------------------------------------------

# (Esempio di colormap personalizzata)
colors = [
    "#FFFFFF",  # Nessuna skill
    "#FFCC99",  # Light orange (basso peso)
    "#FF9966",  # Medium orange
    "#FF6600",  # Dark orange
    "#CC0000",  # Red
    "#990000",  # Dark red
    "#660000"   # Very dark red
]
cmap = LinearSegmentedColormap.from_list("weighted_skills", colors, N=256)

plt.figure(figsize=(18, 12))
ax = plt.gca()

# Visualizza la matrice pesata come heatmap
im = ax.imshow(weighted_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1, interpolation='none')

# Rimuovi tick e label dagli assi (per estetica)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])

# Aggiunge una colorbar che mostra la scala dei pesi
cbar = plt.colorbar(im, label='Skill Weight')
cbar.ax.tick_params(labelsize=8)

# Configuriamo l'interazione con mplcursors
cursor = mplcursors.cursor(ax, hover=True)

@cursor.connect("add")
def on_add(sel):
    # Ricava la riga/colonna su cui si sta puntando
    row = int(sel.target[1] + 0.5)
    col = int(sel.target[0] + 0.5)
    
    if 0 <= row < len(user_ids) and 0 <= col < len(sorted_skills):
        skill_name = sorted_skills[col]
        skill_weight = col_weights[skill_name]
        skill_value = pdf_sorted.iloc[row][skill_name]
        user_score = weighted_scores[row]
        
        # Prende tutte le colonne ID del singolo utente
        id_features = pdf_sorted.filter(regex='^ID-').iloc[row].to_dict()
        id_features_str = "\n".join([f"{key}: {value}" for key, value in id_features.items()])
        
        # Se la skill per quella cella è 1, mostra l'annotazione
        if skill_value == 1:
            sel.annotation.set(text=f"""
            ID Features:
            {id_features_str}
            
            Total Score: {user_score:.2f}
            Skill: {skill_name}
            Weight: {skill_weight}
            """,
            bbox=dict(fc="white", ec="black", alpha=0.9))
        else:
            # Se skill=0, non mostriamo nulla
            sel.annotation.set_visible(False)

# Gestione del click per mostrare TUTTE le skill di quell'utente
def on_click(event):
    if event.inaxes == ax:
        y = int(event.ydata + 0.5)
        if 0 <= y < len(pdf_sorted):
            # Colonne ID
            id_features = pdf_sorted.filter(regex='^ID-').iloc[y].to_dict()
            id_features_str = "\n".join([f"{key}: {value}" for key, value in id_features.items()])
            
            # Skill possedute (quelle = 1)
            user_skills = pdf_sorted.iloc[y][sorted_skills]
            present_skills = user_skills[user_skills == 1].index.tolist()
            skill_list = "\n".join([f"- {skill} (Weight: {col_weights[skill]})" for skill in present_skills])
            
            # Finestra popup con testo
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5,
                     f"ID Features:\n{id_features_str}\n\n"
                     f"Total Score: {weighted_scores[y]:.2f}\n\n"
                     f"Skills:\n{skill_list}",
                     ha='center', va='center', fontsize=12)
            plt.axis('off')
            plt.show()

def on_scroll(event):
    # Se l'evento si verifica fuori dall'area del grafico, non fare nulla
    if event.inaxes is None:
        return
    
    ax = event.inaxes
    xdata = event.xdata
    ydata = event.ydata
    
    # Scelgo un fattore di zoom (1.1 o 0.9)
    # Zoom IN se scrolla 'up', zoom OUT se scrolla 'down'
    if event.button == 'up':
        scale_factor = 1/1.1
    elif event.button == 'down':
        scale_factor = 1.1
    else:
        # Nel caso di altri eventi, non facciamo nulla
        scale_factor = 1.0
    
    # Ottieni i limiti attuali degli assi
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    
    # Calcola la nuova ampiezza dei limiti
    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
    
    # Imposta i nuovi limiti mantenendo il punto di scroll come "centro"
    ax.set_xlim([xdata - new_width/2, xdata + new_width/2])
    ax.set_ylim([ydata - new_height/2, ydata + new_height/2])
    
    ax.figure.canvas.draw_idle()

# Collega l’evento 'scroll_event' al callback on_scroll
plt.gcf().canvas.mpl_connect('scroll_event', on_scroll)


plt.tight_layout()
plt.show()
