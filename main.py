# ============================================================
# Projekt: Analyse kommunaler Beschwerdetexte
# Datensatz: Civic and Municipality Complaint System Dataset
# https://www.kaggle.com/datasets/wajahattaj/civic-and-municipality-complaint-system-dataset?select=municipal_training_set_1100.csv
# Ziel:
# - Textdaten laden und bereinigen
# - Texte mit BOW und TF-IDF vektorisieren
# - Themen mit LSA und LDA extrahieren
# - geeignete Themenanzahl mit Coherence Score untersuchen
# - Ergebnisse vergleichen und auswerten
# ===========================================================


# 1 - Biblitheken importieren

    #Abhängigkeiten:
    #    pandas
    #    scikit-learn
    #    nltk

import string
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download("punkt_tab")

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation



# 2 - Datensatz laden

    # CSV einlesen
datafile = pd.read_csv("municipal_training_set_1100.csv")
'''
print("Erste Zeilen:")
print(datafile.head())
print("Spalten:")
print(datafile.columns)
print("Größe:")
print(datafile.shape)
print("Null-Felder")
print(datafile.isnull().sum())
'''
    # relevante Spalten auswählen
        # -> issue_description
# Spalte mit Beschwerdetexten auswählen:
#texts = datafile["issue_description"]
# Erste Beschwerdetexte anzeigen:
#print("Beispieltexte:")
#print(texts.head())


# 3 - Texte bereinigen

#Stoppwörter und Lemmatizer vorbereiten
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    '''
    Bereinigt Text. Umwandlung in Kleinbuchstaben, Satzzeichen und Stoppwörter entfernen und lemmatieren.
    :param text: CSV-Spalte mit Beschwerden
    :return: CSV-Spalte mit bereinigten Beschwerden
    '''

    # In Kleinbuchstaben umwandeln
    text = text.lower()

    # Satzzeichen entfernen
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Text in einzelne Wörter zerlegen
    tokens = word_tokenize(text)

    # Stopwörter entfernen
    tokens = [word for word in tokens if word not in stop_words]

    # Wörter lemmatisieren
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    #Token
    return " ".join(tokens)


datafile["clean_description"] = datafile["issue_description"].apply(clean_text)
#print(datafile[["issue_description","clean_description"]].head())
    # Text in Kleinbuchstaben umwandeln
    # Satzzeichen entfernen
    # Text tokenisieren
    # Stopwörter entfernen
    # Wörter lemmatisieren

# 4 - Texte vektorisieren

    # BOW

    # TF-IDF

# 5 - BOW und TF-IDF vergleichen
    # ...falls es sich sinnvoll technisch umsetzen lässt. Sonst manuell.

# 6 - Themen mit extrahieren

    # LSA

    # LDA

# 7 - Themenanzahl mit Coherence Score bestimmen

    # mit verschienden Anzahlen testen
        # Topic-Model trainieren
        # wichtigeste Wörter bestimmen
        # Coherence Score berechnen

        # CS speichern
        # grafisch darstellen?
        # Anzahl auswählen

# 8 - LSA und LDA vergleichen
    # ...falls es sich sinnvoll technisch umsetzen lässt. Sonst manuell.

# 9 - Ergebnis ausgeben
    # falls sinnvoll