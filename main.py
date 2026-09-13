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

#Abhängigkeiten:
#    pandas
#    scikit-learn
#    nltk
#    gensim
#    mathplotlib

# =========================================================
# Importe:
# ==========================================================

import string
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

import matplotlib.pyplot as plt

# ==========================================================
# Konstanten:
# ==========================================================

NUMBER_OF_WORDS = 10
DATA_PATH = "municipal_training_set_1100.csv"
TOPIC_NUMBERS=range(2, 16)
PRINT_SCORES = False # Detailausgabe bei der berechnung der einzelnen Coherence Scores

# ==========================================================
# Funktionen
# ==========================================================

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

def print_topics(topics):
    '''
    Gibt Themen lesbar aus.
    :param topics: Themen, die angezeigt werden sollen
    '''
    for index, topic in enumerate(topics):
        print(f"Thema {index + 1}: {', '.join(topic)}")

def get_topics_coherence(model, feature_names, clean_texts, number_words):
    '''
    Extrahiert Themen aus den bereinigten Texten und berechne die Kohärenz
    :param model: LDA- oder LSA-Model
    :param feature_names: Liste der Themen
    :param clean_texts: bereinigte Texte
    :param number_words: Anzahl der Worte pro Thema
    :return: topics, coherence_score
    '''
    topics = []
    #wichtigste Wörter je Thema bestimmen
    for topic in model.components_:
        top_indices = topic.argsort()[-number_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topics.append(top_words)

    # Bereinigte Texte für die verwendung mit Gensim-Funktionen vorbereiten
    tokenized_texts = [
        text.split()
        for text in clean_texts
    ]
    gensim_dictionary = Dictionary(tokenized_texts)

    #Coherence Score berechnen
    coherence_model = CoherenceModel(
        topics=topics,
        texts=tokenized_texts,
        dictionary=gensim_dictionary,
        coherence="c_v",
        processes=1  # Verwendung mehrerer Prozesse kann unter Windows zu Problemen in der Bibliothek gensim führen.
    )
    coherence_score = coherence_model.get_coherence()
    return topics, coherence_score

def test_topic_numbers(model_class, matrix, feature_names, clean_texts, topic_numbers=range(2, 16),print_scores=False):
    '''
    Testet verschiedene Themenanzahlen für ein LSA- oder LDA-Modell
    :param model_class: Modellklasse "LatentDirichletAllocation" oder "TruncatedSVD"
    :param matrix: Vektorisierte Texte (bow_matrix oder tfidf_matrix)
    :param feature_names: Wörter des verwendeten Vektorisierers
    :param clean_texts: Bereinigte Texte
    :param topic_numbers: zu testende Themenzahlen (default 2-16 Themen)
    :param print_scores: Gibt die einzelnen Coherence Scores aus, wenn True
    :return: Themenanzahl und zugehörige Coherence Scores
    '''
    coherence_scores = []
    for topic_number in topic_numbers:
        model = model_class(n_components=topic_number,
                            random_state=42)  # random_state = 42 für reproduzierbare Ergebnisse
        model.fit(matrix)
        topics, coherence_score = get_topics_coherence(model, feature_names, clean_texts, NUMBER_OF_WORDS)
        coherence_scores.append(coherence_score)

        if print_scores:
            print(
                f"{topic_number} Themen: "
                f"Coherence Score = {coherence_score:.4f}"
            )
    return list(topic_numbers), coherence_scores



# ==========================================================
# Datensatz laden
# ==========================================================

# CSV einlesen
datafile = pd.read_csv(DATA_PATH)

# Debuging:
# behalten, falls der Datensatz geändert wird
#print("Erste Zeilen:")
#print(datafile.head())
#print("Spalten:")
#print(datafile.columns)
#print("Größe:")
#print(datafile.shape)
#print("Null-Felder")
#print(datafile.isnull().sum())
# Spalte mit Beschwerdetexten auswählen:
#texts = datafile["issue_description"]
# Erste Beschwerdetexte anzeigen:
#print("Beispieltexte:")
#print(texts.head())

#===========================================================
# Texte bereinigen
# ==========================================================

#Stoppwörter und Lemmatizer vorbereiten
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
datafile["clean_description"] = datafile["issue_description"].apply(clean_text)

# =========================================================
# Texte vektorisieren
# ==========================================================

# BOW
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(datafile["clean_description"])

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(datafile["clean_description"])

# ==========================================================
# Coherence Scores je Themenanzahl bestimmen
# ==========================================================

# LSA
if PRINT_SCORES:
    print("LSA:")
lsa_topic_numbers, lsa_coherence_scores = test_topic_numbers(
    TruncatedSVD,
    tfidf_matrix,
    tfidf_vectorizer.get_feature_names_out(),
    datafile["clean_description"],
    TOPIC_NUMBERS,
    PRINT_SCORES
)

# LDA
if PRINT_SCORES:
    print("LDA:")
lda_topic_numbers, lda_coherence_scores = test_topic_numbers(
    LatentDirichletAllocation,
    bow_matrix,
    bow_vectorizer.get_feature_names_out(),
    datafile["clean_description"],
    TOPIC_NUMBERS,
    PRINT_SCORES
)

# ==========================================================
# grafische darstellung der CS der Themenanzahlen von LSA und LDA
# ==========================================================

plt.plot(lsa_topic_numbers, lsa_coherence_scores, marker="o", label="LSA")
plt.plot(lda_topic_numbers, lda_coherence_scores, marker="o", label="LDA")

plt.xlabel("Anzahl der Themen")
plt.ylabel("Coherence Score")
plt.legend()
plt.title("Coherence Score von LSA und LDA")
plt.show()

# ==========================================================
# beste Themenanzahl für LSA und LDA auswählen
# ==========================================================

best_lsa_topics = lsa_topic_numbers[lsa_coherence_scores.index(max(lsa_coherence_scores))]
best_lda_topics = lsa_topic_numbers[lda_coherence_scores.index(max(lda_coherence_scores))]
print("Best Themenanzahl LSA:", best_lsa_topics)
print("Best Themenanzahl LDA:", best_lda_topics)

# ==========================================================
# Finale Modelle trainieren/erstellen
# ==========================================================

# LSA
final_lsa_model = TruncatedSVD(n_components=best_lsa_topics, random_state=42)
final_lsa_model.fit(tfidf_matrix)

final_lsa_topics, final_lsa_coherence = get_topics_coherence(
    final_lsa_model,
    tfidf_vectorizer.get_feature_names_out(),
    datafile["clean_description"],
    NUMBER_OF_WORDS
)

# LDA
final_lda_model = LatentDirichletAllocation(n_components=best_lda_topics, random_state=42)
final_lda_model.fit(bow_matrix)

final_lda_topics, final_lda_coherence = get_topics_coherence(
    final_lda_model,
    bow_vectorizer.get_feature_names_out(),
    datafile["clean_description"],
    NUMBER_OF_WORDS
)

# ==========================================================
# Ergebnis ausgeben
# ==========================================================

print("\n--- LSA-Themen ---")
print_topics(final_lsa_topics)
print(f"Coherence Score: {final_lsa_coherence:.4f}")

print("\n--- LDA-Themen ---")
print_topics(final_lda_topics)
print(f"Coherence Score: {final_lda_coherence:.4f}")
