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

# 4 - Texte vektorisieren

    # BOW
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(datafile["clean_description"])

#print("BOW: ",bow_matrix)
#print(bow_vectorizer.get_feature_names_out())
#print(bow_matrix.shape)

    # TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(datafile["clean_description"])

#print("TF-IDF: ", tfidf_matrix)
#print(tfidf_matrix.shape)
#print(tfidf_vectorizer.get_feature_names_out())


# 5 - BOW und TF-IDF vergleichen
print("BOW-MAtrix")
bow_df = pd.DataFrame(bow_matrix.toarray(),columns=bow_vectorizer.get_feature_names_out())
print(bow_df)
print("Tfidf-Matrix")
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),columns=tfidf_vectorizer.get_feature_names_out())
print(tfidf_df)
print("Häufigste Wörter nach BOW:")
bow_word_counter = bow_df.sum().sort_values(ascending=False)
print(bow_word_counter.head(20))
print("Höchste TF-IDF-Werte:")
tfidf_word_scores = tfidf_df.sum().sort_values(ascending=False)
print(tfidf_word_scores.head(20))

# 6 - Themen mit extrahieren
number_of_topics = 5
number_of_words = 10

    # LSA
def lsa (number_of_topics, number_of_words,matrix, vectorizer):
    '''

    :param number_of_topics: Anzahl der zu bestimmenden Themen
    :param number_of_words: Anzahl der Wörter pro Thema
    :param matrix: TF-IDF-Matrix zur Verarbeitung
    :param vectorizer: TF-IDF-Vectorizer zur Verarbeitung
    :return:
    '''
    lsa_model = TruncatedSVD(n_components=number_of_topics, random_state=42) # random_state für reproduzierbare Ergebnisse festgelegt
    lsa_model.fit(matrix)

    feature_names = vectorizer.get_feature_names_out()
    for topic_index, topic in enumerate(lsa_model.components_):
        top_indices = topic.argsort()[-number_of_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]

        print("LSA:")
        print(f"Thema {topic_index + 1}: ")
        print(top_words)
lsa(number_of_topics, number_of_words,tfidf_matrix, tfidf_vectorizer)
    # LDA
def lda (number_of_topics, number_of_words, matrix, vectorizer):
    '''

    :param number_of_topics: Anzahl der zu bestimmenden Themen
    :param number_of_words: Anzahl der Wörter pro Thema
    :param matrix: BOW-Matrix zur Verarbeitung
    :param vectorizer: BOW-Vectorizer zur Verarbeitung
    :return:
    '''
    lda_model = LatentDirichletAllocation(n_components=number_of_topics, random_state=42) # random_state für reproduzierbare Ergebnisse festgelegt
    lda_model.fit(matrix)

    feature_names = vectorizer.get_feature_names_out()
    for topic_index, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-number_of_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]

        print("LDA:")
        print(f"Thema {topic_index + 1}: ")
        print(top_words)

lda(number_of_topics, number_of_words, bow_matrix, bow_vectorizer)

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