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
    #    gensim
    #    mathplot

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

from gensim.corpora import Dictionary, dictionary
from gensim.models import CoherenceModel

import matplotlib.pyplot as plt

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
#print("BOW-MAtrix")
#bow_df = pd.DataFrame(bow_matrix.toarray(),columns=bow_vectorizer.get_feature_names_out())
#print(bow_df)
#print("Tfidf-Matrix")
#tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),columns=tfidf_vectorizer.get_feature_names_out())
#print(tfidf_df)
#print("Häufigste Wörter nach BOW:")
#bow_word_counter = bow_df.sum().sort_values(ascending=False)
#print(bow_word_counter.head(20))
#print("Höchste TF-IDF-Werte:")
#tfidf_word_scores = tfidf_df.sum().sort_values(ascending=False)
#print(tfidf_word_scores.head(20))

# 6 - Themen mit extrahieren
number_of_topics = 5
number_of_words = 10

def print_topics(model, feature_names, number_words):
    '''

    :param model: LDA- oder LSA-Model
    :param feature_names: Liste der Themen
    :param number_words: Anzahl der Worte pro Thema
    :return:
    '''
    for topic_index, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-number_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        print(f"{topic_index + 1}: {', '.join(top_words)}")



    # LSA

lsa_model = TruncatedSVD(n_components=number_of_topics, random_state=42) # random_state für reproduzierbare Ergebnisse festgelegt
lsa_model.fit(tfidf_matrix)

#print("LSA:")
#print_topics(lsa_model, tfidf_vectorizer.get_feature_names_out(), number_of_words)

    # LDA

lda_model = LatentDirichletAllocation(n_components=number_of_topics, random_state=42) # random_state für reproduzierbare Ergebnisse festgelegt
lda_model.fit(bow_matrix)

#print("LDA:")
#print_topics(lda_model, bow_vectorizer.get_feature_names_out(), number_of_words)



# 7 - Themenanzahl mit Coherence Score bestimmen
def get_topics_coherence(model, feature_names, clean_texts, number_words):
    '''

    :param model: LDA- oder LSA-Model
    :param feature_names: Liste der Themen
    :param clean_texts: bereinigte Texte
    :param number_words: Anzahl der Worte pro Thema
    :return:
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
        processes=1 # Verwendung mehrerer Prozesse kann unter Windows zu Problemen in der Bibliothek gensim führen.
    )
    coherence_score = coherence_model.get_coherence()
    return topics, coherence_score

def test_topic_numbers(model_class, matrix, feature_names, clean_texts, topic_numbers=range(2,16)):
    '''

    :param model_class: "LatentDirichletAllocation" oder "TruncatedSVD"
    :param matrix: bow_matrix oder tfidf_matrix
    :param feature_names: Themennamen
    :param clean_texts: bereinigte Texte
    :param topic_numbers: zu prüfende Themenzahlen (default 2-16 Themen)
    :return:
    '''
    coherence_scores = []
    for topic_number in topic_numbers:
        model = model_class(n_components=topic_number, random_state=42) # random_state = 42 für reproduzierbare Ergebnisse
        model.fit(matrix)
        topics, coherence_score = get_topics_coherence(model, feature_names, clean_texts, topic_number)
        coherence_scores.append(coherence_score)

        print(
            f"{topic_number} Themen: "
            f"Coherence Score = {coherence_score:.4f}"
        )
    return list(topic_numbers), coherence_scores
#LSA
#lsa_topics, lsa_coherence = get_topics_coherence(lsa_model, tfidf_vectorizer.get_feature_names_out(), datafile["issue_description"], number_of_words)
#print(lsa_topics)
#print(lsa_coherence)
print("LSA:")
lsa_topic_numbers, lsa_coherence_scores = test_topic_numbers(
    TruncatedSVD,
    tfidf_matrix,
    tfidf_vectorizer.get_feature_names_out(),
    datafile["clean_description"]
)

#LDA
#lda_topics, lsa_coherence = get_topics_coherence(lda_model, bow_vectorizer.get_feature_names_out(), datafile["issue_description"], number_of_words)
#print(lda_topics)
#print(lda_coherence)
print("LDA:")
lda_topic_numbers, lda_coherence_scores = test_topic_numbers(
    LatentDirichletAllocation,
    bow_matrix,
    bow_vectorizer.get_feature_names_out(),
    datafile["clean_description"]
)


# grafische darstellung der SC der Themenanzahlen von LSA und LDA
plt.plot(lsa_topic_numbers, lsa_coherence_scores, marker="o", label="LSA")
plt.plot(lda_topic_numbers, lda_coherence_scores, marker="o", label="LDA")

plt.xlabel("Anzahl der Themen")
plt.ylabel("Coherence Score")
plt.legend()
plt.title("Coherence Score von LSA und LDA")
plt.show()

# beste Themenanzahl für LSA und LDA auswählen
best_lsa_topics = lsa_topic_numbers[lsa_coherence_scores.index(max(lsa_coherence_scores))]
best_lda_topics = lsa_topic_numbers[lda_coherence_scores.index(max(lda_coherence_scores))]
print("Best Themenanzahl LSA:", best_lsa_topics)
print("Best Themenanzahl LDA:", best_lda_topics)


    # mit verschienden Anzahlen testen
        # Topic-Model trainieren
        # wichtigeste Wörter bestimmen
        # Coherence Score berechnen

        # CS speichern
        # grafisch darstellen?
        # Anzahl auswählen

# 8 - LSA und LDA vergleichen



# 9 - Ergebnis ausgeben
    # falls sinnvoll