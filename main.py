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

# 2 - Datensatz laden

    # CSV einlesen
    # relevante Spalten auswählen

# 3 - Texte bereinigen

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