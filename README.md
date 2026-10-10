# Projekt Data Analysis: NLP-Techniken anwenden, um eine Textsammlung zu analysieren

## Projektbeschreibung

Dieses Projekt analysiert kommunale Bürgerbeschwerden mithilfe von Python und Topic Modeling. Ziel ist es, wiederkehrende Themen in Beschwerdetexten automatisch zu identifizieren.

Dazu werden die Texte mit NLTK vorverarbeitet, mittels Bag-of-Words (BOW) und TF-IDF vektorisiert und anschließend mit **LSA** und **LDA** analysiert. Die geeignete Themenanzahl wird anhand des **Coherence Scores (`c_v`)** bestimmt.

## Datensatz

Verwendet wird der [Civic and Municipality Complaint System Dataset](https://www.kaggle.com/datasets/wajahattaj/civic-and-municipality-complaint-system-dataset?select=municipal_training_set_1100.csv) von Kaggle.

- CSV-Datei: `municipal_training_set_1100.csv`
- Textspalte: `issue_description`
- Datensätze: 1.100
- Fehlende Beschwerdetexte: 0
- Doppelte Beschwerdetexte: 653

Die Duplikate können die Ergebnisse beeinflussen. Standardmäßig werden alle Datensätze berücksichtigt; ihre Entfernung ist optional möglich.

## Installation und Ausführung

**Voraussetzung:** Python 3.12

1. Repository herunterladen und beispielsweise in PyCharm öffnen.
2. Virtuelle Python-Umgebung erstellen bzw. auswählen.
3. Abhängigkeiten installieren:

```bash
python -m pip install -r requirements.txt
```

4. CSV-Datei im angegebenen Verzeichnis bereitstellen.
5. Python-Hauptprogramm ausführen.

Benötigte NLTK-Ressourcen werden beim Programmstart automatisch heruntergeladen.

Das Programm gibt Informationen zur Datenqualität, die ausgewählten Themenanzahlen, die wichtigsten Keywords und die Coherence Scores aus. Zusätzlich wird ein Diagramm zum Vergleich der Coherence Scores angezeigt.

## Konfiguration

Die Einstellungen können über Konstanten am Anfang des Python-Programms angepasst werden.

| Konstante | Standardwert | Beschreibung                                       |
|---|---|----------------------------------------------------|
| `DATA_PATH` | `"municipal_training_set_1100.csv"` | Pfad zur CSV-Datei                                 |
| `ISSUE_COLUMN` | `"issue_description"` | Name der Spalte, welche die Beschwerden beinhaltet |
| `TOPIC_NUMBERS` | `range(2, 16)` | Zu testende Themenanzahlen (2–16)                  |
| `COHERENCE_WORDS` | `10` | Worte pro Thema für die Coherence-Berechnung       |
| `DISPLAY_WORDS` | `5` | Angezeigte Worte pro Thema                         |
| `PRINT_SCORES` | `False` | Einzelne Coherence Scores zusätzlich ausgeben      |
| `KEEP_DUPLICATES` | `True` | Doppelte Beschwerdetexte beibehalten               |

**Andere Datensätze:** Über `DATA_PATH` und `ISSUE_COLUMN` können eine andere CSV-Datei und Textspalte ausgewählt werden. Die Textspalte muss vorhanden sein und auswertbare Texte ohne fehlende Werte enthalten.

Die aktuelle Textvorverarbeitung ist auf **englischsprachige Texte** ausgelegt.

## Ablauf der Analyse

1. **Datenprüfung:** Anzahl der Einträge, fehlende Texte, Duplikate und durchschnittliche Wortanzahl ermitteln.
2. **Textvorverarbeitung:** Kleinbuchstaben, Satzzeichenentfernung, Tokenisierung, Stoppwortentfernung und Lemmatisierung.
3. **Vektorisierung:** BOW mit `CountVectorizer` und TF-IDF mit `TfidfVectorizer`.
4. **Themenmodellierung:** LSA (`TruncatedSVD`) auf TF-IDF und LDA (`LatentDirichletAllocation`) auf BOW.
5. **Bewertung:** Coherence Scores für verschiedene Themenanzahlen berechnen und grafisch vergleichen.
6. **Ergebnisausgabe:** Ausgewählte Themen mit den Top-5-Worten anzeigen.

## Verwendete Bibliotheken

`pandas`, `nltk`, `scikit-learn`, `gensim`, `matplotlib`

Die benötigten Versionen sowie Abhängigkeiten dieser Bibliotheken sind in `requirements.txt` dokumentiert.
