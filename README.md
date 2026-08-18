# Fake News Detection

An NLP project that explores, analyzes, and classifies news articles as **Fake News** or **Factual News** using linguistic analysis, topic modeling, sentiment analysis, and machine learning classification.

## Problem Statement

A social media company is concerned about the growing spread of fake news on its platform. This project investigates how fake news can be identified — first by exploring and cleaning the data, then by building a classification model to distinguish fake from factual news stories.

## Dataset

`fake_news_data.csv` — news articles with the following columns:
- `title` — article headline
- `text` — full article text
- `date` — publication date
- `fake_or_factual` — target label (`Fake News` / `Factual News`)

## Workflow

### 1. Exploratory Data Analysis
- Load and inspect the dataset
- Visualize the class balance between fake and factual articles

### 2. Part-of-Speech (POS) Tagging
- Use **spaCy** to tag tokens with POS and named entity labels for fake vs. factual articles separately
- Compare frequency of POS tags (nouns, etc.) between the two classes

### 3. Named Entity Recognition (NER)
- Extract and compare the most common named entities (people, organizations, locations, dates) in fake vs. factual news
- Visualize with bar plots

### 4. Text Preprocessing
- Strip leading location/source tags via regex
- Lowercase, remove punctuation
- Remove stopwords (NLTK)
- Tokenize and lemmatize (WordNet Lemmatizer)
- Analyze most common unigrams and bigrams after cleaning

### 5. Sentiment Analysis
- Score article sentiment using **VADER** (`vaderSentiment`)
- Bucket into negative / neutral / positive
- Compare sentiment distribution between fake and factual news

### 6. Topic Modeling
- **LDA** (Latent Dirichlet Allocation) topic modeling on fake news articles, with coherence score tuning to select the optimal number of topics
- **LSA/TF-IDF** (Latent Semantic Analysis with TF-IDF vectorization) as an alternative topic modeling approach

### 7. Classification — Fake vs. Factual News
- Vectorize cleaned text using **CountVectorizer** (bag-of-words)
- Train/test split
- **Logistic Regression** classifier
- **SGD Classifier** (linear SVM) classifier
- Evaluate both models using accuracy score and classification report (precision, recall, F1)

## Tech Stack

- **Python**, `pandas`, `matplotlib`, `seaborn`
- **spaCy** (`en_core_web_sm`) — POS tagging, NER
- **NLTK** — tokenization, stopwords, lemmatization, n-grams
- **VADER Sentiment** — sentiment scoring
- **Gensim** — LDA & LSA topic modeling, coherence scoring
- **scikit-learn** — CountVectorizer, TF-IDF, Logistic Regression, SGDClassifier, evaluation metrics

## Setup

```bash
pip install pandas matplotlib seaborn spacy nltk vaderSentiment gensim scikit-learn
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords punkt wordnet
```

## Usage

1. Place `fake_news_data.csv` in the working directory
2. Run `Fake_news_dec.ipynb` sequentially:
   - EDA → POS/NER analysis → text preprocessing → sentiment analysis → topic modeling → classification
3. Review classification reports at the end of the notebook to compare model performance

## Results

| Model | Metric |
|---|---|
| Logistic Regression | Accuracy, Precision, Recall, F1 (see classification report) |
| SGD Classifier (linear SVM) | Accuracy, Precision, Recall, F1 (see classification report) |

*(Fill in actual accuracy/F1 scores from your notebook run for a quantified comparison.)*

## Key Learnings

- Fake and factual news differ in named entity usage, POS distribution, and sentiment tone
- Simple bag-of-words features with classical ML models (Logistic Regression, SGD) can effectively distinguish fake from factual news
- Topic modeling (LDA/LSA) helps surface dominant themes within fake news articles for further investigation

## Project Structure

```
├── Fake_news_dec.ipynb     # Main notebook
├── fake_news_data.csv      # Dataset
└── README.md
```

## Author

**Soma Sundar Kalla**
[LinkedIn](https://linkedin.com/in/soma-sundar-kalla-08592b389/) · [GitHub](https://github.com/somasundar0708)
