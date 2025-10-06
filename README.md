# Movie Reviews Sentiment Analysis

## Classifying movie reviews as positive or negative using NLP and machine learning

### 🧭 Project Overview

This project applies sentiment analysis techniques to a set of movie reviews to determine whether a review expresses a positive or negative sentiment. The aim is to build, compare, and evaluate models (rule-based, lexicon, machine learning, transformer) that can predict sentiment from raw text, and to derive insights about their strengths and limitations.

### 📁 Dataset

Sourced from NLTK’s Movie Reviews Corpus (2,000 total reviews; 1,000 positive, 1,000 negative)

Data columns:

ID (unique identifier)

review (full text)

sentiment (ground truth label: “positive” or “negative”)

Balanced label distribution ensures fairness in training and evaluation

### 🛠️ Approach & Methods

Preprocessing

Tokenization (splitting into words)

Cleaning: removing stopwords, punctuation, non-alphabetic tokens

Rule / Lexicon-Based Sentiment (Baseline)

VADER: compute polarity scores (neg, neu, pos, compound)

TextBlob: compute sentiment polarity

Machine Learning Model

Train/test split (80 / 20)

Vectorization via CountVectorizer (bag-of-words)

Train MultinomialNB (Naïve Bayes) classifier

Predict and evaluate (confusion matrix, classification metrics)

Transformer / Deep Learning (Optional / Future Extension)

Use Hugging Face sentiment pipeline (e.g. BERT)

Compare results with baseline methods

### 📊 Results & Key Metrics

VADER / Lexicon methods delivered fast, interpretable sentiment scoring

Naïve Bayes model showed reasonable accuracy in prediction, with confusion matrix highlighting misclassifications

Metrics such as precision, recall, F1-score used to assess performance across classes

Comparative analysis revealed trade-offs:

Lexicon methods are lightweight and interpretable

ML methods can better capture patterns but need more data and tuning

### ⚠️ Limitations & Challenges

Small dataset (2,000 reviews) limits model generalizability

Lexicon methods do not capture sarcasm, idioms, or context well

Bag-of-words ignores word order and semantics

Transformer / deep models may perform better but require more computation

Domain shift: models may not transfer to reviews from different genres or styles

### 🔮 Future Directions

Use TF-IDF, word embeddings (Word2Vec, GloVe), or Transformer embeddings (BERT, RoBERTa)

Fine-tune pretrained transformer models for sentiment classification

Expand dataset (e.g. IMDb, Rotten Tomatoes) to increase coverage

Introduce neutral sentiment class (three-label classification)

Analyze sentiment trends over time or across genres

### 🚀 Getting Started / How to Use

Clone this repository

Install dependencies (nltk, scikit-learn, textblob, transformers)

Download / place movie_reviews_dataset.csv in data folder

Run notebook / scripts in order: preprocessing → baseline methods → ML model → evaluation
