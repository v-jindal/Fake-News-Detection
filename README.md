# Fake News Detection System

A Machine Learning and Natural Language Processing (NLP) based application that classifies news articles as **Real** or **Fake**. The project demonstrates an end-to-end text classification pipeline, including data preprocessing, feature extraction, model training, and evaluation using supervised machine learning techniques.

---

## Overview

The rapid spread of misinformation has made automated fake news detection an important real-world problem. This project uses Natural Language Processing (NLP) and Machine Learning to analyze textual content and predict whether a news article is genuine or misleading.

The implementation follows a complete machine learning workflow—from cleaning raw text to training a classification model—and is designed to be modular, readable, and easy to extend.

---

## Features

- Binary classification of news articles (Real/Fake)
- Text preprocessing and cleaning using NLTK
- TF-IDF feature extraction
- Logistic Regression classifier
- Model evaluation using standard performance metrics
- Clean and modular project structure
- Easily extendable with additional ML models

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| NLP | NLTK |
| Feature Engineering | TF-IDF Vectorizer (Scikit-learn) |
| Machine Learning | Logistic Regression |
| Visualization | Matplotlib |

---

## Project Workflow

```
Dataset
   │
   ▼
Text Preprocessing
   │
   ▼
Tokenization & Stopword Removal
   │
   ▼
TF-IDF Vectorization
   │
   ▼
Train-Test Split
   │
   ▼
Model Training
(Logistic Regression)
   │
   ▼
Prediction
   │
   ▼
Performance Evaluation
```

---

## Project Structure

```
fake-news-detection/
│
├── data/
│   └── Dataset files
│
├── fake_news.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/v-jindal/fake-news-detection.git
cd fake-news-detection
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python fake_news.py
```

---

## Model

This project uses **Logistic Regression** as the primary classification model.

Logistic Regression was selected because it:

- Performs efficiently on high-dimensional sparse text data
- Provides fast training and prediction
- Serves as a strong baseline for text classification tasks
- Produces interpretable results

The project can also be extended to compare other classifiers such as:

- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

---

## Evaluation

The trained model is evaluated using standard classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Model performance depends on the dataset, preprocessing techniques, and feature engineering strategy.

---

## Future Enhancements

Some possible improvements include:

- Deploy the model using Flask or Streamlit
- Save trained models using Pickle or Joblib
- Integrate transformer-based models such as BERT
- Build a REST API for inference
- Support real-time news classification using external news APIs

---

## Applications

The system can be applied in several real-world scenarios, including:

- News verification platforms
- Social media content moderation
- Journalism and media research
- Educational demonstrations of NLP techniques
- Misinformation detection systems

---

## Learning Outcomes

Through this project, I gained practical experience with:

- Natural Language Processing
- Text preprocessing techniques
- TF-IDF feature engineering
- Supervised machine learning
- Binary text classification
- Model evaluation and performance analysis
- Building modular and maintainable Python applications

---

## Author

**Vanshika Jindal**

GitHub: https://github.com/v-jindal

LinkedIn: https://linkedin.com/in/vanshika-jindal-b1b3b6272

---

## License

This project is intended for educational and learning purposes.
