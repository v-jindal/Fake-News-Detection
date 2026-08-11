# Fake News Detection System  
### Machine Learning + NLP Powered News Classifier

An intelligent **Machine Learning-based Fake News Detection System** that classifies news articles as **REAL** or **FAKE** using advanced Natural Language Processing (NLP) techniques.

This project demonstrates practical implementation of text preprocessing, TF-IDF feature extraction, and supervised ML classification models to solve real-world misinformation problems.

---

## Repository

https://github.com/v-jindal/fake-news-detection

---

## Key Highlights

End-to-End NLP Pipeline  
TF-IDF Feature Engineering  
Logistic Regression Model Implementation  
Clean & Modular Code Structure  
High Accuracy Text Classification  
Ready for Deployment & Scaling  

---

## Machine Learning Model Used

### Logistic Regression (Primary Model)

Why Logistic Regression?
- Works efficiently with high-dimensional sparse data (like TF-IDF)
- Fast training time
- Strong baseline for text classification
- Interpretable results

You can also experiment with:
- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest

---

## ⚙️ Project Workflow

1️. Data Loading  
2️. Text Cleaning & Preprocessing  
3️. Stopword Removal (NLTK)  
4️. TF-IDF Vectorization  
5️. Train-Test Split  
6️. Model Training (Logistic Regression)  
7️. Accuracy & Performance Evaluation  

---

## 🛠️ Tech Stack

| Category | Tools Used |
|----------|------------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| NLP | NLTK |
| Feature Extraction | TF-IDF (Scikit-learn) |
| ML Model | Logistic Regression |
| Visualization | Matplotlib |

---

## Project Structure

```
fake-news-detection/
│
├── data/                  # Dataset files
├── fake_news.py           # Main Python script
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/v-jindal/fake-news-detection.git
cd fake-news-detection
```

### Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python fake_news.py
```

---

## Model Performance

✔ Strong classification accuracy  
✔ Efficient handling of large text data  
✔ Optimized for binary classification (REAL vs FAKE)

> Accuracy may vary depending on dataset quality and preprocessing techniques.

---

## Future Enhancements

- Deploy as Web App (Flask / Streamlit)
- Add Model Saving (.pkl)
- Implement Deep Learning (LSTM / BERT)
- Add Real-time News API
- Create REST API for production usage

---

## Real-World Applications

- News verification platforms  
- Social media misinformation detection  
- Journalism research tools  
- Content moderation systems  

---

## Author

**Vanshika Jindal**  
GitHub: https://github.com/v-jindal  

---

If you found this project useful, consider giving it a ⭐ on GitHub!
