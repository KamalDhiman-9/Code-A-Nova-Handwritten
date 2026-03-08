# 💬 Sentiment Analysis System
### NLP Project | Text Classification using Machine Learning & Python

---

## 📌 What This Project Does
This system analyzes any text input — reviews, tweets, feedback, or social media posts —
and classifies the underlying sentiment as **Positive**, **Negative**, or **Neutral**
using NLP techniques and Machine Learning models.

---

## 📁 Project Files
| File | Description |
|------|-------------|
| `sentiment_analysis.py` | Main Python script (full pipeline) |
| `sentiment_distribution.png` | Generated: class balance chart |
| `confusion_matrix.png` | Generated: model accuracy grid |
| `model_comparison.png` | Generated: all models compared |

---

## ⚙️ Setup & Installation

### 1. Install Required Libraries
```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```

### 2. Run the Project
```bash
python sentiment_analysis.py
```

### 3. Use Your Own Dataset
Replace the sample data section with:
```python
df = pd.read_csv("your_dataset.csv")
# CSV must have columns: 'text' and 'sentiment'
# Sentiment values: 'positive', 'negative', 'neutral'
```

---

## 🧠 How It Works — Full Pipeline

```
Raw Text Input
     ↓
[ TEXT PREPROCESSING ]
  • Lowercase conversion
  • Remove URLs, mentions, hashtags
  • Remove punctuation & numbers
  • Tokenization
  • Stopword removal
  • Lemmatization
     ↓
[ FEATURE EXTRACTION ]
  • TF-IDF Vectorizer
  • Unigrams + Bigrams
  • Top 5000 features
     ↓
[ MACHINE LEARNING MODELS ]
  • Logistic Regression
  • Naive Bayes
  • Linear SVM
     ↓
[ OUTPUT ]
  😊 Positive | 😞 Negative | 😐 Neutral
```

---

## 🔑 Key Concepts Explained

| Concept | What It Means |
|---------|---------------|
| **Tokenization** | Splitting text into individual words |
| **Stopwords** | Common words removed (e.g., "the", "is", "and") |
| **Lemmatization** | Reducing words to base form ("running" → "run") |
| **TF-IDF** | Scores words by importance across all documents |
| **Logistic Regression** | Predicts probability of each class |
| **Naive Bayes** | Probabilistic classifier, great for text |
| **Linear SVM** | Finds the best boundary between classes |
| **Confusion Matrix** | Table showing correct vs wrong predictions |

---

## 📊 Expected Results
| Model | Expected Accuracy |
|-------|------------------|
| Logistic Regression | ~90–95% |
| Naive Bayes | ~85–92% |
| Linear SVM | ~90–96% |

---

## 🔮 Predict Your Own Text
At the bottom of `sentiment_analysis.py`, uncomment:
```python
interactive_mode()
```
Then run the script and type any text to get instant predictions!

---

## 🚀 Next Steps to Improve
1. **Use a larger dataset** — Try IMDB (50K reviews) or Twitter datasets from Kaggle
2. **Deep Learning** — Replace TF-IDF + ML with LSTM or BERT for higher accuracy
3. **Web App** — Build a Flask/Streamlit interface for a browser-based tool
4. **Aspect-based Sentiment** — Detect sentiment for specific aspects (price, quality, etc.)

---

## 🐛 Common Issues
| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install <module_name>` |
| NLTK errors | The script auto-downloads required NLTK data |
| Low accuracy | Add more training samples to the dataset |
| Unicode errors | Add `encoding='utf-8'` when reading CSV files |
