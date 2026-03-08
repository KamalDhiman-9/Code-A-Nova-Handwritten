# ============================================================
#   Sentiment Analysis System
#   NLP Project | Text Classification using Machine Learning
#   Supports: Positive / Negative / Neutral Sentiments
# ============================================================

# ── STEP 1: Install & Import Libraries ───────────────────────
# Run this in terminal before running the script:
#   pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

# Download required NLTK data
print("📥 Downloading NLTK resources...")
nltk.download('stopwords',    quiet=True)
nltk.download('wordnet',      quiet=True)
nltk.download('punkt',        quiet=True)
nltk.download('punkt_tab',    quiet=True)
nltk.download('omw-1.4',      quiet=True)
print("✅ NLTK resources ready.\n")


# ── STEP 2: Create Sample Dataset ────────────────────────────
# In a real project, replace this with your own CSV file:
#   df = pd.read_csv("your_dataset.csv")
# The CSV should have two columns: 'text' and 'sentiment'

def create_sample_dataset():
    """Create a small labeled dataset of reviews with sentiments."""
    data = {
        "text": [
            # Positive samples
            "This product is absolutely amazing! I love it so much.",
            "Great quality, fast delivery. Highly recommend to everyone!",
            "The movie was fantastic, I enjoyed every single moment.",
            "Excellent customer service, they resolved my issue quickly.",
            "Best purchase I have ever made. Works perfectly!",
            "Outstanding performance and beautiful design. Very happy!",
            "I am so satisfied with this. Exceeded all my expectations.",
            "Wonderful experience from start to finish. Will buy again.",
            "The food was delicious and the staff were very friendly.",
            "Superb quality! This is exactly what I was looking for.",
            "Loved the storyline, characters were brilliant and engaging.",
            "Very impressed with the build quality. Worth every penny.",
            "Incredible value for money. Definitely buying again!",
            "Five stars! Absolutely perfect in every way possible.",
            "This made my day so much better. Truly outstanding!",

            # Negative samples
            "Terrible product, broke after just one day of use.",
            "Worst experience ever. I want a full refund immediately.",
            "Very disappointed. The quality is extremely poor and cheap.",
            "Do not buy this! Complete waste of money and time.",
            "Awful customer support. They ignored all my complaints.",
            "The movie was boring and the plot made no sense at all.",
            "Disgusting food, I felt sick after eating it. Never again.",
            "Package arrived damaged and the item was completely broken.",
            "This is a total scam. Nothing like the pictures online.",
            "Horrible smell and the material feels very cheap and nasty.",
            "Stopped working after a week. Very frustrating experience.",
            "The worst product I have ever purchased in my entire life.",
            "Rude staff and extremely long waiting time. Very unhappy.",
            "Completely useless. Instructions were confusing and wrong.",
            "I regret buying this. Save your money and avoid this item.",

            # Neutral samples
            "The product is okay, nothing special about it really.",
            "It works as described, neither great nor terrible honestly.",
            "Delivery was on time. The item is average in quality.",
            "It is a decent product for the price. Could be better.",
            "The movie was fine, had some good and some bad moments.",
            "Neither impressed nor disappointed. It does the job well.",
            "Received the order. It matches the description on the site.",
            "Average quality product. There are better options available.",
            "It is acceptable for everyday use but nothing outstanding.",
            "The service was standard. No complaints but nothing special.",
            "Works as expected. Not the best but certainly not the worst.",
            "The food was edible. Nothing memorable about the experience.",
            "Okay product overall. Would only recommend if on a budget.",
            "It is a basic product with basic features, nothing more.",
            "The experience was ordinary. Would not go out of my way.",
        ],
        "sentiment": (
            ["positive"] * 15 +
            ["negative"] * 15 +
            ["neutral"]  * 15
        )
    }
    return pd.DataFrame(data)


df = create_sample_dataset()

print("── Dataset Overview ──")
print(f"Total samples : {len(df)}")
print(f"Columns       : {list(df.columns)}")
print(f"\nSentiment Distribution:\n{df['sentiment'].value_counts()}")
print(f"\nSample rows:")
print(df.head(3).to_string(index=False))


# ── STEP 3: Exploratory Data Analysis (EDA) ──────────────────
def plot_sentiment_distribution(df):
    """Bar chart of sentiment class counts."""
    colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#3498db'}
    counts = df['sentiment'].value_counts()

    plt.figure(figsize=(7, 4))
    bars = plt.bar(counts.index, counts.values,
                   color=[colors[s] for s in counts.index], edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(val), ha='center', fontweight='bold', fontsize=12)
    plt.title("Sentiment Class Distribution", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(fontsize=11)
    plt.tight_layout()
    plt.savefig("sentiment_distribution.png", dpi=100)
    plt.show()
    print("✅ Saved: sentiment_distribution.png")

plot_sentiment_distribution(df)


# ── STEP 4: Text Preprocessing ───────────────────────────────
# Clean and normalize text before feeding it to the model

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Full NLP preprocessing pipeline:
    1. Lowercase
    2. Remove URLs, mentions, hashtags
    3. Remove punctuation & numbers
    4. Tokenize
    5. Remove stopwords
    6. Lemmatize
    """
    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs, mentions (@user), hashtags (#tag)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)

    # 3. Remove punctuation and digits
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", "", text)

    # 4. Tokenize
    tokens = word_tokenize(text)

    # 5. Remove stopwords & short tokens
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # 6. Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


print("\n── Text Preprocessing Example ──")
sample = "This product is absolutely amazing! I love it SO much!! 😊"
print(f"Original : {sample}")
print(f"Cleaned  : {preprocess_text(sample)}")

# Apply preprocessing to entire dataset
df['cleaned_text'] = df['text'].apply(preprocess_text)
print("\n✅ Preprocessing complete.")
print(df[['text', 'cleaned_text', 'sentiment']].head(3).to_string(index=False))


# ── STEP 5: Feature Extraction with TF-IDF ───────────────────
# TF-IDF (Term Frequency–Inverse Document Frequency)
# Converts text into numerical vectors the model can understand

print("\n── Feature Extraction (TF-IDF) ──")

X = df['cleaned_text']
y = df['sentiment']

# Encode labels
label_map    = {'positive': 2, 'neutral': 1, 'negative': 0}
label_decode = {v: k for k, v in label_map.items()}
y_encoded    = y.map(label_map)

# Train/Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,    # Use top 5000 words
    ngram_range=(1, 2),   # Unigrams + bigrams (e.g., "not good")
    sublinear_tf=True     # Apply log normalization
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"TF-IDF matrix shape (train): {X_train_tfidf.shape}")


# ── STEP 6: Train Multiple Models ────────────────────────────
# Compare three classic ML classifiers for NLP

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes"        : MultinomialNB(alpha=0.5),
    "Linear SVM"         : LinearSVC(random_state=42, max_iter=2000),
}

results = {}

print("\n── Model Training & Evaluation ──")
for name, clf in models.items():
    clf.fit(X_train_tfidf, y_train)
    preds    = clf.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, preds)
    results[name] = {"model": clf, "predictions": preds, "accuracy": accuracy}
    print(f"{name:<25} → Accuracy: {accuracy * 100:.2f}%")


# ── STEP 7: Detailed Evaluation ──────────────────────────────
best_name = max(results, key=lambda k: results[k]["accuracy"])
best_info = results[best_name]

print(f"\n🏆 Best Model: {best_name}  ({best_info['accuracy']*100:.2f}%)")
print("\n── Classification Report ──")
print(classification_report(
    y_test,
    best_info["predictions"],
    target_names=["negative", "neutral", "positive"]
))


# ── STEP 8: Confusion Matrix ─────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm     = confusion_matrix(y_true, y_pred)
    labels = ["negative", "neutral", "positive"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor='gray')
    plt.title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Predicted", fontsize=11)
    plt.ylabel("Actual", fontsize=11)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=100)
    plt.show()
    print("✅ Saved: confusion_matrix.png")

plot_confusion_matrix(y_test, best_info["predictions"], best_name)


# ── STEP 9: Model Comparison Chart ───────────────────────────
def plot_model_comparison(results):
    names  = list(results.keys())
    scores = [results[n]["accuracy"] * 100 for n in names]
    colors = ['#3498db', '#2ecc71', '#e67e22']

    plt.figure(figsize=(8, 4))
    bars = plt.barh(names, scores, color=colors, edgecolor='white', height=0.5)
    for bar, score in zip(bars, scores):
        plt.text(bar.get_width() - 3, bar.get_y() + bar.get_height() / 2,
                 f"{score:.1f}%", va='center', ha='right',
                 color='white', fontweight='bold', fontsize=12)
    plt.xlim(0, 110)
    plt.xlabel("Accuracy (%)", fontsize=11)
    plt.title("Model Comparison", fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=100)
    plt.show()
    print("✅ Saved: model_comparison.png")

plot_model_comparison(results)


# ── STEP 10: Real-Time Prediction Function ───────────────────
best_model = best_info["model"]

def predict_sentiment(text, model=best_model, vectorizer=tfidf):
    """
    Predict the sentiment of any input text.

    Parameters:
        text      : raw input string (review, tweet, feedback, etc.)
        model     : trained classifier
        vectorizer: fitted TF-IDF vectorizer

    Returns:
        dict with 'sentiment' and 'cleaned_text'
    """
    cleaned   = preprocess_text(text)
    vector    = vectorizer.transform([cleaned])
    pred      = model.predict(vector)[0]
    sentiment = label_decode[pred]

    emoji_map = {'positive': '😊 Positive', 'negative': '😞 Negative', 'neutral': '😐 Neutral'}
    return {
        "input_text"   : text,
        "cleaned_text" : cleaned,
        "sentiment"    : emoji_map[sentiment],
    }


# ── STEP 11: Test with Custom Reviews ────────────────────────
print("\n── Live Sentiment Predictions ──")

test_reviews = [
    "I absolutely love this product! It works perfectly and arrived on time.",
    "This is the worst thing I have ever bought. Complete disappointment.",
    "The item is okay. Nothing special but it gets the job done.",
    "Horrible quality! Fell apart after two days. Very angry!",
    "Decent value for the price. Would consider buying again maybe.",
    "Outstanding service! The team was helpful and very professional.",
]

for review in test_reviews:
    result = predict_sentiment(review)
    print(f"\n📝 Review   : {result['input_text'][:70]}...")
    print(f"   Cleaned  : {result['cleaned_text'][:60]}...")
    print(f"   Sentiment: {result['sentiment']}")


# ── STEP 12: Interactive Prediction Loop ─────────────────────
def interactive_mode():
    """
    Run an interactive loop where the user types text
    and gets instant sentiment predictions.
    """
    print("\n" + "="*55)
    print("   🎯 INTERACTIVE SENTIMENT ANALYSIS SYSTEM")
    print("   Type any review/tweet/feedback and press Enter.")
    print("   Type 'quit' to exit.")
    print("="*55)

    while True:
        user_input = input("\n📝 Enter text: ").strip()
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("👋 Goodbye!")
            break
        if not user_input:
            print("⚠️  Please enter some text.")
            continue
        result = predict_sentiment(user_input)
        print(f"   🔍 Sentiment : {result['sentiment']}")
        print(f"   📄 Cleaned   : {result['cleaned_text']}")

# Uncomment the line below to run the interactive mode:
# interactive_mode()


# ── DONE ─────────────────────────────────────────────────────
print("\n" + "="*55)
print("🎉 Sentiment Analysis System — Complete!")
print("="*55)
print("\nFiles generated:")
print("  • sentiment_distribution.png  — class balance chart")
print("  • confusion_matrix.png        — prediction accuracy grid")
print("  • model_comparison.png        — all models side-by-side")
print("\nTo run interactively, uncomment the last line:")
print("  interactive_mode()")
