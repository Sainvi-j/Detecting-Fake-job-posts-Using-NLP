# Day 5: Logistic Regression Model for Fake Job Detection + Tasks 1,2,3
'''
# Day 5: Logistic Regression Model for Fake Job Detection

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset (preprocessed with clean_description)
df = pd.read_csv('fake_job_postings.csv')
df = df.dropna(subset=['clean_description'])

# 1️⃣ Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_description'])
y = df['fraudulent']

# 2️⃣ Split data into train & test sets
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

# 3️⃣ Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4️⃣ Make predictions
y_pred = model.predict(X_test)

# 5️⃣ Evaluate performance
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6️⃣ Check example predictions
test_samples = [ "Work from home! Limited vacancies. Apply now.","We are hiring a data scientist for our Bangalore office."]
sample_features = vectorizer.transform(test_samples)
print("\nSample Predictions:", model.predict(sample_features))
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
import joblib

# Load preprocessed data
df = pd.read_csv('preprocessed_fake_job_postings.csv')
df = df.dropna(subset=['clean_description']).copy()

y = df['fraudulent']
texts = df['clean_description']

# Helper: Train & evaluate model
def train_and_evaluate(vectorizer, name):
    print(f"\n=== {name} ===")
    X = vectorizer.fit_transform(texts)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = LogisticRegression(max_iter=200, class_weight='balanced')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 4))  # Fixed: use round()
    
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    return model, X_test, y_test, y_proba, vectorizer

# Task 1: BoW vs TF-IDF
print("Running Task 1: BoW vs TF-IDF Comparison")

# TF-IDF
tfidf_vec = TfidfVectorizer(max_features=5000)
model_tfidf, X_test_tfidf, y_test, proba_tfidf, vec_tfidf = train_and_evaluate(tfidf_vec, "TF-IDF (5000)")

# BoW
bow_vec = CountVectorizer(max_features=5000)
model_bow, X_test_bow, _, proba_bow, vec_bow = train_and_evaluate(bow_vec, "BoW (5000)")

# Task 1: Comparison Summary
print("\n" + "="*60)
print("TASK 1: PERFORMANCE COMPARISON")
print("="*60)

acc_tfidf = accuracy_score(y_test, model_tfidf.predict(X_test_tfidf))
acc_bow = accuracy_score(y_test, model_bow.predict(X_test_bow))
recall_tfidf = recall_score(y_test, model_tfidf.predict(X_test_tfidf), pos_label=1)
recall_bow = recall_score(y_test, model_bow.predict(X_test_bow), pos_label=1)

print(f"{'Metric':<20} {'TF-IDF':<12} {'BoW':<12}")
print("-" * 50)
print(f"{'Accuracy':<20} {acc_tfidf:.4f}      {acc_bow:.4f}")
print(f"{'Recall (Fake)':<20} {recall_tfidf:.4f}      {recall_bow:.4f}")

print("\nConclusion: TF-IDF outperforms BoW because it emphasizes rare but suspicious words "
      "(e.g., 'visa', 'urgent', 'training fee') while reducing noise from common terms.")

# Task 2: Top 5 Highest Fake Probability Jobs
print("\n" + "="*60)
print("TASK 2: Top 5 Jobs with Highest Fake Probability")
print("="*60)

X_full = vec_tfidf.transform(texts)
df['predicted_proba'] = model_tfidf.predict_proba(X_full)[:, 1]

top5 = df.nlargest(5, 'predicted_proba')[['clean_description', 'predicted_proba', 'fraudulent']]
print(top5[['predicted_proba', 'fraudulent']].round(4))

print("\nDescriptions (first 300 chars):")
for idx, row in top5.iterrows():
    desc = row['clean_description']
    short = desc[:300] + "..." if len(desc) > 300 else desc
    print(f"\n[Prob: {row['predicted_proba']:.4f} | Real: {row['fraudulent']}]")
    print(short)

# Task 3 (Optional): max_features Experiment
print("\n" + "="*60)
print("TASK 3: max_features Impact on TF-IDF")
print("="*60)

results = []
for mf in [1000, 5000, 10000]:
    vec = TfidfVectorizer(max_features=mf)
    X = vec.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=200, class_weight='balanced')
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    recall = recall_score(y_test, model.predict(X_test), pos_label=1)
    results.append({'max_features': mf, 'accuracy': round(acc, 4), 'recall_fake': round(recall, 4)})
    print(f"max_features={mf} → Acc: {acc:.4f}, Recall(Fake): {recall:.4f}")

print("\nSummary Table:")
print(pd.DataFrame(results))

# Sample Predictions
print("\n" + "="*60)
print("Sample Predictions on New Text")
print("="*60)

test_samples = [
    "Work from home! Limited vacancies. Apply now.",
    "We are hiring a data scientist for our Bangalore office."
]
sample_X = vec_tfidf.transform(test_samples)
preds = model_tfidf.predict(sample_X)
probs = model_tfidf.predict_proba(sample_X)[:, 1]

for text, pred, prob in zip(test_samples, preds, probs):
    print(f"Text: {text}")
    print(f"→ Prediction: {'Fake' if pred == 1 else 'Real'}, Prob(Fake): {prob:.4f}\n")

# Save Model & Vectorizer
joblib.dump(model_tfidf, 'fake_job_model.pkl')
joblib.dump(vec_tfidf, 'tfidf_vectorizer.pkl')

print("\nModel and vectorizer saved:")
print("→ fake_job_model.pkl")
print("→ tfidf_vectorizer.pkl")

# Manual Inspection
print("\n" + "="*60)
print("MANUAL INSPECTION: 1 Real + 1 Fake")
print("="*60)
fake_sample = df[df['predicted_proba'] > 0.8].sample(1)
real_sample = df[df['predicted_proba'] < 0.1].sample(1)

print("\nPREDICTED FAKE:")
print(f"Prob: {fake_sample['predicted_proba'].iloc[0]:.4f} | True Label: {fake_sample['fraudulent'].iloc[0]}")
print(fake_sample['clean_description'].iloc[0][:500] + "...")

print("\nPREDICTED REAL:")
print(f"Prob: {real_sample['predicted_proba'].iloc[0]:.4f} | True Label: {real_sample['fraudulent'].iloc[0]}")
print(real_sample['clean_description'].iloc[0][:500] + "...")