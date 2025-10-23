# Day 4: Feature Extraction using BoW and TF-IDF

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Load preprocessed dataset (from Day 3)
df = pd.read_csv('preprocessed_fake_job_postings.csv')

# Assume we already have the 'clean_description' column
texts = df['clean_description'].fillna('').tolist()

# 1️⃣ Bag-of-Words
bow_vectorizer = CountVectorizer(max_features=2000)  # limit to top 2000 words
X_bow = bow_vectorizer.fit_transform(texts)

print("BoW shape:", X_bow.shape)
print("Sample feature names (BoW):", bow_vectorizer.get_feature_names_out()[:10])

# 2️⃣ TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=2000)
X_tfidf = tfidf_vectorizer.fit_transform(texts)

print("\nTF-IDF shape:", X_tfidf.shape)
print("Sample feature names (TF-IDF):", tfidf_vectorizer.get_feature_names_out()[:10])

# 3️⃣ Compare sparsity and values
print("\nExample BoW vector (first row):")
print(X_bow[0].toarray())

print("\nExample TF-IDF vector (first row):")
print(X_tfidf[0].toarray())


# Task 1: BoW and TF-IDF for clean_company_profile
print("\n=== Task 1: Feature Extraction on clean_company_profile ===")

# Prepare texts from clean_company_profile
texts_profile = df['clean_company_profile'].fillna('').tolist()

# 1️⃣ Bag-of-Words for company_profile
bow_profile_vectorizer = CountVectorizer(max_features=2000)
X_bow_profile = bow_profile_vectorizer.fit_transform(texts_profile)

print("BoW shape (company_profile):", X_bow_profile.shape)
print("Sample feature names (BoW, company_profile):", bow_profile_vectorizer.get_feature_names_out()[:10])

# 2️⃣ TF-IDF for company_profile
tfidf_profile_vectorizer = TfidfVectorizer(max_features=2000)
X_tfidf_profile = tfidf_profile_vectorizer.fit_transform(texts_profile)

print("\nTF-IDF shape (company_profile):", X_tfidf_profile.shape)
print("Sample feature names (TF-IDF, company_profile):", tfidf_profile_vectorizer.get_feature_names_out()[:10])

# 3️⃣ Example vectors for comparison
print("\nExample BoW vector (first row, company_profile):")
print(X_bow_profile[0].toarray())

print("\nExample TF-IDF vector (first row, company_profile):")
print(X_tfidf_profile[0].toarray())

# Shape Comparison
print("\nShape Comparison:")
print("- BoW (description):", X_bow.shape)
print("- TF-IDF (description):", X_tfidf.shape)
print("- BoW (company_profile):", X_bow_profile.shape)
print("- TF-IDF (company_profile):", X_tfidf_profile.shape)


# Task 2: Top 20 most frequent words in job descriptions (using BoW)
print("\n=== Task 2: Top 20 Most Frequent Words in Job Descriptions (BoW) ===")

# Get total frequency of each word across all documents (sum of BoW columns)
word_frequencies = X_bow.sum(axis=0).A1  # .A1 flattens the sparse matrix to a 1D array
feature_names = bow_vectorizer.get_feature_names_out()

# Create a list of (word, frequency) pairs and sort descending
word_freq_pairs = list(zip(feature_names, word_frequencies))
top_20_words = sorted(word_freq_pairs, key=lambda x: x[1], reverse=True)[:20]

# Print top 20
print("Top 20 Most Frequent Words:")
for word, freq in top_20_words:
    print(f"{word}: {int(freq)}")