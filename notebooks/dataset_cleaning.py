# Day 3: Text Cleaning and Preprocessing

import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset (same as Day 2)
df = pd.read_csv('fake_job_postings.csv')

# Define text cleaning function
def clean_text(text):
    if pd.isnull(text):
        return ""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # 4. Remove punctuation and numbers
    text = re.sub(r'[%s\d]' % re.escape(string.punctuation), ' ', text)
    # 5. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # 6. Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

# Apply cleaning to key text columns
df['clean_description'] = df['description'].apply(clean_text)

# Show before and after
print("Original Text:\n", df['description'].iloc[1][:300])
print("\nCleaned Text:\n", df['clean_description'].iloc[1][:300])

# Check for any remaining issues
print("\nExample of Cleaned Data:")
print(df[['description', 'clean_description']].head(3))

# Define cleaning function for company_profile
def clean_company_profile(text):
    if pd.isnull(text):
        return ""
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # 4. Remove punctuation and numbers
    text = re.sub(r'[%s\d]' % re.escape(string.punctuation), ' ', text)
    # 5. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # 6. Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

# Apply cleaning to company_profile column
df['clean_company_profile'] = df['company_profile'].apply(clean_company_profile)

# Show before and after for company_profile
print("\nOriginal Company Profile (first 3 rows):\n", df['company_profile'].head(3))
print("\nCleaned Company Profile (first 3 rows):\n", df['clean_company_profile'].head(3))

# Function to count words in a text string
def count_words(text):
    if pd.isnull(text) or text == "":
        return 0
    return len(text.split())

# Calculate word counts before and after cleaning
df['word_count_before'] = df['description'].apply(count_words)
df['word_count_after'] = df['clean_description'].apply(count_words)

# Compute average word counts
avg_words_before = df['word_count_before'].mean()
avg_words_after = df['word_count_after'].mean()

# Print results
print("\nAverage Word Count Before Cleaning (description):", round(avg_words_before, 2))
print("Average Word Count After Cleaning (description):", round(avg_words_after, 2))
print("Percentage of Words Retained:", round((avg_words_after / avg_words_before) * 100, 2), "%")

# Save preprocessed dataset for Day 4
df.to_csv('preprocessed_fake_job_postings.csv', index=False)
print("\nPreprocessed dataset saved to 'preprocessed_fake_job_postings.csv'")