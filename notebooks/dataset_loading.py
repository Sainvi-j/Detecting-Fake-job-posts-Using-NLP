# Day 2: Understanding and Loading the Dataset
 
import pandas as pd
 
# Load dataset (after downloading from Kaggle)
df = pd.read_csv('fake_job_postings.csv')
 
# Display first few rows
print("Sample Data:")
print(df.head())
 
# Display basic info
print("\nDataset Info:")
print(df.info())
 
# Check for missing values
print("\nMissing Values per Column:")
print(df.isnull().sum())
 
# Check distribution of target variable
print("\nTarget (fraudulent) Distribution:")
print(df['fraudulent'].value_counts())
 
# Basic statistics
print("\nDataset Summary:")
print(df.describe(include='all'))

# Filter for fake jobs (fraudulent == 1) and select the description column
fake_jobs = df[df['fraudulent'] == 1][['job_id', 'title', 'description']].head(3)

# Print 3 examples of fake job descriptions
print("\nThree Examples of Fake Job Descriptions:")
for index, row in fake_jobs.iterrows():
    print(f"\nJob ID: {row['job_id']}")
    print(f"Title: {row['title']}")
    print(f"Description: {row['description']}")