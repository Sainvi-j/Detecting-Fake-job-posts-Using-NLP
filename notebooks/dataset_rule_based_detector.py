# task4 rule based detector
import pandas as pd
import re

# 1. Load cleaned data
df = pd.read_csv('preprocessed_fake_job_postings.csv')
df = df[['clean_description','fraudulent']].copy()

# 2. Suspicious keyword list (add/remove as you like)
SUSPICIOUS = [
    'urgent', 'work from home', 'limited vacancy', 'visa',
    'investment', 'training fee', 'money transfer',
    'guaranteed income', 'no experience needed', 'immediate start',
    'cash daily', 'register now', 'pay to join'
]

# Compile a single regex for speed
pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, SUSPICIOUS)) + r')\b', re.IGNORECASE)

# 3. Rule-based flag function
def rule_based_flag(text):
    if pd.isnull(text):
        return 0
    return 1 if pattern.search(text) else 0

df['suspect_flag'] = df['clean_description'].apply(rule_based_flag)

# 4. Crosstab – overlap with ground truth
print("\n=== Rule-based vs Ground Truth ===")
print(pd.crosstab(df['suspect_flag'], df['fraudulent'],
                  normalize='all').round(4)*100)

# 5. False positives (suspect but NOT fraudulent)
false_pos = df[(df['suspect_flag']==1) & (df['fraudulent']==0)]
print(f"\n{len(false_pos)} suspect jobs that are actually REAL.")
print("Sample of 5 false positives:")
print(false_pos['clean_description'].head(5).tolist())

# Discussion (print for the report)
print("\n--- Why rules are limited ---")
print("""1. Many legitimate remote-work jobs contain “work from home”.
2. Some genuine startups use “urgent” or “immediate start”.
3. Scammers often avoid exact keywords → subtle phrasing.
4. Rules cannot capture context or sarcasm.
=> Machine-learning models that learn patterns from the whole vocabulary are needed.""")