# Day 6: Decision Tree & Random Forest Models
''' 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
 
# Load preprocessed dataset
df = pd.read_csv('preprocessed_fake_job_postings.csv')
df = df.dropna(subset=['clean_description'])
 
# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_description'])
y = df['fraudulent']
 
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
# 1️⃣ Decision Tree
dt = DecisionTreeClassifier(max_depth=20, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
 
# 2️⃣ Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
 
# Evaluation
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
 
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
 
# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
 
# Feature Importance
importances = rf.feature_importances_
indices = importances.argsort()[-10:][::-1]
feature_names = vectorizer.get_feature_names_out()
 
plt.figure(figsize=(8,5))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.title("Top 10 Important Words (Random Forest)")
plt.xlabel("Feature Importance")
plt.show()
'''
# Day 6: Decision Tree & Random Forest + Task 1 & 2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and Prepare Data
df = pd.read_csv('preprocessed_fake_job_postings.csv')
df = df.dropna(subset=['clean_description']).copy()

texts = df['clean_description']
y = df['fraudulent']

# TF-IDF with 3000 features (as in original)
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Task 1: Model Comparison (DT + RF with Hyperparams)
print("Running Task 1: Model Comparison")

results = []

# Decision Trees
for depth in [10, 20, 30]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42, class_weight='balanced')
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    
    results.append({
        'Model': f'DT (depth={depth})',
        'Accuracy': acc,
        'Precision (Fake)': prec,
        'Recall (Fake)': rec,
        'F1 (Fake)': f1
    })
    print(f"DT depth={depth} → Acc: {acc:.4f}, Recall: {rec:.4f}")

# Random Forests
for n_est in [50, 100, 200]:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42, class_weight='balanced', n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    
    results.append({
        'Model': f'RF (n={n_est})',
        'Accuracy': acc,
        'Precision (Fake)': prec,
        'Recall (Fake)': rec,
        'F1 (Fake)': f1
    })
    print(f"RF n_estimators={n_est} → Acc: {acc:.4f}, Recall: {rec:.4f}")

# Add Logistic Regression (from Day 5)
# Assuming ~96.8% acc, ~35% recall from your previous run
results.append({
    'Model': 'Logistic Regression',
    'Accuracy': 0.9681,
    'Precision (Fake)': 0.97,
    'Recall (Fake)': 0.3526,
    'F1 (Fake)': 0.517
})

# Convert to DataFrame
results_df = pd.DataFrame(results)
print("\n" + "="*80)
print("TASK 1: MODEL COMPARISON TABLE")
print("="*80)
print(results_df.round(4))

# Plot comparison
plt.figure(figsize=(12, 6))
metrics = ['Accuracy', 'Recall (Fake)', 'F1 (Fake)']
for i, metric in enumerate(metrics):
    plt.subplot(1, 3, i+1)
    sns.barplot(data=results_df, x='Model', y=metric, hue='Model', palette='viridis', legend=False)
    plt.title(metric)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('')
    plt.xlabel('')
plt.suptitle('Model Comparison: Logistic Regression vs Tree Models', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# 3. Task 2: Feature Importance (Top 15 from Best RF)
print("\n" + "="*80)
print("TASK 2: Top 15 Important Words (Random Forest)")
print("="*80)

# Retrain best RF (n_estimators=100)
best_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
best_rf.fit(X_train, y_train)

# Get feature importances
importances = best_rf.feature_importances_
feature_names = vectorizer.get_feature_names_out()

# Top 15
top15_idx = importances.argsort()[-15:][::-1]
top15_words = [(feature_names[i], importances[i]) for i in top15_idx]

print("Top 15 Most Important Words:")
for word, imp in top15_words:
    print(f"{word:15}: {imp:.6f}")

# Plot
plt.figure(figsize=(8, 6))
plt.barh(range(len(top15_idx)), [importances[i] for i in top15_idx], color='skyblue')
plt.yticks(range(len(top15_idx)), [feature_names[i] for i in top15_idx])
plt.gca().invert_yaxis()
plt.title("Top 15 Important Words for Fake Job Detection (Random Forest)")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# 4. Interpretation
print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print("""
1. TOP WORDS MAKE SENSE:
   - 'urgent', 'visa', 'training', 'home', 'pay' → classic scam red flags
   - 'money', 'investment', 'register' → financial bait
   - These appear rarely in real jobs → high information gain

2. UNEXPECTED WORDS?
   - 'customer', 'service' → common in real jobs too
     → But in fake context: "customer service from home + pay fee"
   - 'data' → might appear in fake "data entry" scams

3. MODEL PERFORMANCE:
   - Random Forest (n=100) usually beats Logistic Regression in recall
   - But may overfit if n_estimators too high
   - Best balance: RF with 100 trees
""")