# Day 8: Saving Model and Vectorizer

import joblib
import pandas as pd
from flask import Flask, request, jsonify

# 1. Load Saved Model & Vectorizer (From Day 5)
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

print("✅ Loaded Existing Model and Vectorizer Successfully!")

# 2. Test Loading with Example Predictions
test_jobs = [
    "Work from home with high pay, no experience required! Apply immediately!",
    "We are hiring a software engineer with 2+ years of Python experience."
]

X_test_jobs = vectorizer.transform(test_jobs)
predictions = model.predict(X_test_jobs)
probas = model.predict_proba(X_test_jobs)[:, 1]  # Probability of being fake

print("\nSample Predictions:")
for job, pred, prob in zip(test_jobs, predictions, probas):
    label = "Fake Job" if pred == 1 else "Real Job"
    print(f"\nJob: {job}")
    print(f"Prediction: {label} (Probability Fake: {prob:.4f})")

# 3. Flask API Prototype (app.py equivalent)
app = Flask(__name__)

@app.route('/')
def home():
    return "Fake Job Detection API is running! Send POST to /predict with {'description': 'job text'}"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    job_text = data.get('description', '')
    if not job_text:
        return jsonify({"error": "No job description provided"}), 400
    
    # Transform and predict
    X_input = vectorizer.transform([job_text])
    prediction = model.predict(X_input)[0]
    prob_fake = model.predict_proba(X_input)[0][1]
    
    label = "Fake Job" if prediction == 1 else "Real Job"
    
    return jsonify({
        "prediction": label,
        "probability_fake": round(prob_fake, 4)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Run on http://localhost:5000