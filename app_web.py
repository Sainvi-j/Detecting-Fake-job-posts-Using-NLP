# Web App with Progress Bar
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    job_desc = request.form['job_description'].strip()
    
    if not job_desc:
        return render_template('index.html', error="Please enter a job description.")

    X_input = vectorizer.transform([job_desc])
    prediction = model.predict(X_input)[0]
    prob_fake = model.predict_proba(X_input)[0][1]

    label = "Fake Job" if prediction == 1 else "Real Job"
    confidence = round(prob_fake * 100, 2) if prediction == 1 else round((1 - prob_fake) * 100, 2)

    return render_template('result.html',
                           label=label,
                           confidence=confidence,
                           description=job_desc)

if __name__ == '__main__':
    app.run(debug=True)