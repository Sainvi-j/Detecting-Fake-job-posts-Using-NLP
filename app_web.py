# ULTIMATE Fake Job Detector (Merged & Perfect)
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Global counters (live dashboard)
fake_count = 0
real_count = 0

@app.route('/')
def home():
    return render_template('index.html', fake=fake_count, real=real_count)

@app.route('/predict', methods=['POST'])
def predict():
    global fake_count, real_count

    job_desc = request.form.get('job_description', '').strip()

    # Smart validation
    if not job_desc or len(job_desc.split()) < 5:
        return render_template('index.html',
                               error="Please enter a detailed job description (at least 5 words).",
                               fake=fake_count, real=real_count)

    # Predict
    X_input = vectorizer.transform([job_desc])
    pred = model.predict(X_input)[0]
    prob_fake = model.predict_proba(X_input)[0][1]

    label = "Fake Job" if pred == 1 else "Real Job"
    confidence = round(prob_fake * 100, 2) if pred == 1 else round((1 - prob_fake) * 100, 2)

    # Update counter
    if pred == 1:
        fake_count += 1
    else:
        real_count += 1

    return render_template('result.html',
                           label=label,
                           confidence=confidence,
                           description=job_desc,
                           fake=fake_count,
                           real=real_count)

if __name__ == '__main__':
    app.run(debug=True)