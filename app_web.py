# FINAL VERSION
from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# Load model
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Global state
fake_count = 0
real_count = 0
last_prediction = None

@app.route('/')
def home():
    return render_template('index.html',
                           fake=fake_count,
                           real=real_count,
                           last=last_prediction)

@app.route('/predict', methods=['POST'])
def predict():
    global fake_count, real_count, last_prediction

    job_desc = request.form.get('job_description', '').strip()

    # Smart Error Handling
    if not job_desc:
        return render_template('index.html', error="Please enter a job description.", fake=fake_count, real=real_count, last=last_prediction)
    if len(job_desc.split()) < 8:
        return render_template('index.html', error="Too short! Please enter at least 8 words.", fake=fake_count, real=real_count, last=last_prediction)
    if re.search(r'[a-zA-Z]', job_desc) is None:
        return render_template('index.html', error="Only symbols/numbers? Please enter real text.", fake=fake_count, real=real_count, last=last_prediction)

    # Predict
    X_input = vectorizer.transform([job_desc])
    pred = model.predict(X_input)[0]
    prob_fake = model.predict_proba(X_input)[0][1]

    label = "Fake Job" if pred == 1 else "Real Job"
    confidence = round(prob_fake * 100, 2) if pred == 1 else round((1 - prob_fake) * 100, 2)

    # Update counters
    if pred == 1:
        fake_count += 1
    else:
        real_count += 1

    # Save last prediction
    last_prediction = {
        'label': label,
        'confidence': confidence,
        'desc': job_desc[:200] + "..." if len(job_desc) > 200 else job_desc
    }

    return render_template('result.html',
                           label=label,
                           confidence=confidence,
                           description=job_desc,
                           fake=fake_count,
                           real=real_count)

if __name__ == '__main__':
    app.run(debug=True)