from flask import Flask, render_template, request

import joblib, sqlite3

from datetime import datetime
 
app = Flask(__name__)
 
# Load model and vectorizer

model = joblib.load('fake_job_model.pkl')

vectorizer = joblib.load('tfidf_vectorizer.pkl')
 
@app.route('/')

def home():

    return render_template('index.html')
 
@app.route('/predict', methods=['POST'])

def predict():

    job_desc = request.form['job_description'].strip()

    if not job_desc or len(job_desc.split()) < 5:

        return render_template('index.html', error="Please enter a meaningful job description.")

    # Predict

    X_input = vectorizer.transform([job_desc])

    pred = model.predict(X_input)[0]

    prob = model.predict_proba(X_input)[0][1]

    label = "Fake Job" if pred == 1 else "Real Job"

    confidence = round(prob * 100, 2) if pred == 1 else round((1 - prob) * 100, 2)

    # Save to DB

    conn = sqlite3.connect('job_predictions.db')

    conn.execute('INSERT INTO predictions (job_description, prediction, confidence) VALUES (?, ?, ?)',

                 (job_desc, label, confidence))

    conn.commit()

    conn.close()
 
    return render_template('result.html', label=label, confidence=confidence, description=job_desc)
 
@app.route('/history')

def history():

    conn = sqlite3.connect('job_predictions.db')

    cursor = conn.execute('SELECT job_description, prediction, confidence, timestamp FROM predictions ORDER BY id DESC')

    records = cursor.fetchall()

    conn.close()

    return render_template('history.html', records=records)
 
if __name__ == '__main__':

    app.run(debug=True)

# HTML Templates (to be placed in 'templates/history' folder)
"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Prediction History</title>
<style>

    body { font-family: Arial; background-color: #f4f6f9; padding: 40px; }

    table { width: 90%; margin: auto; border-collapse: collapse; background: white; border-radius: 10px; }

    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }

    th { background-color: #007bff; color: white; }

    tr:hover { background-color: #f1f1f1; }

    a { text-decoration: none; color: #007bff; }
</style>
</head>
<body>
<h2 style="text-align:center;">Prediction History</h2>
<table>
<tr>
<th>Job Description</th>
<th>Prediction</th>
<th>Confidence (%)</th>
<th>Timestamp</th>
</tr>

    {% for job, label, conf, time in records %}
<tr>
<td>{{ job[:100] }}...</td>
<td>{{ label }}</td>
<td>{{ conf }}</td>
<td>{{ time }}</td>
</tr>

    {% endfor %}
</table>
<p style="text-align:center; margin-top:20px;"><a href="/">🔙 Back to Home</a></p>
</body>
</html>
"""
 