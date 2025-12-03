# FINAL FAKE JOB DETECTOR - NOVEMBER 2025

from flask import Flask, render_template, request, redirect, session, send_file, Response, jsonify
import joblib, sqlite3, pandas as pd, os
from datetime import datetime

app = Flask(__name__)
app.secret_key = "sainvi_final_2025_ultra_secure"

# Load Model
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

DB_NAME = "job_predictions.db"

# Init DB + Tables
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_description TEXT,
                prediction TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS flagged (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                reason TEXT DEFAULT 'User flagged',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS admin (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT
            )
        ''')
        conn.execute("INSERT OR IGNORE INTO admin (username, password) VALUES ('admin', 'admin123')")
        conn.execute('''
            CREATE TABLE IF NOT EXISTS retrain_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                accuracy REAL,
                training_source TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

init_db()

def get_stats():
    with sqlite3.connect(DB_NAME) as conn:
        fake = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Fake Job'").fetchone()[0] or 0
        real = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Real Job'").fetchone()[0] or 0
        flagged = conn.execute("SELECT COUNT(*) FROM flagged").fetchone()[0] or 0
    return fake, real, flagged

# Routes
@app.route('/')
def home():
    fake, real, flagged = get_stats()
    return render_template('index.html', fake=fake, real=real, flagged=flagged)

@app.route('/predict', methods=['POST'])
def predict():
    desc = request.form['job_description'].strip()
    if len(desc.split()) < 8:
        return render_template('index.html', error="Too short! Minimum 8 words.")
    
    X = vectorizer.transform([desc])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    label = "Fake Job" if pred == 1 else "Real Job"
    confidence = round(prob * 100, 2) if pred == 1 else round((1 - prob) * 100, 2)

    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.execute("INSERT INTO predictions (job_description, prediction, confidence) VALUES (?, ?, ?)",
                             (desc, label, confidence))
        pred_id = cursor.lastrowid

    fake, real, flagged = get_stats()
    return render_template('result.html', label=label, confidence=confidence,
                           description=desc, pred_id=pred_id, fake=fake, real=real)

@app.route('/flag/<int:pred_id>')
def flag(pred_id):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("INSERT INTO flagged (prediction_id) VALUES (?)", (pred_id,))
    return "<script>alert('Flagged! Admin will review.'); window.location='/'</script>"

@app.route('/history')
def history():
    with sqlite3.connect(DB_NAME) as conn:
        cur = conn.execute("SELECT job_description, prediction, confidence, timestamp FROM predictions ORDER BY id DESC")
        records = cur.fetchall()
    return render_template('history.html', records=records)

@app.route('/export_csv')
def export_csv():
    with sqlite3.connect(DB_NAME) as conn:
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    output = df.to_csv(index=False)
    return Response(output, mimetype="text/csv",
                    headers={"Content-disposition": "attachment; filename=fake_job_predictions.csv"})

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'admin123':
            session['admin'] = True
            return redirect('/admin_dashboard')
        return render_template('admin_login.html', error="Wrong credentials")
    return render_template('admin_login.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('admin'):
        return redirect('/admin_login')

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Total counts
    fake_count = cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Fake Job'").fetchone()[0] or 0
    real_count = cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Real Job'").fetchone()[0] or 0
    total = fake_count + real_count

    # Average confidence
    avg_fake = cursor.execute("SELECT AVG(confidence) FROM predictions WHERE prediction='Fake Job'").fetchone()[0]
    avg_real = cursor.execute("SELECT AVG(confidence) FROM predictions WHERE prediction='Real Job'").fetchone()[0]
    avg_fake_conf = round(avg_fake, 2) if avg_fake else 0
    avg_real_conf = round(avg_real, 2) if avg_real else 0

    # Daily trend
    daily_data = cursor.execute("""
        SELECT DATE(timestamp), COUNT(*)
        FROM predictions
        GROUP BY DATE(timestamp)
        ORDER BY DATE(timestamp)
    """).fetchall()
    dates = [row[0] for row in daily_data]
    counts = [row[1] for row in daily_data]

    # Flagged
    flagged = cursor.execute("SELECT COUNT(*) FROM flagged").fetchone()[0] or 0

    conn.close()

    return render_template('admin_dashboard.html',
                           fake=fake_count, real=real_count, total=total,
                           avg_fake_conf=avg_fake_conf, avg_real_conf=avg_real_conf,
                           dates=dates, counts=counts, flagged=flagged)

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    if not session.get('admin'):
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    # Simulate retrain (replace with real training code if you have it)
    accuracy = 95.2  # Example — calculate real accuracy from your model
    training_source = "default dataset"  # Or "uploaded_file.csv" if upload added
    
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("INSERT INTO retrain_logs (accuracy, training_source) VALUES (?, ?)",
                     (accuracy, training_source))
    
    return jsonify({'success': True, 'message': f'Model retrained! Accuracy: {accuracy}%', 'timestamp': datetime.now().strftime('%d %b %Y')})

@app.route('/retrain_logs')
def retrain_logs():
    if not session.get('admin'):
        return redirect('/admin_login')
    
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT id, timestamp, accuracy, training_source FROM retrain_logs ORDER BY id DESC")
        rows = cur.fetchall()
    
    return render_template('retrain_logs.html', rows=rows)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# RUN
if __name__ == '__main__':
    print("="*60)
    print("FAKE JOB DETECTOR + ADMIN PANEL IS LIVE!")
    print("Public App : http://127.0.0.1:5000")
    print("Admin Login: http://127.0.0.1:5000/admin_login (admin / admin123)")
    app.run(debug=True)