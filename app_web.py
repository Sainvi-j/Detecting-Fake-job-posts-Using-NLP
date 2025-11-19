# FINAL PROJECT - Fake Job Detector + Admin Panel
# Full-Stack | SQLite DB | Session Auth | Beautiful UI


from flask import Flask, render_template, request, redirect, session
import joblib, sqlite3, re
from datetime import datetime

app = Flask(__name__)
app.secret_key = "sainvi_final_project_2025_secure"

# Load ML Model
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

DB_NAME = "job_predictions.db"

# Initialize Database + Admin User
def init_db():
    with sqlite3.connect(DB_NAME) as conn:
        # Predictions Table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_description TEXT,
                prediction TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Admin Table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS admin (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT
            )
        ''')
        # Default Admin (only once)
        conn.execute("INSERT OR IGNORE INTO admin (username, password) VALUES ('admin', 'admin123')")
    print("Database & Admin Ready!")

init_db()

# Get Live Counts from DB
def get_counts():
    with sqlite3.connect(DB_NAME) as conn:
        fake = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Fake Job'").fetchone()[0] or 0
        real = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Real Job'").fetchone()[0] or 0
    return fake, real

fake_count, real_count = get_counts()
last_prediction = None

# PUBLIC APP
@app.route('/')
def home():
    return render_template('index.html', fake=fake_count, real=real_count, last=last_prediction)

@app.route('/predict', methods=['POST'])
def predict():
    global fake_count, real_count, last_prediction
    job_desc = request.form.get('job_description', '').strip()

    # Smart Validation
    if not job_desc:
        error = "Please enter a job description."
    elif len(job_desc.split()) < 8:
        error = "Too short! Please enter at least 8 words."
    elif re.search(r'[a-zA-Z]', job_desc) is None:
        error = "Only symbols/numbers? Please enter real text."
    else:
        error = None

    if error:
        return render_template('index.html', error=error, fake=fake_count, real=real_count, last=last_prediction)

    # Predict
    X = vectorizer.transform([job_desc])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    label = "Fake Job" if pred == 1 else "Real Job"
    confidence = round(prob * 100, 2) if pred == 1 else round((1 - prob) * 100, 2)

    # Save to DB
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("INSERT INTO predictions (job_description, prediction, confidence) VALUES (?, ?, ?)",
                     (job_desc, label, confidence))

    # Update counters
    if pred == 1: fake_count += 1
    else: real_count += 1

    last_prediction = {
        'label': label, 'confidence': confidence,
        'desc': job_desc[:200] + "..." if len(job_desc) > 200 else job_desc
    }

    return render_template('result.html', label=label, confidence=confidence,
                           description=job_desc, fake=fake_count, real=real_count)

@app.route('/history')
def history():
    with sqlite3.connect(DB_NAME) as conn:
        cur = conn.execute("SELECT job_description, prediction, confidence, timestamp FROM predictions ORDER BY id DESC")
        records = cur.fetchall()
    return render_template('history.html', records=records)

# ADMIN PANEL 
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        with sqlite3.connect(DB_NAME) as conn:
            cur = conn.execute("SELECT * FROM admin WHERE username=? AND password=?", (username, password))
            admin = cur.fetchone()
        if admin:
            session['admin_logged_in'] = True
            return redirect('/admin_dashboard')
        return render_template('admin_login.html', error="Invalid username or password")
    return render_template('admin_login.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect('/admin_login')

    conn = sqlite3.connect('job_predictions.db')
    cursor = conn.cursor()

    # Total counts
    fake_count = cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Fake Job'").fetchone()[0]
    real_count = cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Real Job'").fetchone()[0]

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

    conn.close()

    return render_template('admin_dashboard.html',
                           fake=fake_count, real=real_count,
                           avg_fake_conf=avg_fake_conf, avg_real_conf=avg_real_conf,
                           dates=dates, counts=counts)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# RUN
if __name__ == '__main__':
    print("="*60)
    print("SAINVI'S FAKE JOB DETECTOR + ADMIN PANEL IS LIVE!")
    print("Public App : http://127.0.0.1:5000")
    print("Admin Login: http://127.0.0.1:5000/admin_login (admin / admin123)")
    print("="*60)
    app.run(debug=True)