# (Public App + Secure Admin Panel)
from flask import Flask, render_template, request, redirect, session
import joblib, sqlite3, re
from datetime import datetime

app = Flask(__name__)
app.secret_key = "sainvi_secret_key_2025"   # Change this in real project!

# Load model
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

DB_NAME = "job_predictions.db"

# Initialize DB + admin table (run once)
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
            CREATE TABLE IF NOT EXISTS admin (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT
            )
        ''')
        # Create default admin if not exists
        conn.execute("INSERT OR IGNORE INTO admin (username, password) VALUES ('admin', 'admin123')")

init_db()

# Live counters from DB
def get_counts():
    with sqlite3.connect(DB_NAME) as conn:
        fake = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Fake Job'").fetchone()[0]
        real = conn.execute("SELECT COUNT(*) FROM predictions WHERE prediction='Real Job'").fetchone()[0]
    return fake, real

fake_count, real_count = get_counts()
last_prediction = None

# ==================== PUBLIC ROUTES ====================
@app.route('/')
def home():
    return render_template('index.html', fake=fake_count, real=real_count, last=last_prediction)

@app.route('/predict', methods=['POST'])
def predict():
    global fake_count, real_count, last_prediction
    job_desc = request.form.get('job_description', '').strip()

    # Smart validation
    if not job_desc or len(job_desc.split()) < 8 or re.search(r'[a-zA-Z]', job_desc) is None:
        error = "Please enter a valid job description (min 8 words, real text)."
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

    last_prediction = {'label': label, 'confidence': confidence,
                       'desc': job_desc[:200] + "..." if len(job_desc) > 200 else job_desc}

    return render_template('result.html', label=label, confidence=confidence,
                           description=job_desc, fake=fake_count, real=real_count)

@app.route('/history')
def history():
    with sqlite3.connect(DB_NAME) as conn:
        cur = conn.execute("SELECT job_description, prediction, confidence, timestamp FROM predictions ORDER BY id DESC")
        records = cur.fetchall()
    return render_template('history.html', records=records)

# ==================== ADMIN ROUTES ====================
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
        return render_template('admin_login.html', error="Invalid credentials")
    return render_template('admin_login.html')

@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect('/admin_login')
    fake, real = get_counts()
    total = fake + real
    return render_template('admin_dashboard.html', total=total, fake=fake, real=real)

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# ==================== RUN ====================
if __name__ == '__main__':
    print("Sainvi's Fake Job Detector + Admin Panel is LIVE!")
    app.run(debug=True)