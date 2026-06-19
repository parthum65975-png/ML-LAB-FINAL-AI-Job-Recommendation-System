from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import PyPDF2
import docx
import re

app = Flask(__name__)

# =====================
# Models Load
# =====================
tfidf = pickle.load(open('models/tfidf.pkl', 'rb'))
le    = pickle.load(open('models/label_encoder.pkl', 'rb'))
knn   = pickle.load(open('models/knn_model.pkl', 'rb'))
rf    = pickle.load(open('models/rf_model.pkl', 'rb'))
from tensorflow.keras.models import load_model
ann   = load_model('models/ann_model.keras')

# =====================
# Job Database
# =====================
jobs_db = [
    {"title": "Data Analyst",        "company": "TechCorp",      "location": "Karachi",   "skills": "sql tableau python reporting analytics"},
    {"title": "Data Analyst",        "company": "DataViz Inc",   "location": "Lahore",    "skills": "python excel statistics data visualization"},
    {"title": "Software Engineer",   "company": "SoftSol",       "location": "Karachi",   "skills": "java python git debugging algorithms"},
    {"title": "Software Engineer",   "company": "Systems Ltd",   "location": "Islamabad", "skills": "python git teamwork problem solving"},
    {"title": "Network Admin",       "company": "NetSecure",     "location": "Karachi",   "skills": "cisco firewall linux networking security"},
    {"title": "Social Media Mgr",    "company": "BrandUp",       "location": "Lahore",    "skills": "instagram canva copywriting analytics facebook"},
    {"title": "Procurement Analyst", "company": "SupplyChain Co","location": "Karachi",   "skills": "negotiation vendor supply chain excel reporting"},
    {"title": "UI Designer",         "company": "DesignHub",     "location": "Lahore",    "skills": "figma wireframing adobe xd css prototyping"},
    {"title": "HR Manager",          "company": "PeopleFirst",   "location": "Karachi",   "skills": "recruitment interviewing onboarding communication"},
    {"title": "Accountant",          "company": "FinancePro",    "location": "Islamabad", "skills": "tally taxation auditing excel financial reporting"},
    {"title": "Marketing Manager",   "company": "GrowthX",       "location": "Karachi",   "skills": "seo google ads campaign analytics communication"},
    {"title": "Project Manager",     "company": "ProjektHQ",     "location": "Lahore",    "skills": "scheduling risk management budgeting leadership excel"},
]

# =====================
# Extract Text from CV
# =====================
def extract_text(file):
    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''
        return text
    elif filename.endswith('.docx'):
        doc = docx.Document(file)
        return '\n'.join([p.text for p in doc.paragraphs])
    else:
        return file.read().decode('utf-8', errors='ignore')

# =====================
# Extract Skills from Text
# =====================
def extract_skills(text):
    text = text.lower()
    all_skills = [
        'python', 'java', 'sql', 'excel', 'tableau', 'git', 'linux',
        'cisco', 'firewall', 'instagram', 'canva', 'figma', 'seo',
        'negotiation', 'recruitment', 'tally', 'taxation', 'auditing',
        'adobe xd', 'wireframing', 'copywriting', 'scheduling',
        'budgeting', 'communication', 'teamwork', 'leadership',
        'analytics', 'reporting', 'ms office', 'problem solving',
        'debugging', 'algorithms', 'networking', 'security', 'routing',
        'vendor', 'supply chain', 'interviewing', 'onboarding',
        'google ads', 'campaign', 'risk management', 'prototyping',
        'data visualization', 'statistics', 'facebook', 'css'
    ]
    found = [skill for skill in all_skills if skill in text]
    return found

# =====================
# Match Percentage
# =====================
def get_match_percent(user_skills, job_skills):
    user_set = set(user_skills)
    job_set  = set(job_skills.split())
    if not job_set:
        return 0
    overlap = user_set & job_set
    return round((len(overlap) / len(job_set)) * 100, 1)

# =====================
# Routes
# =====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    skills_text = ''

    # CV upload check
    if 'cv' in request.files and request.files['cv'].filename != '':
        file = request.files['cv']
        skills_text = extract_text(file)
    # Manual skills check
    elif 'skills' in request.form and request.form['skills'].strip() != '':
        skills_text = request.form['skills']

    if not skills_text:
        return jsonify({'error': 'Koi skills ya CV nahi mili!'})

    # Skills extract karo
    extracted = extract_skills(skills_text)
    if not extracted:
        return jsonify({'error': 'Koi recognizable skills nahi mili CV mein!'})

    skills_joined = ' '.join(extracted)

    # TF-IDF transform
    X = tfidf.transform([skills_joined]).toarray()

    # Predictions
    knn_pred = le.classes_[knn.predict(X)[0]]
    rf_pred  = le.classes_[rf.predict(X)[0]]
    ann_pred = le.classes_[np.argmax(ann.predict(X), axis=1)[0]]

    # Top jobs with match %
    results = []
    for job in jobs_db:
        match = get_match_percent(extracted, job['skills'])
        results.append({**job, 'match': match})

    results = sorted(results, key=lambda x: x['match'], reverse=True)[:5]

    return jsonify({
        'extracted_skills': extracted,
        'knn_prediction':   knn_pred,
        'rf_prediction':    rf_pred,
        'ann_prediction':   ann_pred,
        'recommendations':  results
    })

if __name__ == '__main__':
    app.run(debug=True)