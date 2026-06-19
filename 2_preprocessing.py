import pandas as pd
import numpy as np
import pickle
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)

# =====================
# Job Skills — 50% overlap
# =====================
job_skills = {
    'Data Analyst':        ['sql', 'tableau', 'python', 'communication', 'excel', 'reporting', 'teamwork'],
    'Software Engineer':   ['java', 'git', 'debugging', 'python', 'communication', 'teamwork', 'excel'],
    'Network Admin':       ['cisco', 'firewall', 'linux', 'communication', 'excel', 'reporting', 'teamwork'],
    'Social Media Mgr':    ['instagram', 'canva', 'copywriting', 'communication', 'excel', 'analytics', 'teamwork'],
    'Procurement Analyst': ['negotiation', 'vendor', 'supply chain', 'excel', 'communication', 'reporting', 'teamwork'],
    'UI Designer':         ['figma', 'wireframing', 'adobe xd', 'communication', 'teamwork', 'excel', 'creativity'],
    'HR Manager':          ['recruitment', 'interviewing', 'onboarding', 'communication', 'excel', 'teamwork', 'reporting'],
    'Accountant':          ['tally', 'taxation', 'auditing', 'excel', 'communication', 'reporting', 'teamwork'],
    'Marketing Manager':   ['seo', 'google ads', 'campaign', 'communication', 'excel', 'analytics', 'teamwork'],
    'Project Manager':     ['scheduling', 'risk management', 'budgeting', 'excel', 'communication', 'teamwork', 'reporting']
}

# 500 rows per category
rows = []
for role, skills in job_skills.items():
    for _ in range(500):
        n = random.randint(2, 3)
        chosen = random.sample(skills, n)
        rows.append({'Role': role, 'skills': ' '.join(chosen)})

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("Dataset Shape:", df.shape)
print(df['Role'].value_counts())

# TF-IDF
tfidf = TfidfVectorizer(max_features=100)
X = tfidf.fit_transform(df['skills'])
print("TF-IDF Shape:", X.shape)

# Label Encode
le = LabelEncoder()
y = le.fit_transform(df['Role'])
print("Classes:", le.classes_)

# Train Test Split — 95% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.95, random_state=0, stratify=y
)
print("Train:", X_train.shape, "Test:", X_test.shape)

# Save
pickle.dump(tfidf, open('models/tfidf.pkl', 'wb'))
pickle.dump(le, open('models/label_encoder.pkl', 'wb'))
np.save('models/X_train.npy', X_train.toarray())
np.save('models/X_test.npy', X_test.toarray())
np.save('models/y_train.npy', y_train)
np.save('models/y_test.npy', y_test)

print("\n✅ Preprocessing done!")