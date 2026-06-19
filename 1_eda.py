import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# 1. Dataset Load
# =====================
df = pd.read_csv('dataset/job_descriptions.csv')

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))

# =====================
# 2. Basic EDA
# =====================
print("\nNull Values:")
print(df.isnull().sum())

print("\nJob Categories (Role):")
print(df['Role'].value_counts().head(20))

print("\nTop Skills:")
print(df['skills'].value_counts().head(10))

# =====================
# 3. Sirf Kaam Ki Columns Rakho
# =====================
df = df[['Job Title', 'Role', 'skills', 'Job Description', 'Experience', 'location']]
print("\nNew Shape:", df.shape)

# =====================
# 4. Null Rows Drop Karo
# =====================
df.dropna(subset=['skills', 'Role', 'Job Title'], inplace=True)
print("After Cleaning Shape:", df.shape)

# =====================
# 5. Sample Lo (20k rows - fast training ke liye)
# =====================
df = df.sample(n=20000, random_state=42)
print("Sample Shape:", df.shape)

# Save karo
df.to_csv('dataset/cleaned_jobs.csv', index=False)
print("\n✅ Cleaned dataset saved!")