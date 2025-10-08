# train_safety_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
df = pd.read_csv("data/safety_data.csv")

X = df['text']
y = df[['abuse', 'escalation', 'crisis', 'age_inappropriate']]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline with MultiOutputClassifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000)))
])

# Train model
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, 'safety_model.pkl')
print("✅ Model saved as 'safety_model.pkl'")
