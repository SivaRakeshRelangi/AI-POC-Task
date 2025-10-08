
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("data/safety_data.csv")
X = df['text']
y = df[['abuse', 'escalation', 'crisis', 'age_inappropriate']]

# Split (same split as training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load trained model
model = joblib.load("safety_model.pkl")

# Predict
y_pred = model.predict(X_test)

# Convert predictions to DataFrame for comparison
y_pred_df = pd.DataFrame(y_pred, columns=y.columns)

# Evaluate metrics per label
print("\n🔍 Classification Report (per label):\n")
print(classification_report(y_test, y_pred_df, target_names=y.columns))

# Optional: Print exact accuracy for each label
print("\n✅ Label-wise Accuracy:")
for label in y.columns:
    acc = accuracy_score(y_test[label], y_pred_df[label])
    print(f"- {label}: {acc:.2f}")

# Overall accuracy (all labels correct)
exact_match = (y_pred_df == y_test.reset_index(drop=True)).all(axis=1).mean()
print(f"\n🎯 Exact Match Ratio (all labels correct per instance): {exact_match:.2f}")
