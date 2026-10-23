
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "backend"
    / "ml_models"
    / "eyeguard_model.joblib"
)

DATA_PATH = (
    BASE_DIR
    / "dataset"
    / "cleaned"
    / "patient_data_od_cleaned.csv"
)

# Load the saved model and metadata
saved = joblib.load(MODEL_PATH)

model = saved["model"]
feature_names = saved["feature_names"]

print("Model loaded successfully.")
print("Features expected:", len(feature_names))
print("Classes:", saved["classes"])

# Use one existing patient's clinical measurements
df = pd.read_csv(DATA_PATH)
sample = df.iloc[[0]]

X_sample = sample[feature_names]
prediction = model.predict(X_sample)[0]

print("\nTest patient ID:", sample["Patient_ID"].iloc[0])
print("Predicted class:", int(prediction))
print("Actual class:", int(sample["Diagnosis"].iloc[0]))

print("\nPrediction probabilities:")
probabilities = model.predict_proba(X_sample)[0]

for class_label, probability in zip(
    model.classes_, probabilities
):
    print(f"Class {class_label}: {probability:.2%}")

print("\nSaved-model test complete.")
