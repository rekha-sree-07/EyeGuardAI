
from pathlib import Path

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "dataset"
    / "cleaned"
    / "patient_data_od_cleaned.csv"
)

MODEL_DIR = BASE_DIR / "backend" / "ml_models"
MODEL_PATH = MODEL_DIR / "eyeguard_model.joblib"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Patient_ID", "Diagnosis"])
y = df["Diagnosis"]

model = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median",
            add_indicator=True,
        ),
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

model.fit(X, y)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {
        "model": model,
        "feature_names": list(X.columns),
        "classes": [int(value) for value in model.classes_],
        "dataset": "PAPILA v1.1",
        "eye": "OD",
    },
    MODEL_PATH,
)

print("Model saved successfully.")
print("Location:", MODEL_PATH)
print("Features:", len(X.columns))
print("Classes:", model.classes_)
