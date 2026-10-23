
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = (
    BASE_DIR
    / "dataset"
    / "cleaned"
    / "patient_data_od_cleaned.csv"
)

df = pd.read_csv(DATA_PATH)

# Remove rows without a valid target label.
df["Diagnosis"] = pd.to_numeric(df["Diagnosis"], errors="coerce")
df = df.dropna(subset=["Diagnosis"]).copy()
df["Diagnosis"] = df["Diagnosis"].astype(int)

# Exclude the patient ID and use measurements as input features.
X = df.drop(columns=["Patient_ID", "Diagnosis"])
y = df["Diagnosis"]

# Hold out test data that the model will not train on.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

model = make_pipeline(
    SimpleImputer(strategy="median", add_indicator=True),
    RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
    ),
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nEXPERIMENTAL RIGHT-EYE MODEL")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 3),
)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        zero_division=0,
    )
)

print("\nConfusion matrix (labels 0, 1, 2):")
print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))
