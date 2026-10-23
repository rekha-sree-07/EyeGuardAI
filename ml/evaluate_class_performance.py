
from pathlib import Path

import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "dataset" / "cleaned"
    / "combined_patient_data.csv"
)

df = pd.read_csv(DATA_PATH)

od = df[["Patient_ID"] + [
    c for c in df.columns if c.endswith("_OD")
]].copy()

os = df[["Patient_ID"] + [
    c for c in df.columns if c.endswith("_OS")
]].copy()

od.columns = [
    "Patient_ID" if c == "Patient_ID"
    else c.removesuffix("_OD")
    for c in od.columns
]

os.columns = [
    "Patient_ID" if c == "Patient_ID"
    else c.removesuffix("_OS")
    for c in os.columns
]

eyes = pd.concat([od, os], ignore_index=True)
eyes = eyes.dropna(subset=["Diagnosis"]).reset_index(drop=True)

X = eyes.drop(columns=["Patient_ID", "Diagnosis"])
y = eyes["Diagnosis"]
groups = eyes["Patient_ID"]

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

predictions = pd.Series(index=y.index, dtype="float64")

for train_idx, test_idx in cv.split(X, y, groups):
    train_patients = set(groups.iloc[train_idx])
    test_patients = set(groups.iloc[test_idx])

    assert train_patients.isdisjoint(test_patients)

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

    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions.iloc[test_idx] = model.predict(X.iloc[test_idx])

print("Classification report:")
print(
    classification_report(
        y,
        predictions.astype(int),
        labels=[0, 1, 2],
        target_names=[
            "Healthy (0)",
            "Glaucoma (1)",
            "Glaucoma suspect (2)",
        ],
        zero_division=0,
    )
)

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y, predictions.astype(int), labels=[0, 1, 2]))

print("\nOverall balanced accuracy:",
      round(balanced_accuracy_score(y, predictions), 3))

print("Overall macro F1:",
      round(f1_score(y, predictions, average="macro"), 3))
