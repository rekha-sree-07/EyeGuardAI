
from pathlib import Path

import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "dataset"
    / "cleaned"
    / "combined_patient_data.csv"
)

df = pd.read_csv(DATA_PATH)

# Create one row per eye.
# Both eyes retain the same Patient_ID.
od = df[["Patient_ID"] + [
    c for c in df.columns if c.endswith("_OD")
]].copy()

os = df[["Patient_ID"] + [
    c for c in df.columns if c.endswith("_OS")
]].copy()

od.columns = [
    "Patient_ID"
    if c == "Patient_ID"
    else c.removesuffix("_OD")
    for c in od.columns
]

os.columns = [
    "Patient_ID"
    if c == "Patient_ID"
    else c.removesuffix("_OS")
    for c in os.columns
]

eyes = pd.concat([od, os], ignore_index=True)

# Exclude records without a diagnosis.
eyes = eyes.dropna(subset=["Diagnosis"]).reset_index(drop=True)

X = eyes.drop(columns=["Patient_ID", "Diagnosis"])
y = eyes["Diagnosis"]
groups = eyes["Patient_ID"]

# Each patient's two eyes must stay in the same fold.
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

models = {
    "Majority-class baseline": DummyClassifier(
        strategy="most_frequent"
    ),
    "Random Forest": Pipeline([
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
    ]),
}

print("PAPILA: Both-eye patient-grouped evaluation")
print("Eye records:", len(eyes))
print("Unique patients:", groups.nunique())
print("\nDiagnosis counts:")
print(y.value_counts().sort_index())

for name, model in models.items():
    fold_results = []

    for train_idx, test_idx in cv.split(X, y, groups):
        train_patients = set(groups.iloc[train_idx])
        test_patients = set(groups.iloc[test_idx])

        assert train_patients.isdisjoint(test_patients), (
            "Patient leakage detected!"
        )

        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = model.predict(X.iloc[test_idx])

        fold_results.append({
            "accuracy": accuracy_score(
                y.iloc[test_idx], predictions
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y.iloc[test_idx], predictions
            ),
            "macro_f1": f1_score(
                y.iloc[test_idx],
                predictions,
                average="macro",
                zero_division=0,
            ),
        })

    results = pd.DataFrame(fold_results)

    print(f"\n{name}")
    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]:
        print(
            f"{metric}: "
            f"{results[metric].mean():.3f} "
            f"+/- {results[metric].std():.3f}"
        )

print("\nEvaluation complete. Patient groups were kept separate.")
