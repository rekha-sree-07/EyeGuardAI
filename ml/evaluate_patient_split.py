
from pathlib import Path

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = (
    BASE_DIR / "dataset" / "cleaned"
    / "patient_data_od_cleaned.csv"
)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Patient_ID", "Diagnosis"])
y = df["Diagnosis"]

# Each row is one patient in the OD dataset.
# The same Patient_ID must not occur in multiple folds.
cv = StratifiedKFold(
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

print("PAPILA OD: 5-fold cross-validation")
print("Patients:", len(df))
print("\nClass counts:")
print(y.value_counts().sort_index())

for name, model in models.items():
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "macro_f1": "f1_macro",
        },
    )

    print(f"\n{name}")
    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]:
        values = scores[f"test_{metric}"]
        print(
            f"{metric}: "
            f"{values.mean():.3f} "
            f"+/- {values.std():.3f}"
        )

print("\nEvaluation complete.")
