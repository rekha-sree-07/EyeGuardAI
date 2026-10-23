
from pathlib import Path

import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
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
    / "patient_data_od_cleaned.csv"
)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Patient_ID", "Diagnosis"])
y = df["Diagnosis"]

X_reduced = X.drop(
    columns=["IOP_Perkins", "VF_MD"]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

X_reduced_train = X_reduced.loc[X_train.index]
X_reduced_test = X_reduced.loc[X_test.index]

def create_model():
    return Pipeline([
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

models = {
    "Model A - All features": (
        create_model(),
        X_train,
        X_test,
    ),
    "Model B - Fewer missing features": (
        create_model(),
        X_reduced_train,
        X_reduced_test,
    ),
}

print("Comparing models on the same test patients")
print("-" * 55)

for name, (model, train_x, test_x) in models.items():
    model.fit(train_x, y_train)
    predictions = model.predict(test_x)

    print(f"\n{name}")
    print(f"Accuracy:          {accuracy_score(y_test, predictions):.3f}")
    print(
        "Balanced accuracy: "
        f"{balanced_accuracy_score(y_test, predictions):.3f}"
    )
    print(
        "Macro F1-score:    "
        f"{f1_score(y_test, predictions, average='macro'):.3f}"
    )

print("\nCross-validation on the full dataset")
print("Use these results as a more stable comparison.")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

for name, features in [
    ("All features", X),
    ("Fewer missing features", X_reduced),
]:
    scores = cross_validate(
        create_model(),
        features,
        y,
        cv=cv,
        scoring={
            "balanced_accuracy": "balanced_accuracy",
            "macro_f1": "f1_macro",
        },
    )

    print(f"\n{name}")
    print(
        "Mean balanced accuracy: "
        f"{scores['test_balanced_accuracy'].mean():.3f}"
    )
    print(
        "Mean macro F1-score:    "
        f"{scores['test_macro_f1'].mean():.3f}"
    )

print("\nComparison complete.")
