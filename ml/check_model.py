
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = (
    BASE_DIR / "dataset" / "cleaned"
    / "patient_data_od_cleaned.csv"
)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nDiagnosis distribution:")
print(df["Diagnosis"].value_counts().sort_index())

print("\nFeatures with missing values:")
print(df.isna().sum()[df.isna().sum() > 0].to_string())

print("\nDiagnosis versus age:")
print(df.groupby("Diagnosis")["Age"].agg(
    ["count", "mean", "median", "min", "max"]
).round(2))

print("\nDiagnosis versus gender:")
print(pd.crosstab(df["Diagnosis"], df["Gender"]))

print("\nNumeric feature correlations with diagnosis:")
print(
    df.drop(columns=["Patient_ID"])
      .corr(numeric_only=True)["Diagnosis"]
      .drop("Diagnosis")
      .sort_values(key=abs, ascending=False)
      .round(3)
      .to_string()
)

print("\nCheck complete.")
