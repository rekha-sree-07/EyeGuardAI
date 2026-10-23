
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "dataset" / "full_df.csv"

df = pd.read_csv(CSV_PATH)

print("===== CONSISTENCY WITHIN PATIENT ID =====")

for column in ["Patient Age", "Patient Sex", "labels"]:
    counts = df.groupby("ID")[column].nunique()
    inconsistent = counts[counts > 1]

    print(f"\n{column}")
    print("IDs with inconsistent values:", len(inconsistent))

    if len(inconsistent):
        print(inconsistent.head(10))

print("\n===== UNIQUE PATIENT-LEVEL RECORDS =====")
patients = df.drop_duplicates(subset="ID")

print("Unique IDs:", patients["ID"].nunique())
print("\nPatient-level label counts:")
print(patients["labels"].value_counts())

print("\n===== ROW COUNTS PER PATIENT =====")
print(df.groupby("ID").size().value_counts().sort_index())
