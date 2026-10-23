
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "cleaned"

od = pd.read_csv(DATA_DIR / "patient_data_od_cleaned.csv")
os = pd.read_csv(DATA_DIR / "patient_data_os_cleaned.csv")

# Confirm that both files have the expected columns.
print("OD columns:", od.columns.tolist())
print("OS columns:", os.columns.tolist())

# Rename columns explicitly, keeping the patient ID unchanged.
od = od.rename(columns={
    col: f"{col}_OD"
    for col in od.columns
    if col != "Patient_ID"
})

os = os.rename(columns={
    col: f"{col}_OS"
    for col in os.columns
    if col != "Patient_ID"
})

# Match patients using their IDs.
combined = pd.merge(
    od,
    os,
    on="Patient_ID",
    how="inner",
    validate="one_to_one",
    suffixes=("_OD", "_OS"),
)

output_path = DATA_DIR / "combined_patient_data.csv"
combined.to_csv(output_path, index=False)

print("\nCombined dataset created.")
print("Rows:", combined.shape[0])
print("Columns:", combined.shape[1])
print("\nColumn names:")
for col in combined.columns:
    print(col)

print("\nFirst patient, one field per line:")
print(combined.iloc[0].to_string())

print("\nOD diagnosis counts:")
print(combined["Diagnosis_OD"].value_counts(dropna=False).sort_index())

print("\nOS diagnosis counts:")
print(combined["Diagnosis_OS"].value_counts(dropna=False).sort_index())

print("\nNumber of diagnosis differences:",
      (combined["Diagnosis_OD"] != combined["Diagnosis_OS"]).sum())

print("\nSaved to:", output_path)
