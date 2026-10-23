
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "cleaned"

od = pd.read_csv(DATA_DIR / "patient_data_od_cleaned.csv")
os = pd.read_csv(DATA_DIR / "patient_data_os_cleaned.csv")

od_ids = set(od["Patient_ID"].dropna())
os_ids = set(os["Patient_ID"].dropna())

print("OD patient count:", len(od_ids))
print("OS patient count:", len(os_ids))
print("Patients in both files:", len(od_ids & os_ids))
print("Only in OD:", sorted(od_ids - os_ids))
print("Only in OS:", sorted(os_ids - od_ids))

# Check whether diagnosis labels agree for matched patients.
matched = od[["Patient_ID", "Diagnosis"]].merge(
    os[["Patient_ID", "Diagnosis"]],
    on="Patient_ID",
    how="inner",
    suffixes=("_OD", "_OS"),
    validate="one_to_one",
)

print("\nMatched records:", len(matched))
print(
    "Records with different eye diagnosis labels:",
    (matched["Diagnosis_OD"] != matched["Diagnosis_OS"]).sum(),
)

print("\nFirst 10 matched patients:")
print(matched.head(10).to_string(index=False))

print("\nPatients with different eye diagnoses:")
different = matched[
    matched["Diagnosis_OD"] != matched["Diagnosis_OS"]
]
print(different.to_string(index=False))
