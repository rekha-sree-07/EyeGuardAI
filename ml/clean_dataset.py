
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "ClinicalData"
OUTPUT_DIR = BASE_DIR / "dataset" / "cleaned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLUMN_NAMES = [
    "Patient_ID",
    "Age",
    "Gender",
    "Diagnosis",
    "Dioptre_1",
    "Dioptre_2",
    "Astigmatism",
    "Phakic_Pseudophakic",
    "IOP_Pneumatic",
    "IOP_Perkins",
    "Pachymetry",
    "Axial_Length",
    "VF_MD",
]

for eye in ["od", "os"]:
    input_path = DATA_DIR / f"patient_data_{eye}.xlsx"

    # Skip the three non-patient header rows.
    df = pd.read_excel(input_path, header=None, skiprows=3)
    df.columns = COLUMN_NAMES

    df = df.dropna(how="all")
    df["Patient_ID"] = df["Patient_ID"].astype("string").str.strip()
    df = df.dropna(subset=["Patient_ID"])

    numeric_columns = [
        col for col in COLUMN_NAMES if col != "Patient_ID"
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    output_path = OUTPUT_DIR / f"patient_data_{eye}_cleaned.csv"
    df.to_csv(output_path, index=False)

    print(f"\n{eye.upper()} dataset")
    print("Records:", len(df))
    print("Columns:", df.columns.tolist())
    print("Diagnosis counts:")
    print(df["Diagnosis"].value_counts().sort_index())
    print("Saved:", output_path)

print("\nBoth datasets cleaned successfully.")
