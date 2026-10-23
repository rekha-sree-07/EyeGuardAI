
from pathlib import Path
import ast
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "dataset" / "full_df.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"CSV not found: {CSV_PATH}\n"
        "Check the location of full_df.csv."
    )

df = pd.read_csv(CSV_PATH)

print("\n===== DATASET SHAPE =====")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n===== COLUMN NAMES =====")
for column in df.columns:
    print(f"- {column}: {df[column].dtype}")

print("\n===== MISSING VALUES =====")
print(df.isnull().sum().to_string())

print("\n===== FIRST 5 ROWS =====")
print(df.head().to_string())

print("\n===== LABEL DISTRIBUTION =====")
if "labels" in df.columns:
    def parse_label(value):
        try:
            parsed = ast.literal_eval(str(value))
            if isinstance(parsed, list):
                return ",".join(map(str, parsed))
            return str(parsed)
        except (ValueError, SyntaxError):
            return str(value)

    labels = df["labels"].map(parse_label)
    print(labels.value_counts(dropna=False).to_string())

print("\n===== POSSIBLE TARGET-LEAKAGE COLUMNS =====")
for column in [
    "ID", "N", "D", "G", "C", "A", "H", "M", "O",
    "labels", "target", "filepath", "filename",
    "Left-Diagnostic Keywords", "Right-Diagnostic Keywords"
]:
    if column in df.columns:
        print(f"- {column}")

print("\n===== AGE AND SEX SUMMARY =====")
for column in ["Patient Age", "Patient Sex"]:
    if column in df.columns:
        print(f"\n{column}")
        print(df[column].value_counts(dropna=False).head(20))
