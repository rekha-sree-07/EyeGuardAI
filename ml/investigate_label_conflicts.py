
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "dataset" / "full_df.csv"

df = pd.read_csv(CSV_PATH)

# Find IDs whose rows have different labels.
label_counts = df.groupby("ID")["labels"].nunique()
conflict_ids = label_counts[label_counts > 1].index

columns = [
    "ID",
    "Patient Age",
    "Patient Sex",
    "Left-Fundus",
    "Right-Fundus",
    "Left-Diagnostic Keywords",
    "Right-Diagnostic Keywords",
    "N", "D", "G", "C", "A", "H", "M", "O",
    "labels",
    "target",
    "filename",
]

print("Conflicting patient IDs:", len(conflict_ids))
print("\n===== FIRST 12 CONFLICTING PATIENTS =====")

sample = (
    df[df["ID"].isin(conflict_ids)]
    .sort_values(["ID", "filename"])
)

print(sample[columns].head(24).to_string(index=False))

print("\n===== LABEL COMBINATIONS =====")
combinations = (
    df[df["ID"].isin(conflict_ids)]
    .groupby("ID")["labels"]
    .agg(lambda values: " | ".join(sorted(set(values))))
)

print(combinations.value_counts().head(20).to_string())

print("\n===== FILENAME PATTERNS =====")
print(
    df.groupby("ID")["filename"]
    .agg(list)
    .loc[conflict_ids]
    .head(12)
    .to_string()
)
