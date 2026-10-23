
from pathlib import Path
import ast
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "dataset" / "full_df.csv"

df = pd.read_csv(CSV_PATH)

def parse_list(value):
    try:
        result = ast.literal_eval(str(value))
        return result if isinstance(result, list) else None
    except (ValueError, SyntaxError):
        return None

print("===== PATIENT ID CHECK =====")
print("Rows:", len(df))
print("Unique IDs:", df["ID"].nunique())
print("Repeated ID rows:", df["ID"].duplicated().sum())

print("\n===== LABEL VALIDATION =====")
valid_labels = {"N", "D", "G", "C", "A", "H", "M", "O"}

df["parsed_labels"] = df["labels"].apply(parse_list)
df["parsed_target"] = df["target"].apply(parse_list)

invalid_labels = df[
    df["parsed_labels"].apply(
        lambda x: not isinstance(x, list)
        or len(x) == 0
        or not set(x).issubset(valid_labels)
    )
]

print("Invalid or unparseable labels:", len(invalid_labels))

print("\n===== TARGET VECTOR CHECK =====")
def valid_target(value):
    return (
        isinstance(value, list)
        and len(value) == 8
        and all(x in (0, 1) for x in value)
    )

invalid_targets = df[
    ~df["parsed_target"].apply(valid_target)
]
print("Invalid target vectors:", len(invalid_targets))

print("\n===== LABEL AND TARGET CONSISTENCY =====")
class_order = ["N", "D", "G", "C", "A", "H", "M", "O"]

def target_matches(row):
    labels = row["parsed_labels"]
    target = row["parsed_target"]

    if not isinstance(labels, list) or not valid_target(target):
        return False

    expected = [int(label in labels) for label in class_order]
    return target == expected

df["label_target_match"] = df.apply(target_matches, axis=1)
print("Matching rows:", df["label_target_match"].sum())
print("Mismatching rows:", (~df["label_target_match"]).sum())

print("\n===== LABEL COUNTS =====")
print(df["labels"].value_counts(dropna=False))

print("\n===== EXAMPLE MISMATCHES =====")
columns = ["ID", "Patient Age", "Patient Sex", "labels", "target"]
print(
    df.loc[~df["label_target_match"], columns]
    .head(10)
    .to_string(index=False)
)

print("\n===== ROWS PER ID =====")
print(df.groupby("ID").size().value_counts().sort_index().to_string())
