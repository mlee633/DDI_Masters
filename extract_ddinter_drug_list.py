"""
Extract unique DDInter IDs and drug names from all DDInter shards (A–V)
and create a unified ddinter_drug_list.csv for DrugBank mapping.
Automatically detects column naming variations.
"""

import pandas as pd
from pathlib import Path

# === Set your directory path here ===
data_dir = Path(r"C:\Users\minwo\Desktop\Dataset")

# === Match all DDInter shard files ===
ddinter_files = list(data_dir.glob("ddinter_downloads_code_*.csv"))
print(f"📂 Found {len(ddinter_files)} DDInter shards")

all_rows = []

def find_col(cols, candidates):
    """Find first matching column name ignoring case and punctuation."""
    cols_lower = {c.lower().replace(" ", "").replace("_", ""): c for c in cols}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in cols_lower:
            return cols_lower[key]
    return None

for f in ddinter_files:
    df = pd.read_csv(f)
    cols = df.columns.tolist()

    idA = find_col(cols, ["DDInterID_A", "DDInterID A", "DDInterIDA"])
    drugA = find_col(cols, ["Drug A", "DrugA", "drug_A", "drugA"])
    idB = find_col(cols, ["DDInterID_B", "DDInterID B", "DDInterIDB"])
    drugB = find_col(cols, ["Drug B", "DrugB", "drug_B", "drugB"])

    if not idA or not drugA or not idB or not drugB:
        print(f"⚠️ Skipping {f.name} (missing expected columns)")
        continue

    df_a = df[[idA, drugA]].rename(columns={idA: "DDInterID", drugA: "DrugName"})
    df_b = df[[idB, drugB]].rename(columns={idB: "DDInterID", drugB: "DrugName"})

    all_rows.append(df_a)
    all_rows.append(df_b)

# Combine all unique entries
merged = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset=["DDInterID"]).sort_values("DDInterID")

# Export
output_file = data_dir / "ddinter_drug_list.csv"
merged.to_csv(output_file, index=False)

print(f"✅ Exported {len(merged)} unique drugs to {output_file}")
