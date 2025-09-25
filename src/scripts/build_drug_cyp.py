# This script merges multiple CYP training/testing CSV files into a single TSV file
# with binary features indicating whether a drug is a substrate for each CYP enzyme.

import pandas as pd
from pathlib import Path
import re

DATASET_DIR = Path("C:/Users/minwo/Documents/GitHub/DDI_Masters/src/dataset")
OUTPUT_FILE = DATASET_DIR / "drug_cyp.tsv"

def load_and_merge_cyp_files():
    all_files = list(DATASET_DIR.glob("CYP*set.csv"))
    print(f"Found {len(all_files)} CYP files")

    # Group training + testing by enzyme name
    grouped = {}
    for f in all_files:
        name = f.stem  # e.g. CYP1A2_trainingset
        m = re.match(r"(CYP\d+[A-Z]?\d*)_", name)
        if not m:
            print(f"⚠️ Skipping {f.name} (no CYP pattern)")
            continue
        enzyme = m.group(1).lower()  # e.g. cyp1a2
        grouped.setdefault(enzyme, []).append(f)

    data_frames = []
    for enzyme, files in grouped.items():
        print(f"Processing {enzyme} with {len(files)} file(s): {[f.name for f in files]}")
        dfs = []
        for f in files:
            df = pd.read_csv(f)
            if "Name" not in df.columns or "Label" not in df.columns:
                print(f"⚠️ Skipping {f.name}, missing Name/Label columns")
                continue
            df = df.rename(columns={"Name": "drug_id"})
            df = df[["drug_id", "Label"]].copy()
            df["drug_id"] = df["drug_id"].astype(str)
            # Use only substrates (Label==1)
            df[enzyme + "_sub"] = df["Label"].astype(int)
            dfs.append(df[["drug_id", enzyme + "_sub"]])
        if dfs:
            merged = pd.concat(dfs).groupby("drug_id").max().reset_index()
            data_frames.append(merged)

    if not data_frames:
        raise RuntimeError("No CYP data processed successfully!")

    # Merge all CYP features across enzymes
    wide = data_frames[0]
    for df in data_frames[1:]:
        wide = pd.merge(wide, df, on="drug_id", how="outer")

    wide = wide.fillna(0).astype({c: int for c in wide.columns if c != "drug_id"})
    return wide

def main():
    wide = load_and_merge_cyp_files()
    wide.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"✅ Saved combined CYP file → {OUTPUT_FILE} ({wide.shape})")
    print("Columns:", list(wide.columns))

if __name__ == "__main__":
    main()
