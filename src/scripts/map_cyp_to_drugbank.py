# # This script maps drugs from a CYP testing set to standardized drug names/IDs
# # using multiple sources: DDInter and ChCh-Miner. It outputs a TSV file with
# # aligned drug identifiers for further analysis.

# # Run this first then after build_drug_cyp.py

# import pandas as pd
# from pathlib import Path
# import unicodedata

# # ---------- helpers ----------
# def normalize_name(name: str) -> str:
#     """Basic cleanup: lowercase, strip, remove unicode accents."""
#     if not isinstance(name, str):
#         return ""
#     name = name.strip().lower()
#     name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
#     return name

# # ---------- main ----------
# def main():
#     data_dir = Path("C:/Users/minwo/Desktop/Dataset")
#     out_file = Path("C:/Users/minwo/Documents/GitHub/DDI_Masters/src/dataset/drug_cyp_aligned.tsv")

#     # 1. Load ALL CYP files
#     cyp_files = list(data_dir.glob("CYP*.csv"))
#     if not cyp_files:
#         raise FileNotFoundError(f"No CYP csv files found in {data_dir}")
#     all_cyp = []
#     for f in cyp_files:
#         df = pd.read_csv(f)
#         if "Name" not in df.columns:
#             continue
#         df["drug_id"] = df["Name"].apply(normalize_name)
#         all_cyp.append(df)
#     cyp = pd.concat(all_cyp, ignore_index=True)
#     print(f"Loaded {len(cyp)} CYP rows from {len(cyp_files)} files")

#     # 2. Build DDInter mapping
#     dd_map = {}
#     for csv in data_dir.glob("ddinter_downloads_code_*.csv"):
#         df = pd.read_csv(csv)
#         for col in ["Drug_A", "Drug_B"]:
#             for d in df[col].dropna().unique():
#                 dd_map[normalize_name(d)] = d
#     print(f"Built mapping of {len(dd_map)} unique names from DDInter")

#     # 3. Build ChCh-Miner fallback mapping
#     chch = pd.read_csv(data_dir / "ChCh-Miner_durgbank-chem-chem.tsv", sep="\t", header=None)
#     chch.columns = ["drug1", "drug2"]
#     chch = chch.astype(str)

#     chch_map = {}
#     for col in ["drug1", "drug2"]:
#         for d in chch[col].unique():
#             chch_map[normalize_name(d)] = d
#     print(f"Built fallback mapping of {len(chch_map)} names from ChCh-Miner")

#     # 4. Align CYP drugs
#     aligned = []
#     unmapped = []
#     for _, row in cyp.iterrows():
#         name = row["drug_id"]
#         if name in dd_map:
#             mapped = dd_map[name]
#         elif name in chch_map:
#             mapped = chch_map[name]
#         else:
#             unmapped.append(row["Name"])
#             continue
#         aligned.append([mapped] + row.drop("Name").tolist())

#     # 5. Save results
#     if aligned:
#         cols = ["drug_id"] + [c for c in cyp.columns if c != "Name"]
#         pd.DataFrame(aligned, columns=cols).to_csv(out_file, sep="\t", index=False)
#         print(f"✅ Saved aligned CYP table → {out_file}")
#     else:
#         print("⚠️ No CYP drugs could be mapped!")

#     # 6. Log unmapped
#     with open("unmapped_cyp.txt", "w", encoding="utf-8") as f:
#         for u in unmapped:
#             f.write(u + "\n")
#     print(f"Unmapped {len(unmapped)} drugs. Logged to unmapped_cyp.txt")

# if __name__ == "__main__":
#     main()

# src/scripts/map_cyp_to_drugbank.py

import pandas as pd
from pathlib import Path
import unicodedata

def normalize_name(name: str) -> str:
    """Basic cleanup: lowercase, strip, remove unicode accents."""
    if not isinstance(name, str):
        return ""
    name = name.strip().lower()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
    return name

def main():
    data_dir = Path("C:/Users/minwo/Desktop/Dataset")
    out_file = Path("C:/Users/minwo/Documents/GitHub/DDI_Masters/src/dataset/drug_cyp_aligned2.tsv")

    # 1. Load ALL CYP files
    cyp_files = list(data_dir.glob("CYP*.csv"))
    if not cyp_files:
        raise FileNotFoundError(f"No CYP csv files found in {data_dir}")

    all_cyp = []
    for f in cyp_files:
        df = pd.read_csv(f)
        if "Name" not in df.columns:
            continue
        df["drug_id"] = df["Name"].apply(normalize_name)
        # --- Add enzyme column from filename ---
        enzyme = f.stem.split("_")[0].lower()  # e.g. CYP1A2_trainingset.csv -> "cyp1a2"
        df["enzyme"] = enzyme
        all_cyp.append(df)

    cyp = pd.concat(all_cyp, ignore_index=True)
    print(f"Loaded {len(cyp)} CYP rows from {len(cyp_files)} files")

    # 2. Build DDInter mapping
    dd_map = {}
    for csv in data_dir.glob("ddinter_downloads_code_*.csv"):
        df = pd.read_csv(csv)
        for col in ["Drug_A", "Drug_B"]:
            for d in df[col].dropna().unique():
                dd_map[normalize_name(d)] = d
    print(f"Built mapping of {len(dd_map)} unique names from DDInter")

    # 3. Build ChCh-Miner fallback mapping
    chch = pd.read_csv(data_dir / "ChCh-Miner_durgbank-chem-chem.tsv", sep="\t", header=None)
    chch.columns = ["drug1", "drug2"]
    chch = chch.astype(str)

    chch_map = {}
    for col in ["drug1", "drug2"]:
        for d in chch[col].unique():
            chch_map[normalize_name(d)] = d
    print(f"Built fallback mapping of {len(chch_map)} names from ChCh-Miner")

    # 4. Align CYP drugs
    aligned = []
    unmapped = []
    for _, row in cyp.iterrows():
        name = row["drug_id"]
        if name in dd_map:
            mapped = dd_map[name]
        elif name in chch_map:
            mapped = chch_map[name]
        else:
            unmapped.append(row["Name"])
            continue
        aligned.append([mapped, row["SMILES"], row["Label"], row["Source"], row["enzyme"]])

    # 5. Save results
    if aligned:
        cols = ["drug_id", "SMILES", "Label", "Source", "enzyme"]
        pd.DataFrame(aligned, columns=cols).to_csv(out_file, sep="\t", index=False)
        print(f"✅ Saved aligned CYP table → {out_file}")
    else:
        print("⚠️ No CYP drugs could be mapped!")

    # 6. Log unmapped
    with open("unmapped_cyp.txt", "w", encoding="utf-8") as f:
        for u in unmapped:
            f.write(u + "\n")
    print(f"Unmapped {len(unmapped)} drugs. Logged to unmapped_cyp.txt")

if __name__ == "__main__":
    main()
