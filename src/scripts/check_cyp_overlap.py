# This script checks the overlap between drugs in CYP priors and drugs present
# in the interaction networks from ChCh-Miner, Decagon, and DDInter.

import pandas as pd
from pathlib import Path

# --- paths ---
base = Path("C:/Users/minwo/Desktop/Dataset")
cyp_path = Path("C:/Users/minwo/Documents/GitHub/DDI_Masters/src/dataset/drug_cyp.tsv")

chch_file   = base / "ChCh-Miner_durgbank-chem-chem.tsv"
decagon_file = base / "ChChSe-Decagon_polypharmacy.csv"
ddinter_glob = list(base.glob("ddinter_downloads_code_*.csv"))

# --- load CYP table ---
cyp = pd.read_csv(cyp_path, sep="\t")
cyp_ids = set(cyp["drug_id"].astype(str))
print(f"CYP priors: {len(cyp_ids)} drugs")

# --- load ChCh-Miner ---
chch = pd.read_csv(chch_file, sep="\t", header=None)
chch.columns = ["drug_id_1", "drug_id_2"]
chch_ids = set(chch["drug_id_1"].astype(str)) | set(chch["drug_id_2"].astype(str))
print(f"ChCh-Miner: {len(chch_ids)} unique drugs")

# --- load Decagon ---
dec = pd.read_csv(decagon_file)
if "# STITCH 1" in dec.columns and "STITCH 2" in dec.columns:
    dec_ids = set(dec["# STITCH 1"].astype(str)) | set(dec["STITCH 2"].astype(str))
else:
    raise ValueError(f"Unexpected Decagon columns: {dec.columns.tolist()}")
print(f"Decagon: {len(dec_ids)} unique drugs")

# --- load DDInter shards ---
ddi_ids = set()
for f in ddinter_glob:
    df = pd.read_csv(f)
    ddi_ids |= set(df["Drug_A"].astype(str)) | set(df["Drug_B"].astype(str))
print(f"DDInter: {len(ddi_ids)} unique drugs")

# --- union of all network drugs ---
all_ddi_ids = chch_ids | dec_ids | ddi_ids
print(f"Union of all network drugs: {len(all_ddi_ids)}")

# --- overlap ---
overlap = all_ddi_ids & cyp_ids
print(f"Overlap: {len(overlap)} drugs")

if len(overlap) < 50:
    print("Example overlap:", list(overlap)[:20])
