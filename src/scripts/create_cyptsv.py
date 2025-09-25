import pandas as pd

# load aligned CYP table
cyp = pd.read_csv("src/dataset/drug_cyp_aligned.tsv", sep="\t")

# load your network drug vocab (union of ChCh, Decagon, DDInter)
chch = pd.read_csv("C:/Users/minwo/Desktop/Dataset/ChCh-Miner_durgbank-chem-chem.tsv", sep="\t", header=None)
chch.columns = ["drug1", "drug2"]

dec = pd.read_csv("C:/Users/minwo/Desktop/Dataset/ChChSe-Decagon_polypharmacy.csv")
ddinter_files = [
    "C:/Users/minwo/Desktop/Dataset/ddinter_downloads_code_A.csv",
    "C:/Users/minwo/Desktop/Dataset/ddinter_downloads_code_B.csv",
    "C:/Users/minwo/Desktop/Dataset/ddinter_downloads_code_D.csv",
    "C:/Users/minwo/Desktop/Dataset/ddinter_downloads_code_H.csv",
    "C:/Users/minwo/Desktop/Dataset/ddinter_downloads_code_L.csv",
    "C:/Users/minwo/Desktop/Dataset/ddinter_downloads_code_P.csv",
    "C:/Users/minwo/Desktop/Dataset/ddinter_downloads_code_R.csv",
    "C:/Users/minwo/Desktop/Dataset/ddinter_downloads_code_V.csv",
]

ddi_ids = set()
for f in ddinter_files:
    df = pd.read_csv(f)
    ddi_ids |= set(df["Drug_A"].astype(str)) | set(df["Drug_B"].astype(str))

chch_ids = set(chch["drug1"].astype(str)) | set(chch["drug2"].astype(str))
dec_ids = set(dec["STITCH 1"].astype(str)) | set(dec["STITCH 2"].astype(str))

all_ddi_ids = chch_ids | dec_ids | ddi_ids
print("Total drugs in networks:", len(all_ddi_ids))

# filter CYP to only those drugs present in your networks
cyp_filtered = cyp[cyp["drug_id"].astype(str).isin(all_ddi_ids)].copy()
print("CYP before:", cyp["drug_id"].nunique(), "after filtering:", cyp_filtered["drug_id"].nunique())

# save updated file
cyp_filtered.to_csv("src/dataset/drug_cyp_aligned_filtered.tsv", sep="\t", index=False)
print("✅ Saved filtered CYP file → src/dataset/drug_cyp_aligned_filtered.tsv")
