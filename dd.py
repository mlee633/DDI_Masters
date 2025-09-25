# # import pandas as pd

# # # Load your CYP test set
# # cyp = pd.read_csv("CYP_testingset.csv")

# # # Keep only DrugBank entries (those most reliable)
# # cyp_db = cyp[cyp["Source"] == "DrugBank"].copy()

# # # You need a DrugBank vocab file with ID + Name
# # # Example: drugbank_vocab.csv with columns [drugbank_id, name]
# # vocab = pd.read_csv("drugbank_vocab.csv")  

# # # Lowercase + strip for matching
# # cyp_db["name_clean"] = cyp_db["Name"].str.lower().str.strip()
# # vocab["name_clean"] = vocab["name"].str.lower().str.strip()

# # # Merge
# # merged = cyp_db.merge(vocab, on="name_clean", how="left")

# # # Keep only matched ones
# # aligned = merged[["drugbank_id", "Name", "SMILES", "Label"]]

# # print(f"Aligned {aligned['drugbank_id'].notna().sum()} / {len(cyp_db)} drugs with DrugBank IDs")
# # aligned.to_csv("cyp_aligned.csv", index=False)


# import pandas as pd

# # load aligned CYP table
# cyp = pd.read_csv("src/dataset/drug_cyp_aligned.tsv", sep="\t")

# # load your network drug vocab (union of ChCh, Decagon, DDInter)
# chch = pd.read_csv("C:/Users/minwo/Desktop/Dataset/ChCh-Miner_durgbank-chem-chem.tsv", sep="\t", header=None)
# chch.columns = ["drug1", "drug2"]

# dec = pd.read_csv("C:/Users/minwo/Desktop/Dataset/ChChSe-Decagon_polypharmacy.csv")
# ddinter_files = [
#     "C:/Users/minwo/Desktop/Dataset/ddinter_downloads_code_A.csv",
#     "C:/Users/minwo/Desktop/Dataset/ddinter_downloads_code_B.csv",
#     "C:/Users/minwo/Desktop/Dataset/ddinter_downloads_code_D.csv",
#     "C:/Users/minwo/Desktop/Dataset/ddinter_downloads_code_H.csv",
#     "C:/Users/minwo/Desktop/Dataset/ddinter_downloads_code_L.csv",
#     "C:/Users/minwo/Desktop/Dataset/ddinter_downloads_code_P.csv",
#     "C:/Users/minwo/Desktop/Dataset/ddinter_downloads_code_R.csv",
#     "C:/Users/minwo/Desktop/Dataset/ddinter_downloads_code_V.csv",
# ]

# ddi_ids = set()
# for f in ddinter_files:
#     df = pd.read_csv(f)
#     ddi_ids |= set(df["Drug_A"].astype(str)) | set(df["Drug_B"].astype(str))

# chch_ids = set(chch["drug1"].astype(str)) | set(chch["drug2"].astype(str))
# dec_ids = set(dec["STITCH 1"].astype(str)) | set(dec["STITCH 2"].astype(str))

# all_ddi_ids = chch_ids | dec_ids | ddi_ids
# print("Total drugs in networks:", len(all_ddi_ids))

# # filter CYP to only those drugs present in your networks
# cyp_filtered = cyp[cyp["drug_id"].astype(str).isin(all_ddi_ids)].copy()
# print("CYP before:", cyp["drug_id"].nunique(), "after filtering:", cyp_filtered["drug_id"].nunique())

# # save updated file
# cyp_filtered.to_csv("src/dataset/drug_cyp_aligned_filtered.tsv", sep="\t", index=False)
# print("✅ Saved filtered CYP file → src/dataset/drug_cyp_aligned_filtered.tsv")


from src.features.priors import load_cyp_table
import pathlib

# Point directly to your CYP file
cyp_path = pathlib.Path("C:/Users/minwo/Documents/GitHub/DDI_Masters/src/dataset/drug_cyp.tsv")
cyp_df = load_cyp_table(cyp_path)

print("Shape:", cyp_df.shape)
print("Columns:", list(cyp_df.columns)[:10])  # first 10 columns
print("Example rows:")
print(cyp_df.head())