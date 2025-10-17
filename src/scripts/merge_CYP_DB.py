import pandas as pd

cyp = pd.read_csv("src/dataset/drug_cyp.tsv", sep="\t")
vocab = pd.read_csv("drugbank_vocab.csv")  # must contain: drug_id, name

# lower for consistency
cyp["name_clean"] = cyp["drug_id"].str.lower().str.strip()
vocab["name_clean"] = vocab["name"].str.lower().str.strip()

merged = cyp.merge(vocab[["drug_id", "name_clean"]], on="name_clean", how="left")

# replace names with DrugBank IDs
merged = merged.drop(columns=["drug_id"]).rename(columns={"drug_id":"drugbank_id"})
merged.to_csv("src/dataset/drug_cyp_aligned.tsv", sep="\t", index=False)
