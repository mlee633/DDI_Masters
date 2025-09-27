import pandas as pd
from pathlib import Path
from src.data.ingest import load_chch, load_ddinter, load_decagon, merge_sources
from src.utils.io import load_config

def main():
    cfg = load_config("src/config/exp_mhd_v2_cold.yaml")  # or any config with data paths
    data_dir = Path(cfg["data"]["data_dir"])

    # --- Load raw datasets ---
    chch_file = data_dir / cfg["data"]["chch_file"]
    ddinter_glob = list(data_dir.glob(cfg["data"]["ddinter_shards_glob"]))
    decagon_file = data_dir / cfg["data"]["decagon_file"]

    print("=== Raw Dataset Statistics ===")
    chch = load_chch(chch_file, sep=cfg["data"]["sep_chch"])
    ddinter = load_ddinter(sorted(ddinter_glob))
    decagon = load_decagon(decagon_file)

    for name, df in [("ChCh-Miner", chch), ("DDInter", ddinter), ("Decagon", decagon)]:
        drugs = pd.concat([df["drug_u"], df["drug_v"]]).astype(str).str.lower().unique()
        print(f"{name}: {len(drugs)} unique drugs, {len(df)} pairs")

    # --- Merge harmonised dataset ---
    merged = merge_sources(chch, ddinter, decagon)
    merged_drugs = pd.concat([merged["drug_u"], merged["drug_v"]]).astype(str).str.lower().unique()

    print("\n=== Harmonised Dataset ===")
    print(f"Total unique drugs: {len(merged_drugs)}")
    print(f"Total positive pairs: {len(merged)}")

    # sanity check overlaps
    chch_drugs = set(pd.concat([chch["drug_u"], chch["drug_v"]]).astype(str).str.lower())
    ddinter_drugs = set(pd.concat([ddinter["drug_u"], ddinter["drug_v"]]).astype(str).str.lower())
    decagon_drugs = set(pd.concat([decagon["drug_u"], decagon["drug_v"]]).astype(str).str.lower())

    print("\n--- Drug Overlaps ---")
    print("ChCh ∩ DDInter:", len(chch_drugs & ddinter_drugs))
    print("ChCh ∩ Decagon:", len(chch_drugs & decagon_drugs))
    print("DDInter ∩ Decagon:", len(ddinter_drugs & decagon_drugs))
    print("All three:", len(chch_drugs & ddinter_drugs & decagon_drugs))

if __name__ == "__main__":
    main()