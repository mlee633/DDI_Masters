import pandas as pd
from pathlib import Path

DATA_DIR = Path("src/dataset")
OUT_FILE = DATA_DIR / "drug_cyp.tsv"

def load_cyp_file(path, enzyme):
    """
    Expect columns like:
      compound, activity (substrate/inhibitor/inducer)
    or variations thereof.
    """
    df = pd.read_csv(path)
    # normalise column names
    df.columns = [c.lower() for c in df.columns]

    # guess drug identifier column
    id_col = None
    for c in ["drug_id","drug","compound","name","mol","id"]:
        if c in df.columns:
            id_col = c; break
    if id_col is None:
        raise ValueError(f"No drug ID column in {path}")

    # some datasets may have "activity" or "class"
    act_col = None
    for c in ["activity","class","label","role","type"]:
        if c in df.columns:
            act_col = c; break
    if act_col is None:
        raise ValueError(f"No activity/role column in {path}")

    rows = []
    for _, r in df.iterrows():
        did = str(r[id_col])
        act = str(r[act_col]).lower()
        sub = 1 if "substrate" in act else 0
        inh = 1 if "inhibitor" in act else 0
        ind = 1 if "inducer" in act else 0
        rows.append((did, sub, inh, ind))
    out = pd.DataFrame(rows, columns=["drug_id", f"{enzyme}_sub", f"{enzyme}_inh", f"{enzyme}_ind"])
    return out.groupby("drug_id").max().reset_index()

def main():
    all_files = sorted(DATA_DIR.glob("CYP*_*set.csv"))
    if not all_files:
        print("No CYP*.csv files found under", DATA_DIR); return

    merged = None
    for f in all_files:
        # enzyme name (e.g., CYP3A4 from CYP3A4_trainingset.csv)
        enzyme = f.name.split("_")[0].lower()  # cyp3a4
        part = load_cyp_file(f, enzyme)
        merged = part if merged is None else pd.merge(merged, part, on="drug_id", how="outer")

    # fill NaN → 0
    # collapse duplicate _x/_y columns
    for col in list(merged.columns):
        if col.endswith("_x"):
            base = col[:-2]
            col_y = base + "_y"
            if col_y in merged.columns:
                merged[base] = merged[[col, col_y]].max(axis=1)
                merged = merged.drop(columns=[col, col_y])

    # fill NaN -> 0 and cast to int
    merged = merged.fillna(0).astype({c:"int" for c in merged.columns if c!="drug_id"})


    merged.to_csv(OUT_FILE, sep="\t", index=False)
    print(f"Saved combined CYP table -> {OUT_FILE}")
    print("Shape:", merged.shape)
    print("Columns:", list(merged.columns))

if __name__ == "__main__":
    main()
