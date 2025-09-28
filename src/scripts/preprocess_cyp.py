import sys
import pandas as pd

infile, outfile = sys.argv[1], sys.argv[2]
df = pd.read_csv(infile, sep="\t")

# Make column names lowercase for consistency
df.columns = [c.lower() for c in df.columns]

# Check required columns
assert "drug_id" in df.columns, "Missing drug_id column"
assert "enzyme" in df.columns, "Missing enzyme column"
assert "label" in df.columns, "Missing label column"

# Pivot into wide format
wide = df.pivot_table(
    index="drug_id",
    columns="enzyme",
    values="label",
    aggfunc="max",
    fill_value=0
).reset_index()

# Rename columns to match expected style: cypX_sub/inh/ind
wide.columns = [c.lower().replace(" ", "_") for c in wide.columns]

wide.to_csv(outfile, sep="\t", index=False)
print(f"Saved wide-format CYP table to {outfile} with shape {wide.shape}")
