import pandas as pd

def canonicalize_pairs(df):
    # sort pair endpoints lexicographically to avoid duplicates
    a = df["drug_u"].astype(str)
    b = df["drug_v"].astype(str)
    u = a.where(a<=b, b)
    v = b.where(a<=b, a)
    out = df.copy()
    out["drug_u"] = u
    out["drug_v"] = v
    out = out.drop_duplicates(subset=["drug_u","drug_v"]).reset_index(drop=True)
    return out
