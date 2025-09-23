from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

def _safe_read(path, **kw):
    p = Path(path)
    return pd.read_csv(p, **kw) if p.exists() else None

def load_atc_map(data_dir):
    """
    Optional file: {data_dir}/drug_atc.csv with columns:
      drug_id, atc_codes   (atc_codes is '|' separated if multiple)
    """
    df = _safe_read(Path(data_dir) / "drug_atc.csv")
    if df is None: return {}
    out = {}
    for _, r in df.iterrows():
        did = str(r["drug_id"])
        codes = str(r["atc_codes"]).split("|") if pd.notna(r["atc_codes"]) else []
        out[did] = [c.strip() for c in codes if c and c != "nan"]
    return out

def atc_features(u, v, atc_map):
    A = atc_map.get(str(u), [])
    B = atc_map.get(str(v), [])
    if not A or not B:
        return [0,0,0,0,0,0.0]  # level1..level5 match + Jaccard
    def level(c, k): return c[:k] if len(c) >= k else None
    lvl = [0,0,0,0,0]
    for a in A:
        for b in B:
            for k in range(1,6):
                if level(a,k) and level(b,k) and level(a,k) == level(b,k):
                    lvl[k-1] = 1
    # Jaccard on full codes
    JA = set(A); JB = set(B)
    jacc = len(JA & JB) / max(1, len(JA | JB))
    return lvl + [jacc]

def load_cyp_table(data_dir):
    """
    Optional file: {data_dir}/drug_cyp.tsv with columns:
      drug_id, cyp2d6_sub, cyp2d6_inh, cyp3a4_sub, cyp3a4_inh, ... (0/1)
    """
    df = _safe_read(Path(data_dir) / "drug_cyp.tsv", sep="\t")
    if df is None: return None
    df["drug_id"] = df["drug_id"].astype(str)
    return df.set_index("drug_id")

def cyp_features(u, v, cyp_df):
    """
    Automatically generate CYP prior features for any CYP columns in drug_cyp.tsv.

    For each CYP enzyme (prefix), we add:
      - sub_sub: both are substrates
      - inh_inh: both are inhibitors
      - inh_sub: one is inhibitor, the other substrate
      - ind_sub: one is inducer, the other substrate
    """
    if cyp_df is None:
        return []

    u, v = str(u), str(v)
    if u not in cyp_df.index or v not in cyp_df.index:
        return []

    feats = []
    # find CYP prefixes: e.g. cyp3a4 from cyp3a4_sub / cyp3a4_inh / cyp3a4_ind
    enzymes = sorted(set([c[:-4] for c in cyp_df.columns if c.endswith(("_sub","_inh","_ind"))]))

    row_u = cyp_df.loc[u]
    row_v = cyp_df.loc[v]

    for enz in enzymes:
        sub_u, inh_u, ind_u = row_u.get(f"{enz}_sub",0), row_u.get(f"{enz}_inh",0), row_u.get(f"{enz}_ind",0)
        sub_v, inh_v, ind_v = row_v.get(f"{enz}_sub",0), row_v.get(f"{enz}_inh",0), row_v.get(f"{enz}_ind",0)

        # Features per enzyme
        feats.append(int(sub_u and sub_v))         # sub_sub
        feats.append(int(inh_u and inh_v))         # inh_inh
        feats.append(int((inh_u and sub_v) or (inh_v and sub_u)))  # inh_sub either way
        feats.append(int((ind_u and sub_v) or (ind_v and sub_u)))  # ind_sub either way

    return feats
