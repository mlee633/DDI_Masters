# # -- MHD v2 priors -- #
# from pathlib import Path
# import pandas as pd
# import numpy as np
# from collections import defaultdict

# def _safe_read(path, **kw):
#     p = Path(path)
#     return pd.read_csv(p, **kw) if p.exists() else None

# def load_atc_map(data_dir):
#     """
#     Optional file: {data_dir}/drug_atc.csv with columns:
#       drug_id, atc_codes   (atc_codes is '|' separated if multiple)
#     """
#     df = _safe_read(Path(data_dir) / "drug_atc.csv")
#     if df is None: return {}
#     out = {}
#     for _, r in df.iterrows():
#         did = str(r["drug_id"])
#         codes = str(r["atc_codes"]).split("|") if pd.notna(r["atc_codes"]) else []
#         out[did] = [c.strip() for c in codes if c and c != "nan"]
#     return out

# def atc_features(u, v, atc_map):
#     A = atc_map.get(str(u), [])
#     B = atc_map.get(str(v), [])
#     if not A or not B:
#         return [0,0,0,0,0,0.0]  # level1..level5 match + Jaccard
#     def level(c, k): return c[:k] if len(c) >= k else None
#     lvl = [0,0,0,0,0]
#     for a in A:
#         for b in B:
#             for k in range(1,6):
#                 if level(a,k) and level(b,k) and level(a,k) == level(b,k):
#                     lvl[k-1] = 1
#     # Jaccard on full codes
#     JA = set(A); JB = set(B)
#     jacc = len(JA & JB) / max(1, len(JA | JB))
#     return lvl + [jacc]

# def load_cyp_table(data_dir):
#     """
#     Optional file: {data_dir}/drug_cyp.tsv with columns:
#       drug_id, cyp2d6_sub, cyp2d6_inh, cyp3a4_sub, cyp3a4_inh, ... (0/1)
#     """
#     df = _safe_read(Path(data_dir) / "drug_cyp.tsv", sep="\t")
#     if df is None: return None
#     df["drug_id"] = df["drug_id"].astype(str)
#     return df.set_index("drug_id")

# def cyp_features(u, v, cyp_df):
#     """
#     Automatically generate CYP prior features for any CYP columns in drug_cyp.tsv.

#     For each CYP enzyme (prefix), we add:
#       - sub_sub: both are substrates
#       - inh_inh: both are inhibitors
#       - inh_sub: one is inhibitor, the other substrate
#       - ind_sub: one is inducer, the other substrate
#     """
#     if cyp_df is None:
#         return []

#     u, v = str(u), str(v)
#     if u not in cyp_df.index or v not in cyp_df.index:
#         return []

#     feats = []
#     # find CYP prefixes: e.g. cyp3a4 from cyp3a4_sub / cyp3a4_inh / cyp3a4_ind
#     enzymes = sorted(set([c[:-4] for c in cyp_df.columns if c.endswith(("_sub","_inh","_ind"))]))

#     row_u = cyp_df.loc[u]
#     row_v = cyp_df.loc[v]

#     for enz in enzymes:
#         sub_u, inh_u, ind_u = row_u.get(f"{enz}_sub",0), row_u.get(f"{enz}_inh",0), row_u.get(f"{enz}_ind",0)
#         sub_v, inh_v, ind_v = row_v.get(f"{enz}_sub",0), row_v.get(f"{enz}_inh",0), row_v.get(f"{enz}_ind",0)

#         # Features per enzyme
#         feats.append(int(sub_u and sub_v))         # sub_sub
#         feats.append(int(inh_u and inh_v))         # inh_inh
#         feats.append(int((inh_u and sub_v) or (inh_v and sub_u)))  # inh_sub either way
#         feats.append(int((ind_u and sub_v) or (ind_v and sub_u)))  # ind_sub either way

#     return feats

# def atc_pair_features(u, v, atc_map):
#     return atc_features(u, v, atc_map)

# def cyp_pair_features(u, v, cyp_df):
#     return cyp_features(u, v, cyp_df)


# -- MHD v3 priors -- #
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- ATC helpers ----------

def _safe_read(path, **kw):
    p = Path(path)
    return pd.read_csv(p, **kw) if p.exists() else None

def load_atc_map(data_dir):
    """
    Optional file: {data_dir}/drug_atc.csv with columns:
      drug_id, atc_codes   (pipe '|' separated)
    """
    df = _safe_read(Path(data_dir) / "drug_atc.csv")
    if df is None:
        return {}
    df["drug_id"] = df["drug_id"].astype(str)
    out = {}
    for _, r in df.iterrows():
        did = str(r["drug_id"])
        codes = [] if pd.isna(r["atc_codes"]) else [c.strip() for c in str(r["atc_codes"]).split("|") if c.strip()]
        out[did] = codes
    return out

def atc_pair_features(u, v, atc_map):
    A = atc_map.get(str(u), [])
    B = atc_map.get(str(v), [])
    if not A or not B:
        return [0,0,0,0,0,0.0]  # lvl1..lvl5 match + Jaccard
    def lvl(c,k): return c[:k] if len(c) >= k else None
    lv = [0,0,0,0,0]
    for a in A:
        for b in B:
            for k in range(1,6):
                if lvl(a,k) and lvl(b,k) and lvl(a,k) == lvl(b,k):
                    lv[k-1] = 1
    jacc = len(set(A)&set(B)) / max(1, len(set(A)|set(B)))
    return lv + [jacc]

# ---------- CYP helpers ----------

def load_cyp_table(path_like):
    """
    Expect a TSV created by build_drug_cyp.py with columns:
      drug_id, cyp1a2_sub, cyp1a2_inh, cyp1a2_ind, ..., cyp3a4_sub, cyp3a4_inh, cyp3a4_ind
    """
    p = Path(path_like)
    if not p.exists():
        return None
    df = pd.read_csv(p, sep="\t")
    df["drug_id"] = df["drug_id"].astype(str)
    df = df.set_index("drug_id")
    # normalise to int 0/1
    for c in df.columns:
        df[c] = (df[c].fillna(0).astype(int) > 0).astype(int)
    return df

def cyp_pair_features(u, v, cyp_df):
    """
    Auto-generate CYP features for ANY enzymes present:
      For each enzyme prefix 'cypXYZ':
        - sub_sub
        - inh_inh
        - inh_sub (either direction)
        - ind_sub (either direction)
    Returns a flat list concatenating all enzymes in sorted order.
    """
    if cyp_df is None: return []
    u, v = str(u), str(v)
    if u not in cyp_df.index or v not in cyp_df.index: return []
    enzymes = sorted({c[:-4] for c in cyp_df.columns if c.endswith(("_sub","_inh","_ind"))})
    ru, rv = cyp_df.loc[u], cyp_df.loc[v]
    feats = []
    for enz in enzymes:
        su, iu, du = int(ru.get(f"{enz}_sub",0)), int(ru.get(f"{enz}_inh",0)), int(ru.get(f"{enz}_ind",0))
        sv, iv, dv = int(rv.get(f"{enz}_sub",0)), int(rv.get(f"{enz}_inh",0)), int(rv.get(f"{enz}_ind",0))
        feats += [
            int(su and sv),                                 # sub_sub
            int(iu and iv),                                 # inh_inh
            int((iu and sv) or (iv and su)),               # inh_sub either
            int((du and sv) or (dv and su)),               # ind_sub either
        ]
    return feats

def build_pair_priors(df_pairs, atc_map, cyp_df):
    """
    Returns:
      F  : np.ndarray [N, P] prior features per pair (ATC + CYP)
      meta: dict with sizes for bookkeeping
    """
    atc = [atc_pair_features(u,v, atc_map) for u,v in zip(df_pairs["drug_u"], df_pairs["drug_v"])]
    cyp = [cyp_pair_features(u,v, cyp_df) for u,v in zip(df_pairs["drug_u"], df_pairs["drug_v"])]
    # ensure consistent widths even if lists are empty
    atc_dim = 6 if len(atc)==0 or len(atc[0])==0 else len(atc[0])
    cyp_dim = 0 if len(cyp)==0 or len(cyp[0])==0 else len(cyp[0])
    if atc_dim == 0: atc = [[0]*0 for _ in range(len(df_pairs))]
    if cyp_dim == 0: cyp = [[0]*0 for _ in range(len(df_pairs))]
    F = np.concatenate([np.array(atc, dtype=np.float32), np.array(cyp, dtype=np.float32)], axis=1) if (atc_dim+cyp_dim)>0 else np.zeros((len(df_pairs),0),dtype=np.float32)
    meta = {"atc_dim": atc_dim, "cyp_dim": cyp_dim, "total_dim": F.shape[1]}
    return F, meta
