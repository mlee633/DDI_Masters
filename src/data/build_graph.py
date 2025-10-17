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

# # Testing Purposes:
# import os
# import glob
# import pandas as pd
# from typing import Dict, Any, List


# def canonicalize_pairs(df: pd.DataFrame) -> pd.DataFrame:
#     """Sort pair endpoints lexicographically to avoid duplicates."""
#     a = df["drug_u"].astype(str)
#     b = df["drug_v"].astype(str)
#     u = a.where(a <= b, b)
#     v = b.where(a <= b, a)
#     out = df.copy()
#     out["drug_u"] = u
#     out["drug_v"] = v
#     out = out.drop_duplicates(subset=["drug_u", "drug_v"]).reset_index(drop=True)
#     return out


# # ---------------------------------------------------------------------
# # NEW PART: Unified graph builder
# # ---------------------------------------------------------------------
# def load_edges_from_tsv(path: str, sep: str = "\t") -> pd.DataFrame:
#     # Read without header (first row is data)
#     df = pd.read_csv(path, sep=sep, header=None)
#     # Keep only first two columns (drug pairs)
#     df = df.iloc[:, :2]
#     df.columns = ["drug_u", "drug_v"]
#     return canonicalize_pairs(df)

# def load_edges_from_glob(pattern: str, sep: str = ",") -> pd.DataFrame:
#     """Load and unify multiple CSVs of DDI pairs (e.g. DDInter shards)."""
#     import glob
#     files = glob.glob(pattern)
#     if not files:
#         raise FileNotFoundError(f"No files matched pattern: {pattern}")

#     dfs = []
#     for f in files:
#         df = pd.read_csv(f, sep=sep)
#         # --- if file has no header, force header=None
#         if df.columns[0].startswith("DB") or df.columns[0].startswith("CHEM"):
#             df = pd.read_csv(f, sep=sep, header=None)
#             df = df.iloc[:, :2]
#             df.columns = ["drug_u", "drug_v"]
#         else:
#             # lowercase headers and rename typical variants
#             cols = [c.lower().strip() for c in df.columns]
#             df.columns = cols
#             rename_map = {}
#             for c in cols:
#                 if "drug" in c and ("1" in c or "a" in c or c.endswith("u")):
#                     rename_map[c] = "drug_u"
#                 elif "drug" in c and ("2" in c or "b" in c or c.endswith("v")):
#                     rename_map[c] = "drug_v"
#                 elif "chem" in c and ("1" in c or "a" in c or c.endswith("u")):
#                     rename_map[c] = "drug_u"
#                 elif "chem" in c and ("2" in c or "b" in c or c.endswith("v")):
#                     rename_map[c] = "drug_v"
#             df = df.rename(columns=rename_map)
#             if not {"drug_u", "drug_v"}.issubset(df.columns):
#                 # fallback: assume first two columns are the pair
#                 df = df.iloc[:, :2]
#                 df.columns = ["drug_u", "drug_v"]
#         dfs.append(df[["drug_u", "drug_v"]])

#     merged = pd.concat(dfs, ignore_index=True)
#     return canonicalize_pairs(merged)


# def build_graph(cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
#     """
#     Construct the unified DDI edge dataframe(s) according to config toggles.

#     Returns a dict of edge-type -> DataFrame
#     """
#     data_cfg = cfg["data"]
#     src_cfg = cfg.get("sources", {})
#     edges: Dict[str, pd.DataFrame] = {}

#     # --- DrugBank / ChCh-Miner
#     if src_cfg.get("use_chch", True):
#         path = os.path.join(data_cfg["data_dir"], data_cfg["chch_file"])
#         edges["drugbank"] = load_edges_from_tsv(path, sep=data_cfg.get("sep_chch", "\t"))

#     # --- DDInter
#     if src_cfg.get("use_ddinter", True):
#         pattern = os.path.join(data_cfg["data_dir"], data_cfg["ddinter_shards_glob"])
#         edges["ddinter"] = load_edges_from_glob(pattern)

#     # --- Decagon / TWOSIDES
#     if src_cfg.get("use_decagon", True):
#         path = os.path.join(data_cfg["data_dir"], data_cfg["decagon_file"])
#         edges["decagon"] = load_edges_from_tsv(path, sep=",")

#     # --- CYP relations
#     if src_cfg.get("use_cyp", True):
#         path = data_cfg["drug_cyp_file"]
#         edges["cyp"] = load_edges_from_tsv(path, sep="\t")

#     # --- ATC hierarchy (optional)
#     if src_cfg.get("use_atc", False) and "atc_file" in data_cfg:
#         path = data_cfg["atc_file"]
#         edges["atc"] = load_edges_from_tsv(path, sep="\t")

#     # Concatenate all active sources into a single DataFrame for training
#     if not edges:
#         raise ValueError("No data sources enabled in config.")

#     unified = pd.concat(edges.values(), ignore_index=True)
#     unified = canonicalize_pairs(unified)
#     edges["unified"] = unified
#     return edges
