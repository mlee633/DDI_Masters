from pathlib import Path
import pandas as pd
import numpy as np
from .build_graph import canonicalize_pairs

ID_HINTS = ["drug", "stitch", "id", "name", "cid"]

def _find_drug_cols(df):
    cols = list(df.columns)
    lower = [c.lower() for c in cols]
    # try common patterns
    patterns = [
        ("drug1","drug2"), ("drug_1","drug_2"), ("drug_a","drug_b"),
        ("drugid1","drugid2"), ("stitch 1","stitch 2"), ("stitch1","stitch2"),
        ("drugbank1","drugbank2"), ("drugbank id 1","drugbank id 2")
    ]
    for a,b in patterns:
        if a in lower and b in lower:
            i,j = lower.index(a), lower.index(b)
            return cols[i], cols[j]
    # fallback: pick first two cols that look like drug identifiers
    candidates = [c for c in cols if any(h in c.lower() for h in ID_HINTS)]
    if len(candidates) >= 2:
        return candidates[0], candidates[1]
    # absolute fallback: first two columns
    return cols[0], cols[1]

def load_chch(path, sep="\t"):
    df = pd.read_csv(path, sep=sep, low_memory=False)
    c1, c2 = _find_drug_cols(df)
    df = df[[c1, c2]].copy()
    df.columns = ["drug_u","drug_v"]
    df["label"] = 1
    return canonicalize_pairs(df)

def load_ddinter(shards):
    dfs = []
    for p in shards:
        d = pd.read_csv(p, low_memory=False)
        c1, c2 = _find_drug_cols(d)
        d = d[[c1, c2]].copy()
        d.columns = ["drug_u","drug_v"]
        d["label"] = 1
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    return canonicalize_pairs(df)

def load_decagon(path):
    df = pd.read_csv(path, low_memory=False)
    # treat any reported polypharmacy relation as positive pair
    c1, c2 = _find_drug_cols(df)
    df = df[[c1, c2]].copy()
    df.columns = ["drug_u","drug_v"]
    df["label"] = 1
    return canonicalize_pairs(df)

def merge_sources(chch, ddinter, decagon):
    # union of positives
    pos = pd.concat([chch, ddinter, decagon], ignore_index=True)
    pos = pos.drop_duplicates(subset=["drug_u","drug_v"]).reset_index(drop=True)
    return pos
