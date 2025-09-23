import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def warm_split(df, test_size=0.2, val_size=0.1, seed=42):
    train, test = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    train, val = train_test_split(train, test_size=val_size, random_state=seed, shuffle=True)
    return train, val, test

def cold_drug_split(df, test_size=0.2, val_size=0.1, seed=42):
    # hold out a set of drugs entirely
    rng = np.random.RandomState(seed)
    drugs = pd.unique(pd.concat([df["drug_u"], df["drug_v"]])).tolist()
    rng.shuffle(drugs)
    n_test = int(len(drugs)*test_size)
    n_val = int(len(drugs)*val_size)
    test_drugs = set(drugs[:n_test])
    val_drugs = set(drugs[n_test:n_test+n_val])
    def tag(row, S):
        return (row["drug_u"] in S) or (row["drug_v"] in S)
    test = df[df.apply(lambda r: tag(r, test_drugs), axis=1)].copy()
    remain = df[~df.index.isin(test.index)]
    val = remain[remain.apply(lambda r: tag(r, val_drugs), axis=1)].copy()
    train = remain[~remain.index.isin(val.index)].copy()
    # remove any leakage: ensure no val/test drug appears in train
    holdout = test_drugs | val_drugs
    train = train[~train["drug_u"].isin(holdout)]
    train = train[~train["drug_v"].isin(holdout)]
    return train, val, test

def negative_sampling(pos_df, ratio=1, seed=42):
    rng = np.random.RandomState(seed)
    drugs = pd.unique(pd.concat([pos_df["drug_u"], pos_df["drug_v"]])).tolist()
    pos_set = set(zip(pos_df["drug_u"], pos_df["drug_v"]))
    n_neg = int(len(pos_df) * ratio)
    neg = set()
    while len(neg) < n_neg:
        u = rng.choice(drugs); v = rng.choice(drugs)
        if u == v: continue
        a,b = (u,v) if u < v else (v,u)
        if (a,b) in pos_set or (a,b) in neg: continue
        neg.add((a,b))
    neg_df = pd.DataFrame(list(neg), columns=["drug_u","drug_v"])
    neg_df["label"] = 0
    return neg_df
