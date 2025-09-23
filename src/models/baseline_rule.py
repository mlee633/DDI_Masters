import pandas as pd

class RulePresenceModel:
    """Predict 1 if pair is seen in any curated positive set (lookup table)."""
    def __init__(self, pos_pairs_df: pd.DataFrame):
        self.lookup = set(zip(pos_pairs_df["drug_u"], pos_pairs_df["drug_v"]))
    def predict_proba(self, pairs_df: pd.DataFrame):
        return pairs_df.apply(lambda r: 1.0 if (r["drug_u"], r["drug_v"]) in self.lookup else 0.0, axis=1).values
