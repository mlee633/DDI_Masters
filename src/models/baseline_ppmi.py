import pandas as pd
import numpy as np
from collections import Counter

class PPMIBaseline:
    def __init__(self, pos_pairs_df: pd.DataFrame, smoothing=1.0):
        self.smoothing = smoothing
        # estimate co-occurrence stats from observed positives
        drugs = pd.concat([pos_pairs_df["drug_u"], pos_pairs_df["drug_v"]]).tolist()
        self.N = len(pos_pairs_df)
        self.count_u = Counter(drugs)
        self.count_pair = Counter([tuple(x) for x in pos_pairs_df[["drug_u","drug_v"]].itertuples(index=False, name=None)])
        self.total = sum(self.count_u.values())
    def score(self, u, v):
        a,b = (u,v) if u < v else (v,u)
        c_uv = self.count_pair.get((a,b), 0) + self.smoothing
        p_uv = c_uv / (self.N + self.smoothing*self.N)
        p_u = (self.count_u.get(a,0) + self.smoothing) / (self.total + self.smoothing*len(self.count_u))
        p_v = (self.count_u.get(b,0) + self.smoothing) / (self.total + self.smoothing*len(self.count_u))
        pmi = np.log((p_uv)/(p_u*p_v) + 1e-12)
        return max(0.0, pmi)  # PPMI
    def predict_proba(self, pairs_df: pd.DataFrame):
        scores = [self.score(r["drug_u"], r["drug_v"]) for _, r in pairs_df.iterrows()]
        # min-max scale to [0,1] for comparability with other models
        import numpy as np
        s = np.array(scores, dtype=float)
        if s.size == 0:
            return s
        lo, hi = s.min(), s.max()
        if hi > lo:
            s = (s - lo) / (hi - lo)
        else:
            s = np.zeros_like(s)
        return s
