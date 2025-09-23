import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss
from .baseline_ppmi import PPMIBaseline
from .baseline_rule import RulePresenceModel

class MHDHybrid:
    """
    Mechanism-Aware Hybrid for DDI (MHD)
    score(u,v) = alpha * g_phi(u,v) + w^T psi(u,v) + beta*PPMI(u,v) + gamma*Rule(u,v) + b
    We implement this as a logistic regression on concatenated features:
      [ g_phi(u,v), psi(u,v), PPMI(u,v), Rule(u,v) ]
    where g_phi comes from a pre-trained embedding model (DistMult/RotatE).
    """
    def __init__(self, alpha_init=1.0, C=1.0):
        self.alpha_init = alpha_init
        self.C = C
        self.lr = LogisticRegression(max_iter=500, class_weight="balanced", C=C)
        self.ppmi = None
        self.rule = None
        self.temperature_ = 1.0  # post-hoc temperature
    
    def _stack_features(self, g_scores, psi_df, ppmi_scores, rule_scores):
        X = np.column_stack([
            g_scores.reshape(-1),
            psi_df.values,           # deg_u, deg_v, jaccard, adamic_adar, res_alloc
            ppmi_scores.reshape(-1),
            rule_scores.reshape(-1)
        ])
        return X

    def fit(self, G, train_df, val_df, psi_tr, psi_va, g_tr, g_va):
        """
        G: graph built from TRAIN positives (for rule lookup safety)
        train_df/val_df: must include ['drug_u','drug_v','label']
        psi_tr/psi_va: structural feature DataFrames aligned with train/val
        g_tr/g_va: embedding-based scores aligned with train/val (from DistMult/RotatE)
        """
        # Fit PPMI and Rule using TRAIN positives only
        pos_train = train_df[train_df["label"]==1][["drug_u","drug_v"]]
        self.ppmi = PPMIBaseline(pos_train)
        self.rule = RulePresenceModel(pos_train)

        ppmi_tr = self.ppmi.predict_proba(train_df[["drug_u","drug_v"]])
        ppmi_va = self.ppmi.predict_proba(val_df[["drug_u","drug_v"]])
        rule_tr = self._rule_vec(train_df)
        rule_va = self._rule_vec(val_df)

        X_tr = self._stack_features(g_tr, psi_tr, ppmi_tr, rule_tr)
        y_tr = train_df["label"].values
        X_va = self._stack_features(g_va, psi_va, ppmi_va, rule_va)
        y_va = val_df["label"].values

        self.lr.fit(X_tr, y_tr)

        # Temperature scaling on validation (minimise NLL)
        # simple grid over T in [0.5..2.0]
        val_logits = self.lr.decision_function(X_va)
        best_T, best_nll = 1.0, np.inf
        for T in np.linspace(0.5, 2.0, 16):
            nll = log_loss(y_va, 1/(1+np.exp(-val_logits/T)), labels=[0,1])
            if nll < best_nll:
                best_nll, best_T = nll, T
        self.temperature_ = float(best_T)
        return self

    def _rule_vec(self, df):
        return df.apply(lambda r: 1.0 if (r["drug_u"], r["drug_v"]) in self.rule.lookup else 0.0, axis=1).values

    def predict_proba(self, psi_df, g_scores):
        logits = self.lr.decision_function(self._stack_features(
            g_scores, psi_df,
            ppmi_scores=np.zeros(len(psi_df)),  # placeholder, we compute per-batch in predict_proba_pairs
            rule_scores=np.zeros(len(psi_df))
        ))
        # Temperature scaling
        return 1/(1+np.exp(-logits/self.temperature_))

    def predict_proba_pairs(self, df_pairs, psi_df, g_scores):
        # compute fresh PPMI/Rule features for these pairs
        ppmi = self.ppmi.predict_proba(df_pairs[["drug_u","drug_v"]])
        rule = self._rule_vec(df_pairs)
        X = self._stack_features(g_scores, psi_df, ppmi, rule)
        logits = self.lr.decision_function(X)
        return 1/(1+np.exp(-logits/self.temperature_))
