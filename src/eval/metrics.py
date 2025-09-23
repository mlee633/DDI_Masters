import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

def ece_score(probs, y, n_bins=10):
    probs = np.asarray(probs); y = np.asarray(y)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0; n = len(y)
    for i in range(n_bins):
        idx = (probs >= bins[i]) & (probs < bins[i+1])
        if idx.sum() == 0: continue
        conf = probs[idx].mean()
        acc = y[idx].mean()
        ece += (idx.mean()) * abs(acc - conf)
    return ece

def compute_all(y_true, y_score, ks=(10,20,50)):
    out = {}
    out["AUROC"] = roc_auc_score(y_true, y_score)
    out["AUPRC"] = average_precision_score(y_true, y_score)
    out["Brier"] = brier_score_loss(y_true, y_score)
    out["ECE@10"] = ece_score(y_score, y_true, n_bins=10)

    # Precision@K
    order = np.argsort(-y_score)
    y_sorted = np.array(y_true)[order]
    for k in ks:
        k = min(k, len(y_sorted))
        if k <= 0: continue
        out[f"Precision@{k}"] = y_sorted[:k].mean()
    return out
