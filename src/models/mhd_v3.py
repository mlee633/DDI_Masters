import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss
from src.eval.metrics import compute_all

from src.features.priors import load_atc_map, load_cyp_table, build_pair_priors


# --------- utils ---------

def focal_bce_with_logits(input, target, gamma=2.0, reduction="mean"):
    """Focal BCE for imbalance."""
    p = torch.sigmoid(input)
    ce = F.binary_cross_entropy_with_logits(input, target, reduction="none")
    p_t = p*target + (1-p)*(1-target)
    loss = ( (1 - p_t) ** gamma ) * ce
    return loss.mean() if reduction=="mean" else loss.sum()

def temperature_scale(logits, y, grid=np.linspace(0.5, 2.0, 16)):
    """Simple grid-search temperature on validation by NLL."""
    best_T, best_nll = 1.0, 1e9
    lg = logits.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    for T in grid:
        nll = log_loss(y, 1/(1+np.exp(-lg/T)), labels=[0,1])
        if nll < best_nll:
            best_T, best_nll = T, nll
    return float(best_T)

# --------- model ---------

class MHDV3(nn.Module):
    """
    End-to-end mechanism-aware DDI scorer.
    - Learns drug embeddings (contrastive + supervised).
    - Consumes pair priors (ATC/CYP patterns).
    - Counterfactual regulariser that 'turns off' CYP features.

    Inputs at train time:
      - pair indices (u_id, v_id)
      - prior feature matrix F (per pair)
    """
    def __init__(self, n_drugs, emb_dim=128, prior_dim=0, gamma=2.0):
        super().__init__()
        self.emb = nn.Embedding(n_drugs, emb_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        self.gamma = gamma
        in_dim = 4*emb_dim + prior_dim + 1  # [eu, ev, |eu-ev|, eu*ev] + priors + prior-score
        hid = max(128, emb_dim)

        self.prior_head = nn.Linear(prior_dim if prior_dim>0 else 1, 1)  # learnable prior score
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hid, hid//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hid//2, 1)
        )
        self.temperature_ = 1.0

    def pair_embed_feats(self, u_idx, v_idx):
        eu = self.emb(u_idx); ev = self.emb(v_idx)
        return torch.cat([eu, ev, (eu-ev).abs(), eu*ev], dim=-1)

    def forward_logits(self, u_idx, v_idx, F_prior):
        pairz = self.pair_embed_feats(u_idx, v_idx)
        if F_prior.shape[1] == 0:
            prior_score = self.prior_head(torch.ones((F_prior.shape[0],1), device=F_prior.device))
            x = torch.cat([pairz, prior_score], dim=-1)
        else:
            prior_score = self.prior_head(F_prior)
            x = torch.cat([pairz, F_prior, prior_score], dim=-1)
        return self.fuse(x).squeeze(-1), prior_score.squeeze(-1)

    # ----- training utilities -----

    def contrastive_loss(self, u_idx, v_idx, tau=0.2):
        """In-batch InfoNCE on (u,v) positives using cosine sim of embeddings."""
        eu = F.normalize(self.emb(u_idx), dim=-1)
        ev = F.normalize(self.emb(v_idx), dim=-1)
        logits = (eu @ ev.t()) / tau                # [B,B]
        labels = torch.arange(eu.size(0), device=eu.device)
        return F.cross_entropy(logits, labels)

    def supervised_loss(self, logits, y):
        return focal_bce_with_logits(logits, y, gamma=self.gamma)

    def counterfactual_loss(self, u_idx, v_idx, F_prior):
        """Zero-out CYP features (heuristic: last quarter of prior vector often CYP if ATC exists first) and penalise jump."""
        if F_prior.shape[1] == 0:
            return torch.tensor(0.0, device=F_prior.device)
        B, P = F_prior.shape
        # heuristic split: assume ATC first (6 dims), CYP rest
        atc_len = 6 if P >= 6 else 0
        cyp = F_prior.clone()
        if P > atc_len:
            cyp[:, :atc_len] = F_prior[:, :atc_len]
            cyp[:, atc_len:] = 0.0
        else:
            cyp[:, :] = 0.0
        logits_full, _ = self.forward_logits(u_idx, v_idx, F_prior)
        logits_cf, _   = self.forward_logits(u_idx, v_idx, cyp)
        return (logits_full - logits_cf).abs().mean()

    # ----- prediction -----

    def predict_proba(self, u_idx, v_idx, F_prior):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward_logits(u_idx, v_idx, F_prior)
            probs = torch.sigmoid(logits / self.temperature_)
        return probs.detach().cpu().numpy()

# --------- trainer ---------

def train_mhd_v3(model, train_df, val_df, test_df, drug2id, F_tr, F_va, F_te,
                 lr=1e-3, weight_decay=5e-4, max_epochs=20, patience=15,
                 tau=0.2, lambda_sup=0.5, lambda_cf=0.5, device="cpu"):
    model = model.to(device)
    opt = torch.optim.Adam([
        {"params": model.emb.parameters(), "lr": lr * 0.1},  # embeddings fine-tuned slowly
        {"params": [p for n, p in model.named_parameters() if not n.startswith("emb.")], "lr": lr}
    ], weight_decay=weight_decay)

    def to_tensors(df, F):
        u = torch.tensor([drug2id[str(x)] for x in df["drug_u"]], device=device)
        v = torch.tensor([drug2id[str(x)] for x in df["drug_v"]], device=device)
        y = torch.tensor(df["label"].values.astype(np.float32), device=device)
        Ft = torch.tensor(F, dtype=torch.float32, device=device)
        return u, v, y, Ft

    u_tr, v_tr, y_tr, Ft_tr = to_tensors(train_df, F_tr)
    u_va, v_va, y_va, Ft_va = to_tensors(val_df,   F_va)
    u_te, v_te, y_te, Ft_te = to_tensors(test_df,  F_te)

    best = -1.0; best_state = None; wait = 0

    B = 2048  # full-batch is okay; if OOM, we can mini-batch later
    n = len(u_tr)

    for epoch in range(max_epochs):
        model.train()
        total = 0.0
        # simple chunking
        for start in range(0, n, B):
            end = min(n, start+B)
            uu, vv, yy, FF = u_tr[start:end], v_tr[start:end], y_tr[start:end], Ft_tr[start:end]
            opt.zero_grad()
            l_con = model.contrastive_loss(uu, vv, tau=tau)
            logits, _ = model.forward_logits(uu, vv, FF)
            l_sup = model.supervised_loss(logits, yy)
            l_cf  = model.counterfactual_loss(uu, vv, FF)
            loss = l_con + lambda_sup*l_sup + lambda_cf*l_cf
            loss.backward()
            opt.step()
            total += loss.item() * (end-start)

        # validation AUPRC
        model.eval()
        with torch.no_grad():
            va_logits, _ = model.forward_logits(u_va, v_va, Ft_va)
            va_probs = torch.sigmoid(va_logits).cpu().numpy()
        auprc = compute_all(y_va.detach().cpu().numpy(), va_probs)["AUPRC"]
        print(f"Epoch {epoch+1}/{max_epochs} loss={total/n:.4f} val AUPRC={auprc:.4f}")

        if auprc > best:
            best, best_state, wait = auprc, {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    if best_state:
        model.load_state_dict(best_state)

    # temperature scaling on val
    model.eval()
    with torch.no_grad():
        lg = model.forward_logits(u_va, v_va, Ft_va)[0]
    T = temperature_scale(lg, y_va)
    model.temperature_ = T

    # return predictions for val/test
    with torch.no_grad():
        pv = torch.sigmoid(model.forward_logits(u_va, v_va, Ft_va)[0] / T).cpu().numpy()
        pt = torch.sigmoid(model.forward_logits(u_te, v_te, Ft_te)[0] / T).cpu().numpy()
    return model, (y_va.detach().cpu().numpy(), pv), (y_te.detach().cpu().numpy(), pt)
