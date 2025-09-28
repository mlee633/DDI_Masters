# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.metrics import log_loss
# from src.eval.metrics import compute_all

# from src.features.priors import load_atc_map, load_cyp_table, build_pair_priors


# # --------- utils ---------

# def focal_bce_with_logits(input, target, gamma=2.0, reduction="mean"):
#     """Focal BCE for imbalance."""
#     p = torch.sigmoid(input)
#     ce = F.binary_cross_entropy_with_logits(input, target, reduction="none")
#     p_t = p*target + (1-p)*(1-target)
#     loss = ( (1 - p_t) ** gamma ) * ce
#     return loss.mean() if reduction=="mean" else loss.sum()

# def temperature_scale(logits, y, grid=np.linspace(0.5, 2.0, 16)):
#     """Simple grid-search temperature on validation by NLL."""
#     best_T, best_nll = 1.0, 1e9
#     lg = logits.detach().cpu().numpy()
#     y = y.detach().cpu().numpy()
#     for T in grid:
#         nll = log_loss(y, 1/(1+np.exp(-lg/T)), labels=[0,1])
#         if nll < best_nll:
#             best_T, best_nll = T, nll
#     return float(best_T)

# # --------- model ---------

# class MHDV3(nn.Module):
#     def __init__(self, n_drugs, emb_dim=128, prior_dim=0, gamma=2.0):
#         super().__init__()
#         self.emb = nn.Embedding(n_drugs, emb_dim)
#         nn.init.xavier_uniform_(self.emb.weight)

#         self.gamma = gamma
#         in_dim = 4*emb_dim + prior_dim  # no need for prior-score in fusion
#         hid = max(128, emb_dim)

#         self.prior_head = nn.Linear(prior_dim if prior_dim>0 else 1, 1)
#         self.fuse = nn.Sequential(
#             nn.Linear(in_dim, hid),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hid, hid//2),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hid//2, 1)
#         )
#         self.temperature_ = 1.0

#     def pair_embed_feats(self, u_idx, v_idx):
#         eu = self.emb(u_idx); ev = self.emb(v_idx)
#         return torch.cat([eu, ev, (eu-ev).abs(), eu*ev], dim=-1)

#     def forward_logits(self, u_idx, v_idx, F_prior):
#         pairz = self.pair_embed_feats(u_idx, v_idx)
#         if F_prior.shape[1] == 0:
#             F_prior = torch.ones((pairz.size(0),1), device=pairz.device)
#         prior_score = self.prior_head(F_prior)
#         logits = self.fuse(torch.cat([pairz, F_prior], dim=-1))
#         # enforce priors matter: residual connection
#         logits = logits + 0.5 * prior_score
#         return logits.squeeze(-1), prior_score.squeeze(-1)

#     def supervised_loss(self, logits, y):
#         return focal_bce_with_logits(logits, y, gamma=self.gamma)

#     def counterfactual_loss(self, u_idx, v_idx, F_prior):
#         if F_prior.shape[1] == 0:
#             return torch.tensor(0.0, device=F_prior.device)
#         B, P = F_prior.shape
#         atc_len = 6 if P >= 6 else 0
#         cyp = F_prior.clone()
#         if P > atc_len:
#             cyp[:, :atc_len] = F_prior[:, :atc_len]
#             cyp[:, atc_len:] = 0.0
#         else:
#             cyp[:, :] = 0.0
#         logits_full, _ = self.forward_logits(u_idx, v_idx, F_prior)
#         logits_cf, _   = self.forward_logits(u_idx, v_idx, cyp)
#         return (logits_full - logits_cf).abs().mean()

#     def predict_proba(self, u_idx, v_idx, F_prior):
#         self.eval()
#         with torch.no_grad():
#             logits, _ = self.forward_logits(u_idx, v_idx, F_prior)
#             probs = torch.sigmoid(logits / self.temperature_)
#         return probs.detach().cpu().numpy()

# # --------- trainer ---------

# def train_mhd_v3(model, train_df, val_df, test_df, drug2id, F_tr, F_va, F_te,
#                  lr=5e-4, weight_decay=1e-5, max_epochs=50, patience=10,
#                  lambda_sup=1.0, lambda_cf=2.0,
#                  device="cpu", log_writer=None, log_file=None):

#     model = model.to(device)
#     opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

#     def to_tensors(df, F):
#         u = torch.tensor([drug2id[str(x)] for x in df["drug_u"]], device=device)
#         v = torch.tensor([drug2id[str(x)] for x in df["drug_v"]], device=device)
#         y = torch.tensor(df["label"].values.astype(np.float32), device=device)
#         Ft = torch.tensor(F, dtype=torch.float32, device=device)
#         return u, v, y, Ft

#     u_tr, v_tr, y_tr, Ft_tr = to_tensors(train_df, F_tr)
#     u_va, v_va, y_va, Ft_va = to_tensors(val_df,   F_va)
#     u_te, v_te, y_te, Ft_te = to_tensors(test_df,  F_te)

#     best = -1.0; best_state = None; wait = 0
#     B = 2048
#     n = len(u_tr)
#     for epoch in range(max_epochs):
#         model.train()
#         total, tot_sup, tot_cf = 0.0, 0.0, 0.0

#         # --- training batches ---
#         for start in range(0, n, B):
#             end = min(n, start+B)
#             uu, vv, yy, FF = u_tr[start:end], v_tr[start:end], y_tr[start:end], Ft_tr[start:end]
#             opt.zero_grad()
#             logits, _ = model.forward_logits(uu, vv, FF)
#             l_sup = model.supervised_loss(logits, yy)
#             l_cf  = model.counterfactual_loss(uu, vv, FF)
#             loss = lambda_sup*l_sup + lambda_cf*l_cf
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#             opt.step()
#             total   += loss.item() * (end-start)
#             tot_sup += l_sup.item() * (end-start)
#             tot_cf  += l_cf.item()  * (end-start)

#         # --- validation ---
#         model.eval()
#         with torch.no_grad():
#             va_logits, _ = model.forward_logits(u_va, v_va, Ft_va)
#             va_probs = torch.sigmoid(va_logits).cpu().numpy()
#         metrics = compute_all(y_va.detach().cpu().numpy(), va_probs)
#         auprc, auroc = metrics["AUPRC"], metrics["AUROC"]

#         print(f"Epoch {epoch+1}/{max_epochs} "
#               f"loss={total/n:.4f} sup={tot_sup/n:.4f} cf={tot_cf/n:.4f} "
#               f"| val AUPRC={auprc:.4f} AUROC={auroc:.4f}")

#         # --- NEW: log to CSV ---
#         if log_writer:
#             log_writer.writerow([epoch+1, total/n, auprc, auroc])
#             log_file.flush()

#         # --- early stopping ---
#         if auprc > best:
#             best, best_state, wait = auprc, {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}, 0
#         else:
#             wait += 1
#             if wait >= patience:
#                 print("Early stopping.")
#                 break

#     if best_state:
#         model.load_state_dict(best_state)

#     # temperature scaling
#     model.eval()
#     with torch.no_grad():
#         lg = model.forward_logits(u_va, v_va, Ft_va)[0]
#     T = temperature_scale(lg, y_va)
#     model.temperature_ = T

#     with torch.no_grad():
#         pv = torch.sigmoid(model.forward_logits(u_va, v_va, Ft_va)[0] / T).cpu().numpy()
#         pt = torch.sigmoid(model.forward_logits(u_te, v_te, Ft_te)[0] / T).cpu().numpy()
#     return model, (y_va.detach().cpu().numpy(), pv), (y_te.detach().cpu().numpy(), pt)


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss
from src.eval.metrics import compute_all

# --------- utils ---------

def focal_bce_with_logits(input, target, gamma=2.0, reduction="mean"):
    """Focal BCE for imbalance."""
    p = torch.sigmoid(input)
    ce = F.binary_cross_entropy_with_logits(input, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    loss = ((1 - p_t) ** gamma) * ce
    return loss.mean() if reduction == "mean" else loss.sum()

def temperature_scale(logits, y, grid=np.linspace(0.5, 2.0, 16)):
    """Simple grid-search temperature on validation by NLL."""
    best_T, best_nll = 1.0, 1e9
    lg = logits.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    for T in grid:
        nll = log_loss(y, 1 / (1 + np.exp(-lg / T)), labels=[0, 1])
        if nll < best_nll:
            best_T, best_nll = T, nll
    return float(best_T)

# --------- model ---------

class MHDV3(nn.Module):
    def __init__(self, n_drugs, emb_dim=128, prior_dim=0, gamma=2.0,
                 use_rgcn=False, rgcn_cfg=None, graph_data=None):
        """
        If use_rgcn is True, pass graph_data as a dict with:
          - 'num_nodes': int
          - 'edge_index': LongTensor [2, E]
          - 'edge_type' : LongTensor [E]
        """
        super().__init__()
        self.use_rgcn = use_rgcn
        self.graph_data = graph_data or {}
        self.gamma = gamma

        if use_rgcn:
            from src.models.rgcn_encoder import RGCNEncoder
            self.encoder = RGCNEncoder(
                n_drugs=n_drugs,
                emb_dim=emb_dim,
                num_rels=(rgcn_cfg or {}).get("num_rels", 4),
                num_layers=(rgcn_cfg or {}).get("num_layers", 2),
                dropout=(rgcn_cfg or {}).get("dropout", 0.2),
            )
        else:
            self.emb = nn.Embedding(n_drugs, emb_dim)
            nn.init.xavier_uniform_(self.emb.weight)

        in_dim = 4 * emb_dim + prior_dim
        hid = max(128, emb_dim)

        self.prior_head = nn.Linear(prior_dim if prior_dim > 0 else 1, 1)
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hid, hid // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hid // 2, 1),
        )
        self.temperature_ = 1.0

    def _rgcn_encode(self, device):
        x = self.graph_data["x"].to(device)
        h = self.encoder(
            x,
            self.graph_data["edge_index"].to(device),
            self.graph_data["edge_type"].to(device),
        )
        return h

    def pair_embed_feats(self, u_idx, v_idx):
        # u_idx, v_idx MUST be long
        if u_idx.dtype != torch.long: u_idx = u_idx.long()
        if v_idx.dtype != torch.long: v_idx = v_idx.long()

        if self.use_rgcn:
            h = self._rgcn_encode(u_idx.device)
            eu, ev = h[u_idx], h[v_idx]
        else:
            eu, ev = self.emb(u_idx), self.emb(v_idx)

        return torch.cat([eu, ev, (eu - ev).abs(), eu * ev], dim=-1)

    def forward_logits(self, u_idx, v_idx, F_prior):
        pairz = self.pair_embed_feats(u_idx, v_idx)
        if F_prior.shape[1] == 0:
            F_prior = torch.ones((pairz.size(0), 1), device=pairz.device)
        prior_score = self.prior_head(F_prior)
        logits = self.fuse(torch.cat([pairz, F_prior], dim=-1))
        # make priors matter via residual
        logits = logits + 0.5 * prior_score
        return logits.squeeze(-1), prior_score.squeeze(-1)

    def supervised_loss(self, logits, y):
        return focal_bce_with_logits(logits, y, gamma=self.gamma)

    def counterfactual_loss(self, u_idx, v_idx, F_prior):
        if F_prior.shape[1] == 0:
            return torch.tensor(0.0, device=F_prior.device)
        B, P = F_prior.shape
        atc_len = 6 if P >= 6 else 0
        cyp = F_prior.clone()
        if P > atc_len:
            cyp[:, :atc_len] = F_prior[:, :atc_len]
            cyp[:, atc_len:] = 0.0
        else:
            cyp[:, :] = 0.0
        logits_full, _ = self.forward_logits(u_idx, v_idx, F_prior)
        logits_cf, _ = self.forward_logits(u_idx, v_idx, cyp)
        return (logits_full - logits_cf).abs().mean()

    def predict_proba(self, u_idx, v_idx, F_prior):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward_logits(u_idx, v_idx, F_prior)
            probs = torch.sigmoid(logits / self.temperature_)
        return probs.detach().cpu().numpy()

# --------- trainer ---------

def train_mhd_v3(model, train_df, val_df, test_df, drug2id, F_tr, F_va, F_te,
                 lr=5e-4, weight_decay=1e-5, max_epochs=50, patience=10,
                 lambda_sup=1.0, lambda_cf=2.0,
                 device="cpu", log_writer=None, log_file=None):

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def to_tensors(df, F):
        u = torch.tensor([drug2id[str(x)] for x in df["drug_u"]], dtype=torch.long, device=device)
        v = torch.tensor([drug2id[str(x)] for x in df["drug_v"]], dtype=torch.long, device=device)
        y = torch.tensor(df["label"].values.astype(np.float32), dtype=torch.float32, device=device)
        Ft = torch.tensor(F, dtype=torch.float32, device=device)
        return u, v, y, Ft

    u_tr, v_tr, y_tr, Ft_tr = to_tensors(train_df, F_tr)
    u_va, v_va, y_va, Ft_va = to_tensors(val_df, F_va)
    u_te, v_te, y_te, Ft_te = to_tensors(test_df, F_te)

    best = -1.0
    best_state = None
    wait = 0
    B = 2048
    n = len(u_tr)

    for epoch in range(max_epochs):
        model.train()
        total, tot_sup, tot_cf = 0.0, 0.0, 0.0

        for start in range(0, n, B):
            end = min(n, start + B)
            uu, vv, yy, FF = u_tr[start:end], v_tr[start:end], y_tr[start:end], Ft_tr[start:end]
            opt.zero_grad()
            logits, _ = model.forward_logits(uu, vv, FF)
            l_sup = model.supervised_loss(logits, yy)
            l_cf = model.counterfactual_loss(uu, vv, FF)
            loss = lambda_sup * l_sup + lambda_cf * l_cf
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * (end - start)
            tot_sup += l_sup.item() * (end - start)
            tot_cf += l_cf.item() * (end - start)

        # validation
        model.eval()
        with torch.no_grad():
            va_logits, _ = model.forward_logits(u_va, v_va, Ft_va)
            va_probs = torch.sigmoid(va_logits).cpu().numpy()
        metrics = compute_all(y_va.detach().cpu().numpy(), va_probs)
        auprc, auroc = metrics["AUPRC"], metrics["AUROC"]

        print(f"Epoch {epoch+1}/{max_epochs} "
              f"loss={total/n:.4f} sup={tot_sup/n:.4f} cf={tot_cf/n:.4f} "
              f"| val AUPRC={auprc:.4f} AUROC={auroc:.4f}")

        if log_writer:
            log_writer.writerow([epoch + 1, total / n, auprc, auroc])
            log_file.flush()

        if auprc > best:
            best, best_state, wait = auprc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    if best_state:
        model.load_state_dict(best_state)

    # temperature scaling on validation
    model.eval()
    with torch.no_grad():
        lg = model.forward_logits(u_va, v_va, Ft_va)[0]
    T = temperature_scale(lg, y_va)
    model.temperature_ = T

    with torch.no_grad():
        pv = torch.sigmoid(model.forward_logits(u_va, v_va, Ft_va)[0] / T).cpu().numpy()
        pt = torch.sigmoid(model.forward_logits(u_te, v_te, Ft_te)[0] / T).cpu().numpy()
    return model, (y_va.detach().cpu().numpy(), pv), (y_te.detach().cpu().numpy(), pt)