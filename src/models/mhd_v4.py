# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.metrics import log_loss
# from src.eval.metrics import compute_all

# # ---------- utils ----------
# def focal_bce_with_logits(input, target, gamma=2.0, reduction="mean"):
#     p = torch.sigmoid(input)
#     ce = F.binary_cross_entropy_with_logits(input, target, reduction="none")
#     p_t = p*target + (1-p)*(1-target)
#     loss = ((1 - p_t) ** gamma) * ce
#     return loss.mean() if reduction=="mean" else loss.sum()

# def temperature_scale(logits, y, grid=np.linspace(0.5, 2.0, 16)):
#     best_T, best_nll = 1.0, 1e9
#     lg = logits.detach().cpu().numpy()
#     y = y.detach().cpu().numpy()
#     for T in grid:
#         nll = log_loss(y, 1/(1+np.exp(-lg/T)), labels=[0,1])
#         if nll < best_nll:
#             best_T, best_nll = T, nll
#     return float(best_T)

# # ---------- model ----------
# class MHDV4(nn.Module):
#     def __init__(self, n_drugs, emb_dim=256, prior_dim=0, attn_dim=128, gamma=2.0,
#                  prior_dropout=0.1, gate_temp=0.5, prior_scale=0.3):
#         super().__init__()
#         self.emb = nn.Embedding(n_drugs, emb_dim)
#         nn.init.xavier_uniform_(self.emb.weight)
#         self.gamma = gamma
#         self.prior_dim = prior_dim
#         self.gate_temp = gate_temp
#         self.prior_scale = prior_scale

#         pair_dim = 4*emb_dim
#         self.q_proj = nn.Linear(pair_dim, attn_dim)
#         self.k_proj = nn.Linear(max(1, prior_dim), attn_dim)
#         self.v_proj = nn.Linear(max(1, prior_dim), attn_dim)
#         self.prior_dropout = nn.Dropout(prior_dropout)

#         self.gate_net = nn.Sequential(
#             nn.Linear(attn_dim + attn_dim, attn_dim),
#             nn.ReLU(),
#             nn.Linear(attn_dim, 1)
#         )

#         fusion_in = pair_dim + max(1, prior_dim) + attn_dim + pair_dim
#         self.q_to_pair = nn.Linear(attn_dim, pair_dim)
#         hid = max(256, emb_dim)
#         self.fuse = nn.Sequential(
#             nn.Linear(fusion_in, hid),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hid, hid//2),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(hid//2, 1)
#         )
#         self.temperature_ = 1.0
#         if prior_dim == 0:
#             self.null_prior = nn.Parameter(torch.zeros(1,1))
#             nn.init.normal_(self.null_prior, std=0.01)

#     def pair_embed_feats(self, u_idx, v_idx):
#         eu = self.emb(u_idx); ev = self.emb(v_idx)
#         return torch.cat([eu, ev, (eu-ev).abs(), eu*ev], dim=-1)

#     def forward_logits(self, u_idx, v_idx, F_prior):
#         B = u_idx.size(0)
#         q_pair = self.pair_embed_feats(u_idx, v_idx)
#         q = self.q_proj(q_pair)

#         if self.prior_dim == 0:
#             Fp = self.null_prior.expand(B, 1)
#         else:
#             Fp = F_prior
#         Fp = self.prior_dropout(Fp)

#         K = self.k_proj(Fp)
#         V = self.v_proj(Fp)
#         att = (q * K).sum(dim=-1, keepdim=True) / (q.size(-1) ** 0.5)
#         m = V * torch.sigmoid(att)

#         g = torch.sigmoid(self.gate_net(torch.cat([q, m], dim=-1)) * self.gate_temp)
#         g = torch.clamp(g, 0.2, 0.8)
#         m_g = g * m * self.prior_scale

#         m_as_pair = self.q_to_pair(m_g)
#         inter = q_pair * m_as_pair
#         fin = torch.cat([q_pair, Fp, m_g, inter], dim=-1)
#         logits = self.fuse(fin).squeeze(-1)
#         return logits, g.squeeze(-1)

#     def supervised_loss(self, logits, y):
#         return focal_bce_with_logits(logits, y, gamma=self.gamma)

#     def counterfactual_loss(self, u_idx, v_idx, F_prior, atc_len=6):
#         if self.prior_dim == 0:
#             return torch.tensor(0.0, device=F_prior.device)
#         B, P = F_prior.shape
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

# # ---------- trainer ----------
# def train_mhd_v4(model, train_df, val_df, test_df, drug2id, F_tr, F_va, F_te,
#                  lr=5e-4, weight_decay=1e-5, max_epochs=50, patience=10,
#                  lambda_sup=1.0, lambda_cf=1.0, lambda_gate=0.1,   # <— add this
#                  device="cpu", log_gate=False):
#     model = model.to(device)
#     opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

#     def to_tensors(df, F):
#         u = torch.tensor([drug2id[str(x)] for x in df["drug_u"]], device=device)
#         v = torch.tensor([drug2id[str(x)] for x in df["drug_v"]], device=device)
#         y = torch.tensor(df["label"].values.astype(np.float32), device=device)
#         Ft = torch.tensor(F, dtype=torch.float32, device=device) if F is not None else torch.zeros((len(df),0),device=device)
#         return u, v, y, Ft

#     u_tr, v_tr, y_tr, Ft_tr = to_tensors(train_df, F_tr)
#     u_va, v_va, y_va, Ft_va = to_tensors(val_df,   F_va)
#     u_te, v_te, y_te, Ft_te = to_tensors(test_df,  F_te)

#     best = -1.0; best_state = None; wait = 0
#     B = 2048
#     n = len(u_tr)

#     for epoch in range(max_epochs):
#         model.train()
#         total, tot_sup, tot_cf, gate_accum = 0.0, 0.0, 0.0, 0.0
#         for start in range(0, n, B):
#             end = min(n, start+B)
#             uu, vv, yy, FF = u_tr[start:end], v_tr[start:end], y_tr[start:end], Ft_tr[start:end]
#             opt.zero_grad()
#             logits, g = model.forward_logits(uu, vv, FF)
#             l_sup = model.supervised_loss(logits, yy)
#             l_cf  = model.counterfactual_loss(uu, vv, FF)
#             l_gate = ((g - 0.5)**2).mean()
#             loss = lambda_sup*l_sup + lambda_cf*l_cf + lambda_gate*l_gate
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#             opt.step()
#             total   += loss.item() * (end-start)
#             tot_sup += l_sup.item() * (end-start)
#             tot_cf  += l_cf.item()  * (end-start)
#             gate_accum += g.detach().sum().item()

#         # validation
#         model.eval()
#         with torch.no_grad():
#             va_logits, g_va = model.forward_logits(u_va, v_va, Ft_va)
#             va_probs = torch.sigmoid(va_logits).cpu().numpy()
#         auprc = compute_all(y_va.detach().cpu().numpy(), va_probs)["AUPRC"]
#         if log_gate:
#             mean_gate = float(gate_accum / n)
#             print(f"Epoch {epoch+1}/{max_epochs} loss={total/n:.4f} sup={tot_sup/n:.4f} cf={tot_cf/n:.4f} | gate~{mean_gate:.3f} | val AUPRC={auprc:.4f}")
#         else:
#             print(f"Epoch {epoch+1}/{max_epochs} loss={total/n:.4f} sup={tot_sup/n:.4f} cf={tot_cf/n:.4f} | val AUPRC={auprc:.4f}")

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

# ---------- utils ----------
def focal_bce_with_logits(input, target, gamma=2.0, reduction="mean"):
    p = torch.sigmoid(input)
    ce = F.binary_cross_entropy_with_logits(input, target, reduction="none")
    p_t = p*target + (1-p)*(1-target)
    loss = ((1 - p_t) ** gamma) * ce
    return loss.mean() if reduction=="mean" else loss.sum()

def temperature_scale(logits, y, grid=np.linspace(0.5, 2.0, 16)):
    best_T, best_nll = 1.0, 1e9
    lg = logits.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    for T in grid:
        nll = log_loss(y, 1/(1+np.exp(-lg/T)), labels=[0,1])
        if nll < best_nll:
            best_T, best_nll = T, nll
    return float(best_T)

# ---------- model ----------
class MHDV4(nn.Module):
    def __init__(self, n_drugs, emb_dim=256, prior_dim=0, attn_dim=128, gamma=2.0,
                 prior_dropout=0.1, gate_temp=0.5, prior_scale=0.3):
        super().__init__()
        self.emb = nn.Embedding(n_drugs, emb_dim)
        nn.init.xavier_uniform_(self.emb.weight)
        self.gamma = gamma
        self.prior_dim = prior_dim
        self.gate_temp = gate_temp
        self.prior_scale = prior_scale

        pair_dim = 4*emb_dim
        self.q_proj = nn.Linear(pair_dim, attn_dim)
        self.k_proj = nn.Linear(max(1, prior_dim), attn_dim)
        self.v_proj = nn.Linear(max(1, prior_dim), attn_dim)
        self.prior_dropout = nn.Dropout(prior_dropout)

        self.gate_net = nn.Sequential(
            nn.Linear(2*attn_dim, attn_dim),
            nn.ReLU(),
            nn.Linear(attn_dim, 1)
        )

        fusion_in = pair_dim + max(1, prior_dim) + attn_dim + pair_dim
        self.q_to_pair = nn.Linear(attn_dim, pair_dim)
        hid = max(256, emb_dim)
        self.fuse = nn.Sequential(
            nn.Linear(fusion_in, hid),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hid, hid//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hid//2, 1)
        )
        self.temperature_ = 1.0
        if prior_dim == 0:
            self.null_prior = nn.Parameter(torch.zeros(1,1))
            nn.init.normal_(self.null_prior, std=0.01)

    def pair_embed_feats(self, u_idx, v_idx):
        eu, ev = self.emb(u_idx), self.emb(v_idx)
        return torch.cat([eu, ev, (eu-ev).abs(), eu*ev], dim=-1)

    def forward_logits(self, u_idx, v_idx, F_prior):
        B = u_idx.size(0)
        q_pair = self.pair_embed_feats(u_idx, v_idx)
        q = self.q_proj(q_pair)

        if self.prior_dim == 0:
            Fp = self.null_prior.expand(B, 1)
        else:
            Fp = F_prior
        Fp = self.prior_dropout(Fp)

        K, V = self.k_proj(Fp), self.v_proj(Fp)
        att = (q * K).sum(dim=-1, keepdim=True) / (q.size(-1)**0.5)
        m = V * torch.sigmoid(att)

        g = torch.sigmoid(self.gate_net(torch.cat([q, m], dim=-1)) * self.gate_temp)
        g = torch.clamp(g, 0.2, 0.8)
        m_g = g * m * self.prior_scale

        m_as_pair = self.q_to_pair(m_g)
        inter = q_pair * m_as_pair
        fin = torch.cat([q_pair, Fp, m_g, inter], dim=-1)
        logits = self.fuse(fin).squeeze(-1)
        return logits, g.squeeze(-1)

    def supervised_loss(self, logits, y):
        return focal_bce_with_logits(logits, y, gamma=self.gamma)

    def counterfactual_loss(self, u_idx, v_idx, F_prior, atc_len=6):
        if self.prior_dim == 0:
            return torch.tensor(0.0, device=F_prior.device)
        B, P = F_prior.shape
        cyp = F_prior.clone()
        if P > atc_len:
            cyp[:, :atc_len] = F_prior[:, :atc_len]
            cyp[:, atc_len:] = 0.0
        else:
            cyp[:, :] = 0.0
        logits_full, _ = self.forward_logits(u_idx, v_idx, F_prior)
        logits_cf, _   = self.forward_logits(u_idx, v_idx, cyp)
        return (logits_full - logits_cf).abs().mean()

    def predict_proba(self, u_idx, v_idx, F_prior):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward_logits(u_idx, v_idx, F_prior)
            probs = torch.sigmoid(logits / self.temperature_)
        return probs.detach().cpu().numpy()

# ---------- trainer ----------
def train_mhd_v4(model, train_df, val_df, test_df, drug2id, F_tr, F_va, F_te,
                 lr=5e-4, weight_decay=1e-5, max_epochs=50, patience=10,
                 lambda_sup=1.0, lambda_cf=1.0, lambda_gate=0.1,
                 device="cpu", log_gate=False, log_writer=None, log_file=None):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def to_tensors(df, F):
        u = torch.tensor([drug2id[str(x)] for x in df["drug_u"]], device=device)
        v = torch.tensor([drug2id[str(x)] for x in df["drug_v"]], device=device)
        y = torch.tensor(df["label"].values.astype(np.float32), device=device)
        Ft = torch.tensor(F, dtype=torch.float32, device=device) if F is not None else torch.zeros((len(df),0),device=device)
        return u, v, y, Ft

    u_tr, v_tr, y_tr, Ft_tr = to_tensors(train_df, F_tr)
    u_va, v_va, y_va, Ft_va = to_tensors(val_df,   F_va)
    u_te, v_te, y_te, Ft_te = to_tensors(test_df,  F_te)

    best = -1.0; best_state = None; wait = 0
    B = 2048
    n = len(u_tr)

    for epoch in range(max_epochs):
        model.train()
        total, tot_sup, tot_cf, gate_accum = 0.0, 0.0, 0.0, 0.0

        for start in range(0, n, B):
            end = min(n, start+B)
            uu, vv, yy, FF = u_tr[start:end], v_tr[start:end], y_tr[start:end], Ft_tr[start:end]
            opt.zero_grad()
            logits, g = model.forward_logits(uu, vv, FF)
            l_sup = model.supervised_loss(logits, yy)
            l_cf  = model.counterfactual_loss(uu, vv, FF)
            l_gate = ((g - 0.5)**2).mean()
            loss = lambda_sup*l_sup + lambda_cf*l_cf + lambda_gate*l_gate
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total   += loss.item() * (end-start)
            tot_sup += l_sup.item() * (end-start)
            tot_cf  += l_cf.item()  * (end-start)
            gate_accum += g.detach().sum().item()

        # --- validation ---
        model.eval()
        with torch.no_grad():
            va_logits, g_va = model.forward_logits(u_va, v_va, Ft_va)
            va_probs = torch.sigmoid(va_logits).cpu().numpy()
        metrics = compute_all(y_va.detach().cpu().numpy(), va_probs)
        auprc, auroc = metrics["AUPRC"], metrics["AUROC"]

        mean_gate = float(gate_accum / n)
        if log_gate:
            print(f"Epoch {epoch+1}/{max_epochs} loss={total/n:.4f} sup={tot_sup/n:.4f} cf={tot_cf/n:.4f} | gate~{mean_gate:.3f} | val AUPRC={auprc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{max_epochs} loss={total/n:.4f} sup={tot_sup/n:.4f} cf={tot_cf/n:.4f} | val AUPRC={auprc:.4f}")

        # --- NEW: log to CSV ---
        if log_writer:
            log_writer.writerow([epoch+1, total/n, tot_sup/n, tot_cf/n, auprc, auroc, mean_gate])
            log_file.flush()

        # --- early stopping ---
        if auprc > best:
            best, best_state, wait = auprc, {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    if best_state:
        model.load_state_dict(best_state)

    # --- temperature scaling ---
    model.eval()
    with torch.no_grad():
        lg = model.forward_logits(u_va, v_va, Ft_va)[0]
    T = temperature_scale(lg, y_va)
    model.temperature_ = T

    with torch.no_grad():
        pv = torch.sigmoid(model.forward_logits(u_va, v_va, Ft_va)[0] / T).cpu().numpy()
        pt = torch.sigmoid(model.forward_logits(u_te, v_te, Ft_te)[0] / T).cpu().numpy()
    return model, (y_va.detach().cpu().numpy(), pv), (y_te.detach().cpu().numpy(), pt)
