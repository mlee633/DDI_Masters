import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import log_loss
from src.eval.metrics import compute_all

class MLPFusion(nn.Module):
    def __init__(self, in_dim, hidden=128, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class MHDv2:
    """
    Nonlinear fusion of:
      - RotatE score, DistMult score
      - Structural features (deg, jaccard, adamic_adar, RA)
      - Rule presence, PPMI
      - ATC (6 dims), CYP (6 dims) if available
      - Pairwise embedding interactions: [e_u, e_v, |e_u-e_v|, e_u*e_v] from RotatE (and optional DistMult)
    """
    def __init__(self, lr=1e-3, weight_decay=1e-5, max_epochs=200, patience=20, device="cpu"):
        self.lr = lr; self.wd = weight_decay
        self.max_epochs = max_epochs; self.patience = patience
        self.device = device
        self.model = None
        self.temperature_ = 1.0

    def _make_pair_embed_feats(self, df, E_rot, drug2id, E_dm=None):
        # assemble per-pair embedding features
        U = np.array([drug2id[str(u)] for u in df["drug_u"]])
        V = np.array([drug2id[str(v)] for v in df["drug_v"]])
        eu = E_rot[U]; ev = E_rot[V]
        feats = [eu, ev, np.abs(eu-ev), eu*ev]
        if E_dm is not None:
            eu2 = E_dm[U]; ev2 = E_dm[V]
            feats += [eu2, ev2, np.abs(eu2-ev2), eu2*ev2]
        return np.concatenate(feats, axis=1)

    def _stack(self, blocks):
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def fit(self, train_blocks, val_blocks, y_tr, y_va):
        X_tr = self._stack(train_blocks)
        X_va = self._stack(val_blocks)

        self.model = MLPFusion(X_tr.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        bce = nn.BCEWithLogitsLoss()

        best = -1.0; best_state = None; patience = 0
        tr = torch.tensor(X_tr, device=self.device); ytr = torch.tensor(y_tr, dtype=torch.float32, device=self.device)
        va = torch.tensor(X_va, device=self.device); yva = torch.tensor(y_va, dtype=torch.float32, device=self.device)

        for epoch in range(self.max_epochs):
            self.model.train()
            opt.zero_grad()
            logits = self.model(tr)
            loss = bce(logits, ytr)
            loss.backward(); opt.step()

            # val
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(va)
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
            auprc = compute_all(y_va, val_probs)["AUPRC"]
            if auprc > best:
                best = auprc; best_state = {k: v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.patience:
                    break

        # restore best
        if best_state:
            self.model.load_state_dict(best_state)

        # temperature scaling (simple grid)
        with torch.no_grad():
            v_logits = self.model(va).cpu().numpy()
        best_T, best_nll = 1.0, 1e9
        for T in np.linspace(0.5, 2.0, 16):
            nll = log_loss(y_va, 1/(1+np.exp(-v_logits/T)), labels=[0,1])
            if nll < best_nll: best_T, best_nll = T, nll
        self.temperature_ = float(best_T)
        return self

    def predict_proba(self, blocks):
        X = self._stack(blocks)
        x = torch.tensor(X, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x).cpu().numpy()
        return 1/(1+np.exp(-logits/self.temperature_))
