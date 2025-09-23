import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from src.eval.metrics import compute_all

class PairDataset(Dataset):
    def __init__(self, df, drug2id):
        self.u = [drug2id[x] for x in df["drug_u"].astype(str)]
        self.v = [drug2id[x] for x in df["drug_v"].astype(str)]
        self.y = df["label"].values.astype(np.float32)

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        return self.u[idx], self.v[idx], self.y[idx]


class DistMultModel(nn.Module):
    def __init__(self, n_entities, emb_dim=128):
        super().__init__()
        self.emb = nn.Embedding(n_entities, emb_dim)
        nn.init.xavier_uniform_(self.emb.weight.data)

    def score(self, u, v):
        e_u = self.emb(u)
        e_v = self.emb(v)
        return torch.sum(e_u * e_v, dim=-1)

    def forward(self, u, v):
        return torch.sigmoid(self.score(u, v))


class RotatEModel(nn.Module):
    def __init__(self, n_entities, emb_dim=128):
        super().__init__()
        self.emb_dim = emb_dim
        # real and imaginary parts
        self.emb_re = nn.Embedding(n_entities, emb_dim)
        self.emb_im = nn.Embedding(n_entities, emb_dim)
        nn.init.uniform_(self.emb_re.weight.data, -0.1, 0.1)
        nn.init.uniform_(self.emb_im.weight.data, -0.1, 0.1)

    def score(self, u, v):
        re_u, im_u = self.emb_re(u), self.emb_im(u)
        re_v, im_v = self.emb_re(v), self.emb_im(v)
        # elementwise complex product
        re = re_u * re_v - im_u * im_v
        im = re_u * im_v + im_u * re_v
        norm = torch.sqrt(re**2 + im**2).sum(dim=-1)
        return -norm

    def forward(self, u, v):
        return torch.sigmoid(self.score(u, v))

def train_embedding_model(
    model,
    train_df,
    val_df,
    drug2id,
    device="cpu",
    lr=1e-3,
    batch_size=512,
    max_epochs=100,
    patience=10,
):
    train_ds = PairDataset(train_df, drug2id)
    val_ds = PairDataset(val_df, drug2id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCELoss()

    best_val = -1.0
    best_state = None
    patience_ctr = 0
    y_val_best, s_val_best = None, None

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0

        # plain batch loop (no tqdm)
        for u, v, y in train_loader:
            u, v, y = u.to(device), v.to(device), y.to(device)
            pred = model(u, v)
            loss = bce(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for u, v, y in val_loader:
                u, v = u.to(device), v.to(device)
                preds = model(u, v).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())

        val_metrics = compute_all(all_labels, all_preds)
        val_auprc = val_metrics["AUPRC"]

        # epoch summary
        print(f"Epoch {epoch+1}/{max_epochs} "
            f"loss={total_loss/len(train_loader):.4f} "
            f"val_AUPRC={val_auprc:.4f}")

        # early stopping
        if val_auprc > best_val:
            best_val = val_auprc
            best_state = model.state_dict()
            y_val_best, s_val_best = all_labels, all_preds
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # restore best model
    if best_state:
        model.load_state_dict(best_state)

    return model, np.array(y_val_best), np.array(s_val_best)

def predict_embedding_model(model, df, drug2id, device="cpu", batch_size=512):
    ds = PairDataset(df, drug2id)
    loader = DataLoader(ds, batch_size=batch_size)
    model = model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for u, v, _ in loader:
            u, v = u.to(device), v.to(device)
            pred = model(u, v).cpu().numpy()
            preds.extend(pred)
    return np.array(preds)