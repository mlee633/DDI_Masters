# src/train_mhd_v1.py
# Train MHD_v1 (hybrid logistic regression baseline) for DDI prediction
# Uses structural features, KG scores, PPMI, Rule, with temperature scaling

import argparse, datetime, csv, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, log_loss

from src.utils.io import load_config, ensure_dir, save_json, set_seed
from src.data.ingest import load_chch, load_ddinter, load_decagon, merge_sources
from src.data.splits import warm_split, cold_drug_split, negative_sampling
from src.features.basic import build_graph_from_pos, pair_struct_features
from src.models.mhd_v1 import MHDHybrid
from src.models.kg_embeddings import RotatEModel, train_embedding_model, predict_embedding_model
from src.eval.metrics import compute_all

import torch


def plot_curves(y, s, name, split, out_dir):
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC {name} ({split})"); plt.legend()
    plt.savefig(out_dir / f"fig_roc_{name}_{split}.png", dpi=300); plt.close()

    plt.figure()
    plt.plot(rec, prec, label=f"AUC={auc(rec,prec):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR {name} ({split})"); plt.legend()
    plt.savefig(out_dir / f"fig_pr_{name}_{split}.png", dpi=300); plt.close()


def init_logger(out_dir, name):
    log_file = open(out_dir / f"{name}_log.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["epoch", "val_AUPRC", "val_AUROC"])
    return log_file, writer


def dump_preds(y, s, split, out_dir):
    np.savez_compressed(out_dir / f"preds_{split}.npz", y=y, s=s)
    fpr, tpr, _ = roc_curve(y, s)
    rec, prec, _ = precision_recall_curve(y, s)
    pd.DataFrame({"fpr":fpr,"tpr":tpr}).to_csv(out_dir/f"roc_points_{split}.csv", index=False)
    pd.DataFrame({"recall":rec,"precision":prec}).to_csv(out_dir/f"pr_points_{split}.csv", index=False)
    # threshold sweep
    ths = np.linspace(0,1,501)
    rows=[]
    for t in ths:
        p = (s>=t).astype(int)
        tp = ((p==1)&(y==1)).sum(); fp=((p==1)&(y==0)).sum()
        tn = ((p==0)&(y==0)).sum(); fn=((p==0)&(y==1)).sum()
        prec_ = tp/max(1,tp+fp); rec_ = tp/max(1,tp+fn)
        f1 = 0 if (prec_+rec_)==0 else 2*prec_*rec_/(prec_+rec_)
        rows.append({"thr":t,"precision":prec_,"recall":rec_,"f1":f1,
                    "tp":tp,"fp":fp,"tn":tn,"fn":fn})
    pd.DataFrame(rows).to_csv(out_dir/f"threshold_sweep_{split}.csv", index=False)


def pick_threshold(sweep_csv):
    df = pd.read_csv(sweep_csv)
    t_f1 = float(df.loc[df["f1"].idxmax(),"thr"])
    return {"max_f1": t_f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])

    data_dir = Path(cfg["data"]["data_dir"])
    base_out = Path(cfg["output"]["dir"])
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    split = cfg["experiment"]["split_type"]
    out_dir = base_out / f"mhd_v1_{split}_{ts}"
    ensure_dir(out_dir)

    # ---- load datasets ----
    chch = load_chch(data_dir / cfg["data"]["chch_file"], sep=cfg["data"]["sep_chch"])
    ddinter = load_ddinter(sorted(list(data_dir.glob(cfg["data"]["ddinter_shards_glob"]))))
    decagon = load_decagon(data_dir / cfg["data"]["decagon_file"])
    pos_all = merge_sources(chch, ddinter, decagon)
    neg_all = negative_sampling(pos_all, ratio=cfg["experiment"]["n_neg_per_pos"], seed=cfg["experiment"]["seed"])
    pairs = pd.concat([pos_all.assign(label=1), neg_all], ignore_index=True)

    # ---- splits ----
    if split == "warm":
        tr, va, te = warm_split(pairs, cfg["experiment"]["test_size"], cfg["experiment"]["val_size"], cfg["experiment"]["seed"])
    else:
        tr, va, te = cold_drug_split(pairs, cfg["experiment"]["test_size"], cfg["experiment"]["val_size"], cfg["experiment"]["seed"])

    # ---- structural features ----
    G = build_graph_from_pos(tr[tr["label"] == 1])
    psi_tr, psi_va, psi_te = pair_struct_features(G, tr), pair_struct_features(G, va), pair_struct_features(G, te)

    # ---- KG embedding scores (RotatE for g_phi) ----
    drugs = pd.unique(pd.concat([pairs["drug_u"], pairs["drug_v"]])).tolist()
    drug2id = {d:i for i,d in enumerate(drugs)}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Training RotatE for g_phi scores...")
    rt = RotatEModel(len(drug2id), emb_dim=128)
    rt, _, _ = train_embedding_model(rt, tr, va, drug2id, device=device, max_epochs=30, patience=5)
    g_tr = predict_embedding_model(rt, tr, drug2id, device=device)
    g_va = predict_embedding_model(rt, va, drug2id, device=device)
    g_te = predict_embedding_model(rt, te, drug2id, device=device)

    # ---- Train MHD v1 ----
    model = MHDHybrid()
    model.fit(G, tr, va, psi_tr, psi_va, g_tr, g_va)

    # ---- Epoch-like logging during temperature scaling ----
    log_file, writer = init_logger(out_dir, "MHD_v1")
    val_logits = model.lr.decision_function(model._stack_features(
        g_va, psi_va,
        model.ppmi.predict_proba(va[["drug_u","drug_v"]]),
        model._rule_vec(va)
    ))
    y_va = va["label"].values
    for i, T in enumerate(np.linspace(0.5, 2.0, 16), start=1):
        probs = 1/(1+np.exp(-val_logits/T))
        metrics = compute_all(y_va, probs)
        writer.writerow([i, metrics["AUPRC"], metrics["AUROC"]])
        log_file.flush()
    log_file.close()

    # ---- Final eval ----
    rows = []
    for split_name, df, psi, g in [("val", va, psi_va, g_va), ("test", te, psi_te, g_te)]:
        s = model.predict_proba_pairs(df, psi, g)
        y = df["label"].values
        met = compute_all(y, s); met["model"]="MHD_v1"; met["split"]=split_name
        rows.append(met); plot_curves(y, s, "MHD_v1", split_name, out_dir)
        dump_preds(y, s, split_name, out_dir)

    # ---- Select threshold (from val sweep) ----
    sel = pick_threshold(out_dir/"threshold_sweep_val.csv")
    (out_dir/"selected_thresholds.json").write_text(json.dumps(sel, indent=2))

    thr = sel["max_f1"]
    for split_name, df, psi, g in [("val", va, psi_va, g_va), ("test", te, psi_te, g_te)]:
        y = df["label"].values
        s = model.predict_proba_pairs(df, psi, g)
        p = (s>=thr).astype(int)
        cm = confusion_matrix(y, p).tolist()
        (out_dir/f"cmat_{split_name}.json").write_text(json.dumps({"thr":thr,"cm":cm}, indent=2))

    pd.DataFrame(rows).to_csv(out_dir/"metrics_summary.csv", index=False)
    save_json({"config": cfg}, out_dir/"run_config.json")
    print("Saved:", out_dir/"metrics_summary.csv")


if __name__ == "__main__":
    main()
