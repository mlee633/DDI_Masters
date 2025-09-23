import os, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from src.utils.io import load_config, ensure_dir, save_json, set_seed
from src.data.ingest import load_chch, load_ddinter, load_decagon, merge_sources
from src.data.splits import warm_split, cold_drug_split, negative_sampling
from src.features.basic import build_graph_from_pos, pair_struct_features
from src.eval.metrics import compute_all

# Baselines
from src.models.baseline_rule import RulePresenceModel
from src.models.baseline_ppmi import PPMIBaseline
from src.models.baseline_ml import MLModels

# KG embeddings
from src.models.kg_embeddings import (
    DistMultModel, RotatEModel, train_embedding_model, predict_embedding_model, get_entity_embeddings
)

# Priors + MHD v2
from src.features.priors import load_atc_map, atc_features, load_cyp_table, cyp_features
from src.models.mhd_v2 import MHDv2

import numpy as np
import torch


def plot_curves(y_true, y_score, model_name, split, out_dir):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    # ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name} ({split})")
    plt.legend()
    plt.savefig(out_dir / f"fig_roc_{model_name}_{split}.png", dpi=300)
    plt.close()

    # PR
    plt.figure()
    plt.plot(rec, prec, label=f"AUC={auc(rec,prec):.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve: {model_name} ({split})")
    plt.legend()
    plt.savefig(out_dir / f"fig_pr_{model_name}_{split}.png", dpi=300)
    plt.close()


def main():
    cfg = load_config()
    set_seed(cfg["experiment"]["seed"])
    data_dir = Path(cfg["data"]["data_dir"])

    import datetime
    base_out = Path(cfg["output"]["dir"])
    date_tag = datetime.datetime.now().strftime("%m-%d")
    split_name = cfg["experiment"]["split_type"]

    # Experiment name from config, default "exp"
    expname = cfg.get("experiment", {}).get("name", "exp")

    # Detect which models are active
    active_models = []
    if cfg["models"].get("use_rule"): active_models.append("rule")
    if cfg["models"].get("use_ppmi"): active_models.append("ppmi")
    if cfg["models"].get("use_logreg"): active_models.append("logreg")
    if cfg["models"].get("use_xgboost"): active_models.append("xgb")
    if cfg["models"].get("use_distmult") or cfg["models"].get("use_rotate"):
        active_models.append("KG")

    if not active_models:
        algo_tag = "none"
    elif len(active_models) == 1:
        algo_tag = active_models[0]
    else:
        algo_tag = "multi"

    # Name output folder
    base_name = f"{split_name}_{algo_tag}_{expname}_{date_tag}"
    out_dir = base_out / base_name
    run_idx = 1
    while out_dir.exists():
        out_dir = base_out / f"{base_name}_run{run_idx}"
        run_idx += 1
    ensure_dir(out_dir)

    # Load sources
    chch = load_chch(data_dir / cfg["data"]["chch_file"], sep=cfg["data"]["sep_chch"])
    ddinter = load_ddinter(sorted(list(data_dir.glob(cfg["data"]["ddinter_shards_glob"]))))
    decagon = load_decagon(data_dir / cfg["data"]["decagon_file"])
    pos_all = merge_sources(chch, ddinter, decagon)

    # Negatives
    neg_all = negative_sampling(pos_all, ratio=cfg["experiment"]["n_neg_per_pos"], seed=cfg["experiment"]["seed"])
    all_pairs = pd.concat([pos_all.assign(label=1), neg_all], ignore_index=True)

    # Splits
    if cfg["experiment"]["split_type"] == "warm":
        tr, va, te = warm_split(all_pairs, cfg["experiment"]["test_size"], cfg["experiment"]["val_size"], cfg["experiment"]["seed"])
    else:
        tr, va, te = cold_drug_split(all_pairs, cfg["experiment"]["test_size"], cfg["experiment"]["val_size"], cfg["experiment"]["seed"])

    # Graph (TRAIN positives only)
    G = build_graph_from_pos(tr[tr["label"] == 1])

    # Features for classical ML
    X_tr = pair_struct_features(G, tr); y_tr = tr["label"].values
    X_va = pair_struct_features(G, va); y_va = va["label"].values
    X_te = pair_struct_features(G, te); y_te = te["label"].values

    rows = []

    # -------------------
    # B1: Rule
    # -------------------
    if cfg["models"]["use_rule"]:
        rule = RulePresenceModel(tr[tr["label"] == 1][["drug_u", "drug_v"]])
        for split, df, y in [("val", va, y_va), ("test", te, y_te)]:
            score = rule.predict_proba(df[["drug_u", "drug_v"]])
            met = compute_all(y, score); met["model"] = "B1_rule_presence"; met["split"] = split
            rows.append(met); plot_curves(y, score, "B1_rule", split, out_dir)

    # -------------------
    # B0: PPMI
    # -------------------
    if cfg["models"]["use_ppmi"]:
        ppmi = PPMIBaseline(tr[tr["label"] == 1][["drug_u", "drug_v"]])
        for split, df, y in [("val", va, y_va), ("test", te, y_te)]:
            score = ppmi.predict_proba(df[["drug_u", "drug_v"]])
            met = compute_all(y, score); met["model"] = "B0_ppmi"; met["split"] = split
            rows.append(met); plot_curves(y, score, "B0_ppmi", split, out_dir)

    # -------------------
    # B2: Logistic Regression
    # -------------------
    if cfg["models"]["use_logreg"]:
        lr = MLModels("logreg").fit(X_tr, y_tr)
        for split, X, y in [("val", X_va, y_va), ("test", X_te, y_te)]:
            score = lr.predict_proba(X)
            met = compute_all(y, score); met["model"] = "B2_logreg"; met["split"] = split
            rows.append(met); plot_curves(y, score, "B2_logreg", split, out_dir)

    # -------------------
    # B2: XGBoost
    # -------------------
    if cfg["models"]["use_xgboost"]:
        xgb = MLModels("xgboost").fit(X_tr, y_tr)
        for split, X, y in [("val", X_va, y_va), ("test", X_te, y_te)]:
            score = xgb.predict_proba(X)
            met = compute_all(y, score); met["model"] = "B2_xgboost"; met["split"] = split
            rows.append(met); plot_curves(y, score, "B2_xgboost", split, out_dir)

    # -------------------
    # B3: KG Embeddings
    # -------------------
    drug_list = pd.unique(pd.concat([all_pairs["drug_u"], all_pairs["drug_v"]])).tolist()
    drug2id = {d: i for i, d in enumerate(drug_list)}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg["models"].get("use_distmult", False):
        print("Training DistMult...")
        dm = DistMultModel(len(drug2id), emb_dim=128)
        dm, y_val, s_val = train_embedding_model(dm, tr, va, drug2id, device=device, max_epochs=50, patience=10)
        y_te = te["label"].values
        s_te = predict_embedding_model(dm, te, drug2id, device=device)
        for split, y, s in [("val", y_val, s_val), ("test", y_te, s_te)]:
            met = compute_all(y, s); met["model"] = "B3_distmult"; met["split"] = split
            rows.append(met); plot_curves(y, s, "B3_distmult", split, out_dir)

    if cfg["models"].get("use_rotate", False):
        print("Training RotatE...")
        rt = RotatEModel(len(drug2id), emb_dim=128)
        rt, y_val, s_val = train_embedding_model(rt, tr, va, drug2id, device=device, max_epochs=50, patience=10)
        y_te = te["label"].values
        s_te = predict_embedding_model(rt, te, drug2id, device=device)
        for split, y, s in [("val", y_val, s_val), ("test", y_te, s_te)]:
            met = compute_all(y, s); met["model"] = "B3_rotate"; met["split"] = split
            rows.append(met); plot_curves(y, s, "B3_rotate", split, out_dir)

    # -------------------
    # MHD v2
    # -------------------
    atc_map = load_atc_map(data_dir)
    cyp_df  = load_cyp_table(data_dir)
    E_rot = get_entity_embeddings(rt) if "rt" in locals() else None
    E_dm  = get_entity_embeddings(dm) if "dm" in locals() else None

    def score_from(model, df):
        return predict_embedding_model(model, df, drug2id, device)

    g_rot_tr = score_from(rt, tr) if "rt" in locals() else np.zeros(len(tr))
    g_rot_va = score_from(rt, va) if "rt" in locals() else np.zeros(len(va))
    g_rot_te = score_from(rt, te) if "rt" in locals() else np.zeros(len(te))

    g_dm_tr = score_from(dm, tr) if "dm" in locals() else np.zeros(len(tr))
    g_dm_va = score_from(dm, va) if "dm" in locals() else np.zeros(len(va))
    g_dm_te = score_from(dm, te) if "dm" in locals() else np.zeros(len(te))

    # Rule + PPMI
    pos_train = tr[tr["label"]==1][["drug_u","drug_v"]]
    ppmi = PPMIBaseline(pos_train)
    rule = RulePresenceModel(pos_train)
    def rule_vec(df): return df.apply(lambda r: 1.0 if (r["drug_u"], r["drug_v"]) in rule.lookup else 0.0, axis=1).values
    def ppmi_vec(df): return ppmi.predict_proba(df[["drug_u","drug_v"]])

    def prior_block(df):
        atc = np.array([atc_features(u,v, atc_map) for u,v in zip(df["drug_u"], df["drug_v"])], dtype=float)
        cyp = np.array([cyp_features(u,v, cyp_df) for u,v in zip(df["drug_u"], df["drug_v"])], dtype=float)
        return atc, cyp

    def emb_block(df, mhd2):
        blocks = []
        if E_rot is not None: blocks.append(mhd2._make_pair_embed_feats(df, E_rot, drug2id, None))
        if E_dm  is not None: blocks.append(mhd2._make_pair_embed_feats(df, E_dm,  drug2id, None))
        return np.concatenate(blocks, axis=1) if blocks else np.zeros((len(df), 0), dtype=np.float32)

    mhd2 = MHDv2(device=device)

    atc_tr, cyp_tr = prior_block(tr); atc_va, cyp_va = prior_block(va)
    train_blocks = [g_rot_tr.reshape(-1,1), g_dm_tr.reshape(-1,1),
                    X_tr.values, ppmi_vec(tr).reshape(-1,1), rule_vec(tr).reshape(-1,1),
                    atc_tr, cyp_tr, emb_block(tr, mhd2)]
    val_blocks = [g_rot_va.reshape(-1,1), g_dm_va.reshape(-1,1),
                  X_va.values, ppmi_vec(va).reshape(-1,1), rule_vec(va).reshape(-1,1),
                  atc_va, cyp_va, emb_block(va, mhd2)]

    mhd2.fit(train_blocks, val_blocks, tr["label"].values, va["label"].values)

    atc_te, cyp_te = prior_block(te)
    test_blocks = [g_rot_te.reshape(-1,1), g_dm_te.reshape(-1,1),
                   X_te.values, ppmi_vec(te).reshape(-1,1), rule_vec(te).reshape(-1,1),
                   atc_te, cyp_te, emb_block(te, mhd2)]

    for split, blocks, df in [("val", val_blocks, va), ("test", test_blocks, te)]:
        s = mhd2.predict_proba(blocks)
        met = compute_all(df["label"].values, s); met["model"] = "MHD_v2"; met["split"] = split
        rows.append(met); plot_curves(df["label"].values, s, "MHD_v2", split, out_dir)

    # -------------------
    # Save
    # -------------------
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "metrics_summary.csv", index=False)
    save_json({"config": cfg}, out_dir / "run_config.json")
    print("Saved:", out_dir / "metrics_summary.csv")


if __name__ == "__main__":
    main()
