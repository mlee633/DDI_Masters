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
from src.models.baseline_rule import RulePresenceModel
from src.models.baseline_ppmi import PPMIBaseline
from src.models.baseline_ml import MLModels

from src.models.kg_embeddings import (
    DistMultModel, RotatEModel, train_embedding_model, predict_embedding_model
)

def plot_curves(y_true, y_score, model_name, split, out_dir):
    """Save ROC and PR curves for one model/split"""
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

    # Experiment name from config, default "exp"
    expname = cfg.get("experiment", {}).get("name", "exp")

    # Base directory pattern
    base_name = f"{split_name}_{algo_tag}_{expname}_{date_tag}"
    out_dir = base_out / base_name

    # If exists, increment suffix
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

    # # --- DEBUG: subsample for stability test ---
    # sample_size = 50000   # try 50k pairs
    # if len(all_pairs) > sample_size:
    #     all_pairs = all_pairs.sample(n=sample_size, random_state=cfg["experiment"]["seed"])
    #     print(f"⚠️ Subsampled dataset to {len(all_pairs)} pairs for testing")
    # # --- DEBUG: subsample for stability test ---

    # Splits
    if cfg["experiment"]["split_type"] == "warm":
        tr, va, te = warm_split(
            all_pairs,
            cfg["experiment"]["test_size"],
            cfg["experiment"]["val_size"],
            cfg["experiment"]["seed"],
        )
    else:
        tr, va, te = cold_drug_split(
            all_pairs,
            cfg["experiment"]["test_size"],
            cfg["experiment"]["val_size"],
            cfg["experiment"]["seed"],
        )

    # Graph (TRAIN positives only)
    G = build_graph_from_pos(tr[tr["label"] == 1])

    # Features for classical ML
    X_tr = pair_struct_features(G, tr)
    y_tr = tr["label"].values
    X_va = pair_struct_features(G, va)
    y_va = va["label"].values
    X_te = pair_struct_features(G, te)
    y_te = te["label"].values

    rows = []

    # -------------------
    # B1: Rule presence
    # -------------------
    if cfg["models"]["use_rule"]:
        rule = RulePresenceModel(tr[tr["label"] == 1][["drug_u", "drug_v"]])
        for split_name, df, y in [("val", va, y_va), ("test", te, y_te)]:
            score = rule.predict_proba(df[["drug_u", "drug_v"]])
            met = compute_all(y, score)
            met["model"] = "B1_rule_presence"
            met["split"] = split_name
            rows.append(met)
            plot_curves(y, score, "B1_rule", split_name, out_dir)

    # -------------------
    # B0: PPMI baseline
    # -------------------
    if cfg["models"]["use_ppmi"]:
        ppmi = PPMIBaseline(tr[tr["label"] == 1][["drug_u", "drug_v"]])
        for split_name, df, y in [("val", va, y_va), ("test", te, y_te)]:
            score = ppmi.predict_proba(df[["drug_u", "drug_v"]])
            met = compute_all(y, score)
            met["model"] = "B0_ppmi"
            met["split"] = split_name
            rows.append(met)
            plot_curves(y, score, "B0_ppmi", split_name, out_dir)

    # -------------------
    # B2: Logistic Regression
    # -------------------
    if cfg["models"]["use_logreg"]:
        lr = MLModels("logreg").fit(X_tr, y_tr)
        for split_name, X, y in [("val", X_va, y_va), ("test", X_te, y_te)]:
            score = lr.predict_proba(X)
            met = compute_all(y, score)
            met["model"] = "B2_logreg"
            met["split"] = split_name
            rows.append(met)
            plot_curves(y, score, "B2_logreg", split_name, out_dir)

    # -------------------
    # B2: XGBoost
    # -------------------
    if cfg["models"]["use_xgboost"]:
        xgb = MLModels("xgboost").fit(X_tr, y_tr)
        for split_name, X, y in [("val", X_va, y_va), ("test", X_te, y_te)]:
            score = xgb.predict_proba(X)
            met = compute_all(y, score)
            met["model"] = "B2_xgboost"
            met["split"] = split_name
            rows.append(met)
            plot_curves(y, score, "B2_xgboost", split_name, out_dir)

    # -------------------
    # B3: KG Embeddings (DistMult & RotatE)
    # -------------------
    import torch

    drug_list = pd.unique(pd.concat([all_pairs["drug_u"], all_pairs["drug_v"]])).tolist()
    drug2id = {d: i for i, d in enumerate(drug_list)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if cfg["models"].get("use_distmult", False):
        print("Training DistMult...")
        dm = DistMultModel(len(drug2id), emb_dim=128)
        dm, y_val, s_val = train_embedding_model(dm, tr, va, drug2id, device=device, max_epochs=50, patience=10)
        y_te = te["label"].values
        s_te = predict_embedding_model(dm, te, drug2id, device=device)
        for split_name, y, s in [("val", y_val, s_val), ("test", y_te, s_te)]:
            met = compute_all(y, s)
            met["model"] = "B3_distmult"
            met["split"] = split_name
            rows.append(met)
            plot_curves(y, s, "B3_distmult", split_name, out_dir)

    if cfg["models"].get("use_rotate", False):
        print("Training RotatE...")
        rt = RotatEModel(len(drug2id), emb_dim=128)
        rt, y_val, s_val = train_embedding_model(rt, tr, va, drug2id, device=device, max_epochs=50, patience=10)
        y_te = te["label"].values
        s_te = predict_embedding_model(rt, te, drug2id, device=device)
        for split_name, y, s in [("val", y_val, s_val), ("test", y_te, s_te)]:
            met = compute_all(y, s)
            met["model"] = "B3_rotate"
            met["split"] = split_name
            rows.append(met)
            plot_curves(y, s, "B3_rotate", split_name, out_dir)

    # -------------------
    # Save results
    # -------------------
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "metrics_summary.csv", index=False)
    save_json({"config": cfg}, out_dir / "run_config.json")
    print("Saved:", out_dir / "metrics_summary.csv")


if __name__ == "__main__":
    main()
