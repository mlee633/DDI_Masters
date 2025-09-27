# # This script trains the MHD_v4 model for DDI prediction using drug embeddings
# # initialized from a pre-trained RotatE model. It incorporates drug pair priors

# import argparse, datetime, json
# from pathlib import Path
# import pandas as pd
# import torch
# import numpy as np

# from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
# import matplotlib.pyplot as plt
# from src.utils.io import load_config, ensure_dir, save_json, set_seed
# from src.data.ingest import load_chch, load_ddinter, load_decagon, merge_sources
# from src.data.splits import warm_split, cold_drug_split, negative_sampling
# from src.features.priors import load_atc_map, load_cyp_table, build_pair_priors
# from src.models.mhd_v4 import MHDV4, train_mhd_v4
# from src.eval.metrics import compute_all


# def plot_curves(y, s, name, split, out_dir):
#     fpr, tpr, _ = roc_curve(y, s); prec, rec, _ = precision_recall_curve(y, s)
#     plt.figure(); plt.plot(fpr,tpr,label=f"AUC={auc(fpr,tpr):.3f}"); plt.plot([0,1],[0,1],"k--")
#     plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC {name} ({split})"); plt.legend()
#     plt.savefig(out_dir / f"fig_roc_{name}_{split}.png", dpi=300); plt.close()
#     plt.figure(); plt.plot(rec,prec,label=f"AUC={auc(rec,prec):.3f}")
#     plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR {name} ({split})"); plt.legend()
#     plt.savefig(out_dir / f"fig_pr_{name}_{split}.png", dpi=300); plt.close()

# def load_rotate_embeddings(ckpt_path, drug2id_json):
#     ckpt_path = Path(ckpt_path)
#     import torch, json
#     if ckpt_path.is_dir():
#         sd = torch.load(ckpt_path / "data.pkl", map_location="cpu")
#     else:
#         sd = torch.load(ckpt_path, map_location="cpu")
#     E = torch.cat([sd["emb_re.weight"], sd["emb_im.weight"]], dim=1)
#     with open(drug2id_json, "r") as f:
#         rot_d2i = {k: int(v) for k,v in json.load(f).items()}
#     return E, rot_d2i

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", type=str, default="configs/exp_mhd_v4.yaml")
#     args = ap.parse_args()

#     cfg = load_config(args.config); set_seed(cfg["experiment"]["seed"])
#     data_dir = Path(cfg["data"]["data_dir"])
#     base_out = Path(cfg["output"]["dir"])
#     ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
#     split_name = cfg["experiment"]["split_type"]
#     out_dir = base_out / f"mhd_v4_{split_name}_{ts}"; ensure_dir(out_dir)

#     chch = load_chch(data_dir / cfg["data"]["chch_file"], sep=cfg["data"].get("sep_chch","\t"))
#     ddinter = load_ddinter(sorted(list((data_dir if data_dir.exists() else Path(".")).glob(cfg["data"]["ddinter_shards_glob"]))))
#     decagon = load_decagon(data_dir / cfg["data"]["decagon_file"])
#     pos_all = merge_sources(chch, ddinter, decagon)
#     neg_all = negative_sampling(pos_all, ratio=cfg["experiment"]["n_neg_per_pos"], seed=cfg["experiment"]["seed"])
#     pairs = pd.concat([pos_all.assign(label=1), neg_all], ignore_index=True)

#     if split_name == "warm":
#         tr, va, te = warm_split(pairs, cfg["experiment"]["test_size"], cfg["experiment"]["val_size"], cfg["experiment"]["seed"])
#     else:
#         tr, va, te = cold_drug_split(pairs, cfg["experiment"]["test_size"], cfg["experiment"]["val_size"], cfg["experiment"]["seed"])

#     drugs = pd.unique(pd.concat([pairs["drug_u"], pairs["drug_v"]])).astype(str).tolist()
#     drug2id = {d:i for i,d in enumerate(drugs)}

#     atc_map = load_atc_map(data_dir)
#     cyp_df  = load_cyp_table(cfg["data"]["drug_cyp_file"])
#     F_tr, meta_tr = build_pair_priors(tr, atc_map, cyp_df)
#     F_va, _       = build_pair_priors(va, atc_map, cyp_df)
#     F_te, _       = build_pair_priors(te, atc_map, cyp_df)
#     prior_dim = F_tr.shape[1]

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     mcfg = cfg["models"]["mhd_v4"]
#     model = MHDV4(
#     n_drugs=len(drugs),
#     emb_dim=mcfg["emb_dim"],
#     prior_dim=prior_dim,
#     attn_dim=mcfg["attn_dim"],
#     gamma=mcfg["focal_gamma"],
#     prior_dropout=mcfg["prior_dropout"],
#     gate_temp=mcfg.get("gate_temp", 0.5),       
#     prior_scale=mcfg.get("prior_scale", 0.3)    
# )

#     # RotatE init (pretrained on training graph)
#     if cfg["pretrained"].get("rotate_ckpt"):
#         E_rot, rot_d2i = load_rotate_embeddings(cfg["pretrained"]["rotate_ckpt"], cfg["pretrained"]["rotate_map"])
#         with torch.no_grad():
#             copied = 0
#             for d, idx in drug2id.items():
#                 if d in rot_d2i and E_rot.size(1) == model.emb.weight.size(1):
#                     model.emb.weight[idx] = E_rot[rot_d2i[d]]
#                     copied += 1
#         print(f"Init MHD-v4 embeddings from RotatE for {copied}/{len(drug2id)} drugs")

#     model, (y_va, s_va), (y_te, s_te) = train_mhd_v4(
#     model, tr, va, te, drug2id, F_tr, F_va, F_te,
#     lr=mcfg["lr"], weight_decay=1e-5, max_epochs=mcfg["epochs"], patience=mcfg["patience"],
#     lambda_sup=mcfg["lambda_sup"], lambda_cf=mcfg["lambda_cf"],
#     lambda_gate=mcfg.get("lambda_gate", 0.1),
#     device=device, log_gate=True
# )

#     rows = []
#     for split, y, s in [("val", y_va, s_va), ("test", y_te, s_te)]:
#         from src.eval.metrics import compute_all
#         met = compute_all(y, s); met["model"]="MHD_v4"; met["split"]=split
#         rows.append(met); plot_curves(y, s, "MHD_v4", split, out_dir)

#     def dump_preds(y, s, split, out_dir):
#         np.savez_compressed(out_dir / f"preds_{split}.npz", y=y, s=s)
#         fpr, tpr, _ = roc_curve(y, s)
#         rec, prec, _ = precision_recall_curve(y, s)
#         pd.DataFrame({"fpr":fpr,"tpr":tpr}).to_csv(out_dir/f"roc_points_{split}.csv", index=False)
#         pd.DataFrame({"recall":rec,"precision":prec}).to_csv(out_dir/f"pr_points_{split}.csv", index=False)
#         # threshold sweep
#         ths = np.linspace(0,1,501)
#         rows=[]
#         for t in ths:
#             p = (s>=t).astype(int)
#             tp = ((p==1)&(y==1)).sum(); fp=((p==1)&(y==0)).sum()
#             tn = ((p==0)&(y==0)).sum(); fn=((p==0)&(y==1)).sum()
#             prec_ = tp/max(1,tp+fp); rec_ = tp/max(1,tp+fn)
#             f1 = 0 if (prec_+rec_)==0 else 2*prec_*rec_/(prec_+rec_)
#             rows.append({"thr":t,"precision":prec_,"recall":rec_,"f1":f1,
#                         "tp":tp,"fp":fp,"tn":tn,"fn":fn})
#         pd.DataFrame(rows).to_csv(out_dir/f"threshold_sweep_{split}.csv", index=False)

#     def pick_threshold(sweep_csv):
#         df = pd.read_csv(sweep_csv)
#         t_f1 = float(df.loc[df["f1"].idxmax(),"thr"])
#         return {"max_f1": t_f1}

#     for split,(y,s) in [("val",(y_va,s_va)),("test",(y_te,s_te))]:
#         dump_preds(y, s, split, out_dir)

#     sel = pick_threshold(out_dir/"threshold_sweep_val.csv")
#     (out_dir/"selected_thresholds.json").write_text(json.dumps(sel, indent=2))

#     # Confusion matrix at chosen threshold
#     thr = sel["max_f1"]
#     for split,(y,s) in [("val",(y_va,s_va)),("test",(y_te,s_te))]:
#         p = (s>=thr).astype(int)
#         cm = confusion_matrix(y, p).tolist()
#         (out_dir/f"cmat_{split}.json").write_text(json.dumps({"thr":thr,"cm":cm}, indent=2))

#     pd.DataFrame(rows).to_csv(out_dir / "metrics_summary.csv", index=False)
#     save_json({"config": cfg, "prior_meta": meta_tr}, out_dir / "run_config.json")
#     print("Saved ->", out_dir / "metrics_summary.csv")

# if __name__ == "__main__":
#     main()




# This script trains the MHD_v4 model for DDI prediction using drug embeddings
# initialized from a pre-trained RotatE model. It incorporates drug pair priors

import argparse, datetime, json, csv
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

from src.utils.io import load_config, ensure_dir, save_json, set_seed
from src.data.ingest import load_chch, load_ddinter, load_decagon, merge_sources
from src.data.splits import warm_split, cold_drug_split, negative_sampling
from src.features.priors import load_atc_map, load_cyp_table, build_pair_priors
from src.models.mhd_v4 import MHDV4, train_mhd_v4
from src.eval.metrics import compute_all


def plot_curves(y, s, name, split, out_dir):
    """Plot ROC and PR curves and save to disk."""
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC {name} ({split})")
    plt.legend()
    plt.savefig(out_dir / f"fig_roc_{name}_{split}.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(rec, prec, label=f"AUC={auc(rec,prec):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR {name} ({split})")
    plt.legend()
    plt.savefig(out_dir / f"fig_pr_{name}_{split}.png", dpi=300)
    plt.close()


def load_rotate_embeddings(ckpt_path, drug2id_json):
    """Load pretrained RotatE embeddings."""
    ckpt_path = Path(ckpt_path)
    if ckpt_path.is_dir():
        sd = torch.load(ckpt_path / "data.pkl", map_location="cpu")
    else:
        sd = torch.load(ckpt_path, map_location="cpu")
    E = torch.cat([sd["emb_re.weight"], sd["emb_im.weight"]], dim=1)
    with open(drug2id_json, "r") as f:
        rot_d2i = {k: int(v) for k,v in json.load(f).items()}
    return E, rot_d2i


def init_logger(out_dir, name):
    """Prepare CSV logger for training curves."""
    log_file = open(out_dir / f"{name}_log.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["epoch", "loss", "sup_loss", "cf_loss", "val_AUPRC", "val_AUROC", "gate"])
    return log_file, writer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/exp_mhd_v4.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])

    data_dir = Path(cfg["data"]["data_dir"])
    base_out = Path(cfg["output"]["dir"])
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    split_name = cfg["experiment"]["split_type"]
    out_dir = base_out / f"mhd_v4_{split_name}_{ts}"
    ensure_dir(out_dir)

    # ---- Load datasets ----
    chch = load_chch(data_dir / cfg["data"]["chch_file"], sep=cfg["data"].get("sep_chch","\t"))
    ddinter = load_ddinter(sorted(list(data_dir.glob(cfg["data"]["ddinter_shards_glob"]))))
    decagon = load_decagon(data_dir / cfg["data"]["decagon_file"])
    pos_all = merge_sources(chch, ddinter, decagon)
    neg_all = negative_sampling(pos_all, ratio=cfg["experiment"]["n_neg_per_pos"], seed=cfg["experiment"]["seed"])
    pairs = pd.concat([pos_all.assign(label=1), neg_all], ignore_index=True)

    # ---- Splits ----
    if split_name == "warm":
        tr, va, te = warm_split(pairs, cfg["experiment"]["test_size"], cfg["experiment"]["val_size"], cfg["experiment"]["seed"])
    else:
        tr, va, te = cold_drug_split(pairs, cfg["experiment"]["test_size"], cfg["experiment"]["val_size"], cfg["experiment"]["seed"])

    # ---- Drug index ----
    drugs = pd.unique(pd.concat([pairs["drug_u"], pairs["drug_v"]])).astype(str).tolist()
    drug2id = {d:i for i,d in enumerate(drugs)}

    # ---- Priors ----
    atc_map = load_atc_map(data_dir)
    cyp_df  = load_cyp_table(cfg["data"]["drug_cyp_file"])
    F_tr, meta_tr = build_pair_priors(tr, atc_map, cyp_df)
    F_va, _       = build_pair_priors(va, atc_map, cyp_df)
    F_te, _       = build_pair_priors(te, atc_map, cyp_df)
    prior_dim = F_tr.shape[1]

    # ---- Model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg = cfg["models"]["mhd_v4"]
    model = MHDV4(
        n_drugs=len(drugs),
        emb_dim=mcfg["emb_dim"],
        prior_dim=prior_dim,
        attn_dim=mcfg["attn_dim"],
        gamma=mcfg["focal_gamma"],
        prior_dropout=mcfg["prior_dropout"],
        gate_temp=mcfg.get("gate_temp", 0.5),
        prior_scale=mcfg.get("prior_scale", 0.3)
    )

    # ---- RotatE init ----
    if cfg.get("pretrained", {}).get("rotate_ckpt"):
        E_rot, rot_d2i = load_rotate_embeddings(cfg["pretrained"]["rotate_ckpt"], cfg["pretrained"]["rotate_map"])
        with torch.no_grad():
            copied = 0
            for d, idx in drug2id.items():
                if d in rot_d2i and E_rot.size(1) == model.emb.weight.size(1):
                    model.emb.weight[idx] = E_rot[rot_d2i[d]]
                    copied += 1
        print(f"Init MHD-v4 embeddings from RotatE for {copied}/{len(drug2id)} drugs")

    # ---- Training (with logger) ----
    log_file, writer = init_logger(out_dir, "MHD_v4")
    model, (y_va, s_va), (y_te, s_te) = train_mhd_v4(
        model, tr, va, te, drug2id, F_tr, F_va, F_te,
        lr=mcfg["lr"], weight_decay=1e-5, max_epochs=mcfg["epochs"], patience=mcfg["patience"],
        lambda_sup=mcfg["lambda_sup"], lambda_cf=mcfg["lambda_cf"],
        lambda_gate=mcfg.get("lambda_gate", 0.1),
        device=device, log_gate=True, log_writer=writer, log_file=log_file
    )
    log_file.close()

    # ---- Evaluation ----
    rows = []
    for split, y, s in [("val", y_va, s_va), ("test", y_te, s_te)]:
        met = compute_all(y, s)
        met["model"]="MHD_v4"; met["split"]=split
        rows.append(met)
        plot_curves(y, s, "MHD_v4", split, out_dir)

    pd.DataFrame(rows).to_csv(out_dir / "metrics_summary.csv", index=False)
    save_json({"config": cfg, "prior_meta": meta_tr}, out_dir / "run_config.json")
    print("Saved ->", out_dir / "metrics_summary.csv")

      # --- Extra outputs like MHDv3 ---
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

    # dump preds and sweeps for both val + test
    for split,(y,s) in [("val",(y_va,s_va)),("test",(y_te,s_te))]:
        dump_preds(y, s, split, out_dir)

    sel = pick_threshold(out_dir/"threshold_sweep_val.csv")
    (out_dir/"selected_thresholds.json").write_text(json.dumps(sel, indent=2))

    # Confusion matrices
    thr = sel["max_f1"]
    for split,(y,s) in [("val",(y_va,s_va)),("test",(y_te,s_te))]:
        p = (s>=thr).astype(int)
        cm = confusion_matrix(y, p).tolist()
        (out_dir/f"cmat_{split}.json").write_text(json.dumps({"thr":thr,"cm":cm}, indent=2))

if __name__ == "__main__":
    main()
