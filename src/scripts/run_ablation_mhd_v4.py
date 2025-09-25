#This script runs ablation studies for the MHD_v4 model on DDI prediction.
# It trains variants of the model with/without priors and with frozen embeddings,
# and saves the evaluation results for comparison. 

import argparse, datetime, json
from pathlib import Path
import pandas as pd
import torch

from src.utils.io import load_config, ensure_dir, save_json, set_seed
from src.data.ingest import load_chch, load_ddinter, load_decagon, merge_sources
from src.data.splits import warm_split, cold_drug_split, negative_sampling
from src.features.priors import load_atc_map, load_cyp_table, build_pair_priors
from src.models.mhd_v4 import MHDV4, train_mhd_v4

def load_rotate_embeddings(ckpt_path, drug2id_json):
    ckpt_path = Path(ckpt_path)
    if ckpt_path.is_dir():
        sd = torch.load(ckpt_path / "data.pkl", map_location="cpu")
    else:
        sd = torch.load(ckpt_path, map_location="cpu")
    E = torch.cat([sd["emb_re.weight"], sd["emb_im.weight"]], dim=1)
    with open(drug2id_json, "r") as f:
        rot_d2i = {k: int(v) for k,v in json.load(f).items()}
    return E, rot_d2i

def run_one(cfg, name, drug2id, tr, va, te, F_tr, F_va, F_te, freeze_emb=False):
    prior_dim = F_tr.shape[1]
    mcfg = cfg["models"]["mhd_v4"]
    model = MHDV4(n_drugs=len(drug2id), emb_dim=mcfg["emb_dim"], prior_dim=prior_dim,
                  attn_dim=mcfg["attn_dim"], gamma=mcfg["focal_gamma"], prior_dropout=mcfg["prior_dropout"])
    # RotatE init
    if cfg["pretrained"].get("rotate_ckpt"):
        E_rot, rot_d2i = load_rotate_embeddings(cfg["pretrained"]["rotate_ckpt"], cfg["pretrained"]["rotate_map"])
        with torch.no_grad():
            for d, idx in drug2id.items():
                if d in rot_d2i and E_rot.size(1) == model.emb.weight.size(1):
                    model.emb.weight[idx] = E_rot[rot_d2i[d]]
    if freeze_emb:
        for p in model.emb.parameters():
            p.requires_grad = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, (y_va, s_va), (y_te, s_te) = train_mhd_v4(
        model, tr, va, te, drug2id, F_tr, F_va, F_te,
        lr=mcfg["lr"], weight_decay=1e-5, max_epochs=mcfg["epochs"], patience=mcfg["patience"],
        lambda_sup=mcfg["lambda_sup"], lambda_cf=mcfg["lambda_cf"], device=device, log_gate=True
    )
    from src.eval.metrics import compute_all
    rows = []
    for split, y, s in [("val", y_va, s_va), ("test", y_te, s_te)]:
        met = compute_all(y, s)
        met["model"] = name
        met["split"] = split
        rows.append(met)
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/exp_mhd_v4.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])

    data_dir = Path(cfg["data"]["data_dir"])
    chch = load_chch(data_dir / cfg["data"]["chch_file"], sep=cfg["data"].get("sep_chch","\t"))
    ddinter = load_ddinter(sorted(list((data_dir if data_dir.exists() else Path(".")).glob(cfg["data"]["ddinter_shards_glob"]))))
    decagon = load_decagon(data_dir / cfg["data"]["decagon_file"])
    pos_all = merge_sources(chch, ddinter, decagon)
    neg_all = negative_sampling(pos_all, ratio=cfg["experiment"]["n_neg_per_pos"], seed=cfg["experiment"]["seed"])
    pairs = pd.concat([pos_all.assign(label=1), neg_all], ignore_index=True)

    if cfg["experiment"]["split_type"] == "warm":
        tr, va, te = warm_split(pairs, cfg["experiment"]["test_size"], cfg["experiment"]["val_size"], cfg["experiment"]["seed"])
    else:
        tr, va, te = cold_drug_split(pairs, cfg["experiment"]["test_size"], cfg["experiment"]["val_size"], cfg["experiment"]["seed"])

    drugs = pd.unique(pd.concat([pairs["drug_u"], pairs["drug_v"]])).astype(str).tolist()
    drug2id = {d:i for i,d in enumerate(drugs)}

    atc_map = load_atc_map(data_dir)
    cyp_df  = load_cyp_table(cfg["data"]["drug_cyp_file"])
    F_tr, _ = build_pair_priors(tr, atc_map, cyp_df)
    F_va, _ = build_pair_priors(va, atc_map, cyp_df)
    F_te, _ = build_pair_priors(te, atc_map, cyp_df)

    # Run ablations
    dfs = []
    dfs.append(run_one(cfg, "MHD_v4_full", drug2id, tr, va, te, F_tr, F_va, F_te))
    dfs.append(run_one(cfg, "MHD_v4_no_priors", drug2id, tr, va, te,
                       F_tr*0, F_va*0, F_te*0))   # zero-out priors
    dfs.append(run_one(cfg, "MHD_v4_frozen_emb", drug2id, tr, va, te,
                       F_tr, F_va, F_te, freeze_emb=True))

    df_all = pd.concat(dfs, ignore_index=True)

    base_out = Path(cfg["output"]["dir"])
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    out_path = base_out / f"mhd_v4_ablation_{ts}.csv"
    df_all.to_csv(out_path, index=False)
    print("Saved ablation results →", out_path)
    print(df_all)

if __name__ == "__main__":
    main()
