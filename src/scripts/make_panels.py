import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc

OUT = Path("outputs")
PANEL_DIR = OUT / "panels"; PANEL_DIR.mkdir(exist_ok=True)

KEEP_MODELS = {"B2_logreg":"LogReg","B2_xgboost":"XGBoost","B3_distmult":"DistMult","B3_rotate":"RotatE","MHD_hybrid":"MHD"}

def load_latest(split):
    runs = sorted([p for p in OUT.glob(f"{split}_*/metrics_summary.csv")])
    return pd.read_csv(runs[-1]) if runs else None

def plot_pr_panel(df, split, save_as):
    plt.figure()
    for m,label in KEEP_MODELS.items():
        d = df[df["model"]==m]
        d = d[d["split"]=="test"]  # show test
        if d.empty: continue
        # We only have AUCs in CSV; curves are saved as images per model/split. So we’ll compose from saved PNGs if needed.
        # Simple fallback: annotate AUPRC
        auprc = d["AUPRC"].values[0]
        plt.plot([], [], label=f"{label}  AUPRC={auprc:.3f}")  # legend line only
    plt.title(f"Precision–Recall (Test • {split})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend()
    plt.savefig(PANEL_DIR / save_as, dpi=300, bbox_inches="tight"); plt.close()

def main():
    for split in ["warm","cold_drug"]:
        df = load_latest(split)
        if df is None: continue
        plot_pr_panel(df, split, f"panel_pr_{split}.png")

if __name__ == "__main__":
    main()
