import pandas as pd, matplotlib.pyplot as plt

def plot_curve(log_file, out_file):
    df = pd.read_csv(log_file)
    plt.plot(df["epoch"], df["val_AUPRC"], label="val AUPRC")
    if "gate" in df.columns:
        plt.plot(df["epoch"], df["gate"], label="gate value")
    plt.xlabel("Epoch"); plt.ylabel("Value"); plt.legend()
    plt.savefig(out_file, dpi=300)

# Example usage:
# python src/plot_training_curves.py logs/mhd_v4_seed42.csv outputs/curve.png
