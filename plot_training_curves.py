import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

OUTPUT = Path("outputs")
PLOTS = OUTPUT / "training_curves2"
PLOTS.mkdir(parents=True, exist_ok=True)

def load_logs():
    """Recursively find all *_log.csv files and return as dict."""
    logs = {}
    for f in OUTPUT.rglob("*_log.csv"):
        try:
            df = pd.read_csv(f)
            logs[f.parent.name] = df  # use folder name as key (e.g. warm_MHDv3_Final)
        except Exception as e:
            print(f":warning: Skipping {f}: {e}")
    return logs

def plot_comparison(logs, groups, out_file, metric="val_AUPRC"):
    """
    groups: dict of {legend_label: [keys from logs]}.
    Example: {"Baselines": ["warm_multi_exp_09-26_run1"], "MHDv3": ["warm_MHDv3_Final"]}
    """
    plt.figure(figsize=(8,6))
    for label, keys in groups.items():
        for key in keys:
            if key not in logs: 
                print(f":warning: Missing log for {key}")
                continue
            df = logs[key]
            if metric not in df.columns: 
                print(f":warning: {metric} not in {key}")
                continue
            plt.plot(df["epoch"], df[metric], label=f"{label} ({key})")
    plt.xlabel("Epoch"); plt.ylabel(metric)
    plt.title(f"Comparison: {metric}")
    plt.legend()
    plt.savefig(PLOTS / out_file, dpi=300)
    plt.close()
    print(f":white_check_mark: Saved {PLOTS/out_file}")

def main():
    logs = load_logs()

    # ---- Define groups manually ----
    groups1 = {
        "Baselines": [k for k in logs if "multi_exp" in k],   # your baseline exp dirs
    }
    groups2 = {
        "MHDv2": [k for k in logs if "MHDv2" in k],
        "MHDv3": [k for k in logs if "MHDv3" in k],
        "MHDv4": [k for k in logs if "MHDv4" in k],
    }
    groups3 = {**groups1, **groups2}

    plot_comparison(logs, groups1, "compare_baselines.png", metric="val_AUPRC")
    plot_comparison(logs, groups2, "compare_mhd.png", metric="val_AUPRC")
    plot_comparison(logs, groups3, "compare_all.png", metric="val_AUPRC")

if __name__ == "__main__":
    main()