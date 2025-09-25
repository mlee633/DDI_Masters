# This script runs multiple training sessions with different random seeds,
# collects their evaluation metrics, and aggregates the results into a summary CSV.

import subprocess, sys, shutil
from pathlib import Path
import pandas as pd

CONFIG = "src/config/exp_mhd_v4.yaml"
OUTPUT = Path("outputs")
SEEDS = [42, 123, 2025]   # seeds you want to cycle

def run_once(seed):
    """Run training with a specific seed by overriding config on the fly."""
    # Copy config to a temp file with updated seed
    tmp_config = Path(CONFIG).with_name(f"tmp_seed{seed}.yaml")
    with open(CONFIG, "r") as f:
        lines = f.readlines()
    with open(tmp_config, "w") as f:
        for line in lines:
            if line.strip().startswith("seed:"):
                f.write(f"  seed: {seed}\n")
            else:
                f.write(line)

    print(f"\n=== Running with seed={seed} ===")
    subprocess.run([sys.executable, "-m", "src.train_mhd_v4", "--config", str(tmp_config)], check=True)

    # Find latest output folder
    subdirs = [p for p in OUTPUT.glob("mhd_v4_*") if p.is_dir()]
    latest = max(subdirs, key=lambda p: p.stat().st_mtime)
    metrics = latest / "metrics_summary.csv"

    # Copy metrics to a seed-specific filename
    dest = OUTPUT / f"metrics_seed{seed}.csv"
    shutil.copy(metrics, dest)
    print(f"Saved → {dest}")

    tmp_config.unlink()  # clean up
    return dest

def aggregate_results(files):
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        seed = int(str(f).split("seed")[-1].split(".")[0])
        df["seed"] = seed
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)
    out = OUTPUT / "metrics_all_seeds.csv"
    df_all.to_csv(out, index=False)
    print(f"\nAggregated results → {out}")
    print(df_all.groupby(["model","split"]).agg({"AUPRC":["mean","std"], "AUROC":["mean","std"]}))

if __name__ == "__main__":
    files = []
    for s in SEEDS:
        files.append(run_once(s))
    aggregate_results(files)
