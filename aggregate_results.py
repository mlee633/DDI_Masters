import pandas as pd
from pathlib import Path

OUTPUT = Path("outputs")

def main():
    all_rows = []
    for csv in OUTPUT.rglob("metrics_summary.csv"):
        # folder name tells us warm vs cold
        split_tag = csv.parent.name.split("_")[0]
        df = pd.read_csv(csv)
        df["split_run"] = split_tag
        df["source_file"] = str(csv)
        all_rows.append(df)

    if not all_rows:
        print("No metrics_summary.csv found under outputs/")
        return

    df_all = pd.concat(all_rows, ignore_index=True)

    # Pivot to get warm vs cold side by side
    pivot = df_all.pivot_table(
        index=["model", "split"],
        values=["AUROC", "AUPRC"],
        columns="split_run",
        aggfunc="first"
    )

    # Save both combined data and pivot summary
    combined_path = OUTPUT / "all_results_combined.csv"
    pivot_path = OUTPUT / "all_results_summary.csv"

    df_all.to_csv(combined_path, index=False)
    pivot.to_csv(pivot_path)

    print(f"Saved combined results → {combined_path}")
    print(f"Saved pivot summary → {pivot_path}")

    print("\nQuick view (first few rows):")
    print(pivot.head())

if __name__ == "__main__":
    main()
