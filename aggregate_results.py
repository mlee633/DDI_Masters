import pandas as pd
from pathlib import Path

OUTPUT = Path("outputs")

def main():
    all_rows = []
    for csv in OUTPUT.rglob("metrics_summary.csv"):
        split_tag = csv.parent.name.split("_")[0]  # "warm" or "cold"
        df = pd.read_csv(csv)
        df["split_run"] = split_tag
        df["source_file"] = str(csv)
        all_rows.append(df)

    if not all_rows:
        print("No metrics_summary.csv found under outputs/")
        return

    df_all = pd.concat(all_rows, ignore_index=True)

    # Keep a clean set of columns
    metrics = ["AUROC", "AUPRC", "Brier", "ECE@10", 
               "Precision@10", "Precision@20", "Precision@50"]

    # Pivot: warm vs cold side by side
    pivot = df_all.pivot_table(
        index=["model", "split"],
        values=metrics,
        columns="split_run",
        aggfunc="first"
    )

    # Sort models in order B0–B3
    order = ["B0_ppmi", "B1_rule_presence", "B2_logreg", "B2_xgboost",
            "B3_distmult", "B3_rotate", "MHD_v2", "MHD_v3", "MHD_v4"]
    pivot = pivot.reindex(order, level=0)

    # Save outputs
    combined_path = OUTPUT / "all_results_combined.csv"
    pivot_path = OUTPUT / "all_results_summary.csv"

    df_all.to_csv(combined_path, index=False)
    pivot.to_csv(pivot_path)

    # Optional: export to Excel
    pivot_excel = OUTPUT / "all_results_summary.xlsx"
    pivot.to_excel(pivot_excel)

    print(f"Saved combined results → {combined_path}")
    print(f"Saved pivot summary → {pivot_path}")
    print(f"Saved Excel summary → {pivot_excel}")

    print("\nQuick view (first few rows):")
    print(pivot.head(10))

if __name__ == "__main__":
    main()
