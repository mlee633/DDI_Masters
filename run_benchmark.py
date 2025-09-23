import os, subprocess, json, shutil
from pathlib import Path

ROOT = Path(__file__).parent
CFG = ROOT / "src" / "config" / "defaults.yaml"
OUTPUT = ROOT / "outputs"

def set_split(split):
    txt = CFG.read_text()
    txt = txt.replace("split_type: warm", f"split_type: {split}")
    txt = txt.replace("split_type: cold_drug", f"split_type: {split}")
    CFG.write_text(txt)

def run_once():
    subprocess.run(
        ["python", "-m", "src.train_baselines"],
        check=True
    )

def main():
    # keep runs reproducible
    os.environ.setdefault("PYTHONHASHSEED","0")
    for split in ["warm", "cold_drug"]:
        set_split(split)
        print(f"\n=== Running {split} split ===")
        run_once()

    # optional: print a quick manifest of produced metrics files
    print("\nGenerated results:")
    for p in sorted(OUTPUT.rglob("metrics_summary.csv")):
        print(" -", p)

if __name__ == "__main__":
    main()
