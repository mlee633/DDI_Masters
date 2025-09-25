# DDI Project (Baselines + Knowledge Graph Embeddings)

This repository provides ready-to-run **baseline algorithms** for drug–drug interaction (DDI) prediction.  
It supports **warm vs. cold-drug splits**, **classical ML baselines**, and **knowledge graph embedding baselines (DistMult, RotatE)**.

All outputs are stored in **timestamped subfolders** under `outputs/` (e.g. `outputs/warm_run_2025-09-23_16-10/`).  
This way, each run is preserved without overwriting previous results.

---

## Setup

```bash
# (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# Project Layout
ddi_project/
  README.md
  requirements.txt
  src/
    config/defaults.yaml
    utils/io.py
    data/ingest.py
    data/splits.py
    data/build_graph.py
    features/basic.py
    eval/metrics.py
    models/
      baseline_rule.py
      baseline_ppmi.py
      baseline_ml.py
      kg_embeddings.py 
    train_baselines.py
  outputs/
    warm_run_<timestamp>/
    cold_run_<timestamp>/

## Configuration
experiment:
  split_type: warm       # [warm | cold_drug]
  seed: 42
  test_size: 0.2
  val_size: 0.1

models:
  use_logreg: true
  use_xgboost: true
  use_ppmi: true
  use_rule: true
  use_distmult: true     # Knowledge Graph baseline
  use_rotate: true       # Knowledge Graph baseline

## Run the roots
python -m src.train_baselines

# Baseline

# Warm split
python -m src.train_baselines --config src/config/exp_baselines_warm.yaml

# Cold split
python -m src.train_baselines --config src/config/exp_baselines_cold.yaml

# Made Algorithms MHD:

#MHD-v1 (simple ML + priors fusion)
# Warm
python -m src.train_mhd_v1 --config src/config/exp_mhd_v1_warm.yaml
# Cold
python -m src.train_mhd_v1 --config src/config/exp_mhd_v1_cold.yaml

# MHD-v2 (embeddings + priors fusion, no gating)
python -m src.train_mhd_v2 --config src/config/exp_mhd_v2_warm.yaml
python -m src.train_mhd_v2 --config src/config/exp_mhd_v2_cold.yaml

#MHD-v3 (adds counterfactual regularisation, but unstable in cold)
python -m src.train_mhd_v3 --config src/config/exp_mhd_v3_warm.yaml
python -m src.train_mhd_v3 --config src/config/exp_mhd_v3_cold.yaml

#MHD-v4 (final gated version)
python -m src.train_mhd_v4 --config src/config/exp_mhd_v4_warm.yaml
python -m src.train_mhd_v4 --config src/config/exp_mhd_v4_cold.yaml