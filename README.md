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

#B0: PPMI (co-occurrence baseline)
models:
  use_logreg: false
  use_xgboost: false
  use_ppmi: true
  use_rule: false
  use_distmult: false
  use_rotate: false

#B1: Rule presence (lookup baseline)
models:
  use_logreg: false
  use_xgboost: false
  use_ppmi: false
  use_rule: true
  use_distmult: false
  use_rotate: false

#B2: Logistic Regression
models:
  use_logreg: true
  use_xgboost: false
  use_ppmi: false
  use_rule: false
  use_distmult: false
  use_rotate: false

#B3: DistMult
models:
  use_logreg: false
  use_xgboost: false
  use_ppmi: false
  use_rule: false
  use_distmult: true
  use_rotate: false

#B3: RotatE
models:
  use_logreg: false
  use_xgboost: false
  use_ppmi: false
  use_rule: false
  use_distmult: false
  use_rotate: true

# Made Algorithms MHD:

#MHD V2:
python run_benchmark.py

#MHD V3:
python -m src.train_mhd_v3 --config src/config/exp_mhd_v3.yaml

#MHD V4:
python -m src.train_mhd_v4 --config src/config/exp_mhd_v4.yaml

# Comparison Table:
python -m src.scripts.run_ablation_mhd_v4 --config src/config/exp_mhd_v4.yaml