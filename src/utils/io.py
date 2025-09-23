import os
from pathlib import Path
import pandas as pd
import yaml
import json
import numpy as np
import random

def load_config():
    # simple env var interpolation for ${ENV:default}
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "defaults.yaml"
    raw = cfg_path.read_text()
    def repl_env(match):
        inside = match.group(1)
        if ":" in inside:
            k, d = inside.split(":", 1)
            return os.environ.get(k, d)
        return os.environ.get(inside, "")
    import re
    txt = re.sub(r"\$\{([^}]+)\}", repl_env, raw)
    return yaml.safe_load(txt)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))

def set_seed(seed=42):
    import numpy as np, random
    random.seed(seed); np.random.seed(seed)
    try:
        import xgboost as xgb
    except Exception:
        pass
