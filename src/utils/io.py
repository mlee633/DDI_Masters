import os
from pathlib import Path
import yaml
import json
import numpy as np
import random
import re

def load_config(path=None):
    """
    Load a YAML config file.
    If path is None, defaults to config/defaults.yaml.
    Supports ${ENV:default} interpolation.
    """
    if path is None:
        cfg_path = Path(__file__).resolve().parent.parent / "config" / "defaults.yaml"
    else:
        cfg_path = Path(path)

    raw = cfg_path.read_text()

    def repl_env(match):
        inside = match.group(1)
        if ":" in inside:
            k, d = inside.split(":", 1)
            return os.environ.get(k, d)
        return os.environ.get(inside, "")

    txt = re.sub(r"\$\{([^}]+)\}", repl_env, raw)
    return yaml.safe_load(txt)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import xgboost as xgb  # noqa: F401
    except Exception:
        pass
