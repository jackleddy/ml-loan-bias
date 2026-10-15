from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"


def read_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("input", "reference_dir", "processed_dir", "artifact_dir", "report_dir"):
        config[key] = str((path.parent / config[key]).resolve())
    if config["years"] != [2021, 2022, 2023, 2024, 2025]:
        raise ValueError("This study supports Virginia, 2021-2025 only.")
    if config["folds"] < 2 or config["spread_min"] >= config["spread_max"]:
        raise ValueError("Invalid fold count or spread bounds.")
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj).__name__)

    path.write_text(json.dumps(value, indent=2, default=default, allow_nan=False) + "\n", encoding="utf-8")


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def log(message):
    print(message, flush=True)
