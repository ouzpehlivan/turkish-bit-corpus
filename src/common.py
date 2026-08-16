"""Shared loading helpers. Every script reads the published files from data/."""
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")

READ = dict(keep_default_na=False, dtype=str)


def _find(dataset, filename):
    """Locate a file inside data/<dataset>/, tolerating an extra nesting level."""
    base = os.path.join(DATA, dataset)
    direct = os.path.join(base, filename)
    if os.path.exists(direct):
        return direct
    for dirpath, _dirnames, files in os.walk(base):
        if filename in files:
            return os.path.join(dirpath, filename)
    raise FileNotFoundError(
        f"{filename} not found under data/{dataset}. Run src/download_data.py first."
    )


def load(dataset, filename, **kwargs):
    path = _find(dataset, filename)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    opts = dict(READ)
    opts.update(kwargs)
    return pd.read_csv(path, **opts)


def metadata():
    return load("ds1", "treaty_metadata.csv").set_index("treaty_id")


def fps():
    return load("ds1", "fps_classification.csv").set_index("treaty_id")


def extracts():
    return load("ds1", "article_extracts.csv").set_index("treaty_id")


def annotations():
    return load("ds2", "treaty_annotations.csv").set_index("treaty_id")


def clauses():
    return load("ds2", "clause_extracts.csv")


def outdir(name):
    path = os.path.join(OUT, name)
    os.makedirs(path, exist_ok=True)
    return path


def norm(text):
    """Whitespace-normalised comparison form."""
    return re.sub(r"\s+", " ", str(text)).strip()
