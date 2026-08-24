"""Shared loading helpers for downloaded and standalone release layouts.

The toolkit can be run either from a cloned/downloaded repository containing
``ds1/data/{ds1,ds2,ds3}``, or directly from the published release where the
three dataset directories are siblings.  ``TURKISH_BIT_DATA_ROOT`` and
``TURKISH_BIT_OUTPUT_ROOT`` remain available for explicit staging paths.
"""
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.dirname(ROOT)
_downloaded_data = os.path.join(ROOT, "data")
DATA = os.environ.get(
    "TURKISH_BIT_DATA_ROOT",
    _downloaded_data if os.path.isdir(_downloaded_data) else WORKSPACE,
)
OUT = os.environ.get("TURKISH_BIT_OUTPUT_ROOT", os.path.join(ROOT, "outputs"))

READ = dict(keep_default_na=False, dtype=str)


def _find(dataset, filename):
    """Locate a published file, tolerating one additional nesting level."""
    base = os.path.join(DATA, dataset)
    direct = os.path.join(base, filename)
    if os.path.exists(direct):
        return direct
    for dirpath, _dirnames, files in os.walk(base):
        if filename in files:
            return os.path.join(dirpath, filename)
    raise FileNotFoundError(
        f"{filename} not found under {base}. Set TURKISH_BIT_DATA_ROOT or "
        "run download_data.py to stage the release."
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
