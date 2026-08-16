"""Step 0: download the three published datasets from DataverseNO into data/.

Each dataset is fetched as a ZIP through the Dataverse access API and extracted
into data/ds1, data/ds2, data/ds3. Nothing in this repository redistributes the
data; it is retrieved from the DOI at run time.

Usage:  python src/download_data.py
"""
import io
import os
import sys
import zipfile

import requests

BASE = "https://dataverse.no/api/access/dataset/:persistentId"
DOIS = {
    "ds1": "doi:10.18710/JX4WHH",
    "ds2": "doi:10.18710/FPNLKS",
    "ds3": "doi:10.18710/WA7HEO",
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def fetch(key, doi):
    out = os.path.join(DATA, key)
    os.makedirs(out, exist_ok=True)
    print(f"[{key}] fetching {doi} ...", flush=True)
    r = requests.get(BASE, params={"persistentId": doi}, timeout=600)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(out)
    names = sorted(os.listdir(out))
    print(f"[{key}] {len(names)} files -> data/{key}")
    return names


def main():
    os.makedirs(DATA, exist_ok=True)
    for key, doi in DOIS.items():
        try:
            fetch(key, doi)
        except Exception as exc:  # noqa: BLE001
            print(f"[{key}] download failed: {exc}", file=sys.stderr)
            print(f"      Download manually from https://doi.org/{doi.split(':')[1]}"
                  f" and unzip into data/{key}/", file=sys.stderr)
    print("\nNote: DS1 ships the treaty texts as treaty_texts.zip; unzip it inside "
          "data/ds1/ if you need the raw texts.")


if __name__ == "__main__":
    main()
