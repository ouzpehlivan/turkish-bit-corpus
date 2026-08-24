"""Refresh the Dataset 2 explorer's embedded treaty records.

The published page is an offline visualisation whose layout, variable glossary
and JavaScript are hand-authored.  This script preserves that template while
rebuilding every treaty's metadata and annotation object from the canonical
CSV files, preventing the page from retaining an older coding snapshot. The
hand-authored explorer deliberately excludes only 3.13_fps_text because a long
verbatim clause is not a categorical/map-compatible value.

Usage:
    python make_ds2_explorer.py
    python make_ds2_explorer.py --template path/to/explorer.html \
        --output path/to/explorer.html
"""
from __future__ import annotations

import argparse
import json
import os
import re

from common import _find, annotations, metadata, outdir


GENERATION_LABEL = {
    "gen_1_early": "Gen 1 (1962-85)",
    "gen_2_liberalization": "Gen 2 (1986-98)",
    "gen_3_eu_harmonization": "Gen 3 (1999-2009)",
    "gen_4_new_model": "Gen 4 (2010-2024)",
}


def _read_assignment(page: str, name: str):
    marker = f"const {name} = "
    start = page.index(marker) + len(marker)
    value, consumed = json.JSONDecoder().raw_decode(page[start:])
    return value, start, start + consumed


def rebuild_data(existing):
    tm = metadata()
    ta = annotations()
    variables = [dict(item) for item in existing["variables"]]
    keys = [item["key"] for item in variables]

    missing = sorted(set(keys) - set(ta.columns))
    if missing:
        raise ValueError(f"explorer variables absent from annotations: {missing}")
    substantive = {c for c in ta.columns
                   if c not in {"coded_by", "coding_confidence", "coding_date"}}
    expected = substantive - {"3.13_fps_text"}
    if set(keys) != expected:
        raise ValueError(
            "explorer must cover every map-compatible substantive variable; "
            f"missing={sorted(expected - set(keys))}, extra={sorted(set(keys) - expected)}"
        )

    # Options drive every colour, legend and aggregate in the page. Preserve
    # the established order for values still observed, append newly observed
    # values, and remove options no longer present in the canonical table.
    for item in variables:
        observed = list(dict.fromkeys(
            value for value in ta[item["key"]].astype(str) if value != ""
        ))
        prior = [value for value in item.get("options", []) if value in observed]
        item["options"] = prior + [value for value in observed if value not in prior]

    previous = {row["id"]: row for row in existing["treaties"]}
    treaties = []
    for treaty_id in tm.index:
        old = previous.get(treaty_id, {})
        iso = tm.loc[treaty_id, "partner_iso3"]
        treaties.append({
            "id": treaty_id,
            "partner": tm.loc[treaty_id, "partner_state"],
            "iso": iso,
            # Preserve map-specific aliases for historic/special codes.
            "miso": old.get("miso", existing.get("remap", {}).get(iso, iso)),
            "region": tm.loc[treaty_id, "partner_region"],
            "year": int(tm.loc[treaty_id, "signature_date"][:4]),
            "gen": GENERATION_LABEL[tm.loc[treaty_id, "generation"]],
            "status": tm.loc[treaty_id, "status"],
            "v": {key: ta.loc[treaty_id, key] for key in keys},
        })

    result = dict(existing)
    result["variables"] = variables
    result["treaties"] = treaties
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default=_find("ds2", "explorer.html"))
    parser.add_argument("--output", default=os.path.join(outdir("ds2_explorer"),
                                                         "explorer.html"))
    args = parser.parse_args()

    with open(args.template, encoding="utf-8") as fh:
        page = fh.read()
    old, start, end = _read_assignment(page, "DATA")
    new = rebuild_data(old)
    encoded = json.dumps(new, ensure_ascii=False, separators=(",", ":"))
    page = page[:start] + encoded + page[end:]
    page = re.sub(r"Gen 4 \(2010-\d{2,4}\)",
                  GENERATION_LABEL["gen_4_new_model"], page)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    print(f"wrote {args.output}: {len(new['treaties'])} treaties, "
          f"{len(new['variables'])} explorer variables")


if __name__ == "__main__":
    main()
