"""Rebuild the Dataset 3 network page's embedded data from published CSVs.

The visual layout and offline JavaScript bundle remain in the supplied HTML
template; the three authoritative data constants (nodes, edges and family
legend) are regenerated from Dataset 1 and Dataset 3.  This prevents the page
from silently retaining values from an older intermediate snapshot.

Usage:
    python make_ds3_network.py
    python make_ds3_network.py --template path/to/network_visualization.html \
        --output path/to/network_visualization.html
"""
from __future__ import annotations

import argparse
import json
import os
import re

from common import _find, fps, load, metadata, outdir


MULTI_FAMILY_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
    "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324",
    "#fffac8", "#800000",
]
SINGLETON_COLOR = "#cccccc"


def _replace_constant(page: str, name: str, value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    pattern = rf"(?m)^const {re.escape(name)}=.*?;$"
    # Use a replacement callback so JSON escape sequences (notably ``\\n`` in
    # node tooltips) are not interpreted by ``re.sub`` as literal newlines.
    replaced, count = re.subn(
        pattern, lambda _match: f"const {name}={encoded};", page, count=1
    )
    if count != 1:
        raise ValueError(f"template does not contain exactly one const {name}=... line")
    return replaced


def build_embedded_data():
    tm = metadata()
    fc = fps()
    fam = load("ds3", "treaty_families.csv").set_index("treaty_id")
    edges = load("ds3", "genealogy_edges.csv")

    family_rows = fam.reset_index().groupby("family_id", sort=True)
    multi_ids = [family_id for family_id, group in family_rows if len(group) > 1]
    color_of = {
        family_id: MULTI_FAMILY_COLORS[i % len(MULTI_FAMILY_COLORS)]
        for i, family_id in enumerate(multi_ids)
    }
    for family_id, group in fam.reset_index().groupby("family_id", sort=True):
        if len(group) == 1:
            color_of[family_id] = SINGLETON_COLOR

    families = {}
    for family_id, group in fam.reset_index().groupby("family_id", sort=True):
        families[family_id] = {
            "label": group["family_label"].iloc[0],
            "color": color_of[family_id],
            "size": int(len(group)),
        }

    nodes = []
    for treaty_id in tm.index:
        family_id = fam.loc[treaty_id, "family_id"]
        partner = tm.loc[treaty_id, "partner_state"]
        year = int(str(tm.loc[treaty_id, "signature_date"])[:4])
        typology = fc.loc[treaty_id, "fps_typology"]
        label = fam.loc[treaty_id, "family_label"]
        size = int(fam.loc[treaty_id, "family_size"])
        nodes.append({
            "id": treaty_id,
            "label": partner,
            "title": (f"{partner} ({year})\nFamily: {family_id} - {label}\n"
                      f"FPS Type: {typology}\nTreaty ID: {treaty_id}"),
            "year": year,
            "family": family_id,
            "color": color_of[family_id],
            "fps": typology,
            "value": 12 if size > 1 else 7,
        })

    raw_edges = []
    for row in edges.itertuples(index=False):
        similarity = round(float(row.cosine_similarity), 6)
        raw_edges.append({
            "from": row.parent_treaty,
            "to": row.child_treaty,
            "value": similarity,
            "title": f"sim={similarity:.6f}, gap={int(row.year_gap)}y",
            "color": {"opacity": round(min(1.0, similarity + 0.05), 6)},
        })
    return nodes, raw_edges, families


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default=_find("ds3", "network_visualization.html"))
    parser.add_argument("--output",
                        default=os.path.join(outdir("ds3_network"),
                                             "network_visualization.html"))
    args = parser.parse_args()

    with open(args.template, encoding="utf-8") as fh:
        page = fh.read()
    nodes, edges, families = build_embedded_data()
    page = _replace_constant(page, "rawNodes", nodes)
    page = _replace_constant(page, "rawEdges", edges)
    page = _replace_constant(page, "families", families)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    print(f"wrote {args.output}: {len(nodes)} nodes, {len(edges)} edges, "
          f"{len(families)} families")


if __name__ == "__main__":
    main()
