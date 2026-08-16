"""Protocol Step 5: recompute Dataset 3 from Datasets 1 and 2.

Rebuilds the similarity matrix, the pair table, the treaty families and the
genealogy edges with the exact parameters documented in
00_REPRODUCIBILITY_PROTOCOL.txt, writes them to outputs/ds3/, and (if the
published DS3 is present in data/ds3) reports whether the reproduction matches.

Usage:  python src/recompute_ds3.py
"""
import os
import re

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import annotations, extracts, load, metadata, outdir

FAMILY_CUT = 0.65
EDGE_MIN_COSINE = 0.45

GEN2MODEL = {
    "gen_1_early": "Early",
    "gen_2_liberalization": "Liberalization",
    "gen_3_eu_harmonization": "EU-harmonization",
    "gen_4_new_model": "New Turkish",
}

# The nine "new-model" clause packages (protocol Step 5.4).
def packages(ta, tid):
    def yes(col):
        return ta.loc[tid, col] == "Yes"

    return np.array([
        any(yes(c) for c in ("1.02_preamble_sustainable_dev",
                             "1.03_preamble_social",
                             "1.04_preamble_environmental")),
        yes("1.01_preamble_right_to_regulate") or yes("4.05_right_to_regulate"),
        yes("2.19_dob"),
        yes("3.18_expro_indirect_defined") or yes("3.19_expro_regulatory_carveout"),
        yes("5.04_exception_health_env"),
        yes("5.01_essential_security"),
        ta.loc[tid, "3.06_mfn_exception_procedural"] in ("Procedural issues (ISDS)", "Both"),
        any(yes(c) for c in ("4.03_health_environment", "4.04_labour",
                             "4.06_csr", "4.08_not_lowering")),
        any(yes(c) for c in ("7.20_non_disputing", "7.21_transparency_docs",
                             "7.22_transparency_hearings")),
    ])


def build_documents(ae, ids):
    cols = [c for c in ae.columns if c not in ("Anlasma Adi", "Dosya")]
    docs = []
    for tid in ids:
        row = ae.loc[tid]
        docs.append("\n".join(str(row[c]) for c in cols if str(row[c]).strip()))
    return docs


def main():
    tm, ae, ta = metadata(), extracts(), annotations()
    ids = list(tm.index)
    year = {t: int(tm.loc[t, "signature_date"][:4]) for t in ids}
    partner = tm["partner_state"].to_dict()

    docs = build_documents(ae, ids)
    matrix = cosine_similarity(
        TfidfVectorizer(stop_words="english", sublinear_tf=True, max_df=0.9)
        .fit_transform(docs)
    )
    np.fill_diagonal(matrix, 1.0)
    sm = pd.DataFrame(matrix, index=ids, columns=ids)

    tokens = [set(w for w in re.findall(r"[a-z]+", d.lower())
                  if w not in ENGLISH_STOP_WORDS) for d in docs]
    pkg = {t: packages(ta, t) for t in ids}

    rows = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            jac = len(tokens[i] & tokens[j]) / len(tokens[i] | tokens[j])
            rows.append([a, partner[a], year[a], b, partner[b], year[b],
                         round(matrix[i, j], 6), round(jac, 6),
                         int((pkg[a] & pkg[b]).sum()), abs(year[a] - year[b])])
    pairs = pd.DataFrame(rows, columns=[
        "treaty_a", "partner_a", "year_a", "treaty_b", "partner_b", "year_b",
        "cosine_similarity", "jaccard_similarity", "shared_clause_features", "year_gap"])

    # Families: average linkage on cosine distance, cut at 0.65.
    linkage_matrix = linkage(squareform(1 - matrix, checks=False), method="average")
    labels = fcluster(linkage_matrix, t=FAMILY_CUT, criterion="distance")
    frame = pd.DataFrame({"treaty_id": ids, "cluster": labels})
    frame["year"] = frame.treaty_id.map(year)
    size = frame.groupby("cluster").size()
    first_year = frame.groupby("cluster")["year"].min()
    order = sorted(size.index, key=lambda c: (-size[c], first_year[c]))
    fid = {c: "F%02d" % (k + 1) for k, c in enumerate(order)}

    fam_rows = []
    for cluster in order:
        members = frame[frame.cluster == cluster].treaty_id.tolist()
        if len(members) > 1:
            model = pd.Series([GEN2MODEL[tm.loc[t, "generation"]] for t in members]).mode()[0]
            modal_fps = pd.Series([ta.loc[t, "3.12_fps_typology"] for t in members]).mode()[0]
            years = [year[t] for t in members]
            label = (f"{model} model (n={len(members)}, {min(years)}-{max(years)}, "
                     f"dom. FPS={modal_fps})")
        else:
            label = f"Singleton ({year[members[0]]})"
        for t in members:
            fam_rows.append([t, partner[t], year[t], fid[cluster], label,
                             len(members), ta.loc[t, "3.12_fps_typology"]])
    families = pd.DataFrame(fam_rows, columns=[
        "treaty_id", "partner_state", "signature_year", "family_id",
        "family_label", "family_size", "fps_typology"
    ]).sort_values(["family_id", "signature_year", "treaty_id"])

    # Genealogy: each treaty's most similar strictly earlier treaty.
    fam_of = families.set_index("treaty_id").family_id
    chron = sorted(ids, key=lambda t: (tm.loc[t, "signature_date"], t))
    index_of = {t: k for k, t in enumerate(ids)}
    edges = []
    for position, child in enumerate(chron):
        if position == 0:
            continue
        earlier = chron[:position]
        parent = max(earlier, key=lambda p: matrix[index_of[p], index_of[child]])
        value = matrix[index_of[parent], index_of[child]]
        if value >= EDGE_MIN_COSINE:
            edges.append([parent, partner[parent], year[parent], child, partner[child],
                          year[child], round(value, 6), year[child] - year[parent],
                          fam_of[parent] == fam_of[child]])
    edges = pd.DataFrame(edges, columns=[
        "parent_treaty", "parent_partner", "parent_year", "child_treaty",
        "child_partner", "child_year", "cosine_similarity", "year_gap", "same_family"])

    out = outdir("ds3")
    sm.round(6).to_csv(os.path.join(out, "similarity_matrix.csv"))
    pairs.to_csv(os.path.join(out, "treaty_pairs.csv"), index=False)
    families.to_csv(os.path.join(out, "treaty_families.csv"), index=False)
    edges.to_csv(os.path.join(out, "genealogy_edges.csv"), index=False)

    r = np.corrcoef(pairs.cosine_similarity, pairs.shared_clause_features)[0, 1]
    print(f"treaties {len(ids)} | pairs {len(pairs)} | families {families.family_id.nunique()} "
          f"| edges {len(edges)} | text-vs-legal r = {r:.2f}")
    print(f"written to outputs/ds3/")

    # Compare against the published DS3 when it is available.
    try:
        published = load("ds3", "genealogy_edges.csv")
        same_edges = (len(published) == len(edges) and
                      set(zip(published.parent_treaty, published.child_treaty)) ==
                      set(zip(edges.parent_treaty, edges.child_treaty)))
        published_fam = load("ds3", "treaty_families.csv").set_index("treaty_id")
        same_fam = all(published_fam.loc[t, "family_id"] == fam_of[t] for t in ids)
        print(f"reproduction check: edges identical = {same_edges} | "
              f"family assignment identical = {same_fam}")
    except FileNotFoundError:
        print("reproduction check skipped (published DS3 not in data/ds3)")


if __name__ == "__main__":
    main()
