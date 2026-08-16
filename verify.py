"""Protocol Step 6: cross-dataset verification suite.

Runs the checks the datasets are released against: row and column counts, the FPS
block identity across Datasets 1, 2 and 3, provenance of the clause extracts,
coding-log synchronisation, conditional-field logic, and the internal consistency
of Dataset 3 (matrix symmetry, pair table against matrix, family labels against
member rows, genealogy edges against the argmax rule).

Usage:  python src/verify.py
Exit code 0 if every check passes, 1 otherwise.
"""
import sys

import numpy as np
import pandas as pd

from common import annotations, clauses, extracts, fps, load, metadata, norm

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


def main():
    tm, fc, ae, ta, ce = metadata(), fps(), extracts(), annotations(), clauses()
    ids = list(tm.index)
    n = len(ids)

    print("Datasets 1 and 2")
    check("metadata rows == fps rows == annotation rows", len({n, len(fc), len(ta)}) == 1,
          f"{n}/{len(fc)}/{len(ta)}")
    check("annotation table is 122 columns", ta.shape[1] + 1 == 122, str(ta.shape))
    check("treaty_id sets identical", set(fc.index) == set(ta.index) == set(ids))
    pairs = [("fps_typology", "3.12_fps_typology"),
             ("fps_formulation_variant", "3.14_fps_variant"),
             ("fps_article_location", "3.15_fps_location"),
             ("fps_formulation_text", "3.13_fps_text")]
    for ds1_col, ds2_col in pairs:
        bad = [t for t in ids if fc.loc[t, ds1_col] != ta.loc[t, ds2_col]]
        check(f"FPS block sync: {ds1_col}", not bad, str(bad[:3]))
    expected = {"G": "No clause", "D": "With reference to domestic law"}
    check("3.11_fps_clause consistent with typology",
          all(ta.loc[t, "3.11_fps_clause"] == expected.get(fc.loc[t, "fps_typology"], "Standard")
              for t in ids))

    log = load("ds2", "coding_log.csv").set_index("treaty_id")
    check("coding_log partner states match metadata",
          all(log.loc[t, "partner_state"] == tm.loc[t, "partner_state"] for t in ids))
    check("coding_log signature dates match metadata",
          all(log.loc[t, "signature_date"] == tm.loc[t, "signature_date"] for t in ids))

    missing = []
    for _, row in ce.iterrows():
        body = row.clause_text.split("\n", 2)[-1][:100]
        haystack = " ||| ".join(str(v) for v in ae.loc[row.treaty_id].values)
        if body and body not in haystack:
            missing.append((row.treaty_id, row.article))
    check("clause extracts traceable to article extracts", not missing, str(missing[:3]))

    check("total_articles matches extracted articles",
          all(int(tm.loc[t, "total_articles"]) ==
              sum(1 for c in ae.columns
                  if c.startswith("Madde") and "Baslik" not in c and str(ae.loc[t, c]).strip())
              for t in ids))

    print("Dataset 3")
    try:
        sm = load("ds3", "similarity_matrix.csv", index_col=0).astype(float)
        matrix = sm.values
        tp = load("ds3", "treaty_pairs.csv").astype({"cosine_similarity": float})
        fam = load("ds3", "treaty_families.csv").set_index("treaty_id")
        ed = load("ds3", "genealogy_edges.csv").astype({"cosine_similarity": float})
    except FileNotFoundError as exc:
        print(f"  SKIP  Dataset 3 not present ({exc})")
        sm = None

    if sm is not None:
        check("similarity matrix ids match metadata", list(sm.index) == ids == list(sm.columns))
        check("matrix symmetric, unit diagonal, within [0,1]",
              np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix), 1)
              and matrix.min() >= 0 and matrix.max() <= 1 + 1e-9)
        check("pair count == n(n-1)/2", len(tp) == n * (n - 1) // 2, str(len(tp)))
        index_of = {t: k for k, t in enumerate(ids)}
        from_matrix = np.array([matrix[index_of[a], index_of[b]]
                                for a, b in zip(tp.treaty_a, tp.treaty_b)])
        check("pair cosine values equal the matrix",
              np.allclose(from_matrix, tp.cosine_similarity, atol=5e-6))
        check("family assignment covers every treaty", set(fam.index) == set(ids))
        check("family FPS column matches Dataset 1",
              all(fam.loc[t, "fps_typology"] == fc.loc[t, "fps_typology"] for t in ids))
        import re
        bad_labels = []
        for family_id, group in fam.reset_index().groupby("family_id"):
            label = group.family_label.iloc[0]
            match = re.search(r"n=(\d+), (\d{4})-(\d{4}), dom\. FPS=(\w)", label)
            if match:
                years = group.signature_year.astype(int)
                if (int(match.group(1)) != len(group) or int(match.group(2)) != years.min()
                        or int(match.group(3)) != years.max()
                        or match.group(4) != group.fps_typology.mode()[0]):
                    bad_labels.append(family_id)
        check("family labels re-derivable from members", not bad_labels, str(bad_labels))
        check("every edge runs from an earlier to a later treaty",
              all(tm.loc[r.parent_treaty, "signature_date"] <= tm.loc[r.child_treaty, "signature_date"]
                  for _, r in ed.iterrows()))
        check("each treaty has at most one parent", ed.child_treaty.is_unique)
        chron = sorted(ids, key=lambda t: (tm.loc[t, "signature_date"], t))
        position = {t: k for k, t in enumerate(chron)}
        wrong = []
        for _, r in ed.iterrows():
            earlier = chron[:position[r.child_treaty]]
            best = max(earlier, key=lambda p: matrix[index_of[p], index_of[r.child_treaty]])
            if best != r.parent_treaty and not np.isclose(
                    matrix[index_of[best], index_of[r.child_treaty]],
                    matrix[index_of[r.parent_treaty], index_of[r.child_treaty]]):
                wrong.append(r.child_treaty)
        check("parent is the most similar earlier treaty", not wrong, str(wrong[:3]))
        missing_edges = [t for k, t in enumerate(chron)
                         if k and t not in set(ed.child_treaty)
                         and max(matrix[index_of[p], index_of[t]] for p in chron[:k]) >= 0.45]
        check("no edge above the threshold is missing", not missing_edges, str(missing_edges[:3]))

    print(f"\n{len(FAILED)} failed check(s)" if FAILED else "\nAll checks passed.")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
