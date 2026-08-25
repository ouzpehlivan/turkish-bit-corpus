"""Protocol Step 6: cross-dataset verification suite.

Runs the release checks for row and column counts, the
FPS block identity across Datasets 1, 2 and 3, provenance of the clause extracts,
coding-log synchronisation, the internal consistency of Dataset 3 (matrix
symmetry, pair table against matrix, family labels against member rows, genealogy
edges against the argmax rule), the .parquet copies against their .csv originals,
and the family legend embedded in the Dataset 3 network page against
treaty_families.csv.

Usage:  python code/verify.py
Exit code 0 if every check passes, 1 otherwise.
"""
import json
import re
import sys
import zipfile

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import _find, annotations, clauses, extracts, fps, load, metadata, norm
from recompute_ds3 import build_documents, packages

FAILED = []

EXPECTED_CLAUSE_TYPE_COUNTS = {
    "entry_into_force": 153,
    "isds": 150,
    "transfers": 145,
    "ssds": 144,
    "preamble": 141,
    "expropriation": 141,
    "definitions": 140,
    "admission_promotion": 135,
    "subrogation": 134,
    "national_mfn_treatment": 104,
    "scope_application": 90,
    "compensation_losses": 81,
    "general_exceptions": 67,
    "non_derogation": 56,
    "other": 50,
    "denial_benefits": 41,
    "consultations": 30,
    "service_documents": 21,
    "fair_equitable": 17,
    "esg_environment_labour": 13,
    "amendment": 9,
    "transparency": 8,
    "taxation": 7,
    "right_to_regulate": 5,
    "entry_sojourn": 4,
    "umbrella": 3,
    "performance_requirements": 1,
}


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


def _read_text(dataset, filename):
    with open(_find(dataset, filename), encoding="utf-8") as fh:
        return fh.read()


def _embedded_json(page, name):
    """Decode a JSON value assigned to a JavaScript ``const``."""
    marker = f"const {name}="
    start = page.index(marker) + len(marker)
    value, _end = json.JSONDecoder().raw_decode(page[start:])
    return value


def _family_label(group):
    """Re-derive the deterministic, tie-aware family label."""
    if len(group) == 1:
        return f"Singleton ({int(group.signature_year.iloc[0])})"
    generation_names = {
        "gen_1_early": "Early",
        "gen_2_liberalization": "Liberalization",
        "gen_3_eu_harmonization": "EU-harmonization",
        "gen_4_new_model": "New Turkish",
    }
    model_modes = sorted(group.generation.map(generation_names).mode())
    fps_modes = sorted(group.fps_typology.mode())
    model = (f"{model_modes[0]} model" if len(model_modes) == 1 else
             f"Mixed-generation model ({'/'.join(model_modes)} tie)")
    fps_label = (f"dom. FPS={fps_modes[0]}" if len(fps_modes) == 1 else
                 f"FPS tie={'/'.join(fps_modes)}")
    years = group.signature_year.astype(int)
    return (f"{model} (n={len(group)}, {years.min()}-{years.max()}, "
            f"{fps_label})")


def main():
    tm, fc, ae, ta, ce = metadata(), fps(), extracts(), annotations(), clauses()
    ids = list(tm.index)
    n = len(ids)

    print("Datasets 1 and 2")
    check("metadata rows == fps rows == annotation rows", len({n, len(fc), len(ta)}) == 1,
          f"{n}/{len(fc)}/{len(ta)}")
    check("annotation table is 122 columns", ta.shape[1] + 1 == 122, str(ta.shape))
    check("treaty_id sets identical", set(fc.index) == set(ta.index) == set(ids))
    check("row order identical where treaty_id is the canonical sequence",
          list(ae.index) == list(ta.index) == ids)
    check("primary keys are unique", tm.index.is_unique and fc.index.is_unique
          and ae.index.is_unique and ta.index.is_unique)
    check("clause extract key and required fields are complete",
          len(ce) == 1890 and not ce[["treaty_id", "article"]].duplicated().any()
          and not (ce.clause_text.str.strip() == "").any()
          and not (ce.clause_type.str.strip() == "").any())
    observed_clause_types = set(ce.clause_type)
    expected_clause_types = set(EXPECTED_CLAUSE_TYPE_COUNTS)
    check("clause_type uses the controlled 27-value vocabulary",
          observed_clause_types == expected_clause_types,
          str((sorted(observed_clause_types - expected_clause_types),
               sorted(expected_clause_types - observed_clause_types))))
    observed_clause_counts = ce.clause_type.value_counts().to_dict()
    check("clause_type distribution matches the 1,890-row release snapshot",
          observed_clause_counts == EXPECTED_CLAUSE_TYPE_COUNTS,
          str(observed_clause_counts))
    metadata_bad = []
    for treaty_id, row in tm.iterrows():
        if not re.fullmatch(r"TUR_BIT_\d{3}", treaty_id):
            metadata_bad.append((treaty_id, "treaty_id"))
        if not re.fullmatch(r"[A-Z]{3}", row.partner_iso3):
            metadata_bad.append((treaty_id, "partner_iso3"))
        for column in ("signature_date", "entry_into_force_date",
                       "termination_date", "official_gazette_date",
                       "protocol_date"):
            value = row[column]
            if value:
                try:
                    parsed = pd.to_datetime(value, format="%Y-%m-%d", errors="raise")
                except (TypeError, ValueError):
                    metadata_bad.append((treaty_id, column))
                else:
                    if parsed.strftime("%Y-%m-%d") != value:
                        metadata_bad.append((treaty_id, column))
        if row.status == "signed_not_in_force" and row.entry_into_force_date:
            metadata_bad.append((treaty_id, "signed_not_in_force/eif"))
        if row.status in {"in_force", "terminated"} and not row.entry_into_force_date:
            metadata_bad.append((treaty_id, "status/eif"))
        if row.status == "terminated" and not row.termination_date:
            metadata_bad.append((treaty_id, "terminated/date"))
        if row.replaced_by and row.replaced_by not in tm.index:
            metadata_bad.append((treaty_id, "replaced_by"))
        year = int(row.signature_date[:4])
        expected_generation = (
            "gen_1_early" if year <= 1985 else
            "gen_2_liberalization" if year <= 1998 else
            "gen_3_eu_harmonization" if year <= 2009 else
            "gen_4_new_model"
        )
        if row.generation != expected_generation:
            metadata_bad.append((treaty_id, "generation"))
        if not row.source_pdf_filename.lower().endswith(".txt"):
            metadata_bad.append((treaty_id, "source filename"))
    check("metadata formats, status/date rules and generations are consistent",
          not metadata_bad, str(metadata_bad[:5]))
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
    check("coding provenance and source filenames are synchronized",
          all(log.loc[t, "source_filename"] == tm.loc[t, "source_pdf_filename"]
              and log.loc[t, "coded_by"] == ta.loc[t, "coded_by"] == "OKP_verified"
              and log.loc[t, "coding_confidence"] == ta.loc[t, "coding_confidence"] == "high"
              and log.loc[t, "coding_date"] == ta.loc[t, "coding_date"]
              and log.loc[t, "variables_coded"] == "118"
              and log.loc[t, "inconclusive_count"] == "0"
              and log.loc[t, "verification_status"] == "OKP_verified"
              for t in ids))

    missing = []
    for index, row in ce.iterrows():
        if row.article == "Preamble":
            source = ae.loc[row.treaty_id, "Preamble"]
        elif row.article == "Protocol":
            source = ae.loc[row.treaty_id, "Protocol"]
        else:
            source = ae.loc[row.treaty_id, f"{row.article} - Text"]
        source_text = norm(source)
        clause_text = norm(row.clause_text)
        if not source_text:
            missing.append((row.treaty_id, row.article, int(index), "empty_source"))
        elif source_text not in clause_text:
            missing.append((row.treaty_id, row.article, int(index),
                            "full_source_not_found"))
    check("clause extracts traceable to article extracts", not missing, str(missing[:3]))

    check("total_articles matches extracted articles",
          all(int(tm.loc[t, "total_articles"]) ==
              sum(1 for c in ae.columns
                  if c.startswith("Article") and c.endswith("- Text") and str(ae.loc[t, c]).strip())
              for t in ids))
    split_residue = []
    repeated_residue = []
    allowed_sentence_starts = {
        ("TUR_BIT_032", "Article 8 - Title"),
        ("TUR_BIT_127", "Article 17 - Title"),
    }
    for treaty_id, row in ae.iterrows():
        for title_col in [c for c in ae.columns if c.endswith(" - Title")]:
            title = row[title_col].strip()
            body = row[title_col.replace(" - Title", " - Text")].strip()
            if re.search(r"(?i)settlements? of$", title):
                split_residue.append((treaty_id, title_col))
            if (title and body.lower().startswith(title.lower())
                    and (treaty_id, title_col) not in allowed_sentence_starts):
                repeated_residue.append((treaty_id, title_col))
    check("no truncated or redundantly repeated article headings remain",
          not split_residue and not repeated_residue,
          str((split_residue + repeated_residue)[:5]))

    split_regressions = {
        ("TUR_BIT_052", "Article 7"): (
            "Settlement of Investment Disputes between a Contracting Party and an Investor of the other Contracting Party",
            "1. Any disputes arising between a Contracting Party and an investor of the other Contracting Party which involve:",
        ),
        ("TUR_BIT_072", "Article 9"): (
            "Dispute Settlement Between a Contracting Party and Investors of the other Contracting Party",
            "1. In the event of occurrence of a dispute between a Contracting Party in whose territory an investment is made",
        ),
        ("TUR_BIT_109", "Article 13"): (
            "Responsibilities of Nationals and Companies of a Contracting Party In the. Territory of the other Contracting Party",
            "1. Investors of one Contracting Party in the territory of the other Contracting Party shall be bound by the laws",
        ),
    }
    regression_bad = []
    for (treaty_id, article), (expected_title, expected_body_start) in split_regressions.items():
        title = ae.loc[treaty_id, f"{article} - Title"]
        body = ae.loc[treaty_id, f"{article} - Text"]
        if title != expected_title or not body.startswith(expected_body_start):
            regression_bad.append((treaty_id, article, title, body[:120]))
    check("special article title/body split regressions remain fixed",
          not regression_bad, str(regression_bad))

    bad_clause_titles = []
    for row in ce.itertuples(index=False):
        if row.article == "Preamble":
            expected_title = "Preamble"
        elif row.article == "Protocol":
            expected_title = "Protocol"
        else:
            expected_title = ae.loc[row.treaty_id, f"{row.article} - Title"]
        if row.article_title != expected_title:
            bad_clause_titles.append((row.treaty_id, row.article))
    check("clause-extract titles match article extracts",
          not bad_clause_titles, str(bad_clause_titles[:5]))

    conditional_bad = []
    for treaty_id in ids:
        if ta.loc[treaty_id, "3.07_fet_type"] == "None" and any(
                ta.loc[treaty_id, column] != "Not applicable"
                for column in ("3.08_fet_intl_law", "3.09_fet_list",
                               "3.10_fet_combined_nt")):
            conditional_bad.append((treaty_id, "FET"))
        other_yes = ta.loc[treaty_id, "7.11_other_forum"] == "Yes"
        if other_yes != bool(ta.loc[treaty_id, "7.12_other_detail"].strip()):
            conditional_bad.append((treaty_id, "other forum detail"))
    check("conditional annotation fields are logically consistent",
          not conditional_bad, str(conditional_bad[:5]))

    # Every observed coded value must be admitted by the release schema.  The
    # schema's descriptions carry the numeric variable identifiers used by the
    # flat CSV, so the check is independent of its nested JSON property names.
    schema = json.loads(_read_text("ds1", "turkish_bit_schema.json"))
    schema_bad = []
    def inspect_schema(value):
        if isinstance(value, dict):
            match = re.match(r"(\d\.\d\d)", value.get("description", ""))
            if match and "enum" in value:
                column = next((c for c in ta.columns
                               if c.startswith(match.group(1) + "_")), None)
                if column:
                    allowed = {str(item) for item in value["enum"] if item is not None}
                    observed = {item for item in ta[column].unique() if item != ""}
                    missing_values = sorted(observed - allowed)
                    if missing_values:
                        schema_bad.append((column, missing_values))
            for child in value.values():
                inspect_schema(child)
        elif isinstance(value, list):
            for child in value:
                inspect_schema(child)
    inspect_schema(schema)
    check("schema enums admit every observed annotation value",
          not schema_bad, str(schema_bad[:5]))

    logged = load("ds2", "ds2_change_log.csv")
    bad_log_partner = [row.treaty_id for row in logged.itertuples(index=False)
                       if row.partner_state != tm.loc[row.treaty_id, "partner_state"]]
    composite_log = [value for value in logged.variable.unique()
                     if "/" in value or "+" in value]
    check("change log uses canonical partners and atomic variable names",
          not bad_log_partner and not composite_log and not logged.duplicated().any(),
          str((bad_log_partner[:3], composite_log[:3])))

    # The full-corpus audit ledger is a required DS1 release artifact. Every
    # corrected annotation entry must point to the value now published;
    # retained source anomalies must be non-mutating; and review candidates
    # must still carry their disclosed current value. _find supports both the
    # sibling release layout and data/ds1 downloaded by download_data.py.
    audit_path = _find("ds1", "AUDIT_FINDINGS_2026-08-22.csv")
    audit = pd.read_csv(audit_path, dtype=str, keep_default_na=False)
    required_audit = {
        "audit_date", "dataset", "record", "field", "status",
        "old_value", "new_value", "evidence", "source_status",
    }
    audit_bad = []
    if set(audit.columns) != required_audit or audit.duplicated().any():
        audit_bad.append(("schema_or_duplicate", ""))
    allowed_status = {"corrected", "retained", "review_candidate"}
    if not set(audit.status).issubset(allowed_status):
        audit_bad.append(("status", sorted(set(audit.status) - allowed_status)))
    for row in audit.itertuples(index=False):
        if (row.dataset == "DS2" and row.record in ta.index
                and row.field in ta.columns):
            current = ta.loc[row.record, row.field]
            expected = row.new_value if row.status == "corrected" else row.old_value
            if current != expected:
                audit_bad.append((row.record, row.field))
        if row.status == "retained" and row.old_value != row.new_value:
            audit_bad.append((row.record, "retained_mutation"))
    corrected_annotations = audit[
        (audit.dataset == "DS2") & (audit.status == "corrected")
        & audit.field.isin(ta.columns)
    ]
    check("audit ledger agrees with all corrected annotation cells",
          not audit_bad and len(corrected_annotations) == 832,
          str((len(corrected_annotations), audit_bad[:5])))

    # The ZIP is the canonical cleaned-text bundle. Metadata counts and source
    # filenames must therefore be checked against the archive itself.
    with zipfile.ZipFile(_find("ds1", "treaty_texts.zip")) as archive:
        members = [m for m in archive.namelist() if not m.endswith("/")]
        check("ZIP has one unique UTF-8 text per treaty",
              len(members) == n and len(set(members)) == n
              and all(m.lower().endswith(".txt") for m in members)
              and archive.testzip() is None)
        bad_zip = []
        zip_payloads = []
        for treaty_id in ids:
            filename = tm.loc[treaty_id, "source_pdf_filename"]
            # The historical metadata field now stores the published .txt
            # member name directly; tolerate an older .pdf-valued copy too.
            expected = re.sub(r"\.pdf$", ".txt", filename, flags=re.I)
            candidates = [m for m in members if m.rsplit("/", 1)[-1] == expected]
            if len(candidates) != 1:
                bad_zip.append((treaty_id, "filename"))
                continue
            try:
                payload = archive.read(candidates[0])
                text = payload.decode("utf-8")
            except UnicodeDecodeError:
                bad_zip.append((treaty_id, "utf8"))
                continue
            zip_payloads.append(payload)
            if text.startswith("\ufeff") or "\x00" in text or "\ufffd" in text:
                bad_zip.append((treaty_id, "encoding artefact"))
            if len(text.split()) != int(tm.loc[treaty_id, "total_word_count"]):
                bad_zip.append((treaty_id, "word_count"))
        check("ZIP filenames, UTF-8 and metadata word counts agree",
              not bad_zip and len(set(zip_payloads)) == n,
              str(bad_zip[:5]))

    print("Dataset 3")
    try:
        sm = load("ds3", "similarity_matrix.csv", index_col=0).astype(float)
        matrix = sm.values
        tp = load("ds3", "treaty_pairs.csv").astype({
            "cosine_similarity": float, "jaccard_similarity": float,
            "shared_clause_features": int, "year_gap": int,
        })
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
        expected_pair_keys = {(ids[i], ids[j]) for i in range(n) for j in range(i + 1, n)}
        actual_pair_keys = set(zip(tp.treaty_a, tp.treaty_b))
        check("pair keys are unique and complete",
              len(actual_pair_keys) == len(tp) and actual_pair_keys == expected_pair_keys)
        index_of = {t: k for k, t in enumerate(ids)}
        from_matrix = np.array([matrix[index_of[a], index_of[b]]
                                for a, b in zip(tp.treaty_a, tp.treaty_b)])
        check("pair cosine values equal the matrix",
              np.allclose(from_matrix, tp.cosine_similarity, atol=5e-6))

        # Recompute every DS3 source-derived number. Internal matrix/pair
        # agreement alone cannot detect an older, mutually consistent snapshot.
        docs = build_documents(ae, ids)
        expected_matrix = cosine_similarity(
            TfidfVectorizer(stop_words="english", sublinear_tf=True, max_df=0.9)
            .fit_transform(docs)
        )
        np.fill_diagonal(expected_matrix, 1.0)
        check("matrix is reproducible from current article extracts",
              np.allclose(matrix, expected_matrix, atol=5e-7),
              f"max |delta|={np.max(np.abs(matrix - expected_matrix)):.8f}")

        token_sets = [set(w for w in re.findall(r"[a-z]+", d.lower())
                          if w not in ENGLISH_STOP_WORDS) for d in docs]
        package_vectors = {t: packages(ta, t) for t in ids}
        expected_jaccard = []
        expected_shared = []
        for a, b in zip(tp.treaty_a, tp.treaty_b):
            i, j = index_of[a], index_of[b]
            expected_jaccard.append(
                len(token_sets[i] & token_sets[j]) / len(token_sets[i] | token_sets[j]))
            expected_shared.append(int((package_vectors[a] & package_vectors[b]).sum()))
        check("pair Jaccard values reproducible from current extracts",
              np.allclose(tp.jaccard_similarity, expected_jaccard, atol=5e-7))
        check("shared legal-feature counts reproducible from current annotations",
              np.array_equal(tp.shared_clause_features.to_numpy(), expected_shared))
        expected_year_gap = [abs(int(tm.loc[a, "signature_date"][:4]) -
                                 int(tm.loc[b, "signature_date"][:4]))
                             for a, b in zip(tp.treaty_a, tp.treaty_b)]
        check("pair helper columns and year gaps match metadata",
              all(tp.loc[k, "partner_a"] == tm.loc[a, "partner_state"]
                  and tp.loc[k, "partner_b"] == tm.loc[b, "partner_state"]
                  and int(tp.loc[k, "year_a"]) == int(tm.loc[a, "signature_date"][:4])
                  and int(tp.loc[k, "year_b"]) == int(tm.loc[b, "signature_date"][:4])
                  and int(tp.loc[k, "year_gap"]) == expected_year_gap[k]
                  for k, (a, b) in enumerate(zip(tp.treaty_a, tp.treaty_b))))
        check("family assignment covers every treaty", set(fam.index) == set(ids))
        check("family FPS column matches Dataset 1",
              all(fam.loc[t, "fps_typology"] == fc.loc[t, "fps_typology"] for t in ids))
        bad_labels = []
        for family_id, group in fam.reset_index().groupby("family_id"):
            enriched = group.assign(
                generation=group.treaty_id.map(tm.generation),
                fps_typology=group.treaty_id.map(fc.fps_typology),
            )
            if group.family_label.nunique() != 1 or group.family_size.astype(int).nunique() != 1 \
                    or int(group.family_size.iloc[0]) != len(group) \
                    or group.family_label.iloc[0] != _family_label(enriched):
                bad_labels.append(family_id)
        check("family labels re-derivable from members", not bad_labels, str(bad_labels))
        check("every edge follows the (date, treaty_id) total order",
              all((tm.loc[r.parent_treaty, "signature_date"], r.parent_treaty)
                  < (tm.loc[r.child_treaty, "signature_date"], r.child_treaty)
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
        check("parent is the most similar eligible predecessor", not wrong, str(wrong[:3]))
        missing_edges = [t for k, t in enumerate(chron)
                         if k and t not in set(ed.child_treaty)
                         and max(matrix[index_of[p], index_of[t]] for p in chron[:k]) >= 0.45]
        check("no edge above the threshold is missing", not missing_edges, str(missing_edges[:3]))
        edge_bad = []
        fam_of = fam.family_id.to_dict()
        for _, row in ed.iterrows():
            a, b = row.parent_treaty, row.child_treaty
            expected = round(expected_matrix[index_of[a], index_of[b]], 6)
            if (not np.isclose(row.cosine_similarity, expected, atol=5e-7)
                    or row.parent_partner != tm.loc[a, "partner_state"]
                    or row.child_partner != tm.loc[b, "partner_state"]
                    or int(row.parent_year) != int(tm.loc[a, "signature_date"][:4])
                    or int(row.child_year) != int(tm.loc[b, "signature_date"][:4])
                    or int(row.year_gap) != int(row.child_year) - int(row.parent_year)
                    or str(row.same_family).lower() != str(fam_of[a] == fam_of[b]).lower()):
                edge_bad.append(b)
        check("edge values and helper columns match current sources",
              not edge_bad, str(edge_bad[:5]))

    print("Convenience copies and derived pages")

    # The .parquet copies must be cell-for-cell identical to their .csv originals.
    for dataset, stem in (("ds1", "treaty_metadata"), ("ds2", "treaty_annotations")):
        try:
            csv = load(dataset, f"{stem}.csv")
            pq = load(dataset, f"{stem}.parquet").astype(str)
            same = list(csv.columns) == list(pq.columns) and csv.reset_index(drop=True).equals(
                pq.reset_index(drop=True))
        except FileNotFoundError:
            same, detail = True, "parquet not present, skipped"
        else:
            detail = f"{csv.shape} vs {pq.shape}"
        check(f"{stem}.parquet matches {stem}.csv", same, detail)

    # All three embedded constants in the Dataset 3 network page must agree
    # with the source tables. A stale page can otherwise look plausible while
    # silently presenting old edge weights or labels.
    try:
        page = _read_text("ds3", "network_visualization.html")
    except FileNotFoundError:
        check("network page embedded data match published tables", True,
              "page not present, skipped")
    else:
        legend = _embedded_json(page, "families")
        nodes = _embedded_json(page, "rawNodes")
        network_edges = _embedded_json(page, "rawEdges")
        fam = load("ds3", "treaty_families.csv")
        truth = {k: (len(g), g["family_label"].iloc[0])
                 for k, g in fam.groupby("family_id")}
        drift = [k for k in set(truth) | set(legend)
                 if k not in legend or k not in truth
                 or int(legend[k]["size"]) != truth[k][0]
                 or legend[k]["label"] != truth[k][1]]
        check("network page family legend matches treaty_families.csv",
              not drift, str(sorted(drift)[:3]))
        fam_index = fam.set_index("treaty_id")
        node_index = {node["id"]: node for node in nodes}
        node_drift = [t for t in ids if t not in node_index
                      or node_index[t]["label"] != tm.loc[t, "partner_state"]
                      or int(node_index[t]["year"]) != int(tm.loc[t, "signature_date"][:4])
                      or node_index[t]["family"] != fam_index.loc[t, "family_id"]
                      or node_index[t]["fps"] != fc.loc[t, "fps_typology"]]
        check("network page nodes match metadata and families",
              len(node_index) == n and not node_drift, str(node_drift[:3]))
        published_edges = load("ds3", "genealogy_edges.csv")
        edge_index = {(edge["from"], edge["to"]): edge for edge in network_edges}
        edge_drift = []
        for row in published_edges.itertuples(index=False):
            key = (row.parent_treaty, row.child_treaty)
            if key not in edge_index or not np.isclose(
                    float(edge_index[key]["value"]), float(row.cosine_similarity), atol=5e-7):
                edge_drift.append(key)
        check("network page edges match genealogy_edges.csv",
              len(edge_index) == len(published_edges) and not edge_drift,
              str(edge_drift[:3]))

    try:
        explorer_page = _read_text("ds2", "explorer.html")
    except FileNotFoundError:
        check("DS2 explorer records match annotations", True,
              "page not present, skipped")
    else:
        explorer = _embedded_json(explorer_page.replace("const DATA = ",
                                                        "const DATA=", 1), "DATA")
        explorer_rows = {row["id"]: row for row in explorer["treaties"]}
        explorer_keys = [item["key"] for item in explorer["variables"]]
        expected_explorer_keys = {
            column for column in ta.columns
            if column not in {"coded_by", "coding_confidence", "coding_date",
                              "3.13_fps_text"}
        }
        option_drift = []
        for item in explorer["variables"]:
            observed = {value for value in ta[item["key"]] if value != ""}
            if set(item.get("options", [])) != observed:
                option_drift.append(item["key"])
        explorer_drift = []
        for treaty_id in ids:
            row = explorer_rows.get(treaty_id)
            if row is None or row["partner"] != tm.loc[treaty_id, "partner_state"]:
                explorer_drift.append(treaty_id)
                continue
            if any(row["v"].get(key) != ta.loc[treaty_id, key]
                   for key in explorer_keys):
                explorer_drift.append(treaty_id)
        check("DS2 explorer records match annotations",
              len(explorer_rows) == n and set(explorer_keys) == expected_explorer_keys
              and not option_drift and not explorer_drift,
              str((sorted(expected_explorer_keys - set(explorer_keys))[:3],
                   option_drift[:3], explorer_drift[:3])))

    print(f"\n{len(FAILED)} failed check(s)" if FAILED else "\nAll checks passed.")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
