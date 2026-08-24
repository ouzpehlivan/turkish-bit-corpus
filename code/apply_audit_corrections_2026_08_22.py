"""Apply the 22 August 2026 full-corpus audit corrections.

The script is deliberately assertion-heavy and idempotent.  It edits only the
canonical release files, records every annotation cell change in the DS2 change
log and an audit findings table, synchronises CSV/Parquet copies, and preserves
the ZIP members' metadata while correcting the cleaned text payloads.

Derived DS2/DS3 visualisations are rebuilt by the separate make_* scripts after
this script completes.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


def _resolve_root() -> Path:
    """Locate the directory holding ds1/, ds2/ and ds3/.

    Mirrors common.py's resolution order so the script runs both from the
    published release (where the dataset directories are siblings of code/'s
    parent) and from a repository clone (where download_data.py stages them
    under <repo>/data).
    """
    here = Path(__file__).resolve()
    candidates = [
        os.environ.get("TURKISH_BIT_AUDIT_ROOT"),
        os.environ.get("TURKISH_BIT_DATA_ROOT"),
        here.parents[1] / "data",
        here.parents[2],
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "ds1").is_dir():
            return Path(candidate).resolve()
    raise SystemExit(
        "Could not locate the release directories. None of the following "
        "contains a ds1/ folder:\n  "
        + "\n  ".join(str(c) for c in candidates if c)
        + "\nSet TURKISH_BIT_AUDIT_ROOT to the directory holding ds1/, ds2/ "
          "and ds3/, or run download_data.py to stage the release."
    )


ROOT = _resolve_root()
AUDIT_DATE = "2026-08-22"


def tids(spec: str) -> list[str]:
    return [f"TUR_BIT_{int(value):03d}" for value in spec.split()]


def read_csv(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative, dtype=str, keep_default_na=False)


tm = read_csv("ds1/treaty_metadata.csv").set_index("treaty_id")
ae = read_csv("ds1/article_extracts.csv").set_index("treaty_id")
fc = read_csv("ds1/fps_classification.csv").set_index("treaty_id")
ta = read_csv("ds2/treaty_annotations.csv").set_index("treaty_id")
ce = read_csv("ds2/clause_extracts.csv")
coding_log = read_csv("ds2/coding_log.csv").set_index("treaty_id")
change_log = read_csv("ds2/ds2_change_log.csv")

findings: list[dict[str, str]] = []
annotation_changes: list[dict[str, str]] = []


def finding(dataset: str, record: str, field: str, old: str, new: str,
            evidence: str, source_status: str = "treaty_text_verified",
            status: str = "corrected") -> None:
    findings.append({
        "audit_date": AUDIT_DATE,
        "dataset": dataset,
        "record": record,
        "field": field,
        "status": status,
        "old_value": old,
        "new_value": new,
        "evidence": evidence,
        "source_status": source_status,
    })


def set_annotation(treaty_id: str, column: str, new: str, evidence: str) -> None:
    old = ta.loc[treaty_id, column]
    if old == new:
        return
    ta.loc[treaty_id, column] = new
    reason = f"Full-corpus audit {AUDIT_DATE}: {evidence}"
    annotation_changes.append({
        "treaty_id": treaty_id,
        "partner_state": tm.loc[treaty_id, "partner_state"],
        "variable": column,
        "old_value": old,
        "new_value": new,
        "reason": reason,
    })
    finding("DS2", treaty_id, column, old, new, evidence)


def bulk(column: str, treaty_ids: list[str], new: str, evidence: str) -> None:
    for treaty_id in treaty_ids:
        set_annotation(treaty_id, column, new, evidence)


# Normalise the historical change log before appending this audit.  Its
# partner_state column had stored a display label (partner + year), and ten
# rows combined multiple variables in one field.
bad_partner_count = int(sum(
    change_log.loc[index, "partner_state"] != tm.loc[row.treaty_id, "partner_state"]
    for index, row in change_log.iterrows()
))
for index, row in change_log.iterrows():
    change_log.loc[index, "partner_state"] = tm.loc[row.treaty_id, "partner_state"]
if bad_partner_count:
    finding("DS2", "all historical change-log rows", "partner_state",
            f"{bad_partner_count} partner+year labels", "canonical partner_state",
            "The column is named partner_state; signature year is already available "
            "through treaty_id in Dataset 1.", "metadata_verified")

expanded_log_rows = []
meta_mask = change_log.variable == "coded_by/verification_status"
for row in change_log.loc[meta_mask].itertuples(index=False):
    expanded_log_rows.extend([
        {"treaty_id": row.treaty_id, "partner_state": tm.loc[row.treaty_id, "partner_state"],
         "variable": "coded_by", "old_value": "OKP", "new_value": "OKP_verified",
         "reason": row.reason},
        {"treaty_id": row.treaty_id, "partner_state": tm.loc[row.treaty_id, "partner_state"],
         "variable": "verification_status", "old_value": "pending_author_review",
         "new_value": "OKP_verified", "reason": row.reason},
    ])
composite_mask = change_log.variable == "3.12+3.07/3.08+text"
if composite_mask.any():
    row = change_log.loc[composite_mask].iloc[0]
    current_fps = fc.loc[row.treaty_id, "fps_formulation_text"]
    expanded_log_rows.extend([
        {"treaty_id": row.treaty_id, "partner_state": tm.loc[row.treaty_id, "partner_state"],
         "variable": "3.12_fps_typology", "old_value": "B", "new_value": "C",
         "reason": row.reason},
        {"treaty_id": row.treaty_id, "partner_state": tm.loc[row.treaty_id, "partner_state"],
         "variable": "3.07_fet_type", "old_value": "FET unqualified",
         "new_value": "FET qualified", "reason": row.reason},
        {"treaty_id": row.treaty_id, "partner_state": tm.loc[row.treaty_id, "partner_state"],
         "variable": "3.08_fet_intl_law", "old_value": "None",
         "new_value": "CIL/minimum standard of treatment", "reason": row.reason},
        {"treaty_id": row.treaty_id, "partner_state": tm.loc[row.treaty_id, "partner_state"],
         "variable": "3.13_fps_text",
         "old_value": current_fps.replace("protection and security",
                                            "protection and. security", 1),
         "new_value": current_fps, "reason": row.reason},
    ])
if meta_mask.any() or composite_mask.any():
    removed = int(meta_mask.sum() + composite_mask.sum())
    change_log = change_log.loc[~(meta_mask | composite_mask)].copy()
    change_log = pd.concat([change_log, pd.DataFrame(expanded_log_rows)], ignore_index=True)
    finding("DS2", "historical change log", "variable", f"{removed} composite rows",
            f"{len(expanded_log_rows)} atomic rows",
            "Each change-log row now names exactly one field, making the history machine-readable.",
            "schema_verified")

old_082_note = coding_log.loc["TUR_BIT_082", "notes"]
if old_082_note.startswith("No FPS clause (Type G);"):
    coding_log.loc["TUR_BIT_082", "notes"] = old_082_note.replace(
        "No FPS clause (Type G);", "FPS clause present (Type B);", 1)
    finding("DS2", "TUR_BIT_082", "coding_log.notes", old_082_note,
            coding_log.loc["TUR_BIT_082", "notes"],
            "The final FPS classification is Type B and contains a full-protection clause.",
            "cross_dataset_verified")


# ---------------------------------------------------------------------------
# DS1 article-boundary repair.  Seventy-three headings were truncated after
# “Settlement(s) of”; three additional headings were split at another point.
# The heading continuation is the text before the first paragraph marker.
split_cells = []
for treaty_id, row in ae.iterrows():
    for title_col in [c for c in ae.columns if c.endswith(" - Title")]:
        if re.search(r"(?i)settlements? of$", row[title_col].strip()):
            split_cells.append((treaty_id, title_col))
for candidate in [
    ("TUR_BIT_052", "Article 7 - Title"),
    ("TUR_BIT_072", "Article 9 - Title"),
    ("TUR_BIT_109", "Article 13 - Title"),
]:
    treaty_id, title_col = candidate
    body = ae.loc[treaty_id, title_col.replace(" - Title", " - Text")].lstrip()
    if not re.match(r"(?:\(?1\)|1[.\-:])\s", body):
        split_cells.append(candidate)
if len(split_cells) not in (0, 76):
    raise AssertionError(f"expected 76 title/body splits, found {len(split_cells)}")

for treaty_id, title_col in split_cells:
    text_col = title_col.replace(" - Title", " - Text")
    old_title = ae.loc[treaty_id, title_col].strip()
    old_body = ae.loc[treaty_id, text_col].strip()
    marker = re.search(r"\s(?=(?:\(?1\)|1[.\-:])\s)", old_body)
    if marker is None:
        raise AssertionError(f"no paragraph boundary for {treaty_id} {title_col}")
    continuation = old_body[:marker.start()].strip()
    new_body = old_body[marker.start():].strip()
    new_title = f"{old_title} {continuation}".strip()
    ae.loc[treaty_id, title_col] = new_title
    ae.loc[treaty_id, text_col] = new_body
    article = title_col.replace(" - Title", "")
    mask = (ce.treaty_id == treaty_id) & (ce.article == article)
    if not mask.any():
        raise AssertionError(f"no clause extract for {treaty_id} {article}")
    for index in ce.index[mask]:
        clause = ce.loc[index, "clause_text"]
        pattern = re.escape(old_title) + r"\s+" + re.escape(continuation)
        clause, count = re.subn(pattern, new_title, clause, count=1)
        if count != 1:
            raise AssertionError(f"clause heading not found: {treaty_id} {article}")
        ce.loc[index, "article_title"] = new_title
        ce.loc[index, "clause_text"] = clause
    finding("DS1/DS2", treaty_id, f"{article} title/body boundary",
            old_title, new_title,
            "Article heading continuation had been assigned to the body; "
            "the first numbered paragraph provides an unambiguous boundary.",
            "structure_verified")


# Twenty-four additional article bodies redundantly repeated their extracted
# title.  Two starts that resemble titles are genuine sentence text and are
# intentionally excluded (TUR_BIT_032 Art 8; TUR_BIT_127 Art 17).
duplicate_title_cells = [
    ("TUR_BIT_004", "Article 10 - Title"),
    ("TUR_BIT_008", "Article 14 - Title"),
    ("TUR_BIT_026", "Article 2 - Title"),
    ("TUR_BIT_047", "Article 9 - Title"),
    ("TUR_BIT_048", "Article 10 - Title"),
    ("TUR_BIT_066", "Article 11 - Title"),
    ("TUR_BIT_069", "Article 9 - Title"),
    ("TUR_BIT_072", "Article 11 - Title"),
    ("TUR_BIT_072", "Article 12 - Title"),
    ("TUR_BIT_072", "Article 17 - Title"),
    ("TUR_BIT_075", "Article 9 - Title"),
    ("TUR_BIT_082", "Article 4 - Title"),
    ("TUR_BIT_082", "Article 12 - Title"),
    ("TUR_BIT_086", "Article 14 - Title"),
    ("TUR_BIT_109", "Article 6 - Title"),
    ("TUR_BIT_115", "Article 1 - Title"),
    ("TUR_BIT_117", "Article 13 - Title"),
    ("TUR_BIT_120", "Article 13 - Title"),
    ("TUR_BIT_124", "Article 14 - Title"),
    ("TUR_BIT_126", "Article 14 - Title"),
    ("TUR_BIT_127", "Article 8 - Title"),
    ("TUR_BIT_129", "Article 7 - Title"),
    ("TUR_BIT_130", "Article 18 - Title"),
    ("TUR_BIT_132", "Article 14 - Title"),
]
duplicate_titles_applied = 0
for treaty_id, title_col in duplicate_title_cells:
    text_col = title_col.replace(" - Title", " - Text")
    title = ae.loc[treaty_id, title_col].strip()
    body = ae.loc[treaty_id, text_col].strip()
    if not body.lower().startswith(title.lower()):
        # Idempotent path: the duplicate has already been removed.
        continue
    new_body = body[len(title):].lstrip()
    if treaty_id == "TUR_BIT_086" and new_body.startswith(". it "):
        new_body = "It " + new_body[5:]
    ae.loc[treaty_id, text_col] = new_body
    article = title_col.replace(" - Title", "")
    mask = (ce.treaty_id == treaty_id) & (ce.article == article)
    for index in ce.index[mask]:
        clause = ce.loc[index, "clause_text"]
        # Remove only the immediately repeated second occurrence.
        pattern = f"({re.escape(title)}\\s+){re.escape(title)}"
        clause, count = re.subn(pattern, r"\1", clause, count=1,
                                flags=re.IGNORECASE)
        if count != 1:
            raise AssertionError(f"duplicate clause title not found: {treaty_id} {article}")
        if treaty_id == "TUR_BIT_086":
            clause = re.sub(re.escape(title) + r"\s*\.\s*it\s+",
                            f"{title}\nIt ", clause, count=1)
        ce.loc[index, "clause_text"] = clause
    finding("DS1/DS2", treaty_id, f"{article} repeated title", title, "removed from body",
            "The already extracted article title was duplicated verbatim at the start "
            "of the article body and clause extract.", "structure_verified")
    duplicate_titles_applied += 1


# ---------------------------------------------------------------------------
# DS2: investment definition and investor-definition corrections.
set_annotation("TUR_BIT_123", "1.01_preamble_right_to_regulate", "Yes",
               "The preamble expressly recognises the Parties' right to regulate investments for policy objectives.")
bulk("1.04_preamble_environmental", tids("6 72"), "No",
     "The preamble contains no environmental, biodiversity or climate reference; "
     "economic/social development language alone does not meet the definition.")
set_annotation("TUR_BIT_087", "1.03_preamble_social", "Yes",
               "The preamble expressly states the social objective of improving living standards.")
bulk("2.01_investment_def_type", tids("4 25 34 43 53 72 101 118 120"),
     "Asset-based definition",
     "Article 1 expressly defines investment as assets/property and supplies an "
     "illustrative asset list; 'No definition' contradicted the text.")
set_annotation("TUR_BIT_096", "2.09_includes_concessions", "No",
               "The memorandum contains no investment definition or concession list.")
set_annotation("TUR_BIT_096", "2.14_legal_entities", "No",
               "The memorandum contains no covered-investor/legal-entity definition.")
bulk("2.11_natural_persons", tids("1 11"), "Yes",
     "The investor/national definition expressly covers persons or individuals.")
bulk("2.02_excludes_portfolio", tids(
    "74 83 84 85 86 87 88 89 90 92 93 94 95 97 98 99 100 101 102 104 105 "
    "106 108 109 112 117 120 121 122 125 135 140"), "Yes",
     "The investment definition expressly excludes portfolio/stock-exchange holdings "
     "or equity below the stated ownership threshold.")
bulk("2.03_excludes_specific_assets", tids("101 105 107 109 116 117 118 128 130 131 140"), "Yes",
     "The definition expressly excludes commercial claims, public debt, trade finance, "
     "court awards or another specified asset class.")
set_annotation("TUR_BIT_141", "2.04_lists_characteristics", "Yes",
               "The definition lists contribution, gain, risk, development and duration characteristics.")
set_annotation("TUR_BIT_139", "2.04_lists_characteristics", "No",
               "The definition supplies only an illustrative asset list, not required investment characteristics.")
bulk("2.04_lists_characteristics", tids(
    "74 82 84 85 86 87 88 89 91 92 93 94 95 97 98 99 100 102 103 104 106 "
    "108 112 120 121 122 125 126 135 141"), "Yes",
     "The definition expressly requires lasting economic relations, a minimum "
     "holding/duration, contribution, profit, risk or another investment characteristic.")
bulk("2.05_host_state_law", tids("1 3 4 6 7 8 12 14 16 26"), "No",
     "The investment definition contains no host-law legality requirement; admission "
     "duties, incorporation law and licence/concession law do not satisfy this variable.")
set_annotation("TUR_BIT_006", "2.08_includes_debt", "Yes",
               "The definition expressly includes money claims of economic value associated with an investment.")
bulk("2.09_includes_concessions", tids("3 5 15"), "Yes",
     "The asset list expressly includes rights conferred by law/contract, licences or permits.")
bulk("2.12_permanent_residents", tids("101 127"), "No",
     "The investor definition covers nationality and companies only; it does not cover permanent residents/right of abode.")
set_annotation("TUR_BIT_119", "2.13_excludes_dual", "No",
               "The text applies a dominant-and-effective nationality test, which the variable definition does not code as exclusion.")
bulk("2.15_substantial_business", tids("55 59 76 99 110 119 132"), "Yes",
     "The investor definition requires effective economic, business or commercial activities.")
bulk("2.16_ownership_control", tids(
    "1 92 94 95 97 98 99 104 105 107 108 109 110 111 112 113 114 115 116 "
    "119 121 122 123 124 126 132 136 137 138 139 140 141"), "No",
     "The investor/entity definition does not define ownership or control; incidental "
     "property, territorial-control or ordinary-language uses do not satisfy the variable.")
bulk("2.16_ownership_control", tids("53 76 135"), "Yes",
     "The investor definition expressly defines substantial interest/control or a direct/indirect "
     "ownership/control threshold for an enterprise.")
set_annotation("TUR_BIT_088", "2.19_dob", "Yes",
               "The treaty denies benefits to an investor of a third State lacking diplomatic relations.")
set_annotation("TUR_BIT_088", "2.20_dob_substantial", "No",
               "The denial-of-benefits clause has no substantial-business criterion.")
set_annotation("TUR_BIT_088", "2.21_dob_diplomatic", "Yes",
               "The denial-of-benefits clause expressly uses absence of diplomatic relations.")
set_annotation("TUR_BIT_088", "2.22_dob_discretionary", "Mandatory",
               "The clause states that the investor will not benefit, without discretionary wording.")
bulk("2.20_dob_substantial", tids("99 110 118 119 124 126 127 129 130 132 138 139 140"),
     "Yes",
     "The denial-of-benefits clause expressly requires effective, substantive, substantial "
     "or predominant business activities.")
bulk("2.21_dob_diplomatic", tids("118 128"), "Yes",
     "The denial-of-benefits clause covers measures prohibiting transactions with the third State.")
bulk("2.23_excludes_taxation", tids("107 122 123 127"), "Yes",
     "The scope clause expressly excludes taxation or most taxation measures.")
set_annotation("TUR_BIT_026", "2.24_excludes_subsidies", "No",
               "The text says subsidiaries, not subsidies; no subsidy/grant exclusion exists.")
bulk("2.24_excludes_subsidies", tids("123 127 131"), "Yes",
     "The scope clause expressly excludes subsidies or grants.")
bulk("2.25_excludes_procurement", tids("123 127 131"), "Yes",
     "The scope clause expressly excludes government procurement.")
bulk("2.27_temporal_investments", tids("6 10 11 14 20 25 30 38 45 49 66 70 76 93"),
     "Applies to both pre-existing and post-BIT investments",
     "The temporal clause expressly covers investments made before and after entry into force.")
bulk("2.27_temporal_investments", tids("51 72 124"),
     "Applies to post-BIT investments only",
     "The temporal clause limits coverage to investments made after entry into force.")
bulk("2.28_temporal_disputes", tids(
    "45 55 56 59 62 67 68 70 71 74 81 83 101 108 115 118 120 124 131 134 135"),
     "Carves out pre-existing disputes",
     "The treaty expressly excludes disputes or claims arising before entry into force.")


# FET was absent as an operative treatment obligation in these 35 treaties.
# It appears only in aspirational preambular language (and, for TUR_BIT_056,
# as the distinct phrase “fair and equitable compensation”).
fet_preamble_only = tids(
    "5 10 15 18 19 20 21 22 24 25 27 28 29 31 33 35 36 37 39 40 41 42 "
    "44 47 50 51 53 54 56 58 60 61 65 67 73"
)
bulk("3.07_fet_type", fet_preamble_only, "None",
     "'Fair and equitable' occurs only in the preamble; no operative article "
     "creates an FET obligation.")
for column in ("3.08_fet_intl_law", "3.09_fet_list", "3.10_fet_combined_nt"):
    bulk(column, fet_preamble_only, "Not applicable",
         "No operative FET clause exists, so the conditional FET modifier is not applicable.")

set_annotation("TUR_BIT_123", "3.07_fet_type", "FET qualified",
               "Article 5 says a breach may be found only for the exhaustive listed conduct.")
set_annotation("TUR_BIT_123", "3.09_fet_list", "Exhaustive list",
               "Article 5 uses the closed formulation 'may be found only where'.")
set_annotation("TUR_BIT_107", "3.08_fet_intl_law", "CIL/minimum standard of treatment",
               "Article 3 expressly establishes the minimum standard of customary international law.")
bulk("3.07_fet_type", tids("8 23 32 55 57 74"), "FET unqualified",
     "The operative FET sentence contains no international-law, CIL, minimum-standard "
     "or other qualifier; nearby language governs a distinct obligation.")
bulk("3.08_fet_intl_law", tids("8 23 32 55 57 74 116"), "None",
     "No international-law qualifier applies to FET (TUR_BIT_116 is instead qualified "
     "by domestic law).")
for treaty_id, new in {
    "TUR_BIT_105": "Indicative list", "TUR_BIT_109": "None",
    "TUR_BIT_127": "Indicative list", "TUR_BIT_128": "Exhaustive list",
    "TUR_BIT_130": "Indicative list", "TUR_BIT_131": "Indicative list",
    "TUR_BIT_132": "Indicative list", "TUR_BIT_141": "Exhaustive list",
}.items():
    set_annotation(treaty_id, "3.09_fet_list", new,
                   "The operative FET wording was reclassified by its textual marker: "
                   "'includes' is indicative; 'means'/'only where' is exhaustive; an "
                   "external-standard reference without elements is None.")
bulk("3.10_fet_combined_nt", tids("1 2 4 6 9 14 23 32 49 55 57 59 76 123 135"),
     "Yes",
     "The same provision links FET to treatment no less favourable than domestic "
     "or third-country investors; UNCTAD codes linkage in any form as combined.")


# Additional standards and other-clause corrections with direct operative text.
bulk("3.01_nt_type", tids("6 25"), "Post-establishment",
     "The operative treatment clause grants treatment no less favourable than own "
     "or third-country investors after establishment.")
set_annotation("TUR_BIT_004", "3.02_nt_like_circumstances", "No",
               "Article 3 contains no national-treatment clause; similar-situations "
               "wording belongs to the non-derogation article.")
set_annotation("TUR_BIT_110", "3.22_strife_comparator", "MFN and NT",
               "Article 7 compares loss treatment with both own and third-country investors.")
set_annotation("TUR_BIT_110", "3.23_strife_absolute", "Yes",
               "Article 7(2) grants restitution or compensation for requisition/destruction.")
set_annotation("TUR_BIT_007", "3.29_umbrella", "Yes",
               "Article 8 requires observance of any contractual investment obligation.")
set_annotation("TUR_BIT_007", "3.30_umbrella_scope", "broad_any_obligation",
               "Article 8 uses the unrestricted phrase 'any contractual obligation'.")
set_annotation("TUR_BIT_088", "4.02_transparency_investors", "Yes",
               "Article 4 authorises requests for investor corporate-governance and practice information.")
bulk("4.10_non_derogation", tids("8 17 25 30 32 43 46 141"), "Yes",
     "The treaty preserves more favourable treatment available under domestic law "
     "or other international obligations.")
bulk("4.11_promotion", tids("91 96 105 110 111 130"), "Yes",
     "The operative text specifies promotion activities such as information exchange, "
     "missions, training, fairs or technical cooperation.")
set_annotation("TUR_BIT_106", "5.07_reservations", "Negative-list reservations",
               "The treaty expressly reserves existing non-conforming measures, their continuation, "
               "and amendments that do not increase non-conformity.")
set_annotation("TUR_BIT_101", "7.21_transparency_docs", "Yes",
               "Article 22 requires communications, submissions, orders, decisions and awards to be public.")
bulk("7.20_non_disputing", tids("123"), "Yes",
     "Article 12 expressly incorporates the UNCITRAL Rules on Transparency, whose "
     "non-disputing State-party submission rule therefore governs the proceeding.")
bulk("7.23_amicus", tids("123"), "Yes",
     "Article 12 expressly incorporates the UNCITRAL Rules on Transparency, whose "
     "third-person submission rule therefore governs the proceeding.")
bulk("7.14_limitation_period", tids("109 112"), "Yes",
     "The ISDS article contains an express six-year or three-year claim-submission cutoff.")
bulk("8.01_consultations", tids("3 17 48 72"), "Yes",
     "A standalone inter-State treaty consultation mechanism is expressly provided.")
bulk("8.01_consultations", tids("1 135"), "No",
     "The cited consultation language belongs only to SSDS/ISDS, which this variable excludes.")
set_annotation("TUR_BIT_105", "8.03_technical", "Yes",
               "Article 15 expressly provides training and technical cooperation for capacity building.")


# ISDS scope.
bulk("7.03_scope", tids("2 3 4 5 8 12 13 25 52 80"),
     "Lists specific bases of claim beyond treaty",
     "The ISDS article enumerates investment authorisation/agreement and treaty bases of claim.")
bulk("7.03_scope", tids("7 9 14"), "Lists specific bases of claim beyond treaty",
     "The ISDS article includes investment-authorisation/agreement claims beyond treaty breaches.")
bulk("7.03_scope", tids(
    "101 102 105 106 107 108 109 110 111 112 113 114 115 119 123 124 126 127 "
    "128 129 130 131 132 133 136 137 138 139 140 141"),
     "Covers treaty claims only",
     "The ISDS scope is expressly limited to an alleged breach of an obligation under the Agreement.")
bulk("7.03_scope", tids("26 36 37 45 120"),
     "Covers any dispute relating to investment",
     "The clause covers any legal dispute concerning/in connection with an investment "
     "without a treaty-breach limitation.")


# Other arbitral forum and free-text detail.  Appointment-authority references
# that are not submission options were deliberately excluded.
other_forum = {
    18: "ICC", 19: "ICC", 21: "ICC", 22: "ICC", 23: "ICSID Additional Facility",
    25: "ICC", 27: "ICC", 28: "ICC", 29: "ICC", 31: "ICC", 32: "ICC",
    33: "ICC", 35: "ICC", 36: "ICC", 37: "ICC",
    40: "ICC; Regional Cairo Center for International Commercial Arbitration; Istanbul Center for Commercial Arbitration",
    41: "ICC", 42: "ICC", 44: "ICC", 47: "ICC", 50: "ICC", 51: "ICC",
    53: "Arbitration Institute of the Stockholm Chamber of Commerce (SCC)",
    56: "ICSID Additional Facility", 58: "ICC", 59: "ICC", 60: "ICC", 61: "ICC",
    64: "ICC", 66: "ICC", 67: "ICC", 68: "ICC",
    69: "Istanbul Chamber of Commerce arbitration institution; Afghan Chamber of Commerce Arbitration Commission",
    78: "ICC", 83: "ICC", 93: "ICSID Additional Facility",
    94: "Istanbul Chamber of Commerce arbitration/conciliation institution; OHADA",
    95: "ICC", 99: "ICC; ICSID Additional Facility",
    101: "ICSID Additional Facility; any other rules agreed by the disputing parties",
    102: "ICSID Additional Facility",
    107: "ICSID Additional Facility; any other arbitration rules agreed by the disputing parties",
    109: "Ghana Arbitration Centre; any other national or international arbitration institution agreed by the disputing parties",
    110: "Any other arbitration institution or rules agreed by the disputing parties",
    114: "Any other arbitration institution or rules agreed by the disputing parties",
    115: "ISTAC or another relevant Ukrainian institution; other agreed arbitration rules",
    116: "Any other arbitration institution or rules agreed by the disputing parties",
    119: "Any other arbitration institution or rules agreed by the disputing parties",
    120: "ICC Court of Arbitration", 121: "ICSID Additional Facility; ICC",
    123: "Any other arbitration institution or rules agreed by the disputing parties",
    124: "ICSID Additional Facility; ISTAC; other agreed arbitration rules",
    127: "Any other arbitration institution or rules agreed by the disputing parties",
    129: "ICSID Additional Facility; ISTAC; Abu Dhabi Commercial Conciliation and Arbitration Centre; DIAC; Sharjah International Commercial Arbitration Centre; other agreed arbitration rules",
    130: "Any other arbitration institution or rules agreed by the disputing parties",
    131: "Any other arbitration institution or rules agreed by the disputing parties",
    132: "Any other arbitration institution or rules agreed by the disputing parties",
    136: "Any other arbitration institution or rules agreed by the disputing parties",
    137: "Any other arbitration institution or rules agreed by the disputing parties",
    138: "Istanbul Arbitration Centre; other agreed arbitration rules",
    139: "Istanbul Arbitration Centre (printed as STAC); Mauritania International Center of Mediation and Arbitration (CIMAM); other agreed arbitration rules",
    140: "Istanbul Arbitration Center; Ouagadougou Arbitration, Mediation and Conciliation Center (CAMCO); other agreed arbitration rules",
}
for number in tids("18 19 21 22 25 27 28 29 31 32 33 35 36 37 41 42 44 47 50 51 58 60 61 64 66 67 68 83 95 120"):
    other_forum[int(number[-3:])] = "Paris International Chamber of Commerce Court of Arbitration (ICC)"
other_forum[40] = ("Paris International Chamber of Commerce Court of Arbitration (ICC); "
                   "Regional Cairo Center for International Commercial Arbitration; "
                   "Istanbul Center for Commercial Arbitration")
other_forum[59] = ("Paris International Chamber of Commerce Court of Arbitration (ICC); "
                   "previously agreed dispute-settlement procedure")
other_forum[78] = ("Paris International Chamber of Commerce Court of Arbitration (ICC); "
                   "other form agreed by the disputing parties")
other_forum[99] = ("Paris International Chamber of Commerce Court of Arbitration (ICC); "
                   "ICSID Additional Facility")
other_forum[121] = ("Paris International Chamber of Commerce Court of Arbitration (ICC); "
                    "ICSID Additional Facility")
for number, detail in other_forum.items():
    treaty_id = f"TUR_BIT_{number:03d}"
    set_annotation(treaty_id, "7.11_other_forum", "Yes",
                   "The ISDS forum list expressly offers an additional institution/rules beyond ICSID and UNCITRAL.")
    set_annotation(treaty_id, "7.12_other_detail", detail,
                   "Free-text detail now records the additional forum/rules named in the ISDS provision.")


# Relationship between domestic courts and arbitration.
bulk("7.13_forum_relationship", tids(
    "3 6 7 9 14 17 18 19 20 21 22 24 25 26 27 28 29 31 32 35 36 37 38 "
    "39 40 41 42 44 46 47 48 50 51 53 58 63 64 68"),
     "Preserving right to arbitration after domestic court proceedings",
     "Arbitration remains available where domestic proceedings have produced no final "
     "award/judgment within the stated period (or no final award has yet been rendered).")
bulk("7.13_forum_relationship", tids("2 4 8 10 15 16 65 110 118 135"),
     "Fork in the road",
     "Election of the domestic-court route excludes later arbitration, or the forum choice is final.")
set_annotation("TUR_BIT_070", "7.13_forum_relationship", "No U turn (waiver clause)",
               "Article 13 requires waiver of the right to initiate or continue domestic proceedings.")
bulk("7.13_forum_relationship", tids("107 127 130"), "No U turn (waiver clause)",
     "Submission to arbitration requires written waiver/abandonment of court or administrative proceedings.")
bulk("7.13_forum_relationship", tids("105 106 116"), "Local remedies first",
     "The treaty requires domestic administrative review/remedies before arbitration.")


# Public hearings require an express public/open-hearing obligation.  Generic
# uses of “hearing” (fair hearing, scheduling or tribunal procedure) do not meet
# the UNCTAD definition.
false_public_hearings = tids(
    "2 5 10 17 18 19 21 22 24 25 27 28 29 30 31 32 33 34 35 36 37 39 40 41 "
    "42 44 47 50 51 53 54 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 "
    "73 74 75 78 81 82 83 85 86 87 88 89 90 91 92 93 94 95 97 98 99 100 "
    "101 102 103 104 105 106 107 108 109 111 112 113 114 115 116 117 118 "
    "119 120 121 122 124 125 126 129 132 134 136 137 138 139 140"
)
if len(false_public_hearings) != 101:
    raise AssertionError("public-hearing correction set must contain 101 treaties")
bulk("7.22_transparency_hearings", false_public_hearings, "No",
     "No public/open-hearing requirement appears in the authentic treaty text; "
     "generic hearing language does not satisfy the variable definition.")


# Duration, renewal, termination, notice, amendment and survival.
for number, value in {
    17: "15 years", 34: "10 years", 72: "Indefinite", 78: "10 years",
    84: "15 years", 100: "15 years", 123: "15 years", 131: "15 years",
}.items():
    set_annotation(f"TUR_BIT_{number:03d}", "9.01_duration", value,
                   "The duration article's initial term was separated from renewal and survival periods.")
for number, value in {
    3: "10 years", 6: "2 years", 8: "10 years", 17: "10 years",
    34: "5 years", 66: "10 years", 72: "None", 78: "10 years",
    96: "Other", 111: "2 years", 132: "5 years", 134: "15 years",
}.items():
    set_annotation(f"TUR_BIT_{number:03d}", "9.02_renewal", value,
                   "The renewal article states this renewal period; an unlimited initial term is coded None.")
set_annotation("TUR_BIT_008", "9.03_unilateral_term", "Yes",
               "Article 15 provides unilateral termination by prior notice.")
bulk("9.04_notice", tids("11 12 24 34 72 80 118"), "One year prior notice",
     "The duration article expressly requires one year's notice.")
bulk("9.04_notice", tids(
    "9 14 16 38 52 63 66 76 78 79 84 89 101 102 107 116 123 126 130 131 134"),
     "One year prior notice",
     "The duration article expressly requires one year's notice; 'Other period' was incorrect.")
set_annotation("TUR_BIT_045", "9.04_notice", "Six months prior notice",
               "The duration article requires written notification at least six months before expiry.")
bulk("9.05_amendment", tids("72 80 96 111"), "Yes",
     "The treaty expressly provides amendment/modification by mutual consent.")
for number, value in {
    1: "15 years", 3: "15 years", 8: "15 years", 11: "15 years",
    17: "10 years", 33: "20 years", 70: "15 years", 96: "Other",
    118: "5 years", 128: "5 years", 130: "5 years", 132: "5 years",
}.items():
    set_annotation(f"TUR_BIT_{number:03d}", "9.06_survival", value,
                   "The post-termination protection period was read directly from the duration article; "
                   "Other denotes a non-numeric survival commitment.")


# ---------------------------------------------------------------------------
# Clause-extract taxonomy corrections.
clause_type_fixes = {
    ("TUR_BIT_002", "Article 6"): "isds",
    ("TUR_BIT_005", "Article 6"): "isds",
    ("TUR_BIT_007", "Article 8"): "umbrella",
    ("TUR_BIT_008", "Article 7"): "other",
    ("TUR_BIT_008", "Article 9"): "ssds",
    ("TUR_BIT_043", "Article 4"): "national_mfn_treatment",
    ("TUR_BIT_072", "Article 15"): "entry_into_force",
    ("TUR_BIT_096", "Article 4"): "other",
    ("TUR_BIT_101", "Article 25"): "ssds",
    ("TUR_BIT_101", "Article 26"): "ssds",
    ("TUR_BIT_101", "Article 27"): "ssds",
    ("TUR_BIT_101", "Article 28"): "ssds",
    ("TUR_BIT_101", "Article 29"): "ssds",
    ("TUR_BIT_120", "Article 12"): "ssds",
}
for (treaty_id, article), new in clause_type_fixes.items():
    mask = (ce.treaty_id == treaty_id) & (ce.article == article)
    if mask.sum() != 1:
        raise AssertionError(f"expected one clause row for {treaty_id} {article}, got {mask.sum()}")
    index = ce.index[mask][0]
    old = ce.loc[index, "clause_type"]
    if old != new:
        ce.loc[index, "clause_type"] = new
        finding("DS2", treaty_id, f"clause_extracts {article} clause_type", old, new,
                "The article's operative subject was re-read against the controlled clause taxonomy.")


# ---------------------------------------------------------------------------
# Cleaned-text OCR corrections are populated below after source-image review.
text_replacements: dict[str, list[tuple[str, str, str, str]]] = {
    # treaty_id: [(old, new, evidence, source_status)]
    "TUR_BIT_008": [
        ("latter.Contracting", "latter Contracting",
         "Source PDF shows a word boundary, not a full stop.", "source_pdf_verified"),
    ],
    "TUR_BIT_036": [
        ("exploit >natural resources", "exploit natural resources",
         "The stray greater-than glyph is absent from the source PDF.", "source_pdf_verified"),
    ],
    "TUR_BIT_075": [
        ("Investors . . of either", "Investors of either",
         "The spaced dot artefacts are absent from the source PDF.", "source_pdf_verified"),
        ("law^s", "laws",
         "The caret is an OCR artefact; the source PDF prints 'laws'.", "source_pdf_verified"),
    ],
    "TUR_BIT_088": [
        ("establishment, >", "establishment,",
         "The greater-than glyph before subparagraph (ii) is absent from the source PDF.",
         "source_pdf_verified"),
    ],
    "TUR_BIT_104": [
        ("{ree trade area", "free trade area",
         "The source PDF prints 'free trade area'.", "source_pdf_verified"),
        ("Paragraphs (1}", "Paragraphs (1)",
         "The source PDF has a closing parenthesis, not a brace.", "source_pdf_verified"),
    ],
    "TUR_BIT_121": [
        ("protection ofi ts essential", "protection of its essential",
         "The source PDF has the normal word boundary 'of its'.", "source_pdf_verified"),
    ],
    "TUR_BIT_126": [
        ("territory (ho investment was made: or", "territory the investment was made; or",
         "The source PDF prints 'the investment was made; or'.", "source_pdf_verified"),
        ("(b) except as provided in paragraph (5) to; the International Center",
         "(b) except as provided in paragraph (5) to:\n(i) the International Center",
         "The source PDF contains a colon followed by the omitted subparagraph marker '(i)'.",
         "source_pdf_verified"),
    ],
}

# These unusual tokens were checked against the page image and are printed in
# the source itself.  They are findings, not OCR corrections: source fidelity
# takes precedence over silent editorial emendation.
source_printed_anomalies = {
    2: ["bee"], 3: ["rules o"], 13: ["such us"], 24: ["Satates", "Natons"],
    31: ["ail"], 35: ["we 11"], 36: ["ah hoc", "tha", "wno"], 47: ["data"],
    66: ["Agrement"], 67: ["Divestment"], 71: ["tea"], 74: ["die"],
    77: ["taw", "force.."], 78: ["Parlies", "Party,,"], 79: ["cf", "tor"],
    84: ["25 lh"], 88: ["notification hy"], 94: ["THR"],
    98: ["ten (16) years"], 100: ["ten (16) years"], 102: ["fo"],
    104: ["und", "(o", "Lo"], 107: ["source omits 'do not'"],
    108: ["Onc", "te"], 109: ["ot"], 111: ["m", "fo", "Article 16"],
    114: ["jf"], 115: ["pipe glyph"], 119: ["mternational", "pipe glyph"],
    122: ["peer", "ands", "aA eel", "{e)"], 124: ["my"],
    134: ["Thrkey", "Rebuplic"], 136: ["TheOther"],
    137: ["(b)>", "eight (8} months"], 138: ["OtherContracting"],
    139: ["OtherContracting"], 140: ["(a}", "Contrasting Party"],
}
for number, tokens in source_printed_anomalies.items():
    treaty_id = f"TUR_BIT_{number:03d}"
    for token in tokens:
        finding("DS1", treaty_id, "source-printed anomaly", token, token,
                "The token is printed the same way in the source PDF and was retained "
                "to preserve source fidelity.", "source_pdf_verified", "retained")


# Textual candidates whose classification depends on a legal/operational choice
# are disclosed but not silently changed.
review_candidates = [
    ("TUR_BIT_006", "1.01_preamble_right_to_regulate", "Yes",
     "The preamble recognises a right to determine foreign investment's role, but does not use the canonical right-to-regulate formulation."),
    ("TUR_BIT_118", "2.15_substantial_business", "Yes",
     "Principal place of business may or may not be treated as a substantial/real activity requirement."),
    ("TUR_BIT_140", "2.15_substantial_business", "Yes",
     "Principal place of business may or may not be treated as a substantial/real activity requirement."),
    ("TUR_BIT_015", "2.16_ownership_control", "Yes",
     "Control appears only among associated investment activities, not as an investor/entity definition."),
    ("TUR_BIT_031", "2.16_ownership_control", "Yes",
     "Control appears only among associated investment activities, not as an investor/entity definition."),
    ("TUR_BIT_073", "2.16_ownership_control", "Yes",
     "The text refers to a government-owned entity without defining ownership/control."),
    ("TUR_BIT_105", "2.26_excludes_other", "Yes",
     "Prudential-finance/criminal-capital language could be treated as other subject-matter exclusions."),
    ("TUR_BIT_122", "2.26_excludes_other", "Yes",
     "Policy treatment for disadvantaged groups could be treated as another scope exclusion."),
    ("TUR_BIT_123", "2.26_excludes_other", "Yes",
     "Public health-insurance and pension schemes are expressly carved out, but the breadth of 2.26 is not operationalised."),
    ("TUR_BIT_127", "2.26_excludes_other", "Yes",
     "Central-bank/monetary/exchange-rate activities are carved out, but the breadth of 2.26 is not operationalised."),
    ("TUR_BIT_129", "2.26_excludes_other", "Yes",
     "The pre-establishment phase is excluded, but may belong to temporal rather than other-subject scope."),
    ("TUR_BIT_074", "3.10_fet_combined_nt", "Yes",
     "Admission-MFN and FET appear in the same article but as separate obligations."),
    ("TUR_BIT_016", "4.10_non_derogation", "Yes",
     "The more-favourable rule is specific to the Paris Convention rather than a general non-derogation clause."),
    ("TUR_BIT_072", "7.13_forum_relationship", "Fork in the road",
     "The treaty bars arbitration while a domestic case is pending, but its finality language is atypical."),
    ("TUR_BIT_077", "7.13_forum_relationship", "Fork in the road",
     "The text only states that proceedings cannot run at the same time; it does not clearly make election final."),
    ("TUR_BIT_078", "7.13_forum_relationship", "Fork in the road",
     "The text only states that proceedings cannot run at the same time; it does not clearly make election final."),
    ("TUR_BIT_089", "9.02_renewal", "Other",
     "Tacit renewal is explicit but the renewal period is not stated."),
]
for treaty_id, column, suggested, evidence in review_candidates:
    finding("DS2", treaty_id, column, ta.loc[treaty_id, column], suggested,
            evidence, "treaty_text_verified_adjudication_required", "review_candidate")
finding("DS2", "corpus-wide", "1.05_preamble_objective", "mixed labels", "adjudicate rule",
        "Materially identical economic-cooperation preamble templates carry different custom objective labels; "
        "the codebook lacks a sufficiently operational decision rule.",
        "methodology_gap", "review_candidate")


def replace_in_frame_row(frame: pd.DataFrame, treaty_id: str,
                         old: str, new: str) -> int:
    count = 0
    for column in frame.columns:
        value = frame.loc[treaty_id, column]
        if isinstance(value, str) and old in value:
            occurrences = value.count(old)
            frame.loc[treaty_id, column] = value.replace(old, new)
            count += occurrences
    return count


zip_path = ROOT / "ds1/treaty_texts.zip"
zip_payload: dict[str, bytes] = {}
zip_info: dict[str, zipfile.ZipInfo] = {}
with zipfile.ZipFile(zip_path) as archive:
    for info in archive.infolist():
        zip_info[info.filename] = info
        zip_payload[info.filename] = archive.read(info.filename)

ocr_replacements_applied = 0
for treaty_id, replacements in text_replacements.items():
    member = tm.loc[treaty_id, "source_pdf_filename"]
    member_path = next((name for name in zip_payload if name.rsplit("/", 1)[-1] == member), None)
    if member_path is None:
        raise AssertionError(f"ZIP member not found for {treaty_id}: {member}")
    text = zip_payload[member_path].decode("utf-8")
    for old, new, evidence, source_status in replacements:
        if old not in text and new in text:
            continue
        if old not in text:
            raise AssertionError(f"OCR source string absent in {treaty_id}: {old!r}")
        text = text.replace(old, new)
        replace_in_frame_row(ae, treaty_id, old, new)
        replace_in_frame_row(fc, treaty_id, old, new)
        mask = ce.treaty_id == treaty_id
        ce.loc[mask, "clause_text"] = ce.loc[mask, "clause_text"].str.replace(old, new, regex=False)
        finding("DS1/DS2", treaty_id, "cleaned treaty text / extracts", old, new,
                evidence, source_status)
        ocr_replacements_applied += 1
    zip_payload[member_path] = text.encode("utf-8")

# Metadata word counts are derived from the canonical cleaned text ZIP.
for treaty_id in tm.index:
    member = tm.loc[treaty_id, "source_pdf_filename"]
    member_path = next(name for name in zip_payload if name.rsplit("/", 1)[-1] == member)
    count = len(zip_payload[member_path].decode("utf-8").split())
    old = tm.loc[treaty_id, "total_word_count"]
    if old != str(count):
        tm.loc[treaty_id, "total_word_count"] = str(count)
        finding("DS1", treaty_id, "total_word_count", old, str(count),
                "Recomputed as the documented whitespace-token count after OCR repair.",
                "derived")


# Avoid duplicate log entries if the script is rerun on a partially corrected
# package.  Rows are identified by the five substantive log fields.
if annotation_changes:
    additions = pd.DataFrame(annotation_changes, columns=change_log.columns)
    keys = ["treaty_id", "variable", "old_value", "new_value", "reason"]
    existing = set(map(tuple, change_log[keys].itertuples(index=False, name=None)))
    additions = additions[[tuple(row) not in existing
                           for row in additions[keys].itertuples(index=False, name=None)]]
    change_log = pd.concat([change_log, additions], ignore_index=True)


def write_csv(frame: pd.DataFrame, relative: str, index: bool = False) -> None:
    frame.to_csv(ROOT / relative, index=index, lineterminator="\n")


write_csv(tm.reset_index(), "ds1/treaty_metadata.csv")
write_csv(ae.reset_index(), "ds1/article_extracts.csv")
write_csv(fc.reset_index(), "ds1/fps_classification.csv")
write_csv(ta.reset_index(), "ds2/treaty_annotations.csv")
write_csv(ce, "ds2/clause_extracts.csv")
write_csv(coding_log.reset_index(), "ds2/coding_log.csv")
write_csv(change_log, "ds2/ds2_change_log.csv")

tm_parquet = tm.reset_index().copy()
for column in ("total_articles", "total_word_count"):
    tm_parquet[column] = pd.to_numeric(tm_parquet[column], errors="coerce").astype("Int64")
tm_parquet.to_parquet(ROOT / "ds1/treaty_metadata.parquet", index=False)
ta.reset_index().to_parquet(ROOT / "ds2/treaty_annotations.parquet", index=False)

if text_replacements:
    fd, temporary_name = tempfile.mkstemp(suffix=".zip", dir=zip_path.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary_name, "w") as archive:
            for name, payload in zip_payload.items():
                archive.writestr(zip_info[name], payload)
        os.replace(temporary_name, zip_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)

findings_frame = pd.DataFrame(findings)
audit_path = ROOT / "ds1" / "AUDIT_FINDINGS_2026-08-22.csv"
# A second run over an already-corrected package must not erase the evidence
# recorded by the first run.  Merge the prior ledger with findings that are
# still observable (retained source anomalies and review candidates).
if audit_path.exists():
    prior_findings = pd.read_csv(audit_path, dtype=str, keep_default_na=False)
    findings_frame = pd.concat([prior_findings, findings_frame], ignore_index=True)
    findings_frame = findings_frame.drop_duplicates(
        subset=list(findings_frame.columns), keep="first"
    ).reset_index(drop=True)
write_csv(findings_frame, "ds1/AUDIT_FINDINGS_2026-08-22.csv")
print(json.dumps({
    "annotation_cells_corrected": len(annotation_changes),
    "change_log_rows": len(change_log),
    "audit_findings": len(findings_frame),
    "title_body_splits_repaired": len(split_cells),
    "duplicate_titles_removed": duplicate_titles_applied,
    "ocr_replacements": ocr_replacements_applied,
}, ensure_ascii=False, indent=2))
