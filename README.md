# Turkish BIT Corpus: reproduction toolkit

[![DOI](https://img.shields.io/badge/DOI-10.18710%2FJX4WHH-blue)](https://doi.org/10.18710/JX4WHH)

Code that reproduces and verifies the three open datasets describing the complete
corpus of the Republic of Türkiye's international investment agreements, 1962 to 2025.

The datasets themselves live on DataverseNO under CC BY 4.0 and are not copied here.
This repository downloads them, recomputes what is derivable, and checks the result.

| Dataset | Content | DOI |
|---|---|---|
| 1. Treaty Corpus | 141 cleaned full texts, treaty metadata, article extracts, FPS classification | [10.18710/JX4WHH](https://doi.org/10.18710/JX4WHH) |
| 2. Clause-Level Annotation | 141 treaties coded against 118 substantive variables, 1,890 clause extracts | [10.18710/FPNLKS](https://doi.org/10.18710/FPNLKS) |
| 3. Diffusion and Genealogy Network | 9,870 pairwise comparisons, 38 treaty families, 88 descent edges | [10.18710/WA7HEO](https://doi.org/10.18710/WA7HEO) |

An interactive version of the diffusion network is published at
**https://ouzpehlivan.github.io/turkish-bit-corpus/**

## Quick start

```bash
git clone https://github.com/ouzpehlivan/turkish-bit-corpus.git
cd turkish-bit-corpus
pip install -r requirements.txt

python src/download_data.py    # fetch the three datasets from DataverseNO into data/
python src/verify.py           # 24-check cross-dataset verification suite
python src/recompute_ds3.py    # rebuild the network from Datasets 1 and 2
python src/fps_rule_check.py   # re-derive the FPS typology from the clause text
python src/make_figures.py     # regenerate the five figures and the dashboard
```

`verify.py` exits non-zero if any check fails, so it can be wired into CI.

## What each script does

**`download_data.py`** retrieves the three datasets through the Dataverse access API
and extracts them into `data/ds1`, `data/ds2`, `data/ds3`. If a download fails, the
script prints the DOI to fetch manually.

**`verify.py`** runs the checks the datasets are released against: row and column
counts, the FPS block identity across all three datasets, provenance of every clause
extract against the article extracts, coding-log synchronisation with the metadata,
article counts, matrix symmetry, the pair table against the matrix, family labels
re-derived from their member rows, and the genealogy edges against the argmax rule
that generated them.

**`recompute_ds3.py`** rebuilds Dataset 3 from Datasets 1 and 2 with the documented
parameters (TF-IDF with English stop-words, sublinear term frequency, `max_df=0.9`;
average-linkage clustering on cosine distance cut at 0.65; edges at cosine 0.45 or
above), writes the four tables to `outputs/ds3/`, and reports whether the
reproduction matches the published dataset. It does.

**`fps_rule_check.py`** applies the decision procedure of the Pehlivan FPS Typology
mechanically to each recorded clause quote and compares the result with the published
classification, then re-scans every Type G treaty to confirm that no
protection-and-security formulation was missed. A handful of divergences is expected
and correct: the published classification resolves them by reading the whole article,
where an interpretive paragraph or a comparator clause elsewhere in the article
determines the type. The script prints them for inspection rather than treating them
as errors.

**`make_figures.py`** regenerates the five analytical figures and the offline
dashboard from the annotation table.

## Reproducing the datasets from the treaty texts alone

The datasets ship with `00_REPRODUCIBILITY_PROTOCOL.txt` (in Dataset 1), a six-step
protocol covering corpus fixation, article extraction, the FPS typology decision
procedure, clause-level coding conventions, the similarity and genealogy computation,
and the pre-release verification suite. The scripts here implement Steps 5 and 6 in
full and Step 3 as a checker; Steps 2 and 4 involve legal judgment on clause text and
are documented rather than automated.

## The FPS typology in one paragraph

The corpus introduces a seven-part classification of the full protection and security
standard, based not on whether the obligation is present but on how it is formulated
and what it is anchored to. Type A stands alone; B combines with fair and equitable
treatment without an external standard; C is anchored to a minimum standard of
treatment; D is qualified by host-State law; E refers to international law without
using the phrase "minimum standard"; F is tied to a national treatment or
most-favoured-nation comparator; G has no such obligation in the treatment articles.
Fair and equitable treatment counts as combined only when it appears in the same
article, back to back. Across the 141 agreements: A=8, B=31, C=40, D=2, E=10, F=5,
G=45.

## Citation

Cite the datasets. This code has no separate identifier of its own: it is deposited
with Dataset 1 (in its `code/` folder) and is covered by that dataset's DOI.

> Pehlivan, O. K. (2026). *Turkish Bilateral Investment Treaties Corpus (1962-2025)*
> [Dataset]. DataverseNO. https://doi.org/10.18710/JX4WHH

> Pehlivan, O. K. (2026). *Turkish BIT Clause-Level Annotation Dataset (1962-2025)*
> [Dataset]. DataverseNO. https://doi.org/10.18710/FPNLKS

> Pehlivan, O. K. (2026). *Turkish BIT Treaty Diffusion and Genealogy Network
> (1962-2025)* [Dataset]. DataverseNO. https://doi.org/10.18710/WA7HEO

DataverseNO keeps one DOI per dataset across all versions, so the identifiers above
always resolve to the current version. This repository is the working copy and may
run ahead of the deposited toolkit; the deposited version is the version of record.

Machine-readable citation metadata is in `CITATION.cff`.

## Licence

Code: MIT (see `LICENSE`). Datasets: CC BY 4.0, held at DataverseNO. The treaty texts
are official legal acts of the contracting states and are not subject to copyright.

## Author

Dr. Oğuz Kaan Pehlivan, Guest Researcher, Department of Public and International Law,
University of Oslo. ORCID [0000-0003-0136-5389](https://orcid.org/0000-0003-0136-5389).
