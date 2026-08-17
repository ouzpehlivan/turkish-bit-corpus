# -*- coding: utf-8 -*-
"""Build ds3_visual_analysis.html, a self-contained analytical dashboard for
the Turkish BIT Diffusion / Genealogy Network, generated only from the packaged
Dataset 3 CSV files. Five PNG charts are embedded as base64 so the HTML opens
offline in any browser.
"""
import base64
import io
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sm = pd.read_csv(os.path.join(HERE, "similarity_matrix.csv"), index_col=0)
pairs = pd.read_csv(os.path.join(HERE, "treaty_pairs.csv"))
fam = pd.read_csv(os.path.join(HERE, "treaty_families.csv"))
edges = pd.read_csv(os.path.join(HERE, "genealogy_edges.csv"))

NAVY = "#0b3d91"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
                     "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

MODEL_COLOR = {"Liberalization model": "#2c6fbb", "New Turkish model": "#e8743b",
               "EU-harmonization model": "#19a979", "Singleton": "#b0b0b0", "Early": "#945ecf"}


def model_of(label):
    for k in MODEL_COLOR:
        if label.startswith(k):
            return k
    return "Singleton"


FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def save(fig, name):
    p = os.path.join(FIGDIR, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# A. cosine distribution
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.hist(pairs["cosine_similarity"], bins=60, color=NAVY, alpha=0.85)
med = pairs["cosine_similarity"].median()
for x, lab, c in [(0.45, "genealogy ≥0.45", "#e8743b"), (med, f"median {med:.2f}", "#19a979"),
                  (0.8, f"≥0.8 ({int((pairs['cosine_similarity'] >= 0.8).sum())} pairs)", "#945ecf"), (0.9, f"≥0.9 ({int((pairs['cosine_similarity'] >= 0.9).sum())} pairs)", "#c0392b")]:
    ax.axvline(x, color=c, ls="--", lw=1.6); ax.text(x, ax.get_ylim()[1]*0.92, " "+lab, color=c, fontsize=9, rotation=90, va="top")
ax.set_xlabel("TF-IDF cosine similarity"); ax.set_ylabel("number of treaty pairs")
ax.set_title(f"Distribution of pairwise textual similarity ({len(pairs):,} treaty pairs)")
save(fig, "ds3_fig_a_cosine.png")

# B. similarity heatmap ordered by family then year
order = fam.sort_values(["family_id", "signature_year"])["treaty_id"].tolist()
M = sm.loc[order, order].values
fig, ax = plt.subplots(figsize=(9.5, 8.5))
im = ax.imshow(M, cmap="magma", vmin=0, vmax=1, aspect="equal")
fig.colorbar(im, ax=ax, shrink=0.8, label="cosine similarity")
# family block boundaries
fam_seq = fam.set_index("treaty_id").loc[order, "family_id"].tolist()
b = [i for i in range(1, len(order)) if fam_seq[i] != fam_seq[i-1]]
for x in b:
    ax.axhline(x-0.5, color="#39ff14", lw=0.4, alpha=0.5); ax.axvline(x-0.5, color="#39ff14", lw=0.4, alpha=0.5)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Treaty-by-treaty similarity, ordered by family\n(bright blocks = textual families)")
save(fig, "ds3_fig_b_heatmap.png")

# C. family sizes
fl = fam.drop_duplicates("family_id").copy()
fl["model"] = fl["family_label"].map(model_of)
fl = fl.sort_values("family_size", ascending=True)
fig, ax = plt.subplots(figsize=(11, 8))
ax.barh(range(len(fl)), fl["family_size"], color=[MODEL_COLOR[m] for m in fl["model"]])
ax.set_yticks(range(len(fl))); ax.set_yticklabels(fl["family_label"], fontsize=7.5)
for i, v in enumerate(fl["family_size"]):
    ax.text(v+0.2, i, str(int(v)), va="center", fontsize=8)
ax.set_xlabel("treaties in family"); ax.set_title(f"{fam['family_id'].nunique()} textual families by size")
ax.legend(handles=[Patch(color=c, label=k) for k, c in MODEL_COLOR.items() if k != "Early"], fontsize=9, loc="lower right")
ax.grid(axis="y")
save(fig, "ds3_fig_c_families.png")

# D. genealogy diffusion: parent_year vs child_year
fig, ax = plt.subplots(figsize=(9.5, 7))
sc = ax.scatter(edges["child_year"], edges["parent_year"], c=edges["cosine_similarity"],
                cmap="viridis", s=60, edgecolor="white", lw=0.5, vmin=0.45, vmax=1)
fig.colorbar(sc, ax=ax, label="cosine similarity (parent→child)")
lim = [1960, 2026]; ax.plot(lim, lim, color="#aaa", ls=":", lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("child treaty signature year"); ax.set_ylabel("parent (predecessor) signature year")
ax.set_title(f"Genealogy: each treaty's most-similar earlier predecessor ({len(edges)} descent edges)")
save(fig, "ds3_fig_d_genealogy.png")

# E. textual vs legal-feature similarity
fig, ax = plt.subplots(figsize=(10, 6))
hb = ax.hexbin(pairs["cosine_similarity"], pairs["shared_clause_features"], gridsize=28, cmap="Blues", mincnt=1)
fig.colorbar(hb, ax=ax, label="number of pairs")
r = np.corrcoef(pairs["cosine_similarity"], pairs["shared_clause_features"])[0, 1]
ax.set_xlabel("TF-IDF cosine similarity (text)"); ax.set_ylabel("shared clause-feature packages (0–9, from DS2)")
ax.set_title(f"Textual similarity vs shared legal features (Pearson r = {r:.2f})")
save(fig, "ds3_fig_e_text_vs_legal.png")

# ---- HTML ----
FIGS = [
    ("ds3_fig_a_cosine.png", "1 · Pairwise textual similarity", f"How similar treaty texts are across all {len(pairs):,} pairs, with the genealogy (≥0.45) and high-similarity thresholds marked."),
    ("ds3_fig_b_heatmap.png", "2 · Similarity matrix by family", f"The {len(sm)}×{len(sm)} cosine matrix reordered by family - bright square blocks are the textual families the clustering recovers."),
    ("ds3_fig_c_families.png", "3 · Textual families", f"The {fam['family_id'].nunique()} families by size, coloured by the dominant model (Liberalization, New Turkish, EU-harmonization, Singleton)."),
    ("ds3_fig_d_genealogy.png", "4 · Genealogy of descent", "Each treaty linked to its most-similar earlier predecessor; points below the diagonal trace how drafting diffused forward in time."),
    ("ds3_fig_e_text_vs_legal.png", "5 · Text vs legal features", "Whether treaties that read alike (cosine) also share the same Dataset 2 clause coding - the two similarity notions largely agree."),
]


def b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


cards = "\n".join(
    f'''  <section class="card"><h2>{t}</h2><p class="cap">{c}</p>
    <a href="figures/{fn}" target="_blank"><img src="data:image/png;base64,{b64(os.path.join(FIGDIR, fn))}" alt="{t}"/></a></section>'''
    for fn, t, c in FIGS)

html = f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Turkish BIT Diffusion / Genealogy Network - Visual Analysis</title>
<style>
 *{{box-sizing:border-box}} body{{margin:0;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#1a1a2e;background:#faf9f6}}
 header{{background:#0b3d91;color:#fff;padding:28px 24px}} header h1{{margin:0 0 6px;font-size:23px}} header p{{margin:2px 0;opacity:.9;font-size:13.5px}}
 .wrap{{max-width:1100px;margin:0 auto;padding:22px}}
 .card{{background:#fff;border:1px solid #e3e0d8;border-radius:12px;padding:18px 20px;margin:18px 0;box-shadow:0 1px 3px rgba(11,61,145,.06)}}
 .card h2{{margin:0 0 4px;color:#0b3d91;font-size:18px}} .cap{{margin:0 0 12px;color:#666;font-size:13.5px}}
 .card img{{width:100%;height:auto;border-radius:8px;border:1px solid #eee;cursor:zoom-in}}
 footer{{color:#777;font-size:12.5px;text-align:center;padding:22px}} a{{color:inherit;text-decoration:none}}
</style></head><body>
<header><h1>Turkish BIT Diffusion / Genealogy Network - Visual Analysis</h1>
<p>{len(sm)} treaties (1962–2025) · {len(pairs):,} pairwise comparisons · {fam["family_id"].nunique()} families · {len(edges)} genealogy edges</p>
<p>DataverseNO DOI: 10.18710/WA7HEO · Dr. Oğuz Kaan Pehlivan, University of Oslo · CC BY 4.0</p></header>
<div class="wrap">
<p style="color:#555;font-size:14px">Five views generated only from the packaged Dataset 3 files. Click any figure to open it full size. Standalone PNGs are also provided, in figures/.</p>
{cards}
<footer>Generated by code/make_ds3_figures.py, deposited with Dataset 1, from the Dataset 3 CSV files.</footer>
</div></body></html>'''

with open(os.path.join(HERE, "ds3_visual_analysis.html"), "w", encoding="utf-8") as f:
    f.write(html)

print("ds3_visual_analysis.html:", os.path.getsize(os.path.join(HERE, "ds3_visual_analysis.html")) // 1024, "KB")
print("pairs cosine median:", round(pairs["cosine_similarity"].median(), 3),
      "| text-vs-legal r:", round(np.corrcoef(pairs["cosine_similarity"], pairs["shared_clause_features"])[0, 1], 3))
print("families:", fam["family_id"].nunique(), "| edges:", len(edges))
