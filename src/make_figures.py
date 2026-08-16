# -*- coding: utf-8 -*-
"""Build five visualizations of treaty_annotations + a self-contained HTML
dashboard. Outputs to yuklenecek/figures/ : five PNGs + dashboard.html.

The HTML embeds the PNGs as base64 (no external dependencies) so it opens in
any browser tab offline. Every number the figures display is also printed for
double-checking against the dataset.
"""
import base64
import io
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs", "figures")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
from common import annotations as _ann, metadata as _meta

ta = _ann().reset_index()
meta = _meta()
tm = meta.reset_index()
ta["year"] = ta["treaty_id"].map(lambda t: pd.to_datetime(meta.loc[t, "signature_date"]).year)
ta["generation"] = ta["treaty_id"].map(meta["generation"])
ta["partner"] = ta["treaty_id"].map(meta["partner_state"])
ta = ta.sort_values("year").reset_index(drop=True)

GEN_ORDER = ["gen_1_early", "gen_2_liberalization", "gen_3_eu_harmonization", "gen_4_new_model"]
GEN_LABEL = {"gen_1_early": "Gen 1\n1962–85", "gen_2_liberalization": "Gen 2\n1986–98",
             "gen_3_eu_harmonization": "Gen 3\n1999–2009", "gen_4_new_model": "Gen 4\n2010–25"}
gen_n = ta["generation"].value_counts().to_dict()

NAVY = "#1f3b66"
ACCENTS = ["#2c6fbb", "#e8743b", "#19a979", "#945ecf", "#c0392b", "#13a4b4", "#e4b400", "#6b6b6b"]
plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
                     "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

verify = {}


def yes(col):
    return (ta[col] == "Yes").astype(int)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# ===================================================================== FIG 1
# Clause prevalence by treaty generation (the new-model shift)
ESG = ["1.02_preamble_sustainable_dev", "1.03_preamble_social", "1.04_preamble_environmental",
       "4.03_health_environment", "4.04_labour", "4.06_csr", "4.08_not_lowering", "5.04_exception_health_env"]
ta["_esg"] = (ta[ESG] == "Yes").any(axis=1).astype(int)
ta["_fps"] = (ta["3.11_fps_clause"] != "No clause").astype(int)
ta["_mfnisds"] = ta["3.06_mfn_exception_procedural"].isin(["Procedural issues (ISDS)", "Both"]).astype(int)
ta["_fetq"] = (ta["3.07_fet_type"] == "FET qualified").astype(int)

SERIES = [
    ("ISDS clause", yes("7.01_isds")),
    ("Full protection & security", ta["_fps"]),
    ("Denial of benefits", yes("2.19_dob")),
    ("MFN excludes ISDS (3.06)", ta["_mfnisds"]),
    ("Qualified FET", ta["_fetq"]),
    ("ESG-related language", ta["_esg"]),
    ("Essential-security exception", yes("5.01_essential_security")),
    ("Umbrella clause", yes("3.29_umbrella")),
]
fig, ax = plt.subplots(figsize=(12.5, 7))
x = np.arange(len(GEN_ORDER))
v1 = {}
for i, (lab, ser) in enumerate(SERIES):
    pct = [100 * ser[ta["generation"] == g].mean() if (ta["generation"] == g).any() else np.nan for g in GEN_ORDER]
    v1[lab] = [round(p, 1) for p in pct]
    ax.plot(x, pct, marker="o", lw=2.2, ms=7, color=ACCENTS[i % len(ACCENTS)], label=lab)
ax.set_xticks(x)
ax.set_xticklabels([f"{GEN_LABEL[g]}\n(n={gen_n.get(g,0)})" for g in GEN_ORDER])
ax.set_ylabel("Share of treaties (%)")
ax.set_ylim(-3, 108)
ax.set_title("Clause prevalence across treaty generations — the post-2010 'new-model' shift")
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=10)
fig.subplots_adjust(bottom=0.16)
save(fig, "fig1_generation_trends.png")
verify["fig1_prevalence_by_generation_%"] = v1

# ===================================================================== FIG 2
# FPS typology: composition overall + by generation
FPS_DESC = {"A": "A · FPS standalone", "B": "B · FET + FPS", "C": "C · Min. standard + FET + FPS",
            "D": "D · Domestic-law + FPS", "E": "E · Intl law + FET + FPS", "F": "F · NT/MFN + FPS",
            "G": "G · No FPS clause"}
order = ["A", "B", "C", "D", "E", "F", "G"]
fcol = dict(zip(order, ["#2c6fbb", "#19a979", "#e8743b", "#945ecf", "#13a4b4", "#e4b400", "#b0b0b0"]))
counts = ta["3.12_fps_typology"].value_counts().reindex(order).fillna(0).astype(int)
fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.5, 6), gridspec_kw={"width_ratios": [1, 1.05]})
axa.barh([FPS_DESC[k] for k in order][::-1], counts[order].values[::-1],
         color=[fcol[k] for k in order][::-1])
for i, k in enumerate(order[::-1]):
    axa.text(counts[k] + 0.6, i, f"{counts[k]} ({100*counts[k]/132:.1f}%)", va="center", fontsize=9.5)
axa.set_xlim(0, max(counts) * 1.18)
axa.set_xlabel("Number of treaties")
axa.set_title("Pehlivan FPS Typology (all 132 treaties)")
axa.grid(axis="y")
# stacked by generation (share)
bottom = np.zeros(len(GEN_ORDER))
for k in order:
    share = [100 * ((ta["generation"] == g) & (ta["3.12_fps_typology"] == k)).sum() / max(gen_n.get(g, 1), 1) for g in GEN_ORDER]
    axb.bar(np.arange(len(GEN_ORDER)), share, bottom=bottom, color=fcol[k], label=k, edgecolor="white", lw=0.5)
    bottom += np.array(share)
axb.set_xticks(np.arange(len(GEN_ORDER)))
axb.set_xticklabels([f"{GEN_LABEL[g]}\n(n={gen_n.get(g,0)})" for g in GEN_ORDER])
axb.set_ylabel("Share within generation (%)")
axb.set_ylim(0, 100)
axb.set_title("FPS type composition by generation")
axb.legend(title="Type", ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=9)
axb.grid(axis="x")
fig.suptitle("Full Protection and Security: a shift from standalone/FET formulas to the minimum-standard model (Type C)",
             fontsize=13, fontweight="bold", y=1.02)
save(fig, "fig2_fps_typology.png")
verify["fig2_fps_counts"] = counts[order].to_dict()

# ===================================================================== FIG 3
# Heatmap: 132 treaties (chronological) x selected binary variables
HM = [
    "1.01_preamble_right_to_regulate", "1.02_preamble_sustainable_dev", "1.04_preamble_environmental",
    "2.05_host_state_law", "2.06_closed_list", "2.19_dob",
    "3.02_nt_like_circumstances", "3.16_prohibition_unreasonable",
    "3.18_expro_indirect_defined", "3.19_expro_regulatory_carveout", "3.23_strife_absolute",
    "3.25_transfer_bop", "3.29_umbrella", "3.31_entry_sojourn", "3.32_senior_mgmt",
    "4.01_transparency_states", "4.05_right_to_regulate", "4.07_corruption", "4.08_not_lowering",
    "4.10_non_derogation", "5.01_essential_security", "5.04_exception_health_env",
    "7.08_domestic_courts", "7.10_uncitral", "7.14_limitation_period", "7.18_binding_interp",
    "7.22_transparency_hearings", "8.01_consultations", "9.05_amendment",
]
short = lambda c: c.split("_", 1)[1].replace("_", " ")
M = np.array([[1 if ta.loc[i, c] == "Yes" else 0 for c in HM] for i in range(len(ta))])
fig, ax = plt.subplots(figsize=(15, 18))
cmap = ListedColormap(["#eef1f5", NAVY])
ax.imshow(M, aspect="auto", cmap=cmap, interpolation="nearest")
ax.xaxis.set_ticks_position("top")
ax.set_xticks(range(len(HM)))
ax.set_xticklabels([short(c) for c in HM], rotation=55, ha="left", fontsize=8.5)
# y ticks: every treaty would be too dense -> sparse year ticks + generation bands
yt = list(range(0, len(ta), 6))
ax.set_yticks(yt)
ax.set_yticklabels([f"{ta.loc[i,'year']}  {ta.loc[i,'partner'][:16]}" for i in yt], fontsize=7)
# generation boundaries
prevg = None
for i in range(len(ta)):
    g = ta.loc[i, "generation"]
    if g != prevg:
        ax.axhline(i - 0.5, color="#e8743b", lw=1.4)
        ax.text(len(HM) + 0.4, i + 1, GEN_LABEL[g].replace("\n", " "), color="#e8743b", fontsize=8.5, fontweight="bold")
        prevg = g
ax.set_title("Clause-presence heatmap — 132 treaties (oldest→newest, top→bottom) × 29 provisions\n",
             pad=58, loc="left")
ax.legend(handles=[Patch(color=NAVY, label="Present (Yes)"), Patch(color="#eef1f5", label="Absent / NA")],
          loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False, fontsize=11)
fig.subplots_adjust(bottom=0.06, top=0.93)
save(fig, "fig3_clause_heatmap.png")
verify["fig3_rows_cols"] = [int(M.shape[0]), int(M.shape[1])]

# ===================================================================== FIG 4
# Feature prevalence: % Yes across all binary Yes/No variables
SIMPLE = {"Yes", "No", "Not applicable", "Inconclusive", ""}
binvars = [c for c in ta.columns if c not in ("treaty_id", "year", "generation", "partner", "_esg", "_fps", "_mfnisds", "_fetq")
           and set(ta[c].unique()) <= SIMPLE and (ta[c] == "Yes").any()]
prev = sorted([(c, 100 * (ta[c] == "Yes").mean()) for c in binvars], key=lambda kv: kv[1])
SECCOL = {"1": "#2c6fbb", "2": "#19a979", "3": "#e8743b", "4": "#945ecf", "5": "#c0392b",
          "6": "#13a4b4", "7": "#e4b400", "8": "#8e6f4e", "9": "#6b6b6b"}
fig, ax = plt.subplots(figsize=(12, 14.5))
names = [short(c) for c, _ in prev]
vals = [v for _, v in prev]
cols = [SECCOL.get(c.split(".")[0], "#888") for c, _ in prev]
ax.barh(range(len(prev)), vals, color=cols)
ax.set_yticks(range(len(prev)))
ax.set_yticklabels(names, fontsize=8)
for i, v in enumerate(vals):
    ax.text(v + 0.8, i, f"{v:.0f}%", va="center", fontsize=7.5)
ax.set_xlim(0, 108)
ax.set_xlabel("Share of 132 treaties coded 'Yes' (%)")
ax.set_title("Prevalence of every binary provision (coloured by section)")
ax.legend(handles=[Patch(color=SECCOL[k], label=f"Section {k}") for k in sorted(SECCOL)],
          loc="lower right", frameon=True, fontsize=9)
ax.grid(axis="y")
ax.margins(y=0.01)
save(fig, "fig4_feature_prevalence.png")
verify["fig4_n_binary_vars"] = len(prev)

# ===================================================================== FIG 5
# Co-occurrence / phi-correlation among key binary variables
KEY = ["1.02_preamble_sustainable_dev", "1.04_preamble_environmental", "2.19_dob",
       "3.02_nt_like_circumstances", "3.16_prohibition_unreasonable", "3.18_expro_indirect_defined",
       "3.19_expro_regulatory_carveout", "3.29_umbrella", "3.31_entry_sojourn", "3.32_senior_mgmt",
       "4.01_transparency_states", "4.05_right_to_regulate", "4.07_corruption", "4.08_not_lowering",
       "5.01_essential_security", "5.04_exception_health_env", "7.14_limitation_period",
       "7.18_binding_interp", "7.22_transparency_hearings", "_mfnisds", "_esg"]
B = pd.DataFrame({(short(c) if not c.startswith("_") else c[1:]): (ta[c] == "Yes").astype(int) if not c.startswith("_") else ta[c].astype(int) for c in KEY})
C = np.corrcoef(B.values.T)
fig, ax = plt.subplots(figsize=(13.5, 12.5))
cmap = LinearSegmentedColormap.from_list("rb", ["#c0392b", "#ffffff", "#1f3b66"])
im = ax.imshow(C, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(len(B.columns)))
ax.set_xticklabels(B.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(B.columns)))
ax.set_yticklabels(B.columns, fontsize=8)
for i in range(len(B.columns)):
    for j in range(len(B.columns)):
        if abs(C[i, j]) >= 0.45 and i != j:
            ax.text(j, i, f"{C[i,j]:.1f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(C[i, j]) > 0.7 else "#222")
fig.colorbar(im, ax=ax, shrink=0.7, label="phi correlation")
ax.set_title("Which provisions travel together — phi-correlation of key clauses", pad=12)
ax.grid(False)
fig.subplots_adjust(bottom=0.18, left=0.18)
save(fig, "fig5_cooccurrence.png")
verify["fig5_n_vars"] = len(B.columns)

# ===================================================================== HTML
FIGS = [
    ("fig1_generation_trends.png", "1 · Clause prevalence across treaty generations",
     "How the share of treaties carrying each clause changes from Gen 1 (1962–85) to Gen 4 (2010–25)."),
    ("fig2_fps_typology.png", "2 · The Pehlivan FPS Typology",
     "Distribution of Full-Protection-and-Security formulations and their shift toward the minimum-standard model."),
    ("fig3_clause_heatmap.png", "3 · Clause-presence heatmap",
     "Every treaty (chronological) against 29 provisions; newer treaties are visibly denser."),
    ("fig4_feature_prevalence.png", "4 · Prevalence of every binary provision",
     "Share of the 132 treaties coded 'Yes' for each clause, from universal protections to rare modern additions."),
    ("fig5_cooccurrence.png", "5 · Clause co-occurrence",
     "Phi-correlation among key clauses, revealing the 'new-model' bundle of provisions that appear together."),
]


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


cards = "\n".join(
    f'''  <section class="card">
    <h2>{title}</h2>
    <p class="cap">{cap}</p>
    <a href="{fn}" target="_blank" title="Click to open the full-size PNG in a new tab">
      <img src="data:image/png;base64,{b64(os.path.join(OUT, fn))}" alt="{title}"/>
    </a>
  </section>'''
    for fn, title, cap in FIGS
)

html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Turkish BIT Clause-Level Annotation Dataset — Visual Overview</title>
<style>
  :root{{--navy:#1f3b66;--ink:#222;--muted:#666;--line:#e3e7ee;}}
  *{{box-sizing:border-box;}}
  body{{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:#f6f8fb;}}
  header{{background:var(--navy);color:#fff;padding:30px 24px;}}
  header h1{{margin:0 0 6px;font-size:24px;}}
  header p{{margin:2px 0;opacity:.9;font-size:14px;}}
  .wrap{{max-width:1180px;margin:0 auto;padding:24px;}}
  .card{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin:20px 0;
        box-shadow:0 1px 3px rgba(20,40,80,.06);}}
  .card h2{{margin:0 0 4px;color:var(--navy);font-size:18px;}}
  .cap{{margin:0 0 14px;color:var(--muted);font-size:13.5px;}}
  .card img{{width:100%;height:auto;border-radius:8px;border:1px solid var(--line);cursor:zoom-in;}}
  footer{{color:var(--muted);font-size:12.5px;text-align:center;padding:24px;}}
  a{{color:inherit;text-decoration:none;}}
</style></head>
<body>
<header>
  <h1>Turkish BIT Clause-Level Annotation Dataset — Visual Overview</h1>
  <p>132 international investment agreements (1962–2025) × 118 coded clause-level variables</p>
  <p>DataverseNO DOI: 10.18710/FPNLKS · Dr. Oğuz Kaan Pehlivan, University of Oslo · CC BY 4.0</p>
</header>
<div class="wrap">
  <p style="color:#555;font-size:14px;">Five views of <code>treaty_annotations</code>. Click any figure to open it full-size in a new browser tab. Generated from the packaged dataset; all figures are also provided as standalone PNG files in this folder.</p>
{cards}
  <footer>Generated by make_figures.py from treaty_annotations · UNCTAD IIA Mapping Project structure.</footer>
</div>
</body></html>'''

with open(os.path.join(OUT, "dashboard.html"), "w", encoding="utf-8") as f:
    f.write(html)

print("OUTPUT DIR:", OUT)
print("files:", sorted(os.listdir(OUT)))
print(json.dumps(verify, ensure_ascii=False, indent=1))
