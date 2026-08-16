"""Protocol Step 3: re-derive the Pehlivan FPS Typology from the recorded clause text.

For every treaty carrying a type (A-F), the decision procedure of the protocol is
applied mechanically to the quoted clause and to its article, and the result is
compared with the published classification. For Type G treaties, the treatment
articles are re-scanned to confirm that no protection-and-security formulation is
present. Disagreements are printed for manual adjudication: the published dataset
resolves them by reading the full article, which is why a handful of legitimate
divergences are expected rather than an error.

Usage:  python src/fps_rule_check.py
"""
import re

from common import annotations, extracts, fps, metadata, norm

FPS_PHRASE = (
    r"\b(?:full|most constant|constant|complete|adequate|continuous)[\w ,]{0,25}"
    r"(?:legal )?(?:protection|security)\b"
    r"|\bprotection and security\b|\bsecurity and protection\b|full legal protection"
)


def article_body(ae, tid, number):
    prefixed = [c for c in ae.columns if c.startswith(f"Madde {number} (")]
    column = prefixed[0] if prefixed else f"Madde {number}"
    return str(ae.loc[tid, column]) if column in ae.columns else ""


def classify(quote, body):
    """Decision procedure of protocol Step 3.3, applied to quote plus article."""
    text = norm(quote) or norm(body)
    fet = bool(re.search(r"fair and equitable|equitable treatment", text, re.I))
    minimum = bool(re.search(r"minimum standard", text, re.I))
    international = bool(re.search(r"international law|customary", text, re.I))
    domestic = bool(re.search(
        r"(?:legal )?protection[^.]{0,100}(in accordance with|in conformity with|under)"
        r"[^.]{0,50}laws?\b", text, re.I)) or "full legal protection" in text.lower()
    comparator = bool(re.search(
        r"(protection|security)[^.]{0,140}((no |not be |not )?less favou?rable"
        r"|not (be )?less than)", text, re.I))
    if minimum and fet:
        return "C"
    if domestic and not fet:
        return "D"
    if comparator and not fet:
        return "F"
    if international and fet:
        return "E"
    if fet:
        return "B"
    return "A"


def main():
    tm, fc, ae, ta = metadata(), fps(), extracts(), annotations()
    typed = [t for t in fc.index if fc.loc[t, "fps_typology"] != "G"]
    disagreements = []
    for tid in typed:
        quote = fc.loc[tid, "fps_formulation_text"]
        match = re.match(r"Art (\d+)", fc.loc[tid, "fps_article_location"])
        body = article_body(ae, tid, match.group(1)) if match else ""
        derived = classify(quote, body)
        if derived != fc.loc[tid, "fps_typology"]:
            disagreements.append(
                (tid, tm.loc[tid, "partner_state"], fc.loc[tid, "fps_typology"],
                 derived, norm(quote)[:110]))

    ghosts = []
    for tid in fc.index[fc.fps_typology == "G"]:
        for number in range(1, 7):
            if re.search(FPS_PHRASE, article_body(ae, tid, number), re.I):
                ghosts.append((tid, tm.loc[tid, "partner_state"], number))
                break

    sync = [t for t in fc.index
            if fc.loc[t, "fps_typology"] != ta.loc[t, "3.12_fps_typology"]]

    print(f"typed records checked: {len(typed)} | rule/record disagreements: "
          f"{len(disagreements)}")
    for tid, partner, recorded, derived, quote in disagreements:
        print(f"  {tid} {partner}: recorded={recorded} rule={derived} | {quote}")
    print(f"Type G treaties re-scanned: {int((fc.fps_typology == 'G').sum())} | "
          f"with an FPS formulation found: {len(ghosts)}")
    for row in ghosts:
        print(f"  {row}")
    print(f"DS1 / DS2 typology sync mismatches: {len(sync)}")
    print("\nDistribution:",
          fc.fps_typology.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
