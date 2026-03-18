import json
import os
import re
import requests
from pathlib import Path
from typing import List
import csv
import textwrap

# Add caching path
_CACHE_PATH = Path("rag_cache.json")

def _load_cache():
    if _CACHE_PATH.exists():
        return json.loads(_CACHE_PATH.read_text())
    return {}

def _save_cache(cache):
    _CACHE_PATH.write_text(json.dumps(cache, indent=2))


def retrieve_drug_evidence(drug_list: str) -> str:
    cache = _load_cache()
    # naive name extraction: split on ';' and take first token before dose
    candidates: List[str] = []
    for part in drug_list.split(";"):
        words = part.strip().split()
        if not words:
            continue
        token = words[0]
        candidates.append(token)

    snippets = []
    changed = False
    for name in candidates:
        key = name.lower()
        if key in cache:
            text = cache[key]
        else:
            text = _fetch_rxnav(name)
            cache[key] = text
            changed = True
        snippets.append(text)

    if changed:
        _save_cache(cache)

    return "\n".join(snippets)

def _fetch_rxnav(name: str) -> str:
    """
    Fetch richer info from RxNav and OpenFDA for RAG.
    """
    base = "https://rxnav.nlm.nih.gov/REST"

    try:
        # 1) RxNorm: get RxCUI
        rxcui_resp = requests.get(f"{base}/rxcui.json", params={"name": name}, timeout=10)
        data = rxcui_resp.json()
        ids = data.get("idGroup", {}).get("rxnormId")
        if not ids:
            return f"{name}: not found in RxNorm."

        rxcui = ids[0]

        # 2) RxNorm properties
        props_resp = requests.get(f"{base}/rxcui/{rxcui}/properties.json", timeout=10)
        props = props_resp.json().get("properties", {}) or {}

        pref_name = props.get("name", name)
        synonym = props.get("synonym")
        tty = props.get("tty")      # term type
        rxtype = props.get("rxtype")  # brand/generic/etc.

        parts = [f"{name}: recognized drug in RxNorm."]
        if pref_name and pref_name.lower() != name.lower():
            parts.append(f"Preferred name: {pref_name}.")
        if synonym and synonym.lower() != pref_name.lower():
            parts.append(f"Synonym/brand: {synonym}.")
        if rxtype:
            parts.append(f"RxNorm type: {rxtype} (TTY={tty}).")
        elif tty:
            parts.append(f"Term type: {tty}.")

        # 3) OpenFDA enrichment (optional, best-effort)
        fda_summary = _fetch_openfda(pref_name)
        if fda_summary:
            parts.append(fda_summary)

        return " ".join(parts)

    except Exception as e:
        return f"{name}: RxNorm/OpenFDA lookup error ({e})."


def _fetch_openfda(pref_name: str) -> str:
    """
    Query OpenFDA for a drug label by ingredient/brand name and return a short summary.

    This is intentionally minimal: it looks at indications and dosage text if present,
    and compresses them into 1–2 sentences.
    """
    OPENFDA_ENDPOINT = "https://api.fda.gov/drug/label.json"
    try:
        params = {
            "search": f"openfda.generic_name:{pref_name} OR openfda.brand_name:{pref_name}",
            "limit": 1,
        }
        resp = requests.get(OPENFDA_ENDPOINT, params=params, timeout=10)
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return ""

        doc = results[0]
        indications = (doc.get("indications_and_usage") or [])
        dosage = (doc.get("dosage_and_administration") or [])

        parts = []
        if indications:
            # Take first paragraph and squash whitespace
            txt = " ".join(indications[0].split())
            parts.append(f"Indications: {txt}")
        if dosage:
            txt = " ".join(dosage[0].split())
            parts.append(f"Dosage (label excerpt): {txt}")

        summary = " ".join(parts)
        # Keep it reasonably short
        return textwrap.shorten(summary, width=400, placeholder=" ...")
    except Exception:
        return ""

def load_first_brand_case() -> str:
    csv_path = Path("experiments/brand/pokemon.csv")
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader)
    # Column name from your CSV: "pokemon list"
    return first_row["pokemon list"]


if __name__ == "__main__":
    drug_list = load_first_brand_case()
    print("=== Original drug list (truncated) ===")
    print(drug_list[:300] + "...\n")

    print("=== First RAG call (should hit RxNav and fill cache) ===")
    evidence1 = retrieve_drug_evidence(drug_list)
    print(evidence1, "\n")

    print("=== Second RAG call (should use cache, no extra API) ===")
    evidence2 = retrieve_drug_evidence(drug_list)
    print(evidence2)