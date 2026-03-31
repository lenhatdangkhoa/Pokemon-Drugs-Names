import json
import re
import requests
from pathlib import Path
from typing import List
import csv
import textwrap

# Separate caches: RxNorm/OpenFDA vs PokéAPI
RAG_CACHE_RXNORM = Path("rag_cache.json")
RAG_CACHE_POKEMON = Path("rag_cache_pokemon.json")

_STOPWORDS = {
    # Common instruction/boilerplate tokens that are not drug names
    "take", "administer", "give", "start", "continue", "hold", "stop", "use", "apply",
    "inject", "inhale", "instill", "swallow",
    # Articles / filler
    "a", "an", "the", "and", "or", "of", "to", "by", "mouth", "with", "without", "for",
}

_NON_DRUG_TOKENS = {
    # Routes / frequency / units that may appear as early tokens in messy strings
    "po", "iv", "im", "sc", "sq", "subq", "pr", "sl", "topical", "inh", "inhaled",
    "bid", "tid", "qid", "qhs", "qday", "qd", "qod", "q4h", "q6h", "q8h", "q12h", "q24h",
    "mg", "mcg", "g", "kg", "ml", "units",
}


def _load_cache(cache_path: Path):
    if cache_path.exists():
        txt = cache_path.read_text(encoding="utf-8").strip()
        if not txt:
            return {}
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict, cache_path: Path):
    cache_path.write_text(json.dumps(cache, indent=2))


def retrieve_drug_evidence(drug_list: str, use_pokemon: bool = False) -> str:
    """
    RAG evidence for prompts. Default: RxNorm (+ optional OpenFDA). If use_pokemon=True,
    query PokéAPI instead and use rag_cache_pokemon.json.
    """
    cache_path = RAG_CACHE_POKEMON if use_pokemon else RAG_CACHE_RXNORM
    cache = _load_cache(cache_path)

    candidates: List[str] = []

    def _first_plausible_token(text: str) -> str:
        words = re.findall(r"[A-Za-z][A-Za-z\-']*", text.strip())
        filtered = []
        for w in words:
            wl = w.lower()
            if wl in _STOPWORDS:
                continue
            if wl in _NON_DRUG_TOKENS:
                continue
            filtered.append(w)
            if len(filtered) >= 2:
                break
        return " ".join(filtered)

    for part in re.split(r"[;,]", drug_list):
        token = _first_plausible_token(part)
        if token:
            candidates.append(token)

    snippets = []
    changed = False

    def _is_recognized_entry(text: str) -> bool:
        if use_pokemon:
            return "this is a pokemon" in (text or "").lower()
        return "recognized drug in rxnorm" in (text or "").lower()

    def _pick_best_cached_entry(name_key: str):
        key = (name_key or "").lower().strip()
        if not key:
            return None
        parts = key.split()
        if not parts:
            return None
        one = parts[0]
        two = " ".join(parts[:2]) if len(parts) >= 2 else None
        v1 = cache.get(one)
        if v1 and _is_recognized_entry(v1):
            return v1
        if two:
            v2 = cache.get(two)
            if v2 and _is_recognized_entry(v2):
                return v2
        return None

    for name in candidates:
        key = name.lower()
        text = _pick_best_cached_entry(key)
        if text is None:
            if key in cache:
                text = cache[key]
            else:
                if use_pokemon:
                    text = _fetch_pokemon_evidence(name)
                else:
                    text = _fetch_rxnav(name)
                cache[key] = text
                changed = True
        snippets.append(text)

    if changed:
        _save_cache(cache, cache_path)

    return "\n".join(snippets)


def _fetch_pokemon_evidence(name: str) -> str:
    """Query PokéAPI by species name (first token, lowercased)."""
    slug = (name.strip().split()[0].lower() if name.strip() else "") or ""
    if not slug:
        return f"{name}: not found in Pokémon API."
    url = f"https://pokeapi.co/api/v2/pokemon/{slug}/"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 404:
            return f"{name}: not found in Pokémon API."
        r.raise_for_status()
        return f"{name}: this is a pokemon."
    except Exception as e:
        return f"{name}: Pokémon API lookup error ({e})."


def _fetch_rxnav(name: str) -> str:
    """Fetch info from RxNav and OpenFDA for RAG."""
    base = "https://rxnav.nlm.nih.gov/REST"

    try:
        rxcui_resp = requests.get(f"{base}/rxcui.json", params={"name": name}, timeout=10)
        data = rxcui_resp.json()
        ids = data.get("idGroup", {}).get("rxnormId")
        if not ids:
            return f"{name}: not found in RxNorm."

        rxcui = ids[0]
        props_resp = requests.get(f"{base}/rxcui/{rxcui}/properties.json", timeout=10)
        props = props_resp.json().get("properties", {}) or {}

        pref_name = props.get("name", name)
        synonym = props.get("synonym")
        tty = props.get("tty")
        rxtype = props.get("rxtype")

        parts = [f"{name}: recognized drug in RxNorm."]
        if pref_name and pref_name.lower() != name.lower():
            parts.append(f"Preferred name: {pref_name}.")
        if synonym and synonym.lower() != pref_name.lower():
            parts.append(f"Synonym/brand: {synonym}.")
        if rxtype:
            parts.append(f"RxNorm type: {rxtype} (TTY={tty}).")
        elif tty:
            parts.append(f"Term type: {tty}.")

        fda_summary = _fetch_openfda(pref_name)
        if fda_summary:
            parts.append(fda_summary)

        return " ".join(parts)

    except Exception as e:
        return f"{name}: RxNorm/OpenFDA lookup error ({e})."


def _fetch_openfda(pref_name: str) -> str:
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
            txt = " ".join(indications[0].split())
            parts.append(f"Indications: {txt}")
        if dosage:
            txt = " ".join(dosage[0].split())
            parts.append(f"Dosage (label excerpt): {txt}")

        summary = " ".join(parts)
        return textwrap.shorten(summary, width=400, placeholder=" ...")
    except Exception:
        return ""


def load_first_brand_case() -> str:
    csv_path = Path("experiments/brand/pokemon.csv")
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader)
    return first_row["pokemon list"]


if __name__ == "__main__":
    drug_list = load_first_brand_case()
    print("=== Original drug list (truncated) ===")
    print(drug_list[:300] + "...\n")

    print("=== RxNorm RAG ===")
    print(retrieve_drug_evidence(drug_list, use_pokemon=False), "\n")

    print("=== Second RAG call (cache) ===")
    print(retrieve_drug_evidence(drug_list, use_pokemon=False), "\n")

    print("=== Pokémon RAG (separate cache) ===")
    print(retrieve_drug_evidence(drug_list, use_pokemon=True))
