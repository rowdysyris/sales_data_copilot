from __future__ import annotations
import difflib
import re
import pandas as pd


def _roles(roles): return roles.get("roles", roles) if isinstance(roles, dict) else {}


def resolve_entities(question: str, df: pd.DataFrame, column_roles: dict):
    """Resolve values mentioned in a question to dataset columns."""
    roles = _roles(column_roles)
    search_roles = ["category_column","subcategory_column","product_column","region_column","state_column","city_column","country_column","customer_column","segment_column"]
    q = str(question).lower()
    entities = []
    for role in search_roles:
        col = roles.get(role)
        if col not in df.columns:
            continue
        values = df[col].dropna().astype(str).unique().tolist()
        for val in values:
            lv = val.lower().strip()
            if len(lv) < 3:
                # Avoid false matches like customer "A" matching the letter "a" in normal questions.
                continue
            if re.search(r"\b" + re.escape(lv) + r"\b", q):
                entities.append({"text_in_question": val, "matched_value": val, "matched_column": col, "confidence": 1.0, "match_type": "exact"})
                continue
            if lv in q:
                entities.append({"text_in_question": val, "matched_value": val, "matched_column": col, "confidence": 0.85, "match_type": "partial"})
        # fuzzy token matching for shorter lists
        if len(values) <= 1000:
            words = [w for w in re.findall(r"[a-zA-Z0-9]+", q) if len(w) >= 3]
            candidates = [v.lower() for v in values if len(v.strip()) >= 3]
            for w in words:
                matches = difflib.get_close_matches(w, candidates, n=1, cutoff=0.88)
                if matches:
                    mv = next(v for v in values if v.lower() == matches[0])
                    entities.append({"text_in_question": w, "matched_value": mv, "matched_column": col, "confidence": 0.75, "match_type": "fuzzy"})
    # de-duplicate
    dedup = []
    seen = set()
    for e in sorted(entities, key=lambda x: x["confidence"], reverse=True):
        key = (e["matched_column"], e["matched_value"])
        if key not in seen:
            seen.add(key); dedup.append(e)
    return dedup[:5]


def filter_dataframe_by_entities(df, resolved_entities):
    out = df.copy()
    for ent in resolved_entities or []:
        col, val = ent.get("matched_column"), ent.get("matched_value")
        if col in out.columns:
            out = out[out[col].astype(str).str.lower() == str(val).lower()]
    return out
