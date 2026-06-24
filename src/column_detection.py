from __future__ import annotations

import re
from typing import Dict
import pandas as pd
import warnings
from src.utils import safe_numeric

# Patterns are ordered from specific to broad. Broad words such as "total" are
# intentionally excluded from sales because they incorrectly capture columns
# such as "Total Cost".
ROLE_PATTERNS = {
    "ship_mode_column": ["ship mode", "shipping mode", "delivery mode", "shipment mode", "mode of shipment", "shipping method", "delivery method"],
    "ship_date_column": ["ship date", "delivery date", "dispatch date"],
    "date_column": ["order date", "invoice date", "purchase date", "transaction date", "date", "tarikh"],
    "customer_id_column": ["customer id", "client id", "buyer id", "cust id"],
    "customer_column": ["customer name", "customer", "client", "buyer", "grahak"],
    "order_id_column": ["order id", "invoice id", "transaction id", "order number", "bill no"],
    "subcategory_column": ["sub-category", "subcategory", "sub category", "subcat", "product subcategory"],
    "category_column": ["product category", "category"],
    "product_column": ["product name", "product", "item", "sku", "utpad", "maal"],
    "region_column": ["region", "zone", "territory"],
    "city_column": ["city", "town", "shehar"],
    "state_column": ["state", "province"],
    "country_column": ["country", "nation"],
    "segment_column": ["customer segment", "market segment", "segment"],
    "cost_column": ["total cost", "unit cost", "cost", "cogs", "expense", "kharcha", "lagat"],
    "unit_price_column": ["unit price", "price per unit", "item price", "selling price", "rate", "unit selling price"],
    "margin_percent_column": ["margin %", "margin percent", "margin pct", "profit margin", "gross margin", "net margin", "margin rate"],
    "profit_column": ["total profit", "net profit", "gross profit", "profit", "contribution amount", "profit amount", "profit value", "labh", "laabh", "munafa"],
    "sales_column": ["net sales", "gross sales", "total sales", "sales", "revenue", "turnover", "order value", "total amount", "amount", "vikray", "bikri", "sale amount"],
    "revenue_column": ["revenue", "sales", "turnover", "vikray", "bikri"],
    "quantity_column": ["quantity", "qty", "units", "unit sold", "units sold", "matra"],
    "discount_column": ["discount", "rebate", "markdown", "offer", "chhoot"],
}

DATE_VALUE_ROLES = {"date_column", "ship_date_column"}
NUMERIC_VALUE_ROLES = {"sales_column", "revenue_column", "profit_column", "cost_column", "unit_price_column", "quantity_column", "discount_column", "margin_percent_column"}

# Words that make a column unsuitable for a role even if a broad keyword also appears.
ROLE_EXCLUSIONS = {
    "sales_column": ["cost", "cogs", "expense", "discount", "rebate", "margin", "profit", "qty", "quantity", "tax"],
    "revenue_column": ["cost", "cogs", "expense", "discount", "margin", "profit", "qty", "quantity", "tax"],
    "unit_price_column": ["cost", "discount", "margin", "profit", "qty", "quantity"],
    "profit_column": ["margin percent", "margin %", "profit margin", "gross margin", "net margin", "margin rate", "margin pct", "%"],
    "margin_percent_column": ["profit amount", "total profit", "net profit", "gross profit", "profit value"],
    "category_column": ["subcategory", "sub category", "sub-category", "subcat"],
    "product_column": ["category", "subcategory", "sub category", "sub-category"],
    "date_column": ["ship date", "delivery date", "dispatch date"],
}

DATE_NAME_HINTS = ["date", "month", "year", "quarter", "period", "invoice date", "order date", "ship date", "delivery"]
DATE_VALUE_PATTERN = re.compile(
    r"(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\bjan\b|\bfeb\b|\bmar\b|\bapr\b|\bmay\b|\bjun\b|\bjul\b|\baug\b|\bsep\b|\boct\b|\bnov\b|\bdec\b)",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9%]+", " ", str(text).lower()).strip()


def _contains_any(text: str, words: list[str]) -> bool:
    n = _norm(text)
    return any(_norm(w) in n for w in words)


def _series_for_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a single Series even if duplicate labels slipped through."""
    obj = df.loc[:, col]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) or safe_numeric(s).notna().mean() > 0.8



def _looks_like_percent_metric(s: pd.Series) -> bool:
    vals = safe_numeric(s).dropna()
    if vals.empty:
        return False
    # Most margin-percent columns live between -100 and 100, often between -1 and 1 for ratios.
    return (vals.between(-100, 100).mean() >= 0.85)

def _looks_like_yyyymmdd_numeric(s: pd.Series) -> bool:
    st = safe_numeric(s).dropna()
    if st.empty:
        return False
    vals = st.astype("Int64").astype(str)
    if (vals.str.len() == 8).mean() < 0.7:
        return False
    parsed = pd.to_datetime(vals, format="%Y%m%d", errors="coerce")
    return parsed.notna().mean() >= 0.7


def _is_date_like(s: pd.Series, column_name: str | None = None, allow_numeric_date: bool = False) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.empty:
        return False
    name_has_date_hint = _contains_any(column_name or "", DATE_NAME_HINTS)
    if pd.api.types.is_numeric_dtype(s):
        return bool(allow_numeric_date and name_has_date_hint and _looks_like_yyyymmdd_numeric(s))

    text = s.dropna().astype(str).str.strip()
    if text.empty:
        return False
    # Avoid treating plain numeric sales/revenue strings as dates.
    numeric_ratio = safe_numeric(text).notna().mean()
    if numeric_ratio >= 0.8 and not name_has_date_hint:
        return False
    clue_ratio = text.str.contains(DATE_VALUE_PATTERN, regex=True, na=False).mean()
    if clue_ratio < 0.50 and not name_has_date_hint:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=False)
    return parsed.notna().mean() >= 0.70


def _score_name(role: str, col_name: str, patterns: list[str]) -> tuple[float, list[str]]:
    ncol = _norm(col_name)
    if role == "margin_percent_column" and not any(token in ncol for token in ["margin", "percent", "pct", "%", "rate"]):
        return 0.0, ["excluded because column name does not indicate a margin/percent/rate field"]
    if role == "ship_date_column" and ncol in {"date", "order date", "invoice date", "transaction date", "purchase date", "tarikh"}:
        return 0.0, ["excluded because this is a generic/order date, not a ship/delivery date"]
    if role == "subcategory_column" and not any(token in ncol for token in ["sub", "subcategory", "subcat", "sub category", "sub category"]):
        return 0.0, ["excluded because column name does not indicate a subcategory field"]
    if _contains_any(col_name, ROLE_EXCLUSIONS.get(role, [])):
        return 0.0, [f"excluded for {role} by conflicting keyword"]
    score = 0.0
    reason: list[str] = []
    for pat in patterns:
        npat = _norm(pat)
        if ncol == npat:
            score = max(score, 0.98)
            reason.append(f"exact name match '{pat}'")
        elif re.search(rf"\b{re.escape(npat)}\b", ncol):
            score = max(score, 0.88)
            reason.append(f"keyword match '{pat}'")
        elif npat in ncol:
            score = max(score, 0.72)
            reason.append(f"partial keyword match '{pat}'")
    return score, reason


def detect_columns(df: pd.DataFrame) -> dict:
    """Detect business column roles using names and dtype heuristics.

    The detector is deliberately conservative for dates and broad financial names
    to avoid wrong downstream calculations on arbitrary sales datasets.
    """
    roles: Dict[str, str] = {}
    conf: Dict[str, dict] = {}
    used: set[str] = set()

    for role, patterns in ROLE_PATTERNS.items():
        best = None
        best_score = 0.0
        best_reason = ""
        for col in df.columns:
            series = _series_for_col(df, col)
            score, reason = _score_name(role, col, patterns)

            if role in NUMERIC_VALUE_ROLES and score > 0 and _is_numeric(series):
                score += 0.08
                reason.append("numeric values")
            if role == "margin_percent_column" and score > 0:
                if _looks_like_percent_metric(series):
                    score += 0.08
                    reason.append("percent-like margin values")
                else:
                    score = min(score, 0.50)
                    reason.append("values do not look like a percent metric")
            if role in DATE_VALUE_ROLES and score > 0 and _is_date_like(series, col, allow_numeric_date=True):
                score += 0.10
                reason.append("date-like values")

            if score > best_score:
                best, best_score, best_reason = col, min(score, 0.99), "; ".join(reason)

        if best and best_score >= 0.55:
            # Revenue can alias sales. Other roles should not reuse the same column.
            if best not in used or role in {"revenue_column"}:
                roles[role] = best
                conf[role] = {"column": best, "confidence": round(float(best_score), 2), "reason": best_reason}
                if role not in {"revenue_column"}:
                    used.add(best)

    # Conservative fallback for date: only real datetime columns or textual values
    # with date clues. Do not parse arbitrary numeric columns as dates.
    if "date_column" not in roles:
        for col in df.columns:
            series = _series_for_col(df, col)
            if col not in used and _is_date_like(series, col, allow_numeric_date=False):
                roles["date_column"] = col
                conf["date_column"] = {"column": col, "confidence": 0.7, "reason": "date-like textual values"}
                break

    return {"roles": roles, "confidence": conf}
