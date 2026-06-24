from __future__ import annotations

import math
from typing import Any, Iterable
import pandas as pd
import numpy as np


def safe_numeric(series: pd.Series | Any) -> pd.Series:
    """Convert a Series to numeric while preserving invalid values as NaN.

    Handles common sales-dataset formats such as:
    - currency symbols: ₹1,200, Rs. 1,200, $1,200
    - comma separators: 1,200.50
    - percentage strings: 40% -> 40
    - accounting negatives: (1,200) -> -1200

    Percentage strings intentionally convert to 0-100 scale, not 0-1, because the
    analytics modules already normalize discount formats by checking max <= 1.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip()
    # Preserve null-like strings as empty before conversion.
    s = s.replace({"nan": "", "None": "", "NONE": "", "null": "", "NULL": "", "NaN": ""})
    neg_mask = s.str.match(r"^\(.*\)$", na=False)
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    s = (
        s.str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("Rs.", "", regex=False)
        .str.replace("Rs", "", regex=False)
        .str.replace("INR", "", regex=False)
        .str.replace("USD", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(s, errors="coerce")


def get_col(roles: dict, role: str) -> str | None:
    """Return the detected column for a business role."""
    if not roles:
        return None
    if "roles" in roles:
        roles = roles["roles"]
    col = roles.get(role)
    return col if isinstance(col, str) and col else None


def available_cols(roles: dict, *role_names: str) -> list[str]:
    """Return available role columns."""
    return [c for r in role_names if (c := get_col(roles, r))]


def format_currency(value: Any) -> str:
    """Format a number as USD currency for manager-facing output."""
    try:
        v = float(value)
    except Exception:
        return "N/A"
    sign = "-" if v < 0 else ""
    v = abs(v)
    return f"{sign}${v:,.2f}"


def format_percent(value: Any) -> str:
    """Format a number as percentage."""
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "N/A"


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division."""
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        return float(numerator) / float(denominator)
    except Exception:
        return default


def profit_margin(profit: float, sales: float) -> float:
    """Profit margin percent."""
    return safe_div(profit, sales) * 100


def iqr_outlier_mask(series: pd.Series) -> pd.Series:
    """Return boolean mask for IQR outliers."""
    s = safe_numeric(series).dropna()
    if len(s) < 4:
        return pd.Series(False, index=series.index)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or pd.isna(iqr):
        return pd.Series(False, index=series.index)
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    numeric = safe_numeric(series)
    return (numeric < lower) | (numeric > upper)


def normalize_0_100(series: pd.Series, invert: bool = False) -> pd.Series:
    """Normalize a numeric Series to 0-100."""
    s = safe_numeric(series).fillna(0)
    mn, mx = s.min(), s.max()
    if mx == mn:
        out = pd.Series(50.0, index=s.index)
    else:
        out = (s - mn) / (mx - mn) * 100
    if invert:
        out = 100 - out
    return out.clip(0, 100)


def top_records(df: pd.DataFrame, sort_col: str, n: int = 10, ascending: bool = False) -> pd.DataFrame:
    """Sort and return top records safely."""
    if df is None or df.empty or sort_col not in df.columns:
        return pd.DataFrame()
    return df.sort_values(sort_col, ascending=ascending).head(n).reset_index(drop=True)


def as_records(df: pd.DataFrame, n: int = 10) -> list[dict]:
    """DataFrame to records safely."""
    if df is None or getattr(df, "empty", True):
        return []
    return df.head(n).replace({np.nan: None}).to_dict("records")


def _looks_ambiguous_dayfirst(series: pd.Series) -> bool | None:
    """Infer whether slash/dash dates probably use day-first format.

    Returns True for day-first, False for month-first, and None when the sample
    is ambiguous. The rule is conservative: values with first token > 12 prove
    day-first; values with second token > 12 prove month-first.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    text = series.dropna().astype(str).str.strip().head(500)
    dayfirst_votes = 0
    monthfirst_votes = 0
    for value in text:
        m = re_match_date_tokens(value)
        if not m:
            continue
        first, second = m
        if first > 12 and 1 <= second <= 12:
            dayfirst_votes += 1
        elif second > 12 and 1 <= first <= 12:
            monthfirst_votes += 1
    if dayfirst_votes > monthfirst_votes:
        return True
    if monthfirst_votes > dayfirst_votes:
        return False
    return None


def re_match_date_tokens(value: str) -> tuple[int, int] | None:
    import re
    m = re.match(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})(?:\s|$)", str(value))
    if not m:
        return None
    try:
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None


def ensure_datetime(series: pd.Series, *, dayfirst: bool | None = None) -> pd.Series:
    """Convert Series to datetime with conservative Indian/date ambiguity handling.

    If dayfirst is None, the parser auto-detects clear dd/mm/yyyy vs mm/dd/yyyy
    evidence from sampled values. Ambiguous dates are parsed by pandas defaults,
    but clear Indian-style inputs like 31/01/2024 are parsed correctly.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    if dayfirst is None:
        dayfirst = _looks_ambiguous_dayfirst(series)
    if dayfirst is True:
        return pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True)
    if dayfirst is False:
        return pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=False)
    return pd.to_datetime(series, errors="coerce", format="mixed")


def infer_dataset_grain(df: pd.DataFrame, roles: dict) -> dict:
    """Infer whether rows are transaction-level or aggregate-level.

    This does not change calculations. It only records how counts should be
    described so manager reports do not call monthly summary rows "orders".
    """
    roles = _flat_roles(roles)
    if df is None or df.empty:
        return {"grain": "unknown", "count_label": "Rows", "reason": "empty dataset"}
    order_col = roles.get("order_id_column")
    customer_col = roles.get("customer_column") or roles.get("customer_id_column")
    date_col = roles.get("date_column")
    # Explicit order id is the strongest transaction signal.
    if order_col in df.columns:
        return {"grain": "transaction", "count_label": "Orders", "reason": "order id column detected"}
    col_names = " ".join(str(c).lower() for c in df.columns)
    if any(w in col_names for w in ("month", "monthly", "quarter", "period", "fiscal")) and customer_col not in df.columns:
        return {"grain": "aggregate", "count_label": "Records", "reason": "period-style dataset without order/customer identifiers"}
    if date_col in df.columns and customer_col not in df.columns:
        dates = ensure_datetime(df[date_col])
        valid = dates.dropna()
        if len(valid) >= 3:
            periods = valid.dt.to_period("M")
            # Many rows per month with no order/customer id is commonly a monthly
            # aggregate by category/region/product.
            if periods.nunique() <= max(3, len(valid) * 0.45):
                return {"grain": "possibly_aggregate", "count_label": "Records", "reason": "no order/customer id and repeated monthly periods"}
    return {"grain": "row_level_unknown", "count_label": "Records" if customer_col not in df.columns else "Rows", "reason": "no order id detected"}



def _flat_roles(column_roles: dict | None) -> dict:
    """Return a mutable flat role dictionary."""
    if not isinstance(column_roles, dict):
        return {}
    return dict(column_roles.get("roles", column_roles))


def normalize_percent_series(series: pd.Series) -> pd.Series:
    """Return a numeric percent series on a 0-100 scale.

    Accepts either 0.40 or 40 for 40%. This is useful for discount and margin
    fields that appear in different sales systems.
    """
    s = safe_numeric(series)
    if s.dropna().empty:
        return s
    # If most non-null values are between -1 and 1, interpret as a ratio.
    non_null = s.dropna()
    if (non_null.abs() <= 1).mean() >= 0.80:
        return s * 100
    return s


def prepare_analysis_dataset(df: pd.DataFrame, column_roles: dict) -> tuple[pd.DataFrame, dict]:
    """Prepare a dataframe and roles for downstream analytics without mutating inputs.

    Fixes common sales-dataset edge cases:
    - Derives Profit from Sales - Cost when an explicit profit column is missing.
    - Derives estimated Profit from Sales * Margin% only when margin is available.
    - Normalizes discount / margin percent columns to a 0-100 numeric scale.
    - Applies business-aware missing-value treatment and records a full audit log.

    Returns a flat role dictionary with optional metadata keys such as
    ``_profit_derivation_note`` and ``_missing_value_treatment``.
    """
    work = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    roles = _flat_roles(column_roles)

    # Dataset grain metadata is used only for transparent wording in KPIs/Q&A.
    roles.setdefault("_dataset_grain", infer_dataset_grain(work, roles))

    # Normalize percentage-like columns while preserving original business names.
    for role in ("discount_column", "margin_percent_column"):
        col = roles.get(role)
        if col in work.columns:
            work[col] = normalize_percent_series(work[col])

    sales_col = roles.get("sales_column")
    profit_col = roles.get("profit_column")
    cost_col = roles.get("cost_column")
    margin_pct_col = roles.get("margin_percent_column")

    has_profit = bool(profit_col in work.columns)
    if not has_profit and sales_col in work.columns and cost_col in work.columns:
        derived = "__Derived Profit"
        # Keep unresolved rows as NaN. The missing-value treatment step will
        # derive/fill only where safe and log unresolved values transparently.
        work[derived] = safe_numeric(work[sales_col]) - safe_numeric(work[cost_col])
        roles["profit_column"] = derived
        roles["_profit_derivation_note"] = f"Profit was derived as {sales_col} - {cost_col}."
    elif not has_profit and sales_col in work.columns and margin_pct_col in work.columns:
        derived = "__Estimated Profit From Margin"
        margin_pct = normalize_percent_series(work[margin_pct_col])
        work[derived] = safe_numeric(work[sales_col]) * (margin_pct / 100.0)
        roles["profit_column"] = derived
        roles["_profit_derivation_note"] = f"Profit was estimated as {sales_col} × {margin_pct_col}."
        roles["_profit_is_estimated"] = True

    # Import lazily to avoid a circular import: missing_value_treatment uses
    # safe_numeric/format helpers from this module.
    if not roles.get("_missing_treatment_applied"):
        from src.missing_value_treatment import apply_missing_value_treatment

        work, roles = apply_missing_value_treatment(work, roles)

    return work, roles


def infer_has_return_context(df: pd.DataFrame) -> bool:
    """Detect whether negative sales/quantity likely represent returns/refunds.

    This is conservative: it looks for column names or values that explicitly
    mention return/refund/credit/cancel, or a common retail pattern where sales
    and quantity are both negative on some rows.
    """
    if df is None or df.empty:
        return False
    text_cols = [str(c).lower() for c in df.columns]
    return_words = ("return", "refund", "credit", "cancel", "rma", "debit note", "credit note")
    if any(any(w in c for w in return_words) for c in text_cols):
        return True
    for col in df.select_dtypes(exclude="number").columns[:20]:
        vals = df[col].dropna().astype(str).str.lower().head(500)
        if vals.str.contains("return|refund|credit|cancel|rma", regex=True).any():
            return True
    return False


def safe_correlation(left: pd.Series | Any, right: pd.Series | Any) -> float | None:
    """Return Pearson correlation safely, or None when correlation is undefined.

    NumPy/Pandas emit RuntimeWarning when one side has zero variance. In sales
    datasets this commonly happens for small filtered slices, constant discounts,
    or synthetic tests where every record has the same sales value. Undefined
    correlations should be skipped, not shown or warned about.
    """
    a = safe_numeric(left)
    b = safe_numeric(right)
    aligned = pd.concat([a, b], axis=1).dropna()
    if aligned.shape[0] < 3:
        return None
    if aligned.iloc[:, 0].nunique(dropna=True) < 2 or aligned.iloc[:, 1].nunique(dropna=True) < 2:
        return None
    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    if corr is None or pd.isna(corr) or not np.isfinite(corr):
        return None
    return float(corr)


def safe_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build a numeric correlation matrix after removing constant numeric columns."""
    if df is None or df.empty:
        return pd.DataFrame()
    numeric = df.select_dtypes(include="number").copy()
    if numeric.empty:
        return pd.DataFrame()
    usable = [c for c in numeric.columns if numeric[c].dropna().nunique() >= 2]
    if len(usable) < 2:
        return pd.DataFrame()
    return numeric[usable].corr(numeric_only=True)

def metric_summary_text(kpis: dict) -> str:
    """Generate a compact metric summary."""
    return (
        f"Total sales: {format_currency(kpis.get('total_sales'))}; "
        f"total profit: {format_currency(kpis.get('total_profit'))}; "
        f"margin: {format_percent(kpis.get('profit_margin_percent'))}."
    )
