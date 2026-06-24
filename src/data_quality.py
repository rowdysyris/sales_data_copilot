from __future__ import annotations

import pandas as pd

from src.utils import ensure_datetime, get_col, iqr_outlier_mask, safe_numeric, infer_has_return_context


BUSINESS_METRIC_ROLES = {
    "sales_column",
    "revenue_column",
    "profit_column",
    "cost_column",
    "quantity_column",
    "discount_column",
}

DATE_NAME_HINTS = ("date", "datetime", "timestamp", "time", "month", "year", "quarter", "period", "day")
DATE_PARSE_THRESHOLD_WITH_NAME_HINT = 0.60
DATE_PARSE_THRESHOLD_WITHOUT_NAME_HINT = 0.85


def _has_date_name_hint(column_name: str) -> bool:
    normalized = str(column_name).lower().replace("_", " ").replace("-", " ")
    return any(hint in normalized for hint in DATE_NAME_HINTS)


def _looks_like_datetime_column(series: pd.Series, column_name: str) -> bool:
    """Detect real-world date columns even when pandas loaded them as text.

    CSV uploads usually load values such as ``Order Date`` and ``Ship Date`` as
    object/string columns.  If we only check pandas dtype, clear date columns get
    mislabeled as categorical.  This helper uses the column name plus parse success
    rate to identify date columns without accidentally converting normal text.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    non_null = series.dropna()
    if non_null.empty:
        return False

    name_hint = _has_date_name_hint(column_name)

    # Numeric columns are usually metrics/IDs. Only accept compact numeric dates
    # such as 20240131 when the name also clearly indicates a date.
    if pd.api.types.is_numeric_dtype(non_null):
        if not name_hint:
            return False
        as_text = pd.to_numeric(non_null, errors="coerce").dropna().astype("Int64").astype(str)
        if as_text.empty or (as_text.str.len() == 8).mean() < 0.70:
            return False
        parsed = pd.to_datetime(as_text, format="%Y%m%d", errors="coerce")
        return parsed.notna().mean() >= 0.70

    text = non_null.astype(str).str.strip()
    if text.empty:
        return False

    # Avoid treating mostly numeric text such as sales/quantity as dates unless the
    # column name itself says it is a date/time field.
    numeric_ratio = pd.to_numeric(text.str.replace(",", "", regex=False), errors="coerce").notna().mean()
    if numeric_ratio >= 0.80 and not name_hint:
        return False

    parsed = ensure_datetime(text)
    parse_rate = parsed.notna().mean()
    threshold = DATE_PARSE_THRESHOLD_WITH_NAME_HINT if name_hint else DATE_PARSE_THRESHOLD_WITHOUT_NAME_HINT
    return parse_rate >= threshold


def _detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    detected: list[str] = []
    for col in df.columns:
        try:
            if _looks_like_datetime_column(df[col], str(col)):
                detected.append(col)
        except Exception:
            # Data-quality profiling should never fail because one column has
            # unusual mixed values. Leave that column as non-datetime.
            continue
    return detected


def _role_lookup(column_roles: dict) -> dict:
    roles = column_roles.get("roles", column_roles) if isinstance(column_roles, dict) else {}
    return {v: k for k, v in roles.items() if isinstance(v, str) and v}


def analyze_data_quality(df: pd.DataFrame, column_roles: dict) -> dict:
    """Analyze technical data quality separately from business-performance signals.

    Negative profit and loss-making records are valid business outcomes in sales data.
    They are returned under ``business_signals`` and should be handled by Profit
    Intelligence, not treated as dirty data.
    """
    roles = column_roles.get("roles", column_roles) if isinstance(column_roles, dict) else {}
    role_by_column = _role_lookup(roles)
    missing_meta = roles.get("_missing_value_treatment", {}) if isinstance(roles, dict) else {}

    total_cells = max(df.shape[0] * df.shape[1], 1)
    row_count = max(len(df), 1)
    cleaned_missing_total = int(df.isna().sum().sum())
    missing_total = int(missing_meta.get("original_missing_cells_total", cleaned_missing_total))
    duplicates = int(df.duplicated().sum())
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    numeric_cols = list(df.select_dtypes(include="number").columns)
    datetime_cols = _detect_datetime_columns(df)
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]

    column_profiles = {}
    warnings: list[str] = []
    business_signals: list[str] = []
    outlier_summary = []
    business_outlier_summary = []

    business_metric_columns = {roles.get(role) for role in BUSINESS_METRIC_ROLES if roles.get(role)}

    if missing_meta.get("cleaned_dataset_used"):
        warnings.append(missing_meta.get("manager_warning", "Some values were missing in the original dataset. Calculations are based on a cleaned dataset."))

    for c in df.columns:
        miss = int(df[c].isna().sum())
        unique = int(df[c].nunique(dropna=True))
        issues = []
        if miss:
            issues.append("missing values")
        if unique <= 1:
            issues.append("constant or near-empty column")
        if c in numeric_cols:
            mask = iqr_outlier_mask(df[c])
            count = int(mask.sum())
            if count:
                issues.append(f"{count} IQR outliers")
                row = {"column": c, "outlier_count": count, "outlier_percent": count / row_count * 100}
                if c in business_metric_columns:
                    business_outlier_summary.append(row)
                else:
                    outlier_summary.append(row)
        column_profiles[c] = {
            "dtype": str(df[c].dtype),
            "missing_count": miss,
            "missing_percent": miss / row_count * 100,
            "unique_count": unique,
            "unique_percent": unique / row_count * 100,
            "sample_values": [str(x) for x in df[c].dropna().head(5).tolist()],
            "detected_issues": issues,
        }

    sales_col = get_col(roles, "sales_column")
    profit_col = get_col(roles, "profit_column")
    qty_col = get_col(roles, "quantity_column")
    discount_col = get_col(roles, "discount_column")
    cost_col = get_col(roles, "cost_column")
    date_col = get_col(roles, "date_column")

    numeric_checks = {}
    suspicious_negative_penalty_count = 0
    has_return_context = infer_has_return_context(df)
    for role, col in {
        "sales": sales_col,
        "profit": profit_col,
        "quantity": qty_col,
        "discount": discount_col,
        "cost": cost_col,
    }.items():
        if col and col in df.columns:
            s = safe_numeric(df[col])
            negative_count = int((s < 0).sum())
            numeric_checks[role] = {
                "negative_count": negative_count,
                "zero_count": int((s == 0).sum()),
                "min": float(s.min()) if s.notna().any() else None,
                "max": float(s.max()) if s.notna().any() else None,
                "mean": float(s.mean()) if s.notna().any() else None,
                "median": float(s.median()) if s.notna().any() else None,
                "outlier_count": int(iqr_outlier_mask(s).sum()),
            }

            if role in {"sales", "quantity"} and negative_count:
                if has_return_context:
                    business_signals.append(f"{col} has {negative_count} negative values. These likely represent returns/refunds/credits and should be analyzed separately from normal sales.")
                else:
                    warnings.append(f"{col} has {negative_count} negative values. Verify whether these represent returns/credits or data errors.")
                    suspicious_negative_penalty_count += negative_count
            elif role in {"discount", "cost"} and negative_count:
                warnings.append(f"{col} has {negative_count} negative values. Verify whether these represent credits or data errors.")
                suspicious_negative_penalty_count += negative_count
            elif role == "profit" and negative_count:
                business_signals.append(
                    f"{col} has {negative_count} negative values. These are loss-making records and are analyzed in Profit Intelligence, not treated as data-quality errors."
                )

    date_checks = {}
    if date_col and date_col in df.columns:
        parsed = ensure_datetime(df[date_col])
        invalid = int(parsed.isna().sum() - df[date_col].isna().sum())
        date_checks = {
            "invalid_dates": max(invalid, 0),
            "min_date": str(parsed.min().date()) if parsed.notna().any() else None,
            "max_date": str(parsed.max().date()) if parsed.notna().any() else None,
            "date_range_days": int((parsed.max() - parsed.min()).days) if parsed.notna().sum() >= 2 else None,
        }
        if invalid > 0:
            warnings.append(f"{date_col} has {invalid} invalid date values.")

    missing_pct = missing_total / total_cells * 100
    dup_pct = duplicates / row_count * 100
    technical_outlier_pct = sum(x["outlier_count"] for x in outlier_summary) / row_count * 100
    suspicious_negative_pct = suspicious_negative_penalty_count / row_count * 100

    score = 100.0
    score -= min(35, missing_pct * 1.5)
    score -= min(20, dup_pct * 2)
    score -= min(15, len(empty_cols) * 5)
    score -= min(10, len(constant_cols) * 1.5)
    score -= min(8, technical_outlier_pct * 0.4)
    score -= min(10, suspicious_negative_pct * 1.0)
    if date_checks.get("invalid_dates"):
        score -= min(15, date_checks["invalid_dates"] / row_count * 100)
    score = max(0, min(100, round(score, 2)))

    if not warnings and missing_total == 0 and duplicates == 0 and not empty_cols:
        warnings.append("No major technical data-quality issues detected.")

    return {
        "basic_checks": {
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "duplicate_rows_count": duplicates,
            "duplicate_rows_percent": dup_pct,
            "missing_cells_total": missing_total,
            "missing_cells_percent": missing_pct,
            "cleaned_missing_cells_total": cleaned_missing_total,
            "cleaned_missing_cells_percent": cleaned_missing_total / total_cells * 100,
            "columns_with_missing_values": missing_meta.get("columns_with_missing_original", [c for c in df.columns if df[c].isna().any()]),
            "columns_with_missing_after_treatment": [c for c in df.columns if df[c].isna().any()],
            "empty_columns": empty_cols,
            "constant_columns": constant_cols,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "datetime_columns": datetime_cols,
        },
        "column_profiles": column_profiles,
        "numeric_checks": numeric_checks,
        "date_checks": date_checks,
        "outlier_summary": pd.DataFrame(outlier_summary),
        "business_outlier_summary": pd.DataFrame(business_outlier_summary),
        "quality_score": score,
        "warnings": warnings,
        "business_signals": business_signals,
        "missing_value_treatment": missing_meta,
        "score_explanation": {
            "missing_penalty_basis_percent": missing_pct,
            "duplicate_penalty_basis_percent": dup_pct,
            "technical_outlier_penalty_basis_percent": technical_outlier_pct,
            "suspicious_negative_penalty_basis_percent": suspicious_negative_pct,
            "note": "Negative profit is treated as a business signal, not a data-quality penalty. Negative sales/quantity are not penalized when return/refund context is detected.",
        },
    }
