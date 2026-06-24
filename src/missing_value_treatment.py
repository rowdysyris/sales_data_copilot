from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from src.utils import safe_numeric, normalize_percent_series, iqr_outlier_mask


CORE_NUMERIC_ROLES = {
    "sales_column",
    "revenue_column",
    "profit_column",
    "cost_column",
    "quantity_column",
    "discount_column",
    "unit_price_column",
    "margin_percent_column",
}

CATEGORICAL_ROLE_LABELS = {
    "category_column": "Unknown Category",
    "subcategory_column": "Unknown Sub-Category",
    "product_column": "Unknown Product",
    "customer_column": "Unknown Customer",
    "customer_id_column": "Unknown Customer ID",
    "region_column": "Unknown Region",
    "segment_column": "Unknown Segment",
    "city_column": "Unknown City",
    "state_column": "Unknown State",
    "country_column": "Unknown Country",
}

ID_ROLES = {"order_id_column", "customer_id_column"}
DATE_ROLES = {"date_column", "ship_date_column"}


def _flat_roles(column_roles: dict | None) -> dict:
    if not isinstance(column_roles, dict):
        return {}
    return dict(column_roles.get("roles", column_roles))


def _role_col(roles: dict, role: str) -> str | None:
    col = roles.get(role)
    return col if isinstance(col, str) and col else None


def _add_log(
    rows: list[dict[str, Any]],
    *,
    column: str,
    role: str,
    missing_before: int,
    treatment: str,
    values_fixed: int = 0,
    values_derived: int = 0,
    values_filled: int = 0,
    values_excluded: int = 0,
    values_still_missing: int = 0,
    impact: str = "",
    method_value: Any = None,
) -> None:
    if missing_before <= 0:
        return
    rows.append(
        {
            "column": column,
            "role": role.replace("_column", "").replace("_", " ").title(),
            "missing_before": int(missing_before),
            "treatment": treatment,
            "values_fixed": int(values_fixed),
            "values_derived": int(values_derived),
            "values_filled": int(values_filled),
            "values_excluded_or_left_missing": int(values_excluded),
            "values_still_missing": int(values_still_missing),
            "method_value": method_value,
            "business_impact": impact,
        }
    )


def _missing_count(df: pd.DataFrame, col: str | None) -> int:
    if not col or col not in df.columns:
        return 0
    return int(df[col].isna().sum())


def _is_skewed_or_outlier_heavy(s: pd.Series) -> bool:
    values = safe_numeric(s).dropna()
    if len(values) < 6:
        return True
    mean = values.mean()
    median = values.median()
    std = values.std()
    if std and not pd.isna(std) and abs(mean - median) > std * 0.5:
        return True
    try:
        return bool(iqr_outlier_mask(values).sum() > 0)
    except Exception:
        return True


def _fill_numeric_with_stat(work: pd.DataFrame, col: str, role: str, log_rows: list[dict[str, Any]], *, prefer_median: bool = False) -> None:
    if col not in work.columns:
        return
    missing_before = _missing_count(work, col)
    if missing_before <= 0:
        work[col] = safe_numeric(work[col])
        return
    s = safe_numeric(work[col])
    if s.dropna().empty:
        _add_log(
            log_rows,
            column=col,
            role=role,
            missing_before=missing_before,
            treatment="left missing because no valid numeric values were available for imputation",
            values_excluded=missing_before,
            values_still_missing=missing_before,
            impact="This column remains incomplete and affected calculations should be treated as limited.",
        )
        work[col] = s
        return
    use_median = prefer_median or _is_skewed_or_outlier_heavy(s)
    fill_value = float(s.median() if use_median else s.mean())
    work[col] = s.fillna(fill_value)
    method = "median" if use_median else "mean"
    _add_log(
        log_rows,
        column=col,
        role=role,
        missing_before=missing_before,
        treatment=f"filled missing numeric values using {method}",
        values_fixed=missing_before,
        values_filled=missing_before,
        values_still_missing=0,
        method_value=round(fill_value, 4),
        impact=f"{col} was completed for analysis using a transparent {method}-imputation rule.",
    )


def _looks_like_small_or_zero_discount(s: pd.Series) -> bool:
    vals = normalize_percent_series(s).dropna()
    if vals.empty:
        return False
    zero_share = (vals == 0).mean()
    return bool(zero_share >= 0.25 or vals.median() <= 5)


def apply_missing_value_treatment(df: pd.DataFrame, column_roles: dict) -> tuple[pd.DataFrame, dict]:
    """Clean missing values with business-aware rules and return metadata in roles.

    The function is conservative for core financial KPIs: it derives Sales/Profit
    from business formulas when possible, but it does not silently mean-fill core
    revenue/profit values. Every action is recorded in ``_missing_value_treatment``.
    """
    work = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    roles = _flat_roles(column_roles)

    if work.empty or roles.get("_missing_treatment_applied"):
        return work, roles

    original_missing = work.isna().sum()
    original_missing_total = int(original_missing.sum())
    original_columns_with_missing = [str(c) for c in work.columns if int(original_missing.get(c, 0)) > 0]
    log_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    # Normalize core numeric columns first. Missing entries remain NaN until rules below run.
    for role in CORE_NUMERIC_ROLES:
        col = _role_col(roles, role)
        if col in work.columns:
            if role in {"discount_column", "margin_percent_column"}:
                work[col] = normalize_percent_series(work[col])
            else:
                work[col] = safe_numeric(work[col])

    sales_col = _role_col(roles, "sales_column")
    profit_col = _role_col(roles, "profit_column")
    cost_col = _role_col(roles, "cost_column")
    qty_col = _role_col(roles, "quantity_column")
    unit_price_col = _role_col(roles, "unit_price_column")
    margin_pct_col = _role_col(roles, "margin_percent_column")
    discount_col = _role_col(roles, "discount_column")

    # 1) Sales: derive from Quantity * Unit Price when possible. Otherwise do not fabricate revenue.
    if sales_col in work.columns:
        missing_before = _missing_count(work, sales_col)
        derived_count = 0
        if missing_before and qty_col in work.columns and unit_price_col in work.columns:
            mask = work[sales_col].isna()
            derived_values = safe_numeric(work[qty_col]) * safe_numeric(work[unit_price_col])
            can_derive = mask & derived_values.notna()
            work.loc[can_derive, sales_col] = derived_values.loc[can_derive]
            derived_count = int(can_derive.sum())
        still_missing = _missing_count(work, sales_col)
        if missing_before:
            if derived_count:
                treatment = f"derived missing Sales using {qty_col} × {unit_price_col}; unresolved Sales remained missing"
            else:
                treatment = "left missing because Sales is a core KPI and no safe derivation formula was available"
            _add_log(
                log_rows,
                column=sales_col,
                role="sales_column",
                missing_before=missing_before,
                treatment=treatment,
                values_fixed=derived_count,
                values_derived=derived_count,
                values_excluded=still_missing,
                values_still_missing=still_missing,
                impact="Unresolved Sales values are excluded/ignored by affected Sales KPIs instead of being replaced by fake average revenue.",
            )
            if still_missing:
                warnings.append(f"{sales_col}: {still_missing} missing Sales values remain unresolved and can affect Sales totals.")

    # 2) Profit: derive from Sales - Cost, then Sales × Margin %. Do not mean-fill accounting profit.
    if profit_col in work.columns:
        missing_before = _missing_count(work, profit_col)
        derived_count = 0
        if missing_before and sales_col in work.columns and cost_col in work.columns:
            mask = work[profit_col].isna()
            derived_values = safe_numeric(work[sales_col]) - safe_numeric(work[cost_col])
            can_derive = mask & derived_values.notna()
            work.loc[can_derive, profit_col] = derived_values.loc[can_derive]
            derived_count += int(can_derive.sum())
        if _missing_count(work, profit_col) and sales_col in work.columns and margin_pct_col in work.columns:
            mask = work[profit_col].isna()
            margin = normalize_percent_series(work[margin_pct_col]) / 100.0
            derived_values = safe_numeric(work[sales_col]) * margin
            can_derive = mask & derived_values.notna()
            work.loc[can_derive, profit_col] = derived_values.loc[can_derive]
            derived_count += int(can_derive.sum())
        still_missing = _missing_count(work, profit_col)
        if missing_before:
            _add_log(
                log_rows,
                column=profit_col,
                role="profit_column",
                missing_before=missing_before,
                treatment="derived missing Profit from Sales - Cost and/or Sales × Margin % where possible; unresolved Profit left missing",
                values_fixed=derived_count,
                values_derived=derived_count,
                values_excluded=still_missing,
                values_still_missing=still_missing,
                impact="Profit analysis uses derived profit where possible and avoids average-filled profit that could distort loss drivers.",
            )
            if still_missing:
                warnings.append(f"{profit_col}: {still_missing} missing Profit values remain unresolved and can affect profitability analysis.")

    # 3) Cost: derive from Sales - Profit where possible, then small-gap median fill.
    if cost_col in work.columns:
        missing_before = _missing_count(work, cost_col)
        derived_count = 0
        if missing_before and sales_col in work.columns and profit_col in work.columns:
            mask = work[cost_col].isna()
            derived_values = safe_numeric(work[sales_col]) - safe_numeric(work[profit_col])
            can_derive = mask & derived_values.notna()
            work.loc[can_derive, cost_col] = derived_values.loc[can_derive]
            derived_count = int(can_derive.sum())
        still_missing = _missing_count(work, cost_col)
        if still_missing and still_missing / max(len(work), 1) <= 0.05:
            s = safe_numeric(work[cost_col])
            fill_value = float(s.median()) if s.dropna().any() else np.nan
            if not pd.isna(fill_value):
                work[cost_col] = s.fillna(fill_value)
                filled = still_missing
                still_missing = 0
            else:
                filled = 0
        else:
            filled = 0
        if missing_before:
            treatment = "derived missing Cost from Sales - Profit where possible"
            if filled:
                treatment += "; remaining small-gap missing Cost filled with median"
            _add_log(
                log_rows,
                column=cost_col,
                role="cost_column",
                missing_before=missing_before,
                treatment=treatment if derived_count or filled else "left missing because no reliable cost derivation/imputation was available",
                values_fixed=derived_count + filled,
                values_derived=derived_count,
                values_filled=filled,
                values_excluded=still_missing,
                values_still_missing=still_missing,
                impact="Cost was completed only where a business formula or small-gap median rule was safe.",
            )

    # 4) Quantity: derive from Sales / Unit Price, otherwise median fill.
    if qty_col in work.columns:
        missing_before = _missing_count(work, qty_col)
        derived_count = 0
        if missing_before and sales_col in work.columns and unit_price_col in work.columns:
            mask = work[qty_col].isna()
            unit_price = safe_numeric(work[unit_price_col]).replace(0, np.nan)
            derived_values = safe_numeric(work[sales_col]) / unit_price
            can_derive = mask & derived_values.notna()
            work.loc[can_derive, qty_col] = derived_values.loc[can_derive]
            derived_count = int(can_derive.sum())
        if _missing_count(work, qty_col):
            before_fill = _missing_count(work, qty_col)
            s = safe_numeric(work[qty_col])
            fill_value = float(s.median()) if s.dropna().any() else np.nan
            if not pd.isna(fill_value):
                work[qty_col] = s.fillna(fill_value)
                filled = before_fill
            else:
                filled = 0
        else:
            filled = 0
        still_missing = _missing_count(work, qty_col)
        if missing_before:
            _add_log(
                log_rows,
                column=qty_col,
                role="quantity_column",
                missing_before=missing_before,
                treatment="derived missing Quantity from Sales / Unit Price where possible, then filled remaining Quantity with median",
                values_fixed=derived_count + filled,
                values_derived=derived_count,
                values_filled=filled,
                values_excluded=still_missing,
                values_still_missing=still_missing,
                method_value=float(safe_numeric(work[qty_col]).median()) if safe_numeric(work[qty_col]).notna().any() else None,
                impact="Quantity was completed for volume/order analysis using formula-first, median-second logic.",
            )

    # 5) Discount: fill nulls with 0 only when data pattern supports no-discount; else median.
    if discount_col in work.columns:
        missing_before = _missing_count(work, discount_col)
        if missing_before:
            s = normalize_percent_series(work[discount_col])
            if _looks_like_small_or_zero_discount(s):
                fill_value = 0.0
                treatment = "filled missing Discount with 0 because the column pattern suggests blank discounts mean no discount"
            elif s.dropna().any():
                fill_value = float(s.median())
                treatment = "filled missing Discount with median because blank discount meaning was unclear"
            else:
                fill_value = np.nan
                treatment = "left missing because no valid Discount values were available"
            if not pd.isna(fill_value):
                work[discount_col] = s.fillna(fill_value)
                filled = missing_before
                still_missing = 0
            else:
                work[discount_col] = s
                filled = 0
                still_missing = missing_before
            _add_log(
                log_rows,
                column=discount_col,
                role="discount_column",
                missing_before=missing_before,
                treatment=treatment,
                values_fixed=filled,
                values_filled=filled,
                values_excluded=still_missing,
                values_still_missing=still_missing,
                method_value=fill_value if not pd.isna(fill_value) else None,
                impact="Discount impact analysis uses a declared treatment rather than silently ignoring blank discounts.",
            )

    # 6) Categorical dimensions: use explicit Unknown buckets so BI charts keep the records.
    for role, fill_label in CATEGORICAL_ROLE_LABELS.items():
        col = _role_col(roles, role)
        if col in work.columns:
            missing_before = _missing_count(work, col)
            if missing_before:
                work[col] = work[col].fillna(fill_label)
                _add_log(
                    log_rows,
                    column=col,
                    role=role,
                    missing_before=missing_before,
                    treatment=f"filled missing category labels as '{fill_label}'",
                    values_fixed=missing_before,
                    values_filled=missing_before,
                    values_still_missing=0,
                    impact="Rows remain visible in grouping, drilldown, and BI dashboard tables under an explicit Unknown bucket.",
                )

    # 7) Dates: do not fabricate; leave missing and explain that time analyses exclude them.
    for role in DATE_ROLES:
        col = _role_col(roles, role)
        if col in work.columns:
            missing_before = int(work[col].isna().sum())
            if missing_before:
                _add_log(
                    log_rows,
                    column=col,
                    role=role,
                    missing_before=missing_before,
                    treatment="not filled; missing dates are excluded from forecasting and time-series analysis",
                    values_excluded=missing_before,
                    values_still_missing=missing_before,
                    impact="Overall KPIs can still use these rows, but trend/forecast calculations ignore records without valid dates.",
                )
                warnings.append(f"{col}: {missing_before} missing dates remain and are excluded from time-based analysis.")

    # 8) Other numeric columns: mean/median imputation only for non-ID, non-core numeric fields.
    role_by_col = {}
    for k, v in roles.items():
        if isinstance(v, str) and not k.startswith("_"):
            role_by_col.setdefault(v, k)
    protected_cols = {c for c, r in role_by_col.items() if r in CORE_NUMERIC_ROLES or r in ID_ROLES or r in DATE_ROLES or r in CATEGORICAL_ROLE_LABELS}
    for col in list(work.columns):
        if col in protected_cols:
            continue
        if int(original_missing.get(col, 0)) <= 0:
            continue
        if pd.api.types.is_numeric_dtype(work[col]) or safe_numeric(work[col]).notna().mean() >= 0.75:
            _fill_numeric_with_stat(work, col, role_by_col.get(col, "other_numeric_column"), log_rows)

    remaining_missing_total = int(work.isna().sum().sum())
    cleaned_used = original_missing_total > 0 and (len(log_rows) > 0)
    summary_text = "No missing values required treatment."
    if cleaned_used:
        fixed = sum(int(r.get("values_fixed", 0)) for r in log_rows)
        remaining = sum(int(r.get("values_still_missing", 0)) for r in log_rows)
        summary_text = (
            f"Cleaned dataset used. The original dataset had {original_missing_total} missing cells across "
            f"{len(original_columns_with_missing)} columns. {fixed} values were filled or derived; "
            f"{remaining} values remain missing for calculations where safe treatment was not possible."
        )

    roles["_missing_treatment_applied"] = True
    roles["_missing_value_treatment"] = {
        "cleaned_dataset_used": bool(cleaned_used),
        "original_missing_cells_total": original_missing_total,
        "remaining_missing_cells_total": remaining_missing_total,
        "columns_with_missing_original": original_columns_with_missing,
        "treatment_log": log_rows,
        "warnings": warnings,
        "summary_text": summary_text,
        "manager_warning": (
            "Some values were missing in the original dataset. Calculations below are based on a cleaned dataset. "
            "Review the Missing Value Treatment Log before making business decisions."
            if cleaned_used
            else "No missing-value treatment was required."
        ),
    }
    return work, roles
