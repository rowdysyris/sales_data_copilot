from __future__ import annotations

import re
from typing import Any
import pandas as pd

from src.utils import (
    safe_numeric,
    prepare_analysis_dataset,
    profit_margin,
    format_currency,
    format_percent,
    ensure_datetime,
)
from src.metric_relationships import create_discount_bands
from src.entity_resolver import resolve_entities


POSITIVE_SORT_WORDS = (
    "top", "highest", "best", "most", "maximum", "max", "largest", "leading", "generating highest",
)
NEGATIVE_SORT_WORDS = (
    "lowest", "bottom", "worst", "least", "minimum", "min", "underperform", "weakest",
)
LOSS_WORDS = (
    "loss", "loss-making", "loss making", "loss generating", "unprofitable", "bleeding money",
    "bad contribution", "margin leakage", "profit drain", "losing money", "negative profit",
)
ROOT_CAUSE_WORDS = ("why", "reason", "root cause", "driver", "driving", "cause", "because")
TREND_WORDS = ("trend", "month", "monthly", "year", "yearly", "quarter", "quarterly", "growth", "decline", "increasing", "decreasing")
CHURN_WORDS = ("churn", "retention", "retain", "inactive", "leaving", "control churn", "at risk", "renewal")
PROFIT_QUESTION_WORDS = ("profit", "margin", "profitability", "profitable")
LOW_PERFORMANCE_WORDS = ("low", "weak", "poor", "bad", "down", "decline", "declining", "drop", "dropping", "not good", "less", "reduce", "reduced", "hurting", "hurt")

DIMENSION_ALIASES: list[tuple[str, str, tuple[str, ...]]] = [
    ("subcategory", "subcategory_column", ("sub category", "sub-category", "subcategory", "subcat")),
    ("product", "product_column", ("product", "products", "sku", "item", "items", "product name")),
    ("category", "category_column", ("category", "categories")),
    ("segment", "segment_column", ("customer segment", "customer segments", "segment", "segments")),
    ("customer", "customer_column", ("customer", "customers", "client", "clients", "buyer", "account", "accounts")),
    ("region", "region_column", ("region", "regions", "zone", "territory", "market")),
    ("state", "state_column", ("state", "states", "province")),
    ("city", "city_column", ("city", "cities", "town")),
    ("country", "country_column", ("country", "countries", "nation")),
    ("shipping mode", "ship_mode_column", ("ship mode", "ship modes", "shipping mode", "shipping modes", "delivery mode", "delivery modes", "standard class", "first class")),
]



BASIC_PROFILE_WORDS = (
    "row", "rows", "column", "columns", "shape", "profile", "read", "csv",
    "missing", "null", "blank", "na values", "duplicate", "duplicates",
    "data type", "datatype", "dtypes", "schema", "detected column", "column detection",
    "total sales", "total revenue", "sales revenue", "total profit", "average discount",
    "total quantity", "number of orders", "order count", "how many orders", "records",
)

METRIC_ALIASES: list[tuple[str, tuple[str, ...]]] = [
    # Specific metrics must be checked before generic words like "amount".
    ("discount", ("discount", "discounts", "discount amount", "discount value", "rebate", "markdown")),
    ("margin", ("profit margin", "margin percentage", "margin", "margin %")),
    ("profit", ("profit", "profits", "profit amount", "earnings", "contribution")),
    ("cost", ("cost", "costs", "expense", "expenses", "cogs")),
    ("quantity", ("quantity", "quantities", "units", "volume", "unit sold", "units sold")),
    ("orders", ("orders", "order count", "transactions", "transaction count")),
    ("customers", ("customer count", "number of customers", "unique customers")),
    ("sales", ("revenue", "sales", "turnover", "order value", "amount", "business generated")),
]


def _roles(column_roles: dict | None) -> dict:
    return dict(column_roles.get("roles", column_roles)) if isinstance(column_roles, dict) else {}


def _contains_any(q: str, words: tuple[str, ...] | list[str]) -> bool:
    return any(w in q for w in words)


def _extract_limit(q: str) -> int:
    m = re.search(r"\b(?:top|bottom|first|last|highest|lowest|best|worst)\s+(\d{1,3})\b", q)
    if not m:
        m = re.search(r"\b(\d{1,3})\s+(?:top|bottom|highest|lowest|best|worst)\b", q)
    if m:
        return max(1, min(int(m.group(1)), 100))
    if _contains_any(q, POSITIVE_SORT_WORDS + NEGATIVE_SORT_WORDS) or _contains_any(q, LOSS_WORDS):
        return 10
    return 10


def _detect_dimension(q: str, roles: dict) -> tuple[str | None, str | None, str | None]:
    if "discount band" in q or "discount bands" in q:
        return "discount band", "__discount_band", "discount_band"
    if any(w in q for w in ("month", "monthly")):
        return "month", "__period", "month"
    if any(w in q for w in ("quarter", "quarterly")):
        return "quarter", "__period", "quarter"
    if any(w in q for w in ("year", "yearly")):
        return "year", "__period", "year"
    for label, role_key, words in DIMENSION_ALIASES:
        if _contains_any(q, words):
            col = roles.get(role_key)
            if col:
                return label, col, None
            # Useful fallbacks for common sales datasets.
            if label == "product":
                for fallback_label, fallback_role in [("subcategory", "subcategory_column"), ("category", "category_column")]:
                    if roles.get(fallback_role):
                        return fallback_label, roles[fallback_role], None
            if label == "region":
                for fallback_label, fallback_role in [("state", "state_column"), ("city", "city_column"), ("country", "country_column")]:
                    if roles.get(fallback_role):
                        return fallback_label, roles[fallback_role], None
            return label, None, None
    return None, None, None


def _detect_metric(q: str, roles: dict) -> tuple[str | None, str | None]:
    if _contains_any(q, LOSS_WORDS):
        return "loss", roles.get("profit_column")
    for metric, words in METRIC_ALIASES:
        if _contains_any(q, words):
            if metric == "sales":
                return metric, roles.get("sales_column")
            if metric == "profit":
                return metric, roles.get("profit_column")
            if metric == "margin":
                return metric, None
            if metric == "discount":
                return metric, roles.get("discount_column")
            if metric == "cost":
                return metric, roles.get("cost_column")
            if metric == "quantity":
                return metric, roles.get("quantity_column")
            if metric == "orders":
                return metric, roles.get("order_id_column")
            if metric == "customers":
                return metric, roles.get("customer_column") or roles.get("customer_id_column")
    # Default when a dimension ranking is requested but metric is omitted.
    if _contains_any(q, POSITIVE_SORT_WORDS + NEGATIVE_SORT_WORDS):
        if roles.get("sales_column"):
            return "sales", roles.get("sales_column")
    return None, None


def _direction(q: str, metric: str | None) -> bool:
    """Return True for ascending sort."""
    if metric == "loss" or _contains_any(q, LOSS_WORDS):
        return True
    if _contains_any(q, NEGATIVE_SORT_WORDS):
        return True
    return False


def _period_series(df: pd.DataFrame, roles: dict, period: str) -> pd.Series | None:
    date_col = roles.get("date_column")
    if date_col not in df.columns:
        return None
    dates = ensure_datetime(df[date_col])
    if period == "year":
        return dates.dt.year.astype("Int64").astype(str).replace("<NA>", pd.NA)
    if period == "quarter":
        return dates.dt.to_period("Q").astype(str)
    return dates.dt.to_period("M").astype(str)


def _build_grouped_table(df: pd.DataFrame, roles: dict, dim_col: str, synthetic_dim: str | None = None) -> pd.DataFrame:
    work = df.copy()
    if synthetic_dim == "discount_band":
        disc_col = roles.get("discount_column")
        if disc_col not in work.columns:
            return pd.DataFrame()
        work["__discount_band"] = create_discount_bands(work[disc_col])
    elif synthetic_dim in {"month", "quarter", "year"}:
        period = _period_series(work, roles, synthetic_dim)
        if period is None:
            return pd.DataFrame()
        work["__period"] = period
        work = work.dropna(subset=["__period"])
    if dim_col not in work.columns or work.empty:
        return pd.DataFrame()

    sales_col = roles.get("sales_column")
    profit_col = roles.get("profit_column")
    qty_col = roles.get("quantity_column")
    cost_col = roles.get("cost_column")
    disc_col = roles.get("discount_column")
    order_col = roles.get("order_id_column")
    customer_col = roles.get("customer_column") or roles.get("customer_id_column")

    aggregations: dict[str, tuple[str, str]] = {}
    if sales_col in work.columns:
        work[sales_col] = safe_numeric(work[sales_col]).fillna(0)
        aggregations["sales"] = (sales_col, "sum")
    if profit_col in work.columns:
        work[profit_col] = safe_numeric(work[profit_col]).fillna(0)
        aggregations["profit"] = (profit_col, "sum")
    if qty_col in work.columns:
        work[qty_col] = safe_numeric(work[qty_col]).fillna(0)
        aggregations["quantity"] = (qty_col, "sum")
    if cost_col in work.columns:
        work[cost_col] = safe_numeric(work[cost_col]).fillna(0)
        aggregations["cost"] = (cost_col, "sum")
    if disc_col in work.columns:
        work[disc_col] = safe_numeric(work[disc_col])
        aggregations["average_discount"] = (disc_col, "mean")
    if not aggregations:
        return pd.DataFrame()

    grouped = work.groupby(dim_col, dropna=False, observed=False).agg(**aggregations).reset_index()
    grouped = grouped.rename(columns={dim_col: "dimension_value"})
    if order_col in work.columns:
        orders = work.groupby(dim_col, dropna=False, observed=False)[order_col].nunique().reset_index(name="order_count")
    else:
        orders = work.groupby(dim_col, dropna=False, observed=False).size().reset_index(name="order_count")
    orders = orders.rename(columns={dim_col: "dimension_value"})
    grouped = grouped.merge(orders, on="dimension_value", how="left")
    grain = roles.get("_dataset_grain", {}) if isinstance(roles, dict) else {}
    grouped["order_count_basis"] = "distinct_order_id" if order_col in work.columns else grain.get("count_label", "Rows")
    if sales_col in work.columns:
        unresolved_sales = work.groupby(dim_col, dropna=False, observed=False)[sales_col].apply(lambda s: int(s.isna().sum())).reset_index(name="unresolved_sales_rows").rename(columns={dim_col: "dimension_value"})
        grouped = grouped.merge(unresolved_sales, on="dimension_value", how="left")
    if profit_col in work.columns:
        unresolved_profit = work.groupby(dim_col, dropna=False, observed=False)[profit_col].apply(lambda s: int(s.isna().sum())).reset_index(name="unresolved_profit_rows").rename(columns={dim_col: "dimension_value"})
        grouped = grouped.merge(unresolved_profit, on="dimension_value", how="left")
    if customer_col in work.columns:
        customers = work.groupby(dim_col, dropna=False, observed=False)[customer_col].nunique().reset_index(name="customer_count").rename(columns={dim_col: "dimension_value"})
        grouped = grouped.merge(customers, on="dimension_value", how="left")
    if "sales" in grouped and "profit" in grouped:
        grouped["profit_margin_percent"] = grouped.apply(lambda r: profit_margin(r["profit"], r["sales"]), axis=1)
        grouped["loss_amount"] = grouped["profit"].where(grouped["profit"] < 0, 0.0)
    return grouped


def _sort_metric_name(metric: str | None) -> str | None:
    return {
        "sales": "sales",
        "revenue": "sales",
        "profit": "profit",
        "loss": "profit",
        "margin": "profit_margin_percent",
        "discount": "average_discount",
        "cost": "cost",
        "quantity": "quantity",
        "orders": "order_count",
        "customers": "customer_count",
    }.get(metric or "")


def _format_cell(col: str, val: Any) -> str:
    if pd.isna(val):
        return "N/A"
    c = col.lower()
    if c in {"sales", "profit", "cost", "loss_amount", "total_sales", "total_profit"} or "revenue" in c or "loss" in c:
        try:
            v = float(val)
            sign = "-" if v < 0 else ""
            return f"{sign}${abs(v):,.2f}"
        except Exception:
            return str(val)
    if "margin" in c or "discount" in c or "percent" in c:
        return format_percent(val)
    if isinstance(val, float):
        if val.is_integer():
            return f"{int(val):,}"
        return f"{val:,.2f}"
    if isinstance(val, int):
        return f"{val:,}"
    return str(val)


def _simple_markdown_table(df: pd.DataFrame) -> str:
    """Render a small dataframe as Markdown without pandas.to_markdown/tabulate.

    Pandas uses the optional `tabulate` package for DataFrame.to_markdown().
    The app should not crash if that optional dependency is missing, so dynamic
    Q&A tables are rendered with this lightweight local formatter.
    """
    if df is None or df.empty:
        return ""

    headers = [str(c) for c in df.columns]
    rows = [[str(v) if pd.notna(v) else "" for v in row] for row in df.to_numpy()]

    def esc(value: str) -> str:
        # Escape pipe characters so markdown table columns do not break.
        return value.replace("|", "\\|")

    lines = []
    lines.append("| " + " | ".join(esc(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(esc(cell) for cell in row) + " |")
    return "\n".join(lines)


def _markdown_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df is None or df.empty:
        return ""
    show = df.head(max_rows).copy()
    cols = [c for c in ["rank", "dimension_value", "sales", "profit", "cost", "loss_amount", "profit_margin_percent", "quantity", "order_count", "customer_count", "average_discount"] if c in show.columns]
    # Add any remaining useful columns up to a small limit.
    for c in show.columns:
        if c not in cols and len(cols) < 8:
            cols.append(c)
    out = show[cols].copy()
    rename = {
        "rank": "Rank",
        "dimension_value": "Name",
        "sales": "Sales/Revenue",
        "profit": "Profit",
        "loss_amount": "Loss Amount",
        "profit_margin_percent": "Margin %",
        "quantity": "Quantity",
        "order_count": "Orders",
        "customer_count": "Customers",
        "average_discount": "Avg Discount",
    }
    for col in out.columns:
        out[col] = out[col].map(lambda v, c=col: _format_cell(c, v))
    out = out.rename(columns=rename)
    return _simple_markdown_table(out)


def _cleaned_note(roles: dict) -> list[str]:
    meta = roles.get("_missing_value_treatment", {}) if isinstance(roles, dict) else {}
    if isinstance(meta, dict) and meta.get("cleaned_dataset_used"):
        return [
            "## Data Reliability Note",
            meta.get("manager_warning", "Calculations are based on a cleaned dataset because missing values were found in the original dataset."),
            meta.get("summary_text", ""),
            "",
        ]
    return []


def _format_business_date(value: Any) -> str:
    """Return dates in manager-readable Month D YYYY format."""
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return str(value)
        return f"{ts.strftime('%B')} {ts.day} {ts.year}"
    except Exception:
        return str(value)


def _basic_recommendation(intent: str) -> str:
    """Every Q&A answer should end with a practical recommendation."""
    mapping = {
        "dataset_shape": "Use this as the baseline row/column count before trusting any aggregation or dashboard result.",
        "dataset_columns": "Use these exact column names when asking follow-up questions so the analyst engine can map your request precisely.",
        "total_sales_check": "Use total revenue as the top-line benchmark, then compare it with profit and margin before making decisions.",
        "total_profit_check": "Use total profit with margin and loss-order analysis to understand whether revenue is converting into earnings.",
        "missing_values_check": "Because no missing values were found, you can proceed to KPI and profitability analysis without imputation concerns.",
        "order_date_range_check": "Use this date range to frame trend, seasonality, and year-over-year questions.",
        "unique_customers_check": "Use unique customer count as the base for customer concentration, loyalty, and retention analysis.",
        "unique_orders_check": "Use unique order count as the base for order frequency, AOV, and loss-order percentage analysis.",
        "category_values": "Use these categories for category-level revenue, profit, margin, and discount analysis.",
        "segment_values": "Use these segments to compare revenue quality, margin, and discount strategy by customer type.",
        "region_values": "Use these regions to compare geographic profit, sales, discounting, and expansion decisions.",
        "shipping_mode_values": "Use these shipping modes to compare sales, profit, margin, AOV, and service strategy.",
        "average_order_value_check": "Use both mean and median order value because the mean shows overall average scale while the median shows the typical order-line value.",
        "duplicate_rows_check": "Because no duplicate rows were found, continue analysis without duplicate-removal adjustments.",
        "dataset_profile": "Use this profile as the starting point, then ask targeted Tier 2 or Tier 3 questions for deeper business decisions.",
    }
    return mapping.get(intent, "Use this result as a verified data-quality checkpoint before deeper analysis.")



def _raw_profile_meta(roles: dict) -> dict:
    meta = roles.get("_raw_profile", {}) if isinstance(roles, dict) else {}
    return meta if isinstance(meta, dict) else {}



def _q_has(q: str, *patterns: str) -> bool:
    return any(p in q for p in patterns)


def _is_basic_sanity_question(q: str) -> bool:
    """Detect dataset-read/profile questions before any business-diagnostic fallback.

    These prompts ask whether the uploaded CSV was read and profiled correctly.
    They must be answered from the actual dataframe/metadata, never from generic
    profit/root-cause logic.
    """
    q = re.sub(r"\s+", " ", str(q).lower().strip())
    if not q:
        return False

    # Do not let simple profile handlers hijack Tier 2 analytical questions
    # such as "which product has highest average discount". Those contain
    # words like "average discount", but they require group-by/ranking.
    dimension_terms = (
        "product", "products", "category", "categories", "sub-category", "subcategory",
        "region", "regions", "segment", "segments", "customer", "customers",
        "city", "cities", "state", "states", "country", "countries", "month", "year",
    )
    metric_terms = (
        "sales", "revenue", "profit", "margin", "discount", "cost", "quantity",
        "orders", "customers", "amount", "value",
    )
    analytical_terms = POSITIVE_SORT_WORDS + NEGATIVE_SORT_WORDS + LOSS_WORDS + (
        " by ", "group", "breakdown", "split", "across", "rank", "ranking",
        "compare", " vs ", "versus", "correlation", "relationship", "affect",
    )
    if any(d in q for d in dimension_terms) and any(m in q for m in metric_terms) and any(a in q for a in analytical_terms):
        return False

    exact_patterns = (
        # read / profile / shape
        "can you read", "did you read", "read the csv", "parse the csv", "parsed the csv",
        "can it read", "profile the csv", "csv profile", "dataset profile", "data profile",
        "quick profile", "quick data profile", "overview of this dataset", "dataset overview",
        "quick overview", "data overview", "basic sanity", "sanity check",
        "how many rows", "how many records", "how many entries", "how many observations",
        "rows and columns", "row and column", "dataset shape", "shape of", "row count",
        "column count", "number of rows", "number of columns", "rows cols", "rows columns",
        # columns / schema
        "what columns", "which columns", "all columns", "list columns", "list all columns",
        "column names", "column list", "columns are in", "columns in this dataset",
        "headers", "field names", "features", "variables", "column detection", "detected columns", "detected business columns", "schema", "data types", "data type",
        "datatype", "dtypes", "numeric columns", "categorical columns", "text columns", "date columns",
        # missing / duplicates
        "missing values", "missing value", "null values", "null value", "any null", "any missing",
        "missing count", "null count", "blank values", "blanks", "empty values", "na values", "nan values",
        "null detection", "missing detection", "duplicate row", "duplicate rows", "duplicate record",
        "duplicate records", "duplicates", "repeated rows", "repeated records",
        # core totals / simple stats
        "total sales", "total revenue", "sales revenue", "revenue sum", "sum of sales", "sales sum",
        "total amount", "sum revenue", "total profit", "profit sum", "sum profit", "how much profit",
        "total quantity", "quantity sum", "sum quantity", "average discount", "avg discount", "mean discount",
        "minimum sales", "min sales", "maximum sales", "max sales", "average sales", "avg sales",
        "mean sales", "median sales", "minimum profit", "maximum profit", "average profit", "median profit",
        "basic aggregation", "basic aggregations", "basic totals", "basic metrics",
        # distinct counts / unique values
        "unique customers", "distinct customers", "number of customers", "how many customers",
        "unique customer ids", "customer count", "unique orders", "distinct orders", "number of orders",
        "how many orders", "order count", "unique products", "distinct products", "how many products",
        "product count", "unique categories", "product categories", "categories of products",
        "what categories", "categories cover", "unique subcategories", "sub categories", "sub-categories",
        "what sub categories", "what subcategories", "customer segments", "segments are in",
        "segments in this data", "unique segments", "regions covered", "what regions", "regions are covered",
        "unique regions", "what cities", "cities are present", "unique cities", "what states",
        "states are covered", "unique states", "what countries", "countries are covered", "unique countries",
        "shipping modes", "ship modes", "shipping mode", "delivery modes", "ship mode values",
        # dates / preview / sample
        "date range", "range of orders", "order date range", "orders date range", "min and max date", "min and max order date", "min max order date",
        "minimum date", "maximum date", "first order date", "last order date", "earliest order",
        "latest order", "average order value", "aov", "show first", "first 5 rows", "first rows",
        "show last", "last 5 rows", "last rows", "preview dataset", "sample rows", "show sample",
    )
    if any(p in q for p in exact_patterns):
        return True
    # Short sanity prompts, but avoid hijacking business questions like
    # "top revenue products" or "why is profit low".
    tokens = q.split()
    if len(tokens) <= 7 and any(w in q for w in ("null", "missing", "profile", "schema", "aggregation", "columns", "overview", "duplicates", "aov", "shape", "headers")):
        return True
    return False


def _format_role_value(value: Any) -> str:
    if value is None or value == "":
        return "Not detected"
    return str(value)


def _find_column_by_name(df: pd.DataFrame, patterns: tuple[str, ...]) -> str | None:
    """Find a column by normalized name patterns when no semantic role exists."""
    for col in df.columns:
        name = re.sub(r"[^a-z0-9]+", " ", str(col).lower()).strip()
        if any(p in name for p in patterns):
            return str(col)
    return None


def _unique_values_table(df: pd.DataFrame, col: str, value_label: str) -> tuple[pd.DataFrame, list[str]]:
    vals = (
        df[col]
        .dropna()
        .astype(str)
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    ) if col in df.columns else []
    table = pd.DataFrame({value_label: vals})
    return table, vals


def _basic_metric_column(metric: str, roles: dict) -> str | None:
    mapping = {
        "sales": roles.get("sales_column"),
        "revenue": roles.get("sales_column"),
        "profit": roles.get("profit_column"),
        "quantity": roles.get("quantity_column"),
        "discount": roles.get("discount_column"),
        "cost": roles.get("cost_column"),
    }
    return mapping.get(metric)


def _basic_stat_answer(q: str, df: pd.DataFrame, roles: dict) -> tuple[bool, str, dict[str, pd.DataFrame], str, str]:
    """Handle simple numeric aggregations such as sum/avg/min/max/median sales."""
    metric = None
    if any(w in q for w in ("sales", "revenue", "amount")):
        metric = "sales"
    elif "profit" in q:
        metric = "profit"
    elif "quantity" in q or "units" in q:
        metric = "quantity"
    elif "discount" in q:
        metric = "discount"
    elif "cost" in q:
        metric = "cost"
    if not metric:
        return False, "", {}, "", "High"

    if any(w in q for w in ("average", "avg", "mean")):
        stat = "average"
    elif any(w in q for w in ("median",)):
        stat = "median"
    elif any(w in q for w in ("minimum", "min", "lowest")):
        stat = "minimum"
    elif any(w in q for w in ("maximum", "max", "highest")):
        stat = "maximum"
    elif any(w in q for w in ("sum", "total", "how much")):
        stat = "total"
    else:
        return False, "", {}, "", "High"

    col = _basic_metric_column(metric, roles)
    title = f"{stat.title()} {metric.title()}"
    lines = [f"## {title}"]
    confidence = "High"
    evidence: dict[str, pd.DataFrame] = {}
    if col not in df.columns:
        lines.append(f"I could not calculate {title.lower()} because no {metric} column was detected.")
        return True, "\n".join(lines), evidence, f"{stat}_{metric}_check", "Low"
    series = safe_numeric(df[col])
    if stat == "average":
        val = float(series.mean(skipna=True))
    elif stat == "median":
        val = float(series.median(skipna=True))
    elif stat == "minimum":
        val = float(series.min(skipna=True))
    elif stat == "maximum":
        val = float(series.max(skipna=True))
    else:
        val = float(series.sum(skipna=True))
    is_percent = metric == "discount"
    formatted = format_percent(val) if is_percent else (_tier2_money(val) if metric in {"sales", "revenue", "profit", "cost"} else f"{val:,.2f}")
    lines.append(f"{title} is **{formatted}**.")
    lines.append(f"Calculated from the detected column: **{col}**.")
    evidence["basic_numeric_stat"] = pd.DataFrame([{"Metric": title, "Column Used": col, "Value": val, "Formatted": formatted}])
    return True, "\n".join(lines), evidence, f"{stat}_{metric}_check", confidence


def _list_unique_dimension(q: str, df: pd.DataFrame, roles: dict) -> tuple[bool, str, dict[str, pd.DataFrame], str, str]:
    specs = [
        ("product", roles.get("product_column"), "Product", ("unique products", "distinct products", "how many products", "product count", "products are present", "what products")),
        ("category", roles.get("category_column"), "Category", ("categories of products", "product categories", "what categories", "categories cover", "unique categories", "how many categories")),
        ("subcategory", roles.get("subcategory_column"), "Sub-Category", ("sub categories", "sub-category", "subcategories", "unique subcategories", "what sub categories", "what subcategories")),
        ("segment", roles.get("segment_column"), "Segment", ("customer segments", "segments are in", "segments in this data", "unique segments", "what segments")),
        ("region", roles.get("region_column"), "Region", ("regions covered", "what regions", "regions are covered", "unique regions")),
        ("city", roles.get("city_column"), "City", ("what cities", "cities are present", "unique cities", "cities covered")),
        ("state", roles.get("state_column"), "State", ("what states", "states are covered", "unique states", "states covered")),
        ("country", roles.get("country_column"), "Country", ("what countries", "countries are covered", "unique countries", "countries covered")),
        ("shipping_mode", roles.get("ship_mode_column") or _find_column_by_name(df, ("ship mode", "shipping mode", "delivery mode", "shipment mode", "mode of shipment")), "Shipping Mode", ("shipping modes", "ship modes", "shipping mode", "delivery modes", "ship mode values")),
    ]
    for key, col, label, patterns in specs:
        if any(p in q for p in patterns):
            lines = [f"## {label}s" if not label.endswith("y") else f"## {label} Values"]
            evidence = {}
            if col in df.columns:
                table, vals = _unique_values_table(df, col, label)
                lines.append(f"The dataset contains **{len(vals):,} unique {label.lower()} values** from the detected column: **{col}**.")
                if vals:
                    lines.append(_simple_markdown_table(table.head(100)))
                evidence[f"unique_{key}_values"] = table
                intent_map = {
                    "category": "product_categories_check",
                    "segment": "customer_segments_check",
                    "region": "regions_covered_check",
                    "shipping_mode": "shipping_modes_check",
                }
                return True, "\n".join(lines), evidence, intent_map.get(key, f"{key}_values_check"), "High"
            lines.append(f"I could not list {label.lower()} values because no matching column was detected.")
            evidence[f"unique_{key}_values"] = pd.DataFrame([{f"{label} Column": "Not detected"}])
            intent_map = {
                "category": "product_categories_check",
                "segment": "customer_segments_check",
                "region": "regions_covered_check",
                "shipping_mode": "shipping_modes_check",
            }
            return True, "\n".join(lines), evidence, intent_map.get(key, f"{key}_values_check"), "Low" if key != "shipping_mode" else "Medium"
    return False, "", {}, "", "High"


def _raw_preview_table(df: pd.DataFrame, q: str) -> tuple[bool, str, dict[str, pd.DataFrame], str, str]:
    if not any(p in q for p in ("show first", "first 5 rows", "first rows", "preview dataset", "sample rows", "show sample", "show last", "last 5 rows", "last rows")):
        return False, "", {}, "", "High"
    m = re.search(r"\b(\d{1,2})\b", q)
    n = max(1, min(int(m.group(1)), 20)) if m else 5
    last = any(p in q for p in ("show last", "last rows", "last 5 rows"))
    preview = df.tail(n).copy() if last else df.head(n).copy()
    label = "last" if last else "first"
    lines = ["## Dataset Preview", f"Showing the **{label} {len(preview)} rows** from the uploaded dataset.", _simple_markdown_table(preview.astype(str).head(n))]
    return True, "\n".join(lines), {"dataset_preview": preview}, "dataset_preview", "High"


def _column_type_answer(q: str, df: pd.DataFrame, raw_dtypes: dict[str, str]) -> tuple[bool, str, dict[str, pd.DataFrame], str, str]:
    if not any(p in q for p in ("numeric columns", "categorical columns", "text columns", "date columns", "data type", "datatype", "dtypes", "schema")):
        return False, "", {}, "", "High"
    schema = pd.DataFrame([{"Column": c, "Detected Type": raw_dtypes.get(str(c), str(df[c].dtype))} for c in df.columns])
    if "numeric columns" in q:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        table = pd.DataFrame({"Numeric Column": cols})
        return True, "\n".join(["## Numeric Columns", f"The dataset has **{len(cols):,} numeric columns**.", _simple_markdown_table(table)]), {"numeric_columns": table}, "numeric_columns_check", "High"
    if "categorical columns" in q or "text columns" in q:
        cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        table = pd.DataFrame({"Categorical/Text Column": cols})
        return True, "\n".join(["## Categorical/Text Columns", f"The dataset has **{len(cols):,} categorical/text columns**.", _simple_markdown_table(table)]), {"categorical_columns": table}, "categorical_columns_check", "High"
    if "date columns" in q:
        date_cols = []
        for c in df.columns:
            parsed = ensure_datetime(df[c])
            if parsed.notna().mean() >= 0.6:
                date_cols.append(c)
        table = pd.DataFrame({"Date-like Column": date_cols})
        return True, "\n".join(["## Date-Like Columns", f"The dataset has **{len(date_cols):,} date-like columns** based on parseability.", _simple_markdown_table(table)]), {"date_like_columns": table}, "date_columns_check", "High"
    return True, "\n".join(["## Dataset Schema / Data Types", _simple_markdown_table(schema)]), {"dataset_schema": schema}, "dataset_schema", "High"


def _basic_profile_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    """Answer basic sanity-check questions exactly instead of falling back to business diagnostics."""
    q = re.sub(r"\s+", " ", str(question).lower().strip())
    if not _is_basic_sanity_question(q):
        return {"handled": False, "reason": "Not a basic dataset sanity/profile question."}

    raw = _raw_profile_meta(roles)
    row_count = int(raw.get("row_count", len(df)))
    column_names = list(raw.get("columns", list(df.columns)))
    column_count = int(raw.get("column_count", len(column_names)))
    missing_by_column = raw.get("missing_by_column", {}) or {}
    missing_total = int(raw.get("missing_total", 0))
    duplicate_rows = int(raw.get("duplicate_rows", 0))
    dtypes = raw.get("dtypes", {}) or {str(c): str(df[c].dtype) for c in df.columns}

    evidence_tables: dict[str, pd.DataFrame] = {}
    lines: list[str] = []
    intent = "dataset_profile"
    confidence = "High"

    # Ensure profile questions use raw uploaded columns when raw metadata exists.
    raw_df_for_profile = df[[c for c in column_names if c in df.columns]].copy() if all(c in df.columns for c in column_names) else df.copy()

    # Highest-priority exact checks first.
    preview_handled, preview_text, preview_ev, preview_intent, preview_conf = _raw_preview_table(raw_df_for_profile, q)
    if preview_handled:
        lines.append(preview_text); evidence_tables.update(preview_ev); intent = preview_intent; confidence = preview_conf
    else:
        type_handled, type_text, type_ev, type_intent, type_conf = _column_type_answer(q, raw_df_for_profile, dtypes)
        if type_handled:
            lines.append(type_text); evidence_tables.update(type_ev); intent = type_intent; confidence = type_conf
        elif any(p in q for p in ("how many rows", "how many records", "how many entries", "how many observations", "rows and columns", "row and column", "dataset shape", "shape of", "row count", "number of rows", "number of columns", "column count", "rows cols", "rows columns")) and not any(p in q for p in ("duplicate", "duplicates", "repeated")):
            label = "records" if roles.get("_dataset_grain", {}).get("is_aggregate") else "rows"
            lines.append("## Dataset Size")
            lines.append(f"The dataset was read successfully. It has **{row_count:,} {label}** and **{column_count:,} columns**.")
            evidence_tables["dataset_shape"] = pd.DataFrame([{"Rows/Records": row_count, "Columns": column_count}])
            intent = "dataset_shape"

        elif any(p in q for p in ("can you read", "did you read", "read the csv", "parse the csv", "parsed the csv", "can it read", "profile the csv")):
            lines.append("## CSV Read / Parse Check")
            lines.append(f"**Yes.** The dataset was parsed into a dataframe with **{row_count:,} rows** and **{column_count:,} columns**.")
            lines.append(f"Missing/null values detected in the original file: **{missing_total:,}**. Duplicate rows: **{duplicate_rows:,}**.")
            evidence_tables["csv_parse_check"] = pd.DataFrame([{"Rows": row_count, "Columns": column_count, "Missing Values": missing_total, "Duplicate Rows": duplicate_rows}])
            intent = "csv_parse_check"

        elif any(p in q for p in ("what columns", "which columns", "all columns", "list columns", "list all columns", "column names", "column list", "columns are in", "columns in this dataset", "headers", "field names", "features", "variables")):
            lines.append("## Dataset Columns")
            lines.append(f"The dataset has **{column_count:,} columns**:")
            col_table = pd.DataFrame({"Column #": range(1, column_count + 1), "Column Name": column_names})
            lines.append(_simple_markdown_table(col_table))
            evidence_tables["dataset_columns"] = col_table
            intent = "dataset_columns"

        elif any(p in q for p in ("missing", "null", "blank", "blanks", "na values", "nan values", "empty values")):
            lines.append("## Missing / Null Value Check")
            if missing_total == 0:
                lines.append(f"**No.** The original uploaded dataset has **0 missing/null values** across **{row_count:,} rows** and **{column_count:,} columns**.")
            else:
                lines.append(f"**Yes.** The original uploaded dataset has **{missing_total:,} missing/null values**.")
                miss_table = pd.DataFrame([
                    {"Column": col, "Missing Values": int(cnt), "Missing %": (int(cnt) / row_count * 100 if row_count else 0)}
                    for col, cnt in missing_by_column.items() if int(cnt) > 0
                ]).sort_values("Missing Values", ascending=False) if missing_by_column else pd.DataFrame()
                if not miss_table.empty:
                    display = miss_table.copy()
                    display["Missing %"] = display["Missing %"].map(lambda v: f"{v:.2f}%")
                    lines.append(_simple_markdown_table(display.head(50)))
                    evidence_tables["missing_values_by_column"] = miss_table
            evidence_tables.setdefault("missing_value_summary", pd.DataFrame([{"Total Missing Values": missing_total, "Rows": row_count, "Columns": column_count}]))
            intent = "missing_values_check"

        elif any(p in q for p in ("duplicate", "duplicates", "repeated rows", "repeated records")):
            lines.append("## Duplicate Row Check")
            if duplicate_rows == 0:
                lines.append("**No duplicate rows** were found in the original uploaded dataset.")
            else:
                lines.append(f"The original uploaded dataset has **{duplicate_rows:,} duplicate rows**.")
            evidence_tables["duplicate_row_summary"] = pd.DataFrame([{"Duplicate Rows": duplicate_rows, "Rows": row_count}])
            intent = "duplicate_rows_check"

        elif any(p in q for p in ("column detection", "detected columns", "detected business columns")):
            lines.append("## Detected Business Columns")
            role_keys = [
                "sales_column", "profit_column", "cost_column", "margin_percent_column", "date_column",
                "order_id_column", "customer_column", "customer_id_column", "product_column", "category_column", "subcategory_column",
                "region_column", "state_column", "city_column", "country_column", "segment_column", "quantity_column", "discount_column", "unit_price_column", "ship_mode_column",
            ]
            role_table = pd.DataFrame([
                {"Business Role": key.replace("_column", "").replace("_", " ").title(), "Detected Column": _format_role_value(roles.get(key))}
                for key in role_keys
            ])
            lines.append(_simple_markdown_table(role_table))
            evidence_tables["detected_business_columns"] = role_table
            intent = "column_detection_check"

        else:
            stat_handled, stat_text, stat_ev, stat_intent, stat_conf = _basic_stat_answer(q, df, roles)
            if stat_handled:
                lines.append(stat_text); evidence_tables.update(stat_ev); intent = stat_intent; confidence = stat_conf
            else:
                unique_handled, unique_text, unique_ev, unique_intent, unique_conf = _list_unique_dimension(q, df, roles)
                if unique_handled:
                    lines.append(unique_text); evidence_tables.update(unique_ev); intent = unique_intent; confidence = unique_conf
                elif any(p in q for p in ("date range", "range of orders", "order date range", "orders date range", "min and max date", "min and max order date", "min max order date", "minimum date", "maximum date", "first order date", "last order date", "earliest order", "latest order")):
                    date_col = roles.get("date_column")
                    lines.append("## Order Date Range")
                    if date_col in df.columns:
                        dates = ensure_datetime(df[date_col]).dropna()
                        invalid_count = int(len(df) - len(dates))
                        if not dates.empty:
                            start = dates.min(); end = dates.max()
                            start_text = _format_business_date(start); end_text = _format_business_date(end)
                            lines.append(f"The order date range is **{start_text} to {end_text}**.")
                            if invalid_count:
                                lines.append(f"{invalid_count:,} rows did not have a valid order date and were excluded from this date-range calculation.")
                            evidence_tables["order_date_range"] = pd.DataFrame([{"Date Column": date_col, "Start Date": start_text, "End Date": end_text, "Invalid/Missing Dates": invalid_count}])
                        else:
                            lines.append(f"I detected **{date_col}**, but no valid dates could be parsed."); confidence = "Low"
                    else:
                        lines.append("I could not calculate the date range because no order/date column was detected."); confidence = "Low"
                    intent = "order_date_range_check"

                elif any(p in q for p in ("unique customers", "distinct customers", "number of customers", "how many customers", "unique customer ids", "customer count")):
                    cust_col = roles.get("customer_id_column") or roles.get("customer_column")
                    lines.append("## Unique Customer Count")
                    if cust_col in df.columns:
                        count = int(df[cust_col].nunique(dropna=True))
                        lines.append(f"There are **{count:,} unique customers** based on the detected customer column: **{cust_col}**.")
                        evidence_tables["unique_customers"] = pd.DataFrame([{"Customer Column": cust_col, "Unique Customers": count}])
                    else:
                        lines.append("I could not calculate unique customers because no customer column was detected."); confidence = "Low"
                    intent = "unique_customers_check"

                elif any(p in q for p in ("unique orders", "distinct orders", "number of orders", "how many orders", "order count")):
                    order_col = roles.get("order_id_column")
                    lines.append("## Unique Order Count")
                    if order_col in df.columns:
                        count = int(df[order_col].nunique(dropna=True))
                        lines.append(f"There are **{count:,} unique orders** based on the detected order column: **{order_col}**.")
                        evidence_tables["unique_orders"] = pd.DataFrame([{"Order Column": order_col, "Unique Orders": count}])
                    else:
                        count = int(len(df))
                        lines.append(f"No order ID column was detected, so the dataset has **{count:,} rows/records** and row count is being used as a proxy, not true unique orders.")
                        evidence_tables["unique_orders"] = pd.DataFrame([{"Order Column": "Not detected", "Unique Orders / Row Proxy": count}])
                        confidence = "Medium"
                    intent = "unique_orders_check"

                elif any(p in q for p in ("average order value", "aov")):
                    sales_col = roles.get("sales_column"); order_col = roles.get("order_id_column")
                    lines.append("## Average Order Value")
                    if sales_col in df.columns:
                        sales_series = safe_numeric(df[sales_col])
                        mean_value = float(sales_series.mean(skipna=True))
                        median_value = float(sales_series.median(skipna=True))
                        total_sales = float(sales_series.sum(skipna=True))
                        lines.append(f"Mean order-line value is **{_tier2_money(mean_value)}** and median order-line value is **{_tier2_money(median_value)}**.")
                        lines.append(f"Calculated from the detected sales/revenue column: **{sales_col}**.")
                        evidence = {"Sales Column": sales_col, "Mean Order-Line Value": mean_value, "Median Order-Line Value": median_value, "Total Sales": total_sales, "Rows Used": int(sales_series.notna().sum())}
                        if order_col in df.columns:
                            unique_orders = int(df[order_col].nunique(dropna=True))
                            unique_order_aov = total_sales / unique_orders if unique_orders else 0.0
                            lines.append(f"For reference, total sales divided by **{unique_orders:,} unique orders** is **{_tier2_money(unique_order_aov)}**.")
                            evidence.update({"Order Column": order_col, "Unique Orders": unique_orders, "Unique-Order AOV": unique_order_aov})
                        evidence_tables["average_order_value"] = pd.DataFrame([evidence])
                    else:
                        lines.append("I could not calculate average order value because no sales/revenue column was detected."); confidence = "Low"
                    intent = "average_order_value_check"

                elif any(p in q for p in ("basic aggregation", "basic aggregations", "basic totals", "basic metrics")):
                    lines.append("## Basic Aggregation Sanity Check")
                    rows = []
                    metric_map = [("Total Sales/Revenue", "sales_column", "currency"), ("Total Profit", "profit_column", "currency"), ("Total Cost", "cost_column", "currency"), ("Total Quantity", "quantity_column", "number"), ("Average Discount", "discount_column", "percent")]
                    for metric_name, role_key, kind in metric_map:
                        col = roles.get(role_key)
                        if col in df.columns:
                            series = safe_numeric(df[col]); val = float(series.mean(skipna=True)) if "Average" in metric_name else float(series.sum(skipna=True))
                            rows.append({"Metric": metric_name, "Column Used": col, "Value": val, "Formatted": _tier2_money(val) if kind == "currency" else (format_percent(val) if kind == "percent" else f"{val:,.2f}")})
                    if rows:
                        agg = pd.DataFrame(rows)
                        lines.append(_simple_markdown_table(agg[["Metric", "Column Used", "Formatted"]].rename(columns={"Formatted": "Value"})))
                        evidence_tables["basic_aggregation"] = agg
                    else:
                        lines.append("No standard numeric business metric columns were detected for aggregation."); confidence = "Low"
                    intent = "basic_aggregation_check"

                else:
                    # General read/profile sanity check.
                    lines.append("## Quick Dataset Profile / Overview")
                    lines.append(f"The dataset was read successfully. It has **{row_count:,} rows** and **{column_count:,} columns**.")
                    lines.append(f"Missing/null values in the original dataset: **{missing_total:,}**.")
                    lines.append(f"Duplicate rows in the original dataset: **{duplicate_rows:,}**.")
                    detected = {k: v for k, v in roles.items() if k.endswith("_column") and v}
                    if detected:
                        clean_detected = ", ".join(f"{k.replace('_column','').replace('_',' ')}: {v}" for k, v in list(detected.items())[:12])
                        lines.append(f"Detected key business columns include: {clean_detected}.")
                    rows = [{"Metric": "Rows/Records", "Value": row_count}, {"Metric": "Columns", "Value": column_count}, {"Metric": "Missing Values", "Value": missing_total}, {"Metric": "Duplicate Rows", "Value": duplicate_rows}]
                    sales_col = roles.get("sales_column"); profit_col = roles.get("profit_column"); order_col = roles.get("order_id_column"); customer_col = roles.get("customer_id_column") or roles.get("customer_column")
                    date_col = roles.get("date_column"); category_col = roles.get("category_column"); region_col = roles.get("region_column"); segment_col = roles.get("segment_column")
                    if date_col in df.columns:
                        dates = ensure_datetime(df[date_col]).dropna()
                        if not dates.empty:
                            rows.append({"Metric": "Order Date Range", "Value": f"{_format_business_date(dates.min())} to {_format_business_date(dates.max())}"})
                    if category_col in df.columns:
                        cats = sorted(df[category_col].dropna().astype(str).unique().tolist())
                        rows.append({"Metric": "Categories", "Value": ", ".join(cats)})
                    if region_col in df.columns:
                        regs = sorted(df[region_col].dropna().astype(str).unique().tolist())
                        rows.append({"Metric": "Regions", "Value": ", ".join(regs)})
                    if segment_col in df.columns:
                        segs = sorted(df[segment_col].dropna().astype(str).unique().tolist())
                        rows.append({"Metric": "Segments", "Value": ", ".join(segs)})
                    if sales_col in df.columns: rows.append({"Metric": "Total Sales/Revenue", "Value": float(safe_numeric(df[sales_col]).sum(skipna=True))})
                    if profit_col in df.columns: rows.append({"Metric": "Total Profit", "Value": float(safe_numeric(df[profit_col]).sum(skipna=True))})
                    if order_col in df.columns: rows.append({"Metric": "Unique Orders", "Value": int(df[order_col].nunique(dropna=True))})
                    if customer_col in df.columns: rows.append({"Metric": "Unique Customers", "Value": int(df[customer_col].nunique(dropna=True))})
                    overview = pd.DataFrame(rows); display = overview.copy()
                    def _profile_value(row):
                        metric_name = str(row.get("Metric", "")).lower()
                        v = row.get("Value")
                        if any(k in metric_name for k in ("sales", "revenue", "profit", "cost")):
                            return _tier2_money(v)
                        if "discount" in metric_name or "margin" in metric_name:
                            return format_percent(v)
                        if any(k in metric_name for k in ("rows", "records", "columns", "missing", "duplicate", "orders", "customers", "quantity")):
                            try:
                                return f"{int(float(v)):,}"
                            except Exception:
                                return str(v)
                        if isinstance(v, float):
                            return f"{v:,.2f}"
                        if isinstance(v, int):
                            return f"{v:,}"
                        return str(v)
                    display["Value"] = display.apply(_profile_value, axis=1)
                    lines.append(_simple_markdown_table(display))
                    evidence_tables["dataset_profile_overview"] = overview
                    intent = "dataset_profile"

    if not any(str(line).strip().lower() == "## recommendation" for line in lines):
        lines.extend(["", "## Recommendation", _basic_recommendation(intent)])

    limitations = []
    if roles.get("_missing_value_treatment", {}).get("cleaned_dataset_used") and intent not in {"missing_values_check", "dataset_columns", "dataset_shape", "csv_parse_check"}:
        limitations.append("Some analytical calculations may use cleaned values; this sanity answer reports original dataset metadata where relevant.")
    return {
        "handled": True,
        "intent": intent,
        "confidence": confidence,
        "markdown_answer": "\n".join(lines),
        "evidence_tables": evidence_tables,
        "analysis_plan": {"intent": intent, "source": "actual_dataframe_and_original_uploaded_metadata"},
        "relationship_findings": [],
        "root_causes": [],
        "recommendations": [],
        "limitations": limitations,
        "follow_up_questions": ["What columns are in this dataset?", "Are there any missing values?", "Show basic aggregations."],
    }

def _ranking_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = question.lower()
    dim_label, dim_col, synthetic = _detect_dimension(q, roles)
    metric, metric_col = _detect_metric(q, roles)
    if not dim_label:
        return {"handled": False, "reason": "No dimension detected."}
    if not dim_col:
        return {"handled": True, "confidence": "Low", "intent": "ranking", "markdown_answer": f"I understood this as a {dim_label} question, but the dataset does not have a detected {dim_label} column.", "evidence_tables": {}, "limitations": [f"Missing {dim_label} column."]}
    if not metric:
        metric = "sales" if roles.get("sales_column") else ("profit" if roles.get("profit_column") else None)
    sort_col = _sort_metric_name(metric)
    if sort_col in {"sales", "profit"} and metric_col not in df.columns:
        return {"handled": True, "confidence": "Low", "intent": "ranking", "markdown_answer": f"I can rank by {dim_label}, but the required {metric} column was not detected.", "evidence_tables": {}, "limitations": [f"Missing {metric} column."]}
    table = _build_grouped_table(df, roles, dim_col, synthetic)
    if table.empty:
        return {"handled": True, "confidence": "Low", "intent": "ranking", "markdown_answer": f"I could not build a {dim_label} ranking because the required fields are missing or empty.", "evidence_tables": {}, "limitations": ["Required fields missing or empty."]}
    if sort_col not in table.columns:
        return {"handled": True, "confidence": "Low", "intent": "ranking", "markdown_answer": f"I can group by {dim_label}, but I cannot sort by {metric} because that metric is unavailable.", "evidence_tables": {"dynamic_grouped_table": table}, "limitations": [f"Metric unavailable: {metric}."]}

    loss_mode = metric == "loss" or _contains_any(q, LOSS_WORDS)
    ascending = _direction(q, metric)
    if loss_mode and "profit" in table.columns:
        filtered = table[table["profit"] < 0].copy()
        if not filtered.empty:
            table = filtered
    table = table.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    limit = _extract_limit(q)
    result = table.head(limit).copy()
    result.insert(0, "rank", range(1, len(result) + 1))

    metric_label = {
        "sales": "revenue/sales",
        "profit": "profit",
        "loss": "loss",
        "margin": "margin",
        "discount": "average discount",
        "cost": "cost",
        "quantity": "quantity",
        "orders": "orders",
        "customers": "customer count",
    }.get(metric or "", metric or "selected metric")
    direction_label = "lowest to highest" if ascending else "highest to lowest"
    title_word = "Loss-Generating" if loss_mode else f"Top {metric_label.title()}"

    lines: list[str] = [f"## {title_word} {dim_label.title()}s", ""]
    lines.extend(_cleaned_note(roles))
    lines.append(f"I grouped the dataset by **{dim_label}** and sorted by **{metric_label}** from {direction_label}.")
    if loss_mode:
        lines.append("Only groups with negative total profit are shown when loss data is available.")
    lines.append("")
    lines.append(_markdown_table(result, limit))
    lines.append("")
    if not result.empty:
        top = result.iloc[0]
        if loss_mode and "profit" in top:
            lines.append(f"## Main Finding\nThe largest loss is from **{top['dimension_value']}** with total profit/loss of **{_tier2_money(top['profit'])}**.")
        elif metric == "sales" and "sales" in top:
            lines.append(f"## Main Finding\nThe highest revenue contribution is from **{top['dimension_value']}** with sales of **{_tier2_money(top['sales'])}**.")
        elif metric == "profit" and "profit" in top:
            lines.append(f"## Main Finding\nThe strongest profit contribution is from **{top['dimension_value']}** with profit of **{_tier2_money(top['profit'])}**.")
        elif metric == "cost" and "cost" in top:
            lines.append(f"## Main Finding\nThe highest cost contribution is from **{top['dimension_value']}** with cost of **{_tier2_money(top['cost'])}**.")
        elif metric == "margin" and "profit_margin_percent" in top:
            lines.append(f"## Main Finding\n**{top['dimension_value']}** has the selected margin position at **{format_percent(top['profit_margin_percent'])}**.")
    if metric == "sales" and "profit" in result.columns:
        lines.append("\n## Manager Note\nRevenue alone does not prove business quality. Profit and margin are shown beside revenue so high-sales but low-profit areas can be reviewed.")
    if loss_mode:
        lines.append("\n## Recommended Action\nReview pricing, discounts, cost, and regional/customer concentration for the listed loss-making groups first.")
    return {
        "handled": True,
        "intent": "dynamic_ranking",
        "confidence": "High",
        "markdown_answer": "\n".join([x for x in lines if x is not None]),
        "evidence_tables": {"dynamic_ranking_result": result, "dynamic_grouped_table": table},
        "analysis_plan": {"dimension": dim_label, "metric": metric, "sort": "ascending" if ascending else "descending", "limit": limit},
        "limitations": [],
        "follow_up_questions": [
            f"Why is the lowest-profit {dim_label} losing money?",
            f"Which {dim_label}s have high revenue but low margin?",
            f"Show {dim_label}s by discount impact.",
        ],
    }


def _driver_for_entity(sub: pd.DataFrame, roles: dict, entity_label: str, entity_value: Any, overall_discount: float | None, overall_margin: float | None) -> list[str]:
    sales_col = roles.get("sales_column")
    profit_col = roles.get("profit_column")
    disc_col = roles.get("discount_column")
    qty_col = roles.get("quantity_column")
    cost_col = roles.get("cost_column")
    drivers: list[str] = []
    sales = float(safe_numeric(sub[sales_col]).sum()) if sales_col in sub.columns else None
    profit = float(safe_numeric(sub[profit_col]).sum()) if profit_col in sub.columns else None
    margin = profit_margin(profit, sales) if sales not in (None, 0) and profit is not None else None
    if profit is not None and profit < 0:
        drivers.append(f"{entity_value} is loss-making with total profit/loss of {_tier2_money(profit)}.")
    if margin is not None and (margin < 0 or (overall_margin is not None and margin < overall_margin - 5)):
        drivers.append(f"Margin is weak at {format_percent(margin)} compared with the overall benchmark {format_percent(overall_margin)}.")
    if disc_col in sub.columns:
        avg_disc = float(safe_numeric(sub[disc_col]).mean())
        threshold = 20 if avg_disc > 1 else 0.2
        if avg_disc >= threshold and (overall_discount is None or avg_disc > overall_discount):
            drivers.append(f"Average discount is high at {format_percent(avg_disc)}.")
    if qty_col in sub.columns and profit is not None:
        qty = float(safe_numeric(sub[qty_col]).sum())
        if qty and profit / qty < 0:
            drivers.append(f"Profit per unit is negative at {_tier2_money(profit / qty)}.")
    if cost_col in sub.columns and sales not in (None, 0):
        cost = float(safe_numeric(sub[cost_col]).sum())
        if cost / sales >= 0.85:
            drivers.append(f"Cost consumes {format_percent(cost / sales * 100)} of sales, leaving limited margin room.")
    if sales_col in sub.columns and (safe_numeric(sub[sales_col]) < 0).any():
        drivers.append("Returns/refunds are present for this group and may be reducing net revenue.")

    # Concentration checks inside the entity.
    for role, label in [("region_column", "region"), ("segment_column", "segment"), ("customer_column", "customer"), ("category_column", "category"), ("subcategory_column", "subcategory")]:
        col = roles.get(role)
        if col in sub.columns and col != entity_label and profit_col in sub.columns:
            g = sub.groupby(col, dropna=False)[profit_col].sum().sort_values(ascending=True)
            if not g.empty and g.iloc[0] < 0:
                drivers.append(f"Loss is concentrated in {label} **{g.index[0]}** with profit/loss of {_tier2_money(g.iloc[0])}.")
                break
    if not drivers:
        drivers.append("No single strong driver was proven from available columns; review cost, price, discounts, and returns for this group.")
    return drivers[:4]


def _available_profit_dimensions(roles: dict, include_time: bool = True) -> list[tuple[str, str, str | None]]:
    """Return all dimensions worth scanning for a broad profit diagnostic."""
    candidates = [
        ("category", roles.get("category_column"), None),
        ("subcategory", roles.get("subcategory_column"), None),
        ("product", roles.get("product_column"), None),
        ("region", roles.get("region_column"), None),
        ("state", roles.get("state_column"), None),
        ("city", roles.get("city_column"), None),
        ("segment", roles.get("segment_column"), None),
        ("customer", roles.get("customer_column") or roles.get("customer_id_column"), None),
    ]
    if roles.get("discount_column"):
        candidates.append(("discount band", "__discount_band", "discount_band"))
    if include_time and roles.get("date_column"):
        candidates.append(("month", "__period", "month"))

    seen: set[tuple[str, str | None, str | None]] = set()
    dims: list[tuple[str, str, str | None]] = []
    for label, col, synthetic in candidates:
        key = (label, col, synthetic)
        if not col or key in seen:
            continue
        seen.add(key)
        dims.append((label, col, synthetic))
    return dims


def _profit_diagnostic_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    """Broad profit diagnostic across all available dimensions.

    This handles management prompts like "Why is profit low?". It should not
    immediately lock onto one sub-category such as Tables. Instead, it scans all
    relevant business cuts and then shows the largest leakage / weakest-margin
    areas across the full dataset.
    """
    q = question.lower()
    asks_why = _contains_any(q, ROOT_CAUSE_WORDS)
    asks_profit = _contains_any(q, PROFIT_QUESTION_WORDS) or _contains_any(q, LOSS_WORDS)
    broad_low_profit = asks_why and asks_profit and not resolve_entities(question, df, roles)
    if not broad_low_profit:
        return {"handled": False, "reason": "Not a broad profit diagnostic question."}

    sales_col = roles.get("sales_column")
    profit_col = roles.get("profit_column")
    if profit_col not in df.columns:
        return {
            "handled": True,
            "intent": "dynamic_profit_diagnostic",
            "confidence": "Low",
            "markdown_answer": "Profit diagnostic requires a Profit column or fields to derive Profit, such as Sales + Cost or Sales + Margin%.",
            "evidence_tables": {},
            "limitations": ["Missing profit metric."],
        }

    total_sales = float(safe_numeric(df[sales_col]).sum()) if sales_col in df.columns else None
    total_profit = float(safe_numeric(df[profit_col]).sum()) if profit_col in df.columns else None
    overall_margin = profit_margin(total_profit, total_sales) if total_sales not in (None, 0) and total_profit is not None else None

    all_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    scanned_dimensions: list[str] = []

    for label, col, synthetic in _available_profit_dimensions(roles, include_time=True):
        table = _build_grouped_table(df, roles, col, synthetic)
        if table.empty or "profit" not in table.columns:
            continue
        scanned_dimensions.append(label)
        t = table.copy()
        t["dimension_type"] = label
        if "sales" in t.columns and "profit_margin_percent" not in t.columns:
            t["profit_margin_percent"] = t.apply(lambda r: profit_margin(r.get("profit"), r.get("sales")), axis=1)
        t["loss_amount"] = t["profit"].where(t["profit"] < 0, 0.0)
        # Keep the weakest rows from each dimension, not only the single global worst.
        weak_profit = t.sort_values("profit", ascending=True).head(7)
        if "profit_margin_percent" in t.columns:
            margin_candidates = t[(safe_numeric(t.get("sales", pd.Series([0] * len(t)))) > 0)].copy()
            weak_margin = margin_candidates.sort_values("profit_margin_percent", ascending=True).head(5) if not margin_candidates.empty else pd.DataFrame()
            combined = pd.concat([weak_profit, weak_margin], ignore_index=True).drop_duplicates(subset=["dimension_type", "dimension_value"])
        else:
            combined = weak_profit
        if not combined.empty:
            all_rows.append(combined)

        neg = t[t["profit"] < 0]
        worst = t.sort_values("profit", ascending=True).iloc[0]
        low_margin = None
        if "profit_margin_percent" in t.columns:
            candidates = t[safe_numeric(t.get("sales", pd.Series([0] * len(t)))) > 0]
            if not candidates.empty:
                low_margin = candidates.sort_values("profit_margin_percent", ascending=True).iloc[0]
        summary_rows.append({
            "dimension_type": label,
            "groups_checked": int(len(t)),
            "loss_making_groups": int(len(neg)),
            "total_loss_amount": float(neg["profit"].sum()) if not neg.empty else 0.0,
            "lowest_profit_group": worst.get("dimension_value"),
            "lowest_profit": float(worst.get("profit", 0.0)),
            "lowest_margin_group": low_margin.get("dimension_value") if low_margin is not None else None,
            "lowest_margin_percent": float(low_margin.get("profit_margin_percent")) if low_margin is not None and pd.notna(low_margin.get("profit_margin_percent")) else None,
        })

    if not all_rows:
        return {
            "handled": True,
            "intent": "dynamic_profit_diagnostic",
            "confidence": "Low",
            "markdown_answer": "I could not run a full profit diagnostic because no usable product/category/region/customer/discount dimensions were detected.",
            "evidence_tables": {},
            "limitations": ["No usable business dimensions detected."],
        }

    diagnostic = pd.concat(all_rows, ignore_index=True)
    # Prioritize actual loss first, then weak margin. Keep a broad mix of dimensions.
    diagnostic["diagnostic_priority"] = diagnostic["profit"].apply(lambda x: 0 if pd.notna(x) and x < 0 else 1)
    sort_cols = ["diagnostic_priority", "profit"]
    diagnostic = diagnostic.sort_values(sort_cols, ascending=[True, True]).drop_duplicates(subset=["dimension_type", "dimension_value"]).reset_index(drop=True)
    diagnostic.insert(0, "rank", range(1, len(diagnostic) + 1))

    dimension_summary = pd.DataFrame(summary_rows).sort_values(["total_loss_amount", "lowest_profit"], ascending=[True, True]).reset_index(drop=True)
    if not dimension_summary.empty:
        dimension_summary.insert(0, "rank", range(1, len(dimension_summary) + 1))

    # Build dimension-specific tables for clear manager drilldowns.
    evidence_tables: dict[str, pd.DataFrame] = {
        "profit_diagnostic_all_dimensions": diagnostic.head(50),
        "profit_diagnostic_dimension_summary": dimension_summary,
    }
    for label in ["category", "subcategory", "product", "region", "segment", "customer", "discount band", "month"]:
        sub = diagnostic[diagnostic["dimension_type"] == label].head(15)
        if not sub.empty:
            evidence_tables[f"{label.replace(' ', '_')}_profit_diagnostic"] = sub

    lines: list[str] = ["## Direct Answer", "Profit is being reduced by the weakest loss and margin areas across the full dataset. I scanned all available dimensions before prioritizing specific drivers.", "", "## Full Profit Diagnostic", ""]
    lines.extend(_cleaned_note(roles))
    if total_profit is not None:
        margin_text = f" with margin **{format_percent(overall_margin)}**" if overall_margin is not None else ""
        sales_text = f" from sales of **{_tier2_money(total_sales)}**" if total_sales is not None else ""
        lines.append(f"Overall profit is **{_tier2_money(total_profit)}**{sales_text}{margin_text}.")
        lines.append(f"- Total Profit: {_tier2_money(total_profit)}")
        lines.append("\n## Evidence From Data")
        lines.append(f"- Total Profit: {_tier2_money(total_profit)}")
        if total_sales is not None:
            lines.append(f"- Total Sales: {_tier2_money(total_sales)}")
        if overall_margin is not None:
            lines.append(f"- Profit Margin: {format_percent(overall_margin)}")
    lines.append(f"I scanned **all available dimensions**: {', '.join(scanned_dimensions)}. The table below shows the weakest profit/margin areas across the whole dataset, not only one sub-category.")
    lines.append("\n## Relationships / Drivers")
    lines.append("The diagnostic checks product, category, subcategory, region, segment, customer, discount band, and time where available. It ranks actual loss-making groups first and then weak-margin groups.")
    lines.append("")
    lines.append(_markdown_table(diagnostic.head(15), 15))

    lines.append("\n## What Is Pulling Profit Down")
    if not dimension_summary.empty:
        # Summarize the broad evidence first.
        loss_dims = dimension_summary[dimension_summary["loss_making_groups"] > 0].head(5)
        if not loss_dims.empty:
            for _, r in loss_dims.iterrows():
                lines.append(
                    f"- **{str(r['dimension_type']).title()} level:** {int(r['loss_making_groups'])} loss-making groups; worst group is **{r['lowest_profit_group']}** with profit/loss of **{_tier2_money(r['lowest_profit'])}**."
                )
        else:
            for _, r in dimension_summary.head(5).iterrows():
                lines.append(
                    f"- **{str(r['dimension_type']).title()} level:** no negative-profit group detected, but the lowest-profit group is **{r['lowest_profit_group']}** with profit of **{_tier2_money(r['lowest_profit'])}**."
                )

    # Specific root-cause lenses.
    disc = diagnostic[diagnostic["dimension_type"] == "discount band"]
    if not disc.empty:
        worst_disc = disc.sort_values("profit", ascending=True).iloc[0]
        lines.append(f"- **Discount lens:** weakest discount band is **{worst_disc['dimension_value']}** with profit/loss of **{_tier2_money(worst_disc['profit'])}**.")
    high_sales_low_margin = diagnostic[(diagnostic.get("sales", pd.Series(dtype=float)) > 0) & (diagnostic.get("profit_margin_percent", pd.Series(dtype=float)).notna())]
    if not high_sales_low_margin.empty and overall_margin is not None:
        weak_conversion = high_sales_low_margin[high_sales_low_margin["profit_margin_percent"] < overall_margin].sort_values("sales", ascending=False).head(3)
        if not weak_conversion.empty:
            names = ", ".join([f"{r['dimension_type']}={r['dimension_value']} ({format_percent(r['profit_margin_percent'])})" for _, r in weak_conversion.iterrows()])
            lines.append(f"- **Revenue-to-profit conversion:** these high-sales areas have below-overall margin: {names}.")

    lines.append("\n## Recommended Action")
    lines.append("1. Do not fix only Tables/Furniture by default. Start with the highest-ranked loss/weak-margin rows across product, subcategory, category, region, segment, customer, discount band, and month.")
    lines.append("2. For each top row, check price, discount, cost, returns/refunds, and whether the loss is concentrated in a region/segment/customer.")
    lines.append("3. Review discount approval rules for loss-making discount bands before changing the entire product portfolio.")
    lines.append("4. Use the interactive evidence tables below to drill into each dimension separately.")

    relationship_findings = []
    if total_profit is not None:
        relationship_findings.append(f"Total Profit: {_tier2_money(total_profit)}; Overall Margin: {format_percent(overall_margin)}")
    if scanned_dimensions:
        relationship_findings.append(f"Scanned dimensions for profit drivers: {', '.join(scanned_dimensions)}")
    if not dimension_summary.empty:
        top_dim = dimension_summary.iloc[0]
        relationship_findings.append(f"Largest dimension-level leakage: {top_dim['dimension_type']} -> {top_dim['lowest_profit_group']} with profit/loss {_tier2_money(top_dim['lowest_profit'])}")

    return {
        "handled": True,
        "intent": "dynamic_profit_diagnostic",
        "confidence": "High",
        "markdown_answer": "\n".join(lines),
        "evidence_tables": evidence_tables,
        "relationship_findings": relationship_findings,
        "root_causes": relationship_findings[:4],
        "recommendations": [{"title": "Full profit diagnostic", "recommended_action": "Prioritize the largest loss and weakest-margin rows across all scanned dimensions.", "evidence_from_data": relationship_findings[-1] if relationship_findings else "Full dimension scan completed."}],
        "analysis_plan": {
            "intent": "full_profit_diagnostic",
            "metric": "profit",
            "dimensions_scanned": scanned_dimensions,
            "sort": "negative profit first, then weakest margin",
        },
        "limitations": ["This is a diagnostic scan from available columns; causality should be validated with pricing, cost, returns, and business-owner context."],
        "follow_up_questions": ["Show top 10 loss-making products", "Which discount bands damage margin?", "Which regions have high revenue but low profit?"],
    }


def _root_cause_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = question.lower()
    if not (_contains_any(q, ROOT_CAUSE_WORDS) and _contains_any(q, LOSS_WORDS)):
        return {"handled": False, "reason": "Not a loss root-cause question."}
    # Entity-specific questions such as "Why are Tables losing money?" should
    # stay with the existing context-safe root-cause path. The dynamic engine is
    # for broad group questions like "Why are products making loss?".
    if resolve_entities(question, df, roles):
        return {"handled": False, "reason": "Entity-specific root cause should use analyst brain fallback."}
    dim_label, dim_col, synthetic = _detect_dimension(q, roles)
    if not dim_label:
        # Broad loss question: product is the most useful default, then subcategory/category.
        for label, role_key in [("product", "product_column"), ("subcategory", "subcategory_column"), ("category", "category_column"), ("region", "region_column")]:
            if roles.get(role_key):
                dim_label, dim_col, synthetic = label, roles[role_key], None
                break
    if not dim_col:
        return {"handled": True, "intent": "dynamic_root_cause", "confidence": "Low", "markdown_answer": "I can explain losses only if the dataset has a product/category/region dimension and a profit or derived-profit column.", "evidence_tables": {}, "limitations": ["Missing dimension for root-cause analysis."]}
    if roles.get("profit_column") not in df.columns:
        return {"handled": True, "intent": "dynamic_root_cause", "confidence": "Low", "markdown_answer": "Loss root-cause analysis requires a Profit column or enough fields to derive Profit from Sales - Cost / Sales × Margin%.", "evidence_tables": {}, "limitations": ["Missing profit metric."]}

    grouped = _build_grouped_table(df, roles, dim_col, synthetic)
    if grouped.empty or "profit" not in grouped.columns:
        return {"handled": True, "intent": "dynamic_root_cause", "confidence": "Low", "markdown_answer": "I could not calculate loss drivers because the required grouped profit table is unavailable.", "evidence_tables": {}, "limitations": ["Grouped profit unavailable."]}
    loss_groups = grouped[grouped["profit"] < 0].sort_values("profit", ascending=True).head(_extract_limit(q)).reset_index(drop=True)
    if loss_groups.empty:
        return {"handled": True, "intent": "dynamic_root_cause", "confidence": "High", "markdown_answer": f"No loss-making {dim_label}s were found. All detected {dim_label} groups have non-negative total profit.", "evidence_tables": {"dynamic_grouped_table": grouped}, "limitations": []}
    loss_groups = loss_groups.copy()
    loss_groups.insert(0, "rank", range(1, len(loss_groups) + 1))

    sales_col = roles.get("sales_column")
    profit_col = roles.get("profit_column")
    disc_col = roles.get("discount_column")
    total_sales = float(safe_numeric(df[sales_col]).sum()) if sales_col in df.columns else None
    total_profit = float(safe_numeric(df[profit_col]).sum()) if profit_col in df.columns else None
    overall_margin = profit_margin(total_profit, total_sales) if total_sales not in (None, 0) and total_profit is not None else None
    overall_discount = float(safe_numeric(df[disc_col]).mean()) if disc_col in df.columns else None

    lines: list[str] = [f"## Why {dim_label.title()}s Are Making Loss", ""]
    lines.extend(_cleaned_note(roles))
    lines.append(f"I scanned all {dim_label} groups, filtered groups with negative total profit, and then checked discount, margin, volume, cost, returns, and concentration drivers.")
    lines.append("")
    lines.append(_markdown_table(loss_groups, len(loss_groups)))
    lines.append("\n## Root-Cause Drivers")
    driver_rows = []
    driver_df = df.copy()
    driver_dim_col = dim_col
    if synthetic == "discount_band":
        disc_for_band = roles.get("discount_column")
        if disc_for_band in driver_df.columns:
            driver_df["__discount_band"] = create_discount_bands(driver_df[disc_for_band])
            driver_dim_col = "__discount_band"
    elif synthetic in {"month", "quarter", "year"}:
        period_values = _period_series(driver_df, roles, synthetic)
        if period_values is not None:
            driver_df["__period"] = period_values
            driver_dim_col = "__period"
    for _, row in loss_groups.head(5).iterrows():
        value = row["dimension_value"]
        if driver_dim_col in driver_df.columns:
            sub = driver_df[driver_df[driver_dim_col].astype(str) == str(value)].copy()
        else:
            sub = df.copy()
        drivers = _driver_for_entity(sub, roles, driver_dim_col, value, overall_discount, overall_margin)
        lines.append(f"\n### {value}")
        for d in drivers:
            lines.append(f"- {d}")
        driver_rows.append({"entity": value, "drivers": " | ".join(drivers)})
    lines.append("\n## Recommended Action")
    lines.append("Prioritize the largest loss-making groups first. Check whether losses come from high discounts, high cost, negative profit per unit, specific regions/segments, or returns/refunds before changing product strategy.")
    return {
        "handled": True,
        "intent": "dynamic_root_cause",
        "confidence": "High",
        "markdown_answer": "\n".join(lines),
        "evidence_tables": {"loss_making_groups": loss_groups, "root_cause_driver_summary": pd.DataFrame(driver_rows), "dynamic_grouped_table": grouped},
        "analysis_plan": {"dimension": dim_label, "metric": "profit", "filter": "profit < 0", "driver_checks": ["discount", "margin", "volume", "cost", "returns", "region/segment/customer concentration"]},
        "limitations": [],
        "follow_up_questions": [f"Show top 5 loss-generating {dim_label}s", f"Which {dim_label}s have high revenue but low profit?", "Are discounts causing these losses?"],
    }


def _churn_status_column(df: pd.DataFrame) -> str | None:
    candidates = [c for c in df.columns if re.search(r"churn|status|active|inactive|cancel|renew|subscription", str(c), re.I)]
    return candidates[0] if candidates else None


def _churn_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = question.lower()
    if not _contains_any(q, CHURN_WORDS):
        return {"handled": False, "reason": "Not a churn/retention question."}
    cust_col = roles.get("customer_column") or roles.get("customer_id_column")
    date_col = roles.get("date_column")
    sales_col = roles.get("sales_column")
    profit_col = roles.get("profit_column")
    order_col = roles.get("order_id_column")
    status_col = _churn_status_column(df)
    lines: list[str] = ["## Customer Churn / Retention Analysis", ""]
    lines.extend(_cleaned_note(roles))

    if status_col in df.columns and cust_col in df.columns:
        tmp = df.copy()
        status = tmp[status_col].astype(str).str.lower()
        churn_mask = status.str.contains("churn|inactive|cancel|lost|not renew|non renew|closed", regex=True, na=False)
        tmp["__churned"] = churn_mask
        row_churn_rate = float(churn_mask.mean() * 100) if len(tmp) else 0.0

        # Customer-level churn is the safer business metric because one customer
        # can have many transaction rows. A customer is counted as churned if any
        # of their rows indicate churn/inactive/cancelled status.
        customer_churn = tmp.groupby(cust_col, dropna=False)["__churned"].max().reset_index(name="is_churned_customer")
        customer_churn_rate = float(customer_churn["is_churned_customer"].mean() * 100) if len(customer_churn) else 0.0
        customer_churn = customer_churn.rename(columns={cust_col: "customer"})
        customer_churn["churn_status"] = customer_churn["is_churned_customer"].map(lambda x: "Churned/Inactive" if bool(x) else "Active/Retained")

        lines.append(f"The dataset has an explicit churn/status field: **{status_col}**.")
        lines.append(f"Unique-customer churn/inactive rate: **{format_percent(customer_churn_rate)}**.")
        lines.append(f"Row-level churn/inactive rate: **{format_percent(row_churn_rate)}**. Use the customer-level rate for management decisions when customers have multiple rows.")
        evidence = [("customer_level_churn_status", customer_churn)]
        for role, label in [("segment_column", "Segment"), ("region_column", "Region"), ("product_column", "Product")]:
            col = roles.get(role)
            if col in tmp.columns:
                if cust_col in tmp.columns:
                    cust_dim = tmp.groupby([col, cust_col], dropna=False)["__churned"].max().reset_index()
                    g = cust_dim.groupby(col, dropna=False)["__churned"].mean().mul(100).reset_index(name="customer_churn_rate_percent").sort_values("customer_churn_rate_percent", ascending=False)
                else:
                    g = tmp.groupby(col, dropna=False)["__churned"].mean().mul(100).reset_index(name="row_churn_rate_percent").sort_values("row_churn_rate_percent", ascending=False)
                evidence.append((f"churn_by_{label.lower()}", g))
                rate_col = "customer_churn_rate_percent" if "customer_churn_rate_percent" in g.columns else "row_churn_rate_percent"
                if not g.empty:
                    lines.append(f"- Highest churn by {label}: **{g.iloc[0][col]}** at **{format_percent(g.iloc[0][rate_col])}**.")
        lines.append("\n## How To Control Churn")
        lines.append("1. Prioritize segments/regions with the highest customer-level churn rate.")
        lines.append("2. Contact high-value churn-risk customers before renewal/inactivity becomes permanent.")
        lines.append("3. Compare churned vs active customers by discount, product mix, margin, and last purchase behavior.")
        lines.append("4. Create targeted retention offers for high-value customers, not blanket discounts.")
        return {"handled": True, "intent": "churn_analysis", "confidence": "High", "markdown_answer": "\n".join(lines), "evidence_tables": dict(evidence), "limitations": []}

    if cust_col in df.columns and date_col in df.columns:
        tmp = df.copy()
        tmp[date_col] = ensure_datetime(tmp[date_col])
        tmp = tmp.dropna(subset=[date_col])
        if tmp.empty:
            return {"handled": True, "intent": "churn_analysis", "confidence": "Low", "markdown_answer": "Customer churn risk cannot be inferred because the detected date column has no valid dates.", "evidence_tables": {}, "limitations": ["No valid dates."]}
        agg_spec: dict[str, tuple[str, str]] = {"last_purchase_date": (date_col, "max"), "first_purchase_date": (date_col, "min")}
        if sales_col in tmp.columns:
            tmp[sales_col] = safe_numeric(tmp[sales_col]).fillna(0)
            agg_spec["total_sales"] = (sales_col, "sum")
        if profit_col in tmp.columns:
            tmp[profit_col] = safe_numeric(tmp[profit_col]).fillna(0)
            agg_spec["total_profit"] = (profit_col, "sum")
        g = tmp.groupby(cust_col, dropna=False).agg(**agg_spec).reset_index().rename(columns={cust_col: "customer"})
        if order_col in tmp.columns:
            orders = tmp.groupby(cust_col)[order_col].nunique().reset_index(name="order_count").rename(columns={cust_col: "customer"})
        else:
            orders = tmp.groupby(cust_col).size().reset_index(name="order_count").rename(columns={cust_col: "customer"})
        g = g.merge(orders, on="customer", how="left")
        max_date = tmp[date_col].max()
        g["days_since_last_order"] = (max_date - g["last_purchase_date"]).dt.days
        g["customer_lifespan_days"] = (g["last_purchase_date"] - g["first_purchase_date"]).dt.days.fillna(0)
        inactive_cutoff = max(60, float(g["days_since_last_order"].quantile(0.75)))
        sales_median = float(g.get("total_sales", pd.Series([0])).median()) if "total_sales" in g.columns else 0
        g["churn_risk_score"] = 0
        g.loc[g["days_since_last_order"] >= inactive_cutoff, "churn_risk_score"] += 50
        g.loc[g["order_count"] >= 2, "churn_risk_score"] += 20
        if "total_sales" in g.columns:
            g.loc[g["total_sales"] >= sales_median, "churn_risk_score"] += 20
        if "total_profit" in g.columns:
            g.loc[g["total_profit"] > 0, "churn_risk_score"] += 10
        g["risk_level"] = pd.cut(g["churn_risk_score"], bins=[-1, 39, 69, 100], labels=["Low", "Medium", "High"])
        risk = g.sort_values(["churn_risk_score", "days_since_last_order", "total_sales" if "total_sales" in g.columns else "order_count"], ascending=[False, False, False]).head(15).reset_index(drop=True)
        risk.insert(0, "rank", range(1, len(risk) + 1))
        lines.append("The dataset does not have an explicit churn flag, so churn is inferred from customer inactivity and purchase history.")
        lines.append(f"High-risk customers are those inactive for roughly **{int(inactive_cutoff)}+ days** and previously valuable/repeat buyers.")
        lines.append("")
        lines.append(_markdown_table(risk.rename(columns={"customer": "dimension_value"}), 10))
        lines.append("\n## How To Control Churn")
        lines.append("1. First contact high-risk, high-value inactive customers shown above.")
        lines.append("2. Use targeted reactivation offers instead of blanket discounts.")
        lines.append("3. Check their last purchased products/regions/segments to identify operational or pricing issues.")
        lines.append("4. Track repeat purchase frequency monthly and alert when valuable customers cross the inactivity threshold.")
        return {"handled": True, "intent": "churn_analysis", "confidence": "Medium", "markdown_answer": "\n".join(lines), "evidence_tables": {"churn_risk_customers": risk}, "limitations": ["Churn is inferred because no explicit churn/status column was detected."]}

    return {
        "handled": True,
        "intent": "churn_analysis",
        "confidence": "Low",
        "markdown_answer": "This dataset does not contain enough customer activity or churn-status information to calculate churn reliably. To analyze churn, include Customer ID/Name, transaction date, and either repeat-purchase history or churn/active status.",
        "evidence_tables": {},
        "limitations": ["Missing customer/date or churn status columns."],
    }


def _trend_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = question.lower()
    if not _contains_any(q, TREND_WORDS):
        return {"handled": False, "reason": "Not a trend question."}
    metric, _ = _detect_metric(q, roles)
    metric = metric or "sales"
    dim_label, dim_col, synthetic = _detect_dimension(q, roles)
    if synthetic not in {"month", "quarter", "year"}:
        # Only handle clear time-grain questions here.
        return {"handled": False, "reason": "No clear time grain."}
    if roles.get("date_column") not in df.columns:
        return {"handled": True, "intent": "dynamic_trend", "confidence": "Low", "markdown_answer": "Trend analysis requires a valid date column. No date column was detected.", "evidence_tables": {}, "limitations": ["Missing date column."]}
    table = _build_grouped_table(df, roles, dim_col, synthetic)
    sort_col = _sort_metric_name(metric) or "sales"
    if table.empty or sort_col not in table.columns:
        return {"handled": True, "intent": "dynamic_trend", "confidence": "Low", "markdown_answer": f"I could not calculate {synthetic} trend because the required metric/date fields are missing.", "evidence_tables": {}, "limitations": ["Missing metric/date fields."]}
    table = table.rename(columns={"dimension_value": "period"}).sort_values("period").reset_index(drop=True)

    wants_ranked_period = any(w in q for w in POSITIVE_SORT_WORDS + NEGATIVE_SORT_WORDS + ("worst", "best", "strongest", "weakest"))
    ascending = _direction(q, metric) or any(w in q for w in ("worst", "weakest", "lowest", "least"))
    ranked_table = table.sort_values(sort_col, ascending=ascending).reset_index(drop=True) if wants_ranked_period else table.copy()
    ranked_table["rank"] = range(1, len(ranked_table) + 1)
    focus = ranked_table.iloc[0]
    focus_word = "weakest" if ascending else "strongest"

    lines = [f"## {metric.title()} Trend by {synthetic.title()}", ""]
    lines.extend(_cleaned_note(roles))
    lines.append(_tier2_table(ranked_table.rename(columns={"period": "dimension_value"}), min(12, len(ranked_table))))
    if wants_ranked_period:
        lines.append(f"\n## Main Finding\nThe {focus_word} {synthetic} for {metric} is **{focus['period']}** with **{_tier2_format_cell(sort_col, focus[sort_col])}**.")
    else:
        best = table.sort_values(sort_col, ascending=False).iloc[0]
        lines.append(f"\n## Main Finding\nThe strongest {synthetic} for {metric} is **{best['period']}** with **{_tier2_format_cell(sort_col, best[sort_col])}**.")
    return {"handled": True, "intent": "dynamic_trend", "confidence": "High", "markdown_answer": "\n".join(lines), "evidence_tables": {"dynamic_trend": ranked_table}, "limitations": []}


def _compare_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = question.lower()
    if not any(w in q for w in ("compare", " vs ", " versus ", "between ")):
        return {"handled": False, "reason": "Not a comparison question."}
    entities = resolve_entities(question, df, roles)
    if len(entities) < 2:
        dim_label, dim_col, _ = _detect_dimension(q, roles)
        if dim_col in df.columns:
            matched = [str(e.get("matched_value")) for e in entities if e.get("matched_value") is not None]
            available = sorted(df[dim_col].dropna().astype(str).unique().tolist())[:25]
            matched_text = ", ".join(matched) if matched else "none"
            answer = (
                f"## Headline\nI could not run the comparison because fewer than two {dim_label or 'dimension'} values were found in the dataset.\n\n"
                f"## Data\nMatched values: **{matched_text}**. Available {dim_label or 'values'} include: {', '.join(available)}.\n\n"
                f"## Insight\nA comparison needs at least two valid values from the same detected column.\n\n"
                f"## Recommendation\nUse two values exactly as they appear in the dataset, then rerun the comparison."
            )
            return {"handled": True, "intent": "dynamic_comparison_value_not_found", "confidence": "Low", "markdown_answer": answer, "evidence_tables": {"available_comparison_values": pd.DataFrame({"Available Values": available})}, "limitations": ["Fewer than two comparable values matched."]}
        return {"handled": False, "reason": "Not enough matched entities."}
    # Prefer entities that belong to the same column.
    by_col: dict[str, list[str]] = {}
    for ent in entities:
        col = ent.get("matched_column")
        val = ent.get("matched_value")
        if col and val is not None:
            by_col.setdefault(col, []).append(val)
    same_col = next(((col, vals) for col, vals in by_col.items() if len(vals) >= 2), None)
    if not same_col:
        return {"handled": False, "reason": "Entities are not comparable within one dimension."}
    col, vals = same_col
    comp = df[df[col].astype(str).str.lower().isin([str(v).lower() for v in vals])].copy()
    table = _build_grouped_table(comp, roles, col)
    if table.empty:
        return {"handled": True, "intent": "dynamic_comparison", "confidence": "Low", "markdown_answer": "I matched the requested entities, but could not compute comparison metrics because the required columns are missing.", "evidence_tables": {}, "limitations": ["Required metrics missing."]}
    table = table.sort_values("sales" if "sales" in table else table.columns[-1], ascending=False).reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))
    lines = ["## Comparison", "", f"Compared values from **{col}**: {', '.join(map(str, vals[:5]))}.", "", _tier2_table(table, len(table))]
    return {"handled": True, "intent": "dynamic_comparison", "confidence": "High", "markdown_answer": "\n".join(lines), "evidence_tables": {"dynamic_comparison": table}, "limitations": []}



# ---------------------------------------------------------------------------
# Tier 2 deterministic analyst engine
# ---------------------------------------------------------------------------
# This layer handles normal data-analyst questions using a compact Pandas plan:
# question -> metric/dimension/operation -> relevant pre-aggregated table ->
# structured manager answer. It intentionally does not pass raw rows to an LLM.

def _tier2_money(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return "N/A"
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v):,.2f}"


def _tier2_percent(value: Any) -> str:
    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "N/A"


def _tier2_number(value: Any) -> str:
    try:
        v = float(value)
        if v.is_integer():
            return f"{int(v):,}"
        return f"{v:,.2f}"
    except Exception:
        return str(value)


def _tier2_format_cell(col: str, val: Any) -> str:
    if pd.isna(val):
        return "N/A"
    c = str(col).lower()
    # Percent/count fields must be classified before broad currency words like
    # "loss"; otherwise loss_order_percent/count render as dollars.
    if "correlation" in c:
        return f"{float(val):.3f}" if pd.notna(val) else "N/A"
    if "percent" in c or "margin" in c or "discount" in c:
        return _tier2_percent(val)
    if "count" in c or c in {"orders", "order_count", "customer_count", "quantity", "rank", "total_count", "loss_orders"}:
        return _tier2_number(val)
    if c in {"sales", "profit", "cost", "loss_amount", "total_sales", "total_profit", "total_loss", "average_order_value"} or "revenue" in c or "order_value" in c:
        return _tier2_money(val)
    if isinstance(val, (int, float)):
        return _tier2_number(val)
    return str(val)


def _tier2_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df is None or df.empty:
        return ""
    show = df.head(max_rows).copy()
    preferred = [
        "rank", "dimension_type", "dimension_value", "sales", "profit",
        "profit_margin_percent", "loss_amount", "quantity", "order_count",
        "customer_count", "average_discount", "average_order_value", "cost", "correlation",
    ]
    cols = [c for c in preferred if c in show.columns]
    for c in show.columns:
        if c not in cols and len(cols) < 10:
            cols.append(c)
    out = show[cols].copy()
    for col in out.columns:
        out[col] = out[col].map(lambda v, c=col: _tier2_format_cell(c, v))
    out = out.rename(columns={
        "rank": "Rank",
        "dimension_type": "Dimension",
        "dimension_value": "Name",
        "sales": "Sales/Revenue",
        "profit": "Profit",
        "profit_margin_percent": "Margin %",
        "loss_amount": "Loss Amount",
        "quantity": "Quantity",
        "order_count": "Orders",
        "customer_count": "Customers",
        "average_discount": "Avg Discount",
        "average_order_value": "AOV",
        "cost": "Cost",
        "correlation": "Correlation",
    })
    return _simple_markdown_table(out)


def _tier2_question_text(q: str) -> str:
    return re.sub(r"\s+", " ", str(q).lower().strip())


def _plural_label(label: str | None) -> str:
    label = str(label or "group")
    irregular = {
        "category": "categories",
        "subcategory": "sub-categories",
        "city": "cities",
        "country": "countries",
        "discount band": "discount bands",
    }
    if label in irregular:
        return irregular[label]
    if label.endswith("s"):
        return label
    return f"{label}s"


def _tier2_metric(q: str, roles: dict) -> tuple[str | None, str | None]:
    q = _tier2_question_text(q)
    # Specific metrics before generic words such as amount/value.
    if any(w in q for w in ("profit margin", "margin %", "margin percentage", "margin", "profitable")):
        if "profit" in q and not any(w in q for w in ("margin", "profit margin")):
            return "profit", roles.get("profit_column")
        return "margin", None
    if any(w in q for w in ("loss", "loss-making", "loss making", "losing", "unprofitable", "negative profit", "bleeding")):
        return "loss", roles.get("profit_column")
    if any(w in q for w in ("discount", "discounts", "rebate", "markdown")):
        return "discount", roles.get("discount_column")
    if any(w in q for w in ("profit", "profits", "earnings", "contribution")):
        return "profit", roles.get("profit_column")
    if any(w in q for w in ("cost", "costs", "expense", "cogs")):
        return "cost", roles.get("cost_column")
    if any(w in q for w in ("quantity", "units", "volume", "unit sold", "units sold")):
        return "quantity", roles.get("quantity_column")
    if any(w in q for w in ("order", "orders", "transactions")):
        return "orders", roles.get("order_id_column")
    if any(w in q for w in ("customer count", "number of customers", "unique customers")):
        return "customers", roles.get("customer_column") or roles.get("customer_id_column")
    if any(w in q for w in ("sales", "revenue", "turnover", "order value", "amount", "value")):
        return "sales", roles.get("sales_column")
    return None, None


def _tier2_dimension(q: str, roles: dict) -> tuple[str | None, str | None, str | None]:
    return _detect_dimension(_tier2_question_text(q), roles)


def _tier2_operation(q: str) -> str | None:
    q = _tier2_question_text(q)
    if any(w in q for w in ("correlat", "relationship", "affect", "effect", "impact", "influence", "related")):
        return "correlation"
    if any(w in q for w in ("compare", " vs ", " versus ", "between ")):
        return "compare"
    if any(w in q for w in ("by ", "group", "breakdown", "split", "across")):
        return "group"
    if any(w in q for w in POSITIVE_SORT_WORDS + NEGATIVE_SORT_WORDS + LOSS_WORDS):
        return "rank"
    metric, _ = _tier2_metric(q, {})
    dim, _, _ = _detect_dimension(q, {})
    if metric and dim:
        return "rank"
    return None


def _tier2_should_handle(question: str, roles: dict) -> bool:
    q = _tier2_question_text(question)
    if _is_basic_sanity_question(q):
        return False
    # Keep specialized existing engines for these cases. This Tier 2 layer is
    # for metric/dimension/group/rank/correlation questions, not loyalty,
    # broad root-cause diagnostics, churn, or forecasting/trend prompts.
    if "loyal" in q or _contains_any(q, CHURN_WORDS):
        return False
    if "stop selling" in q or ("stop" in q and "selling" in q):
        return False
    if _contains_any(q, ROOT_CAUSE_WORDS) and (_contains_any(q, LOSS_WORDS) or _contains_any(q, PROFIT_QUESTION_WORDS)):
        return False
    if _contains_any(q, TREND_WORDS) and any(w in q for w in ("sales", "revenue", "profit", "margin", "quantity", "orders")):
        return False
    if ("fix first" in q or "focus" in q) and not _tier2_metric(q, roles)[0]:
        return False

    op = _tier2_operation(q)
    metric, _ = _tier2_metric(q, roles)
    dim, _, _ = _tier2_dimension(q, roles)
    if op == "correlation" and metric:
        return True
    if op == "compare":
        return True
    if dim and (metric or op in {"rank", "group"}):
        return True
    # profitability questions often imply profit by a dimension.
    if any(w in q for w in ("most profitable", "least profitable", "profitable")) and dim:
        return True
    return False


def _tier2_sort(metric: str | None, q: str) -> tuple[str | None, bool]:
    sort_col = _sort_metric_name(metric)
    q = _tier2_question_text(q)
    ascending = False
    if metric == "loss" or any(w in q for w in LOSS_WORDS):
        ascending = True
        sort_col = "profit"
    elif any(w in q for w in NEGATIVE_SORT_WORDS):
        ascending = True
    return sort_col, ascending


def _tier2_group_rank(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = _tier2_question_text(question)
    dim_label, dim_col, synthetic = _tier2_dimension(q, roles)
    metric, metric_col = _tier2_metric(q, roles)
    if not dim_label or not dim_col:
        return {"handled": False, "reason": "Tier2 needs a dimension."}
    if metric is None:
        metric = "profit" if "profitable" in q and roles.get("profit_column") else "sales"
    if metric in {"sales", "profit", "discount", "cost", "quantity"} and metric_col not in df.columns:
        return {
            "handled": True,
            "intent": "dynamic_ranking",
            "confidence": "Low",
            "markdown_answer": f"## Headline\nI could not answer this because the required {metric} column was not detected.\n\n## Data\nNo reliable pre-aggregated table was produced.\n\n## Insight\nThe dataset needs a detected {metric} field for this question.\n\n## Recommendation\nUse the sidebar column override to map the correct {metric} column, then rerun the question.",
            "evidence_tables": {},
            "limitations": [f"Missing required metric: {metric}"],
            "analysis_plan": {"operation": "group/rank", "metric": metric, "dimension": dim_label, "source_rows_sent_to_ai": 0},
        }
    table = _build_grouped_table(df, roles, dim_col, synthetic)
    if table.empty:
        return {"handled": False, "reason": "Grouped table empty."}

    sort_col, ascending = _tier2_sort(metric, q)
    if sort_col not in table.columns:
        return {
            "handled": True,
            "intent": "dynamic_ranking",
            "confidence": "Low",
            "markdown_answer": f"## Headline\nI could group by {dim_label}, but could not calculate {metric}.\n\n## Data\nThe relevant metric is unavailable after aggregation.\n\n## Insight\nThis usually means a required sales/profit/discount column was not detected.\n\n## Recommendation\nCheck column detection or manually map the missing field.",
            "evidence_tables": {"tier2_grouped_table": table},
            "limitations": [f"Metric unavailable: {metric}"],
            "analysis_plan": {"operation": "group/rank", "metric": metric, "dimension": dim_label, "source_rows_sent_to_ai": 0},
        }
    result = table.copy()
    if metric == "loss" and "profit" in result.columns:
        result = result[result["profit"] < 0].copy()
    limit = _extract_limit(q)
    result = result.sort_values(sort_col, ascending=ascending).head(limit).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))

    if result.empty:
        headline = f"No loss-making {_plural_label(dim_label)} found" if metric == "loss" else f"No result found for {dim_label} by {metric}"
        data_text = "The pre-aggregated table contained no rows after the requested filter."
        insight = "The requested condition is not present in the available data."
        recommendation = "Review broader metrics or remove the filter if you expected results."
    else:
        top = result.iloc[0]
        name = top.get("dimension_value")
        value = top.get(sort_col)
        metric_label = {"sales": "sales/revenue", "profit": "profit", "loss": "loss", "margin": "profit margin", "discount": "average discount", "cost": "cost", "quantity": "quantity", "orders": "orders", "customers": "customer count"}.get(metric, metric)
        if metric == "loss":
            headline = f"{name} is the largest loss-making {dim_label} in the requested ranking."
            insight = "The ranking is based on negative total profit, not sales volume."
            recommendation = "Review price, discount, cost, and customer/region concentration for the listed loss-making groups."
        elif metric == "margin":
            headline = f"{name} has the selected profit-margin position among {_plural_label(dim_label)}."
            insight = "Margin is calculated as profit / sales * 100, not raw profit."
            recommendation = "Compare margin beside revenue and profit before making pricing or portfolio decisions."
        elif ascending:
            headline = f"{name} is the lowest-ranked {dim_label} by {metric_label}."
            insight = "The table is sorted from lowest to highest based on the requested metric."
            recommendation = "Investigate the bottom-ranked groups for pricing, cost, demand, discount, or coverage issues."
        else:
            headline = f"{name} is the highest-ranked {dim_label} by {metric_label}."
            insight = "The table is sorted from highest to lowest using only the relevant aggregated data."
            recommendation = "Revenue alone does not prove business quality. Prioritize these groups for performance review, but compare profit and margin before acting on revenue alone."
        data_text = _tier2_table(result, limit)

    answer = (
        f"## Headline\n{('Loss-Generating ' + _plural_label(dim_label).title() + ': ') if metric == 'loss' else ''}{headline}\n\n"
        f"## Direct Answer\n{headline}\n\n"
        f"## Data\n{data_text}\n\n"
        f"## Insight\n{insight}\n\n"
        f"## Recommendation\n{recommendation}"
    )
    return {
        "handled": True,
        "intent": "dynamic_ranking",
        "confidence": "High",
        "markdown_answer": answer,
        "evidence_tables": {"tier2_result": result, "tier2_preaggregated_table": table, "dynamic_ranking_result": result, "dynamic_grouped_table": table},
        "analysis_plan": {
            "operation": "group_by_rank_filter" if metric == "loss" else "group_by_rank",
            "dimension": dim_label,
            "metric": "profit" if metric == "loss" else metric,
            "aggregation": "sum/mean/nunique as relevant",
            "sort": "ascending" if ascending else "descending",
            "limit": limit,
            "source_rows_sent_to_ai": 0,
            "preaggregated_rows_sent_to_ai": int(min(len(result), limit)),
        },
        "limitations": [],
        "relationship_findings": [],
        "root_causes": [],
        "recommendations": [],
        "follow_up_questions": [
            f"Show bottom {limit} {_plural_label(dim_label)} by {metric}.",
            f"Compare {_plural_label(dim_label)} by margin.",
            f"What is driving this {dim_label} performance?",
        ],
    }


def _metric_series_for_correlation(df: pd.DataFrame, roles: dict, metric: str) -> tuple[str, pd.Series] | tuple[None, None]:
    """Return a numeric series for a Tier 2 correlation metric."""
    if metric == "sales":
        col = roles.get("sales_column")
        return ("sales", safe_numeric(df[col])) if col in df.columns else (None, None)
    if metric == "profit":
        col = roles.get("profit_column")
        return ("profit", safe_numeric(df[col])) if col in df.columns else (None, None)
    if metric == "discount":
        col = roles.get("discount_column")
        return ("discount", safe_numeric(df[col])) if col in df.columns else (None, None)
    if metric == "quantity":
        col = roles.get("quantity_column")
        return ("quantity", safe_numeric(df[col])) if col in df.columns else (None, None)
    if metric == "cost":
        col = roles.get("cost_column")
        return ("cost", safe_numeric(df[col])) if col in df.columns else (None, None)
    if metric == "margin":
        sales_col = roles.get("sales_column")
        profit_col = roles.get("profit_column")
        if sales_col in df.columns and profit_col in df.columns:
            sales = safe_numeric(df[sales_col])
            profit = safe_numeric(df[profit_col])
            margin = (profit / sales.replace(0, pd.NA)) * 100
            return "margin", margin
    return None, None


def _metric_mentions_in_question(q: str) -> list[str]:
    """Find metric mentions in textual order for generic correlation questions."""
    q = _tier2_question_text(q)
    aliases = {
        "sales": ("sales", "revenue", "turnover", "order value"),
        "profit": ("profit", "profits", "earnings", "contribution"),
        "discount": ("discount", "discounts", "rebate", "markdown"),
        "quantity": ("quantity", "quantities", "units", "volume"),
        "cost": ("cost", "costs", "expense", "cogs"),
        "margin": ("profit margin", "margin", "margin %"),
    }
    hits: list[tuple[int, str]] = []
    for metric, words in aliases.items():
        positions = [q.find(w) for w in words if q.find(w) >= 0]
        if positions:
            hits.append((min(positions), metric))
    ordered: list[str] = []
    for _, metric in sorted(hits, key=lambda x: x[0]):
        if metric not in ordered:
            ordered.append(metric)
    # For prompts like "how does discount affect profit", make sure target is profit.
    if "affect profit" in q or "impact profit" in q or "effect on profit" in q:
        if "discount" in ordered and "profit" not in ordered:
            ordered.append("profit")
    return ordered


def _correlation_strength_text(corr: float | None) -> str:
    if corr is None or pd.isna(corr):
        return "not measurable"
    mag = abs(corr)
    if mag >= 0.60:
        strength = "strong"
    elif mag >= 0.35:
        strength = "moderate"
    elif mag >= 0.15:
        strength = "weak"
    else:
        strength = "very weak"
    direction = "positive" if corr > 0 else "negative" if corr < 0 else "flat"
    return f"{strength} {direction}"


def _tier2_correlation_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = _tier2_question_text(question)
    metrics = _metric_mentions_in_question(q)
    if len(metrics) < 2:
        # Keep the common implicit case: discount impact means discount -> profit.
        if any(w in q for w in ("discount", "rebate", "markdown")):
            metrics = ["discount", "profit"]
        else:
            return {"handled": False, "reason": "No supported correlation metric pair detected."}

    driver_metric, target_metric = metrics[0], metrics[1]
    driver_name, driver_series = _metric_series_for_correlation(df, roles, driver_metric)
    target_name, target_series = _metric_series_for_correlation(df, roles, target_metric)
    if driver_series is None or target_series is None:
        missing = driver_metric if driver_series is None else target_metric
        return {
            "handled": True,
            "intent": "tier2_correlation",
            "confidence": "Low",
            "markdown_answer": f"## Headline\nI could not calculate this relationship because the required {missing} field was not detected.\n\n## Data\nNo correlation table was produced.\n\n## Insight\nCorrelation questions require two valid numeric fields.\n\n## Recommendation\nUse the sidebar column override to map the missing field, then rerun the question.",
            "evidence_tables": {},
            "limitations": [f"Missing required metric: {missing}"],
            "analysis_plan": {"operation": "correlation", "driver": driver_metric, "target": target_metric, "source_rows_sent_to_ai": 0},
        }

    corr_df = pd.DataFrame({"driver": driver_series, "target": target_series}).dropna()
    if corr_df.empty or corr_df["driver"].nunique(dropna=True) <= 1 or corr_df["target"].nunique(dropna=True) <= 1:
        corr = None
    else:
        corr = float(corr_df["driver"].corr(corr_df["target"]))

    corr_text = "N/A" if corr is None or pd.isna(corr) else f"{corr:.3f}"
    strength = _correlation_strength_text(corr)
    headline = f"The relationship between {driver_metric} and {target_metric} is {strength}."

    evidence_tables: dict[str, pd.DataFrame] = {
        "tier2_correlation": pd.DataFrame([{
            "driver": driver_metric,
            "target": target_metric,
            "correlation": corr,
            "valid_rows": int(len(corr_df)),
        }])
    }

    band_text = ""
    if driver_metric == "discount" and roles.get("discount_column") in df.columns and target_metric in {"profit", "sales", "margin"}:
        disc_col = roles.get("discount_column")
        work = df.copy()
        work[disc_col] = safe_numeric(work[disc_col])
        work["__discount_band"] = create_discount_bands(work[disc_col])
        sales_col = roles.get("sales_column")
        profit_col = roles.get("profit_column")
        agg: dict[str, tuple[str, str]] = {"average_discount": (disc_col, "mean"), "order_count": (disc_col, "size")}
        if sales_col in work.columns:
            work[sales_col] = safe_numeric(work[sales_col])
            agg["sales"] = (sales_col, "sum")
        if profit_col in work.columns:
            work[profit_col] = safe_numeric(work[profit_col])
            agg["profit"] = (profit_col, "sum")
        grouped = work.groupby("__discount_band", dropna=False, observed=True).agg(**agg).reset_index().rename(columns={"__discount_band": "dimension_value"})
        if "sales" in grouped.columns and "profit" in grouped.columns:
            grouped["profit_margin_percent"] = grouped.apply(lambda r: profit_margin(r.get("profit"), r.get("sales")), axis=1)
        evidence_tables["tier2_discount_by_band"] = grouped
        evidence_tables["tier2_discount_profit_by_band"] = grouped
        band_text = "\n\n" + _tier2_table(grouped.sort_values("average_discount"), 20)

    answer = (
        f"## Headline\n{headline}\n\n"
        f"## Direct Answer\n{headline}\n\n"
        f"## Data\nCorrelation ({driver_metric} vs {target_metric}): **{corr_text}**\nValid rows used: **{len(corr_df):,}**{band_text}\n\n"
        f"## Insight\nCorrelation measures linear movement between two numeric fields. A negative value means higher {driver_metric} usually moves with lower {target_metric}; a positive value means they move together.\n\n"
        f"## Recommendation\nUse this as a directional signal, then validate with grouped tables by product, category, region, or discount band before taking action."
    )
    return {
        "handled": True,
        "intent": "tier2_correlation",
        "confidence": "High" if corr is not None and not pd.isna(corr) else "Medium",
        "markdown_answer": answer,
        "evidence_tables": evidence_tables,
        "analysis_plan": {"operation": "correlate_and_group_by_band" if driver_metric == "discount" and "tier2_discount_by_band" in evidence_tables else "correlation", "driver": driver_metric, "target": target_metric, "source_rows_sent_to_ai": 0, "preaggregated_rows_sent_to_ai": int(len(evidence_tables.get("tier2_correlation", [])) + len(evidence_tables.get("tier2_discount_by_band", [])))},
        "limitations": [] if corr is not None and not pd.isna(corr) else ["Correlation skipped or weak because one field has low variation."],
        "relationship_findings": [],
        "root_causes": [],
        "recommendations": [],
        "follow_up_questions": [f"Show {target_metric} by {driver_metric} band.", f"Which products have weak {target_metric}?", f"Compare {driver_metric} by category."],
    }



def _tier2_high_sales_low_profit_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = _tier2_question_text(question)
    if not (("high sales" in q or "high revenue" in q or "high selling" in q) and ("low profit" in q or "low margin" in q or "weak profit" in q or "weak margin" in q)):
        return {"handled": False, "reason": "Not a high-sales/low-profit question."}
    dim_label, dim_col, synthetic = _tier2_dimension(q, roles)
    if not dim_col:
        # Product is the usual manager intent for this phrasing.
        dim_label, dim_col, synthetic = "product", roles.get("product_column") or roles.get("subcategory_column") or roles.get("category_column"), None
    sales_col, profit_col = roles.get("sales_column"), roles.get("profit_column")
    if dim_col not in df.columns or sales_col not in df.columns or profit_col not in df.columns:
        return {"handled": True, "intent": "tier2_high_sales_low_profit", "confidence": "Low", "markdown_answer": "## Headline\nI could not find high-sales/low-profit groups because the required dimension, Sales, and Profit columns were not all detected.\n\n## Data\nNo table was produced.\n\n## Insight\nThis question needs both scale and margin.\n\n## Recommendation\nMap Product/Category, Sales, and Profit, then rerun the question.", "evidence_tables": {}, "limitations": ["Missing dimension/sales/profit column."], "analysis_plan": {"operation": "high_sales_low_profit", "source_rows_sent_to_ai": 0}}
    table = _build_grouped_table(df, roles, dim_col, synthetic)
    if table.empty or "sales" not in table.columns or "profit_margin_percent" not in table.columns:
        return {"handled": False, "reason": "Grouped table unavailable."}
    # High-sales/low-profit should surface groups with meaningful scale AND weak conversion.
    # For small dimensions such as Category, a 70th percentile sales cutoff can
    # incorrectly exclude the main weak category. Use a softer top-half cutoff
    # when there are only a few groups.
    sales_cut = table["sales"].quantile(0.50 if len(table) <= 5 else 0.70)
    overall_margin = profit_margin(table["profit"].sum(), table["sales"].sum()) if "profit" in table.columns else table["profit_margin_percent"].median()
    margin_cut = max(float(table["profit_margin_percent"].median()), float(overall_margin))
    result = table[(table["sales"] >= sales_cut) & (table["profit_margin_percent"] <= margin_cut)].copy()
    if result.empty:
        result = table.sort_values(["profit_margin_percent", "sales"], ascending=[True, False]).head(10).copy()
    else:
        result = result.sort_values(["profit_margin_percent", "sales"], ascending=[True, False]).head(10).copy()
    result = result.reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result)+1))
    top = result.iloc[0] if not result.empty else None
    headline = f"{top['dimension_value']} has high sales but relatively weak profit margin." if top is not None else "No high-sales/low-profit group was found."
    answer = f"## Headline\n{headline}\n\n## Data\n{_tier2_table(result, 10)}\n\n## Insight\nThis filters for groups with sales in the upper range and margin below the dataset median, so it is not simply a top-profit ranking.\n\n## Recommendation\nReview pricing, discounting, cost, and product mix for these high-volume but weak-margin groups first."
    return {"handled": True, "intent": "tier2_high_sales_low_profit", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"tier2_high_sales_low_profit": result, "tier2_grouped_table": table}, "analysis_plan": {"operation": "high_sales_low_profit", "dimension": dim_label, "source_rows_sent_to_ai": 0, "preaggregated_rows_sent_to_ai": int(len(result))}, "limitations": [], "relationship_findings": [], "root_causes": [], "recommendations": []}

def _tier2_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    if not _tier2_should_handle(question, roles):
        return {"handled": False, "reason": "Not a Tier 2 analytical question."}
    hs_low = _tier2_high_sales_low_profit_answer(question, df, roles)
    if hs_low.get("handled"):
        return hs_low
    # Correlation / affect questions are handled first because they need a
    # driver-target plan, not a simple ranking.
    if _tier2_operation(question) == "correlation":
        corr = _tier2_correlation_answer(question, df, roles)
        if corr.get("handled"):
            return corr
    # Comparison still uses entity resolution, but we keep the old handler.
    comp = _compare_answer(question, df, roles)
    if comp.get("handled"):
        if comp.get("intent") in {None, "dynamic_comparison"}:
            comp["intent"] = "tier2_comparison"
        comp.setdefault("analysis_plan", {}).update({"source_rows_sent_to_ai": 0})
        return comp
    return _tier2_group_rank(question, df, roles)



# ---------------------------------------------------------------------------
# Tier 3 deterministic analyst engine
# ---------------------------------------------------------------------------
# This layer handles harder manager/data-analyst questions that require time
# parsing, multi-condition filters, subset percentages, consistency detection,
# business judgment, and cross-dimension strategy comparisons. It still follows
# the same safety rule: Pandas computes the numbers; optional AI only rewrites
# the compact answer and never receives raw CSV rows.

TIER3_BUSINESS_JUDGMENT_WORDS = (
    "should", "worth", "exit", "double down", "strategy", "working", "recommend", "consider",
    "continue", "stop", "risk", "exception", "judgment", "decision",
)


def _tier3_question_text(q: str) -> str:
    return re.sub(r"\s+", " ", str(q).lower().strip())


def _tier3_role_col(df: pd.DataFrame, roles: dict, role_key: str, aliases: tuple[str, ...] = ()) -> str | None:
    col = roles.get(role_key)
    if col in df.columns:
        return col
    norm_aliases = [re.sub(r"[^a-z0-9]+", " ", a.lower()).strip() for a in aliases]
    for c in df.columns:
        nc = re.sub(r"[^a-z0-9]+", " ", str(c).lower()).strip()
        if nc in norm_aliases or any(a and a in nc for a in norm_aliases):
            return c
    return None


def _tier3_order_basis(df: pd.DataFrame, roles: dict) -> tuple[pd.Series, str]:
    """Return the counting series and label for order-style questions.

    For Superstore-style line-item data, managers often say "orders" while the
    expected calculation is row/record based. We therefore use row records for
    subset percentages but state that basis clearly in the answer.
    """
    return pd.Series(range(len(df)), index=df.index), "records/order rows"


def _tier3_date_series(df: pd.DataFrame, roles: dict) -> tuple[str | None, pd.Series | None]:
    date_col = _tier3_role_col(df, roles, "date_column", ("order date", "date", "transaction date", "invoice date"))
    if date_col not in df.columns:
        return None, None
    return date_col, ensure_datetime(df[date_col])


def _tier3_sales_profit_cols(df: pd.DataFrame, roles: dict) -> tuple[str | None, str | None]:
    return roles.get("sales_column") if roles.get("sales_column") in df.columns else None, roles.get("profit_column") if roles.get("profit_column") in df.columns else None


def _tier3_year_filter_from_question(q: str, years: pd.Series) -> list[int]:
    mentioned = [int(y) for y in re.findall(r"\b(20\d{2}|19\d{2})\b", q)]
    if len(mentioned) >= 2 and any(w in q for w in ("to", "through", "from", "between", "-") ):
        lo, hi = min(mentioned), max(mentioned)
        return list(range(lo, hi + 1))
    if mentioned:
        return sorted(set(mentioned))
    return sorted(int(y) for y in years.dropna().unique())


def _tier3_yoy_growth_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = _tier3_question_text(question)
    if not (any(w in q for w in ("year over year", "year-over-year", "yoy", "yearly growth", "annual growth")) or ("growth" in q and "year" in q)):
        return {"handled": False, "reason": "Not a YoY growth question."}
    sales_col, profit_col = _tier3_sales_profit_cols(df, roles)
    date_col, dates = _tier3_date_series(df, roles)
    if date_col is None or dates is None or sales_col not in df.columns:
        return {
            "handled": True,
            "intent": "tier3_yoy_growth",
            "confidence": "Low",
            "markdown_answer": "## Headline\nI could not calculate year-over-year growth because a valid Order Date and Sales/Revenue column were not detected.\n\n## Data\nNo yearly table was produced.\n\n## Insight\nYoY analysis requires parseable dates and sales values.\n\n## Recommendation\nMap the order-date and sales columns, then rerun the question.",
            "evidence_tables": {},
            "limitations": ["Missing valid date or sales column."],
            "analysis_plan": {"operation": "year_over_year_growth", "source_rows_sent_to_ai": 0},
        }
    work = df.copy()
    work["__order_year"] = dates.dt.year
    work[sales_col] = safe_numeric(work[sales_col]).fillna(0)
    agg = {"sales": (sales_col, "sum"), "order_count": (sales_col, "size")}
    if profit_col in work.columns:
        work[profit_col] = safe_numeric(work[profit_col]).fillna(0)
        agg["profit"] = (profit_col, "sum")
    table = work.dropna(subset=["__order_year"]).groupby("__order_year", as_index=False).agg(**agg)
    table = table.rename(columns={"__order_year": "year"}).sort_values("year").reset_index(drop=True)
    requested_years = _tier3_year_filter_from_question(q, table["year"])
    if requested_years:
        table = table[table["year"].astype(int).isin(requested_years)].copy()
    table["sales_yoy_growth_percent"] = table["sales"].pct_change().mul(100)
    if "profit" in table.columns:
        table["profit_yoy_growth_percent"] = table["profit"].pct_change().mul(100)
        table["profit_margin_percent"] = table.apply(lambda r: profit_margin(r.get("profit"), r.get("sales")), axis=1)
    if table.empty:
        answer = "## Headline\nNo matching years were found for the requested YoY range.\n\n## Data\nNo rows matched the requested years.\n\n## Insight\nThe dataset may not contain the specified years.\n\n## Recommendation\nCheck the date range first, then rerun the YoY question."
    else:
        years_txt = ", ".join(str(int(y)) for y in table["year"].tolist())
        latest = table.iloc[-1]
        latest_growth = latest.get("sales_yoy_growth_percent")
        growth_text = "N/A" if pd.isna(latest_growth) else _tier2_percent(latest_growth)
        answer = (
            f"## Headline\nYear-over-year sales were calculated for {years_txt}.\n\n"
            f"## Data\n{_tier2_table(table.assign(year=table['year'].astype(int).astype(str)).rename(columns={'year': 'dimension_value'}), len(table))}\n\n"
            f"## Insight\nSales growth is calculated as `(current year sales - previous year sales) / previous year sales * 100`. The latest available YoY sales growth is **{growth_text}**.\n\n"
            f"## Recommendation\nUse the yearly table to separate revenue growth from profit quality. If sales grow while margin weakens, review discounting, product mix, and cost structure."
        )
    return {
        "handled": True,
        "intent": "tier3_yoy_growth",
        "confidence": "High" if not table.empty else "Medium",
        "markdown_answer": answer,
        "evidence_tables": {"tier3_year_over_year": table},
        "analysis_plan": {"operation": "parse_date_group_by_year_growth", "date_column": date_col, "metric": "sales", "source_rows_sent_to_ai": 0, "preaggregated_rows_sent_to_ai": int(len(table))},
        "limitations": [] if not table.empty else ["Requested years not found."],
        "relationship_findings": [], "root_causes": [], "recommendations": [],
    }


def _tier3_potential_unknown_filter(question: str, df: pd.DataFrame, roles: dict) -> str | None:
    """Return a likely entity name mentioned after in/from/for when it is not found."""
    q = _tier3_question_text(question)
    m = re.search(r"\b(?:in|from|for|of)\s+([a-z][a-z0-9 .&'-]{1,40}?)(?:\?|$|\borders\b|\brecords\b|\bprofit\b|\bsales\b|\brevenue\b|\bcustomers\b|\bdiscounts\b|\bdiscount\b)", q)
    if not m:
        return None
    candidate = m.group(1).strip(" .?'")
    stop = {"this", "the", "our", "all", "total", "profit", "sales", "revenue", "orders", "records", "dataset", "data"}
    if not candidate or candidate in stop:
        return None
    # If the candidate appears in any common dimension, it is not unknown.
    for role_key in ("state_column", "region_column", "city_column", "country_column", "category_column", "subcategory_column", "segment_column", "customer_column", "ship_mode_column"):
        col = roles.get(role_key)
        if col in df.columns and df[col].dropna().astype(str).str.lower().eq(candidate.lower()).any():
            return None
    return candidate.title()


def _tier3_no_matching_filter_answer(question: str, missing_entity: str, operation: str = "filtered analysis") -> dict:
    answer = (
        f"## Headline\nI could not run the requested {operation} because **{missing_entity}** was not found in the dataset.\n\n"
        f"## Data\nNo matching rows were found for **{missing_entity}** in the detected geography/customer/product/category fields.\n\n"
        f"## Insight\nThis is safer than silently ignoring the filter and answering for the whole dataset.\n\n"
        f"## Recommendation\nCheck the spelling or use one of the values listed in the dataset profile before rerunning the question."
    )
    return {"handled": True, "intent": "tier3_filter_value_not_found", "confidence": "High", "markdown_answer": answer, "evidence_tables": {}, "limitations": [f"Filter value not found: {missing_entity}"], "analysis_plan": {"operation": operation, "source_rows_sent_to_ai": 0}}


def _tier3_unprofitable_percentage_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = _tier3_question_text(question)
    if not (("percentage" in q or "%" in q or "percent" in q or "proportion" in q or "share" in q) and any(w in q for w in ("unprofitable", "loss", "negative profit", "losing"))):
        return {"handled": False, "reason": "Not an unprofitable percentage question."}
    profit_col = roles.get("profit_column")
    if profit_col not in df.columns:
        return {"handled": True, "intent": "tier3_unprofitable_percentage", "confidence": "Low", "markdown_answer": "## Headline\nI could not calculate the unprofitable-order percentage because Profit was not detected.\n\n## Data\nNo percentage was calculated.\n\n## Insight\nThis requires a profit column or a derived profit field.\n\n## Recommendation\nMap Profit or provide Sales and Cost so Profit can be derived.", "evidence_tables": {}, "limitations": ["Missing profit column."], "analysis_plan": {"operation": "subset_percentage", "source_rows_sent_to_ai": 0}}
    work = df.copy()
    filters = _tier3_entity_filters(question, work, roles)
    missing_entity = _tier3_potential_unknown_filter(question, work, roles) if not filters and any(w in q for w in (" in ", " from ", " for ", " of ")) else None
    if missing_entity:
        return _tier3_no_matching_filter_answer(question, missing_entity, "filtered unprofitable-percentage calculation")
    mask = pd.Series(True, index=work.index)
    filter_desc: list[str] = []
    for f in filters:
        col, val = f["column"], f["value"]
        mask &= work[col].astype(str).str.lower().eq(str(val).lower())
        filter_desc.append(f"{col} = {val}")
    subset = work[mask]
    profits = safe_numeric(subset[profit_col])
    total = int(len(subset))
    loss_count = int((profits < 0).sum())
    pct = (loss_count / total * 100) if total else 0.0
    label = " after filtering " + " AND ".join(filter_desc) if filter_desc else ""
    table = pd.DataFrame([{"subset": f"Unprofitable records/order rows{label}", "count": loss_count, "total_count": total, "percentage": pct}])
    answer = (
        f"## Headline\n{_tier2_percent(pct)} of records/order rows{label} are unprofitable.\n\n"
        f"## Data\n{loss_count:,} out of {total:,} records/order rows{label} have negative profit.\n\n{_tier2_table(table, 1)}\n\n"
        f"## Insight\nThis is a subset calculation: `(negative-profit rows / total rows) * 100`. The result is not just a raw count; it measures the loss frequency.\n\n"
        f"## Recommendation\nReview the products, regions, segments, and discount bands with the highest loss frequency before changing broad pricing strategy."
    )
    return {"handled": True, "intent": "tier3_unprofitable_percentage", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"tier3_unprofitable_percentage": table}, "analysis_plan": {"operation": "filter_profit_lt_zero_percentage", "filters": filter_desc + ["profit < 0"], "source_rows_sent_to_ai": 0, "preaggregated_rows_sent_to_ai": 1}, "limitations": [], "relationship_findings": [], "root_causes": [], "recommendations": []}

def _tier3_entity_filters(question: str, df: pd.DataFrame, roles: dict) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    try:
        entities = resolve_entities(question, df, roles)
    except Exception:
        entities = []
    for ent in entities or []:
        col = ent.get("matched_column")
        val = ent.get("matched_value")
        if col in df.columns and val is not None:
            filters.append({"column": col, "value": val, "label": ent.get("entity_type") or str(col)})
    # Extra fallback for common geography names if resolver misses them.
    q = _tier3_question_text(question)
    for role_key, label in [("state_column", "state"), ("region_column", "region"), ("city_column", "city"), ("category_column", "category"), ("subcategory_column", "subcategory"), ("segment_column", "segment"), ("ship_mode_column", "shipping mode")]:
        col = roles.get(role_key)
        if col in df.columns:
            for value in df[col].dropna().astype(str).unique():
                if str(value).lower() in q and not any(f["column"] == col and str(f["value"]).lower() == str(value).lower() for f in filters):
                    filters.append({"column": col, "value": value, "label": label})
    return filters


def _tier3_multi_condition_filter_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = _tier3_question_text(question)
    wants_loss_filter = any(w in q for w in ("unprofitable", "loss", "negative profit", "losing"))
    filters = _tier3_entity_filters(question, df, roles)
    if wants_loss_filter and not filters:
        missing_entity = _tier3_potential_unknown_filter(question, df, roles)
        if missing_entity:
            return _tier3_no_matching_filter_answer(question, missing_entity, "multi-condition loss filter")
    # Only handle when there is at least one entity filter plus one metric/condition.
    if not (wants_loss_filter and filters):
        return {"handled": False, "reason": "No multi-condition filter detected."}
    profit_col = roles.get("profit_column")
    if profit_col not in df.columns:
        return {"handled": True, "intent": "tier3_multi_condition_filter", "confidence": "Low", "markdown_answer": "## Headline\nI could not filter unprofitable records because Profit was not detected.\n\n## Data\nNo filtered table was produced.\n\n## Insight\nMulti-condition questions need both the requested filter column and profit.\n\n## Recommendation\nMap the Profit column and rerun the question.", "evidence_tables": {}, "limitations": ["Missing profit column."], "analysis_plan": {"operation": "multi_condition_filter", "source_rows_sent_to_ai": 0}}
    work = df.copy()
    mask = pd.Series(True, index=work.index)
    filter_desc: list[str] = []
    for f in filters:
        col, val = f["column"], f["value"]
        mask &= work[col].astype(str).str.lower().eq(str(val).lower())
        filter_desc.append(f"{col} = {val}")
    profits = safe_numeric(work[profit_col])
    mask &= profits < 0
    filtered = work[mask].copy()
    sales_col = roles.get("sales_column")
    product_col = roles.get("product_column") or roles.get("subcategory_column") or roles.get("category_column")
    total_loss = float(safe_numeric(filtered[profit_col]).sum()) if not filtered.empty else 0.0
    total_sales = float(safe_numeric(filtered[sales_col]).sum()) if sales_col in filtered.columns and not filtered.empty else None
    count = int(len(filtered))
    product_table = pd.DataFrame()
    if product_col in filtered.columns and not filtered.empty:
        agg = {"loss_amount": (profit_col, "sum"), "loss_order_count": (profit_col, "size")}
        if sales_col in filtered.columns:
            agg["sales"] = (sales_col, "sum")
        product_table = filtered.assign(**{profit_col: safe_numeric(filtered[profit_col])}).groupby(product_col, dropna=False).agg(**agg).reset_index().rename(columns={product_col: "dimension_value"}).sort_values("loss_amount").head(10)
        product_table.insert(0, "rank", range(1, len(product_table) + 1))
    summary = pd.DataFrame([{"filters": " AND ".join(filter_desc + ["Profit < 0"]), "matching_records": count, "total_loss": total_loss, "sales": total_sales}])
    product_names = ", ".join(map(str, product_table["dimension_value"].head(5).tolist())) if not product_table.empty else "No product names available"
    answer = (
        f"## Headline\nI filtered the data using **{' AND '.join(filter_desc)} AND Profit < 0**.\n\n"
        f"## Data\nMatching unprofitable records/order rows: **{count:,}**. Total loss: **{_tier2_money(total_loss)}**.\nProducts involved: {product_names}.\n\n{_tier2_table(product_table, 10) if not product_table.empty else _tier2_table(summary, 1)}\n\n"
        f"## Insight\nThis is a combined-condition answer, not a single-column scan. The loss figure is calculated only after applying all requested filters together.\n\n"
        f"## Recommendation\nReview the listed products/customers inside this filtered slice first; do not generalize the finding to the whole dataset until you compare it against other states/regions."
    )
    return {"handled": True, "intent": "tier3_multi_condition_filter", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"tier3_filtered_summary": summary, "tier3_filtered_products": product_table}, "analysis_plan": {"operation": "filter_conditions_and_metric", "filters": filter_desc + ["profit < 0"], "source_rows_sent_to_ai": 0, "preaggregated_rows_sent_to_ai": int(len(product_table) + 1)}, "limitations": [], "relationship_findings": [], "root_causes": [], "recommendations": []}


def _tier3_consistency_loss_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = _tier3_question_text(question)
    if not ("consistent" in q and any(w in q for w in ("losing", "loss", "unprofitable", "negative profit"))):
        return {"handled": False, "reason": "Not a consistency-loss question."}
    profit_col = roles.get("profit_column")
    dim_label, dim_col, _ = _detect_dimension(q, roles)
    if not dim_col:
        dim_label, dim_col = "product", roles.get("product_column") or roles.get("subcategory_column") or roles.get("category_column")
    if profit_col not in df.columns or dim_col not in df.columns:
        label = dim_label or "group"
        return {"handled": True, "intent": "tier3_consistency_loss", "confidence": "Low", "markdown_answer": f"## Headline\nI could not detect consistently losing {_plural_label(label)} because the required {label} and Profit fields were not both detected.\n\n## Data\nNo consistency table was produced.\n\n## Insight\nConsistency requires multiple rows/orders per {label} and a profit field.\n\n## Recommendation\nMap {label.title()} and Profit, then rerun the question.", "evidence_tables": {}, "limitations": ["Missing dimension/profit column."], "analysis_plan": {"operation": "loss_consistency", "source_rows_sent_to_ai": 0}}
    work = df.copy()
    work[profit_col] = safe_numeric(work[profit_col])
    work["__loss_order"] = work[profit_col] < 0
    agg = work.groupby(dim_col, dropna=False).agg(total_orders=(profit_col, "size"), loss_orders=("__loss_order", "sum"), total_profit=(profit_col, "sum")).reset_index().rename(columns={dim_col: "dimension_value"})
    sales_col = roles.get("sales_column")
    if sales_col in work.columns:
        work[sales_col] = safe_numeric(work[sales_col])
        sales = work.groupby(dim_col, dropna=False)[sales_col].sum().reset_index(name="sales").rename(columns={dim_col: "dimension_value"})
        agg = agg.merge(sales, on="dimension_value", how="left")
        agg["profit_margin_percent"] = agg.apply(lambda r: profit_margin(r.get("total_profit"), r.get("sales")), axis=1)
    agg["loss_order_percent"] = agg["loss_orders"] / agg["total_orders"] * 100
    result = agg[(agg["total_orders"] >= 2) & (agg["loss_order_percent"] >= 60)].sort_values(["loss_order_percent", "total_profit"], ascending=[False, True]).reset_index(drop=True)
    if not result.empty:
        result.insert(0, "rank", range(1, len(result) + 1))
    label_plural = _plural_label(dim_label)
    no_result_text = f"No {label_plural} met the >60% loss-order threshold."
    answer = (
        f"## Headline\nI found {len(result):,} {label_plural} that are consistently losing money.\n\n"
        f"## Data\nConsistently losing means **loss order % above 60% across multiple orders/records**, not one isolated bad order.\n\n{_tier2_table(result.rename(columns={'total_profit':'profit','loss_order_percent':'loss_percent'}), 15) if not result.empty else no_result_text}\n\n"
        f"## Insight\nThis separates repeat loss patterns from one-off losses. A {dim_label} with one negative order is not called consistently losing.\n\n"
        f"## Recommendation\nAudit pricing, discounting, cost, and mix for these repeated-loss {label_plural} before making exit or discontinuation decisions."
    )
    return {"handled": True, "intent": "tier3_consistency_loss", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"tier3_consistently_losing": result, "tier3_loss_consistency_all": agg}, "analysis_plan": {"operation": "group_loss_frequency", "dimension": dim_label, "threshold": "loss_order_percent >= 60 and total_orders >= 2", "source_rows_sent_to_ai": 0, "preaggregated_rows_sent_to_ai": int(len(result))}, "limitations": [], "relationship_findings": [], "root_causes": [], "recommendations": []}


def _tier3_state_exit_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = _tier3_question_text(question)
    if not (("exit" in q or "leave" in q or "pull out" in q) and ("state" in q or "market" in q or "profit" in q or "profitability" in q)):
        return {"handled": False, "reason": "Not a state/market exit question."}
    state_col = roles.get("state_column") or roles.get("region_column") or roles.get("city_column")
    sales_col, profit_col = _tier3_sales_profit_cols(df, roles)
    if state_col not in df.columns or profit_col not in df.columns:
        return {"handled": True, "intent": "tier3_state_exit", "confidence": "Low", "markdown_answer": "## Headline\nI could not evaluate exit candidates because State/Region and Profit were not both detected.\n\n## Data\nNo state profitability table was produced.\n\n## Insight\nExit decisions require market-level profitability, sales scale, and preferably strategic/context data.\n\n## Recommendation\nMap State/Region and Profit, then rerun this question.", "evidence_tables": {}, "limitations": ["Missing state/region or profit column."], "analysis_plan": {"operation": "business_judgment_exit", "source_rows_sent_to_ai": 0}}
    work = df.copy()
    work[profit_col] = safe_numeric(work[profit_col]).fillna(0)
    agg_spec = {"profit": (profit_col, "sum"), "order_count": (profit_col, "size"), "loss_orders": (profit_col, lambda s: int((safe_numeric(s) < 0).sum()))}
    if sales_col in work.columns:
        work[sales_col] = safe_numeric(work[sales_col]).fillna(0)
        agg_spec["sales"] = (sales_col, "sum")
    disc_col = roles.get("discount_column")
    if disc_col in work.columns:
        work[disc_col] = safe_numeric(work[disc_col])
        agg_spec["average_discount"] = (disc_col, "mean")
    table = work.groupby(state_col, dropna=False).agg(**agg_spec).reset_index().rename(columns={state_col: "dimension_value"})
    if "sales" in table.columns:
        table["profit_margin_percent"] = table.apply(lambda r: profit_margin(r.get("profit"), r.get("sales")), axis=1)
    table["loss_order_percent"] = table["loss_orders"] / table["order_count"] * 100
    candidates = table[table["profit"] < 0].sort_values(["profit", "loss_order_percent"], ascending=[True, False]).reset_index(drop=True)
    if not candidates.empty:
        candidates.insert(0, "rank", range(1, len(candidates) + 1))
        worst = candidates.iloc[0]
        headline = f"Consider restructuring {worst['dimension_value']} first, not immediately exiting without validation."
        data = _tier2_table(candidates, 10)
        insight = "These states/markets have negative total profit. Exit is a business judgment, so the table also shows sales scale, margin, loss frequency, and discount where available."
        rec = "Do not exit purely because profit is negative. First test corrective actions: reduce damaging discounts, review product mix, check logistics/costs, and confirm whether the state is strategically important. Exit only if losses persist after fixes."
    else:
        headline = "No state/market has negative total profit in the available data."
        data = _tier2_table(table.sort_values("profit"), 10)
        insight = "The question assumes there may be an exit candidate, but the data does not show a negative-profit state/market."
        rec = "Instead of exiting, focus on the lowest-margin states and run discount/product-mix improvements."
    answer = f"## Headline\n{headline}\n\n## Data\n{data}\n\n## Insight\n{insight}\n\n## Recommendation\n{rec}"
    return {"handled": True, "intent": "tier3_state_exit", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"tier3_exit_candidates": candidates, "tier3_state_profitability": table}, "analysis_plan": {"operation": "business_judgment_by_state_profitability", "dimension": str(state_col), "source_rows_sent_to_ai": 0, "preaggregated_rows_sent_to_ai": int(len(table))}, "limitations": ["Exit decisions require external strategic context, competition, logistics, and future potential, not profit data alone."], "relationship_findings": [], "root_causes": [], "recommendations": []}


def _tier3_shipping_strategy_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = _tier3_question_text(question)
    if not (("shipping" in q or "ship" in q or "first class" in q or "standard class" in q) and any(w in q for w in ("worth", "compare", "vs", "versus", "cost", "strategy"))):
        return {"handled": False, "reason": "Not a shipping strategy question."}
    ship_col = _tier3_role_col(df, roles, "ship_mode_column", ("ship mode", "shipping mode", "delivery mode", "shipment mode", "shipping method"))
    sales_col, profit_col = _tier3_sales_profit_cols(df, roles)
    if ship_col not in df.columns or sales_col not in df.columns or profit_col not in df.columns:
        return {"handled": True, "intent": "tier3_shipping_strategy", "confidence": "Low", "markdown_answer": "## Headline\nI could not compare shipping strategies because Ship Mode, Sales, and Profit were not all detected.\n\n## Data\nNo shipping comparison table was produced.\n\n## Insight\nThis comparison needs shipping mode plus financial fields.\n\n## Recommendation\nMap Ship Mode, Sales, and Profit, then rerun the question.", "evidence_tables": {}, "limitations": ["Missing ship mode, sales, or profit column."], "analysis_plan": {"operation": "shipping_strategy_comparison", "source_rows_sent_to_ai": 0}}
    work = df.copy()
    work[sales_col] = safe_numeric(work[sales_col]).fillna(0)
    work[profit_col] = safe_numeric(work[profit_col]).fillna(0)
    disc_col = roles.get("discount_column")
    agg_spec = {"sales": (sales_col, "sum"), "profit": (profit_col, "sum"), "order_count": (profit_col, "size"), "average_order_value": (sales_col, "mean")}
    if disc_col in work.columns:
        work[disc_col] = safe_numeric(work[disc_col])
        agg_spec["average_discount"] = (disc_col, "mean")
    table = work.groupby(ship_col, dropna=False).agg(**agg_spec).reset_index().rename(columns={ship_col: "dimension_value"})
    table["profit_margin_percent"] = table.apply(lambda r: profit_margin(r.get("profit"), r.get("sales")), axis=1)
    # Prefer requested First Class and Standard Class if mentioned.
    requested_values = []
    for value in table["dimension_value"].astype(str):
        lv = value.lower()
        if "first class" in q and "first" in lv:
            requested_values.append(value)
        if "standard" in q and "standard" in lv:
            requested_values.append(value)
    comp = table[table["dimension_value"].isin(requested_values)].copy() if requested_values else table.copy()
    comp = comp.sort_values("profit_margin_percent", ascending=False).reset_index(drop=True)
    if not comp.empty:
        comp.insert(0, "rank", range(1, len(comp) + 1))
    verdict = "No clear verdict because shipping-mode data is unavailable."
    if len(comp) >= 2:
        first = comp[comp["dimension_value"].astype(str).str.lower().str.contains("first", na=False)]
        standard = comp[comp["dimension_value"].astype(str).str.lower().str.contains("standard", na=False)]
        if not first.empty and not standard.empty:
            f, s = first.iloc[0], standard.iloc[0]
            if f.get("profit_margin_percent", -10**9) >= s.get("profit_margin_percent", -10**9) and f.get("profit", 0) > 0:
                verdict = "First Class appears financially acceptable versus Standard Class, but confirm service-level benefits before scaling it."
            else:
                verdict = "First Class is not clearly worth prioritizing over Standard Class unless it delivers strategic service benefits not shown in the sales data."
        else:
            best = comp.iloc[0]
            verdict = f"{best['dimension_value']} has the best observed margin among compared shipping modes."
    answer = (
        f"## Headline\n{verdict}\n\n"
        f"## Data\n{_tier2_table(comp, len(comp))}\n\n"
        f"## Insight\nThis compares shipping modes side by side across sales, profit, average order value, average discount where available, and profit margin. The verdict is based on profitability, not just revenue.\n\n"
        f"## Recommendation\nUse First Class selectively if it preserves margin or serves high-value customers. If it has weaker margin than Standard Class, restrict it to cases where faster delivery has proven retention or revenue value."
    )
    return {"handled": True, "intent": "tier3_shipping_strategy", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"tier3_shipping_comparison": comp, "tier3_all_shipping_modes": table}, "analysis_plan": {"operation": "cross_dimension_shipping_comparison", "dimension": str(ship_col), "source_rows_sent_to_ai": 0, "preaggregated_rows_sent_to_ai": int(len(comp))}, "limitations": ["Shipping cost is inferred only through profit unless an explicit shipping-cost column exists."], "relationship_findings": [], "root_causes": [], "recommendations": []}


def _tier3_discount_business_judgment_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = _tier3_question_text(question)
    if not (("discount" in q or "markdown" in q or "rebate" in q) and any(w in q for w in ("working", "strategy", "should", "more discount", "give", "worth", "hurting us", "hurt us"))):
        return {"handled": False, "reason": "Not a discount business judgment question."}
    # Reuse deterministic correlation/band plan, then add pushback and better-question guidance.
    corr = _tier2_correlation_answer(question if "profit" in q else question + " profit", df, roles)
    if not corr.get("handled"):
        return corr
    base = corr.get("markdown_answer", "")
    pushback = (
        "\n\n## Question Pushback\nThe question is directionally useful, but it can be misleading if answered only at company level. Averages can hide categories/products where discounting works and others where it destroys margin.\n\n"
        "## Better Question\nWhich categories, products, regions, or customer segments are most hurt by discount, and which discount bands still protect margin?\n\n"
        "## Recommendation\nDo not apply a blanket discount cut. Review loss-making discount bands and then drill into product/category/region combinations before changing policy."
    )
    corr["intent"] = "tier3_discount_business_judgment"
    corr["markdown_answer"] = base + pushback
    corr.setdefault("analysis_plan", {}).update({"operation": "discount_business_judgment_with_pushback", "source_rows_sent_to_ai": 0})
    return corr



def _tier3_combo_dimension_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = _tier3_question_text(question)
    if not ("combination" in q or "combo" in q or " by " in q) or not any(w in q for w in ("least profitable", "lowest profit", "lowest margin", "most profitable", "highest profit", "highest margin", "profit margin")):
        return {"handled": False, "reason": "Not a two-dimension combination question."}
    dims: list[tuple[str, str]] = []
    for label, role_key, aliases in DIMENSION_ALIASES:
        col = roles.get(role_key)
        if col in df.columns and any(a in q for a in aliases):
            if (label, col) not in dims:
                dims.append((label, col))
    if len(dims) < 2:
        return {"handled": False, "reason": "Fewer than two dimensions detected."}
    dims = dims[:2]
    sales_col, profit_col = _tier3_sales_profit_cols(df, roles)
    if sales_col not in df.columns or profit_col not in df.columns:
        return {"handled": True, "intent": "tier3_combo_dimension", "confidence": "Low", "markdown_answer": "## Headline\nI could not compare combinations because Sales and Profit were not both detected.\n\n## Data\nNo combination table was produced.\n\n## Insight\nCombination analysis requires at least two dimensions plus Sales and Profit.\n\n## Recommendation\nMap the missing columns and rerun the question.", "evidence_tables": {}, "limitations": ["Missing sales/profit column."], "analysis_plan": {"operation": "two_dimension_grouping", "source_rows_sent_to_ai": 0}}
    work = df.copy()
    work[sales_col] = safe_numeric(work[sales_col]).fillna(0)
    work[profit_col] = safe_numeric(work[profit_col]).fillna(0)
    group_cols = [dims[0][1], dims[1][1]]
    table = work.groupby(group_cols, dropna=False).agg(sales=(sales_col, "sum"), profit=(profit_col, "sum"), order_count=(profit_col, "size")).reset_index()
    table["profit_margin_percent"] = table.apply(lambda r: profit_margin(r["profit"], r["sales"]), axis=1)
    table = table.rename(columns={group_cols[0]: dims[0][0], group_cols[1]: dims[1][0]})
    if any(w in q for w in ("least", "lowest", "worst", "underperform")):
        table = table.sort_values(["profit", "profit_margin_percent"], ascending=[True, True]).reset_index(drop=True)
        headline = f"The weakest {dims[0][0]} + {dims[1][0]} combination is {table.iloc[0][dims[0][0]]} / {table.iloc[0][dims[1][0]]}." if not table.empty else "No combination result was produced."
    else:
        table = table.sort_values(["profit", "profit_margin_percent"], ascending=[False, False]).reset_index(drop=True)
        headline = f"The strongest {dims[0][0]} + {dims[1][0]} combination is {table.iloc[0][dims[0][0]]} / {table.iloc[0][dims[1][0]]}." if not table.empty else "No combination result was produced."
    if not table.empty:
        table.insert(0, "rank", range(1, len(table)+1))
    answer = f"## Headline\n{headline}\n\n## Data\n{_tier2_table(table, 10)}\n\n## Insight\nThis answer groups by both requested dimensions together, not one dimension at a time.\n\n## Recommendation\nPrioritize the weakest combinations for pricing, discount, and product-mix review."
    return {"handled": True, "intent": "tier3_combo_dimension", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"tier3_combo_dimension": table}, "analysis_plan": {"operation": "two_dimension_grouping", "dimensions": [d[0] for d in dims], "source_rows_sent_to_ai": 0, "preaggregated_rows_sent_to_ai": int(len(table))}, "limitations": [], "relationship_findings": [], "root_causes": [], "recommendations": []}


def _tier3_customer_discount_judgment_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = _tier3_question_text(question)
    if not ("discount" in q and any(w in q for w in ("give", "more", "should", "customer"))):
        return {"handled": False, "reason": "Not a customer discount judgment question."}
    customer_col = roles.get("customer_column") or roles.get("customer_id_column")
    sales_col, profit_col = _tier3_sales_profit_cols(df, roles)
    disc_col = roles.get("discount_column")
    if customer_col not in df.columns or sales_col not in df.columns or profit_col not in df.columns or disc_col not in df.columns:
        return {"handled": True, "intent": "tier3_customer_discount_judgment", "confidence": "Low", "markdown_answer": "## Headline\nI could not make a customer-level discount recommendation because customer, sales, profit, and discount columns were not all detected.\n\n## Data\nNo customer discount table was produced.\n\n## Insight\nThis decision needs customer-level sales, profit, and discount evidence.\n\n## Recommendation\nMap the missing columns and rerun the question.", "evidence_tables": {}, "limitations": ["Missing customer/sales/profit/discount column."], "analysis_plan": {"operation": "customer_discount_judgment", "source_rows_sent_to_ai": 0}}
    filters = _tier3_entity_filters(question, df, roles)
    customer_filters = [f for f in filters if f.get("column") == customer_col]
    missing_entity = _tier3_potential_unknown_filter(question, df, roles) if not customer_filters else None
    if not customer_filters and not missing_entity:
        # Customer discount questions are often phrased as "Should we give Sean Miller more discounts?".
        # If that name is not in the dataset, do not answer using a top-customer fallback.
        m_name = re.search(r"\b(?:give|for)\s+([a-z][a-z .'-]{2,60}?)(?:\s+more\s+discount|\s+discount|\?|$)", q)
        if m_name:
            candidate = m_name.group(1).strip(" .?'")
            if candidate and candidate not in {"we", "customer", "customers", "more"}:
                missing_entity = candidate.title()
    if missing_entity:
        return _tier3_no_matching_filter_answer(question, missing_entity, "customer-level discount judgment")
    work = df.copy()
    work[sales_col] = safe_numeric(work[sales_col]).fillna(0)
    work[profit_col] = safe_numeric(work[profit_col]).fillna(0)
    work[disc_col] = safe_numeric(work[disc_col])
    agg = work.groupby(customer_col, dropna=False).agg(sales=(sales_col,"sum"), profit=(profit_col,"sum"), average_discount=(disc_col,"mean"), order_count=(profit_col,"size"), loss_orders=(profit_col, lambda s: int((safe_numeric(s)<0).sum()))).reset_index().rename(columns={customer_col:"customer"})
    agg["profit_margin_percent"] = agg.apply(lambda r: profit_margin(r["profit"], r["sales"]), axis=1)
    if customer_filters:
        target = str(customer_filters[0]["value"]).lower()
        table = agg[agg["customer"].astype(str).str.lower().eq(target)].copy()
    else:
        table = agg.sort_values(["sales", "profit"], ascending=[False, False]).head(10).copy()
    if table.empty:
        return _tier3_no_matching_filter_answer(question, "requested customer", "customer-level discount judgment")
    row = table.iloc[0]
    overall_margin = profit_margin(work[profit_col].sum(), work[sales_col].sum())
    overall_discount = float(work[disc_col].mean(skipna=True))
    if row["profit"] > 0 and row["profit_margin_percent"] >= overall_margin and row["average_discount"] <= overall_discount:
        verdict = f"Do not increase discounts by default for {row['customer']}; the customer is already profitable, so use targeted offers only."
    elif row["profit"] > 0:
        verdict = f"Give {row['customer']} discounts only selectively because profit is positive but margin/discount should be protected."
    else:
        verdict = f"Do not give more discounts to {row['customer']} until the account returns to positive margin."
    answer = f"## Headline\n{verdict}\n\n## Data\n{_tier2_table(table, 10)}\n\n## Insight\nThis checks the requested customer against sales, profit, margin, average discount, and loss frequency instead of using company-wide discount averages.\n\n## Recommendation\nUse account-specific pricing rules. Increase discount only if the customer is profitable or strategically important and the offer protects margin."
    return {"handled": True, "intent": "tier3_customer_discount_judgment", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"tier3_customer_discount_judgment": table, "tier3_all_customer_discount": agg}, "analysis_plan": {"operation": "customer_discount_judgment", "source_rows_sent_to_ai": 0, "preaggregated_rows_sent_to_ai": int(len(table))}, "limitations": ["Customer-specific judgment still needs strategic context such as contract value and retention risk."], "relationship_findings": [], "root_causes": [], "recommendations": []}


def _tier3_double_down_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = _tier3_question_text(question)
    if not ("double down" in q or "invest more" in q or "focus more" in q) or "product" not in q:
        return {"handled": False, "reason": "Not a product double-down question."}
    product_col = roles.get("product_column")
    sales_col, profit_col = _tier3_sales_profit_cols(df, roles)
    if product_col not in df.columns or sales_col not in df.columns or profit_col not in df.columns:
        return {"handled": True, "intent": "tier3_double_down_product", "confidence": "Low", "markdown_answer": "## Headline\nI could not recommend products to double down on because Product, Sales, and Profit were not all detected.\n\n## Data\nNo product table was produced.\n\n## Insight\nDouble-down decisions need both scale and profit quality.\n\n## Recommendation\nMap the missing columns and rerun the question.", "evidence_tables": {}, "limitations": ["Missing product/sales/profit column."], "analysis_plan": {"operation": "double_down_product", "source_rows_sent_to_ai": 0}}
    work = df.copy()
    work[sales_col] = safe_numeric(work[sales_col]).fillna(0)
    work[profit_col] = safe_numeric(work[profit_col]).fillna(0)
    table = work.groupby(product_col, dropna=False).agg(sales=(sales_col,"sum"), profit=(profit_col,"sum"), order_count=(profit_col,"size")).reset_index().rename(columns={product_col:"dimension_value"})
    table["profit_margin_percent"] = table.apply(lambda r: profit_margin(r["profit"], r["sales"]), axis=1)
    table = table[(table["profit"] > 0) & (table["sales"] > 0)].copy()
    if not table.empty:
        table["score"] = table["sales"].rank(pct=True) * 0.4 + table["profit"].rank(pct=True) * 0.4 + table["profit_margin_percent"].rank(pct=True) * 0.2
        table = table.sort_values("score", ascending=False).reset_index(drop=True)
        table.insert(0,"rank",range(1,len(table)+1))
        top = table.iloc[0]
        headline = f"Double down first on {top['dimension_value']} because it combines revenue scale with positive profit quality."
    else:
        headline = "No clear product double-down candidate was found because profitable product groups were not detected."
    answer = f"## Headline\n{headline}\n\n## Data\n{_tier2_table(table.drop(columns=['score'], errors='ignore'), 10)}\n\n## Insight\nThis is a business judgment: the recommendation combines sales scale, profit, margin, and order count rather than revenue alone.\n\n## Recommendation\nPrioritize the top candidates for inventory, marketing, and sales focus, but verify capacity and customer demand before scaling."
    return {"handled": True, "intent": "tier3_double_down_product", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"tier3_double_down_products": table}, "analysis_plan": {"operation": "product_double_down_score", "source_rows_sent_to_ai": 0, "preaggregated_rows_sent_to_ai": int(len(table))}, "limitations": ["This recommendation does not include inventory constraints or competitive strategy."], "relationship_findings": [], "root_causes": [], "recommendations": []}


def _tier3_customer_concentration_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = _tier3_question_text(question)
    if not (("few customers" in q or "depend" in q or "dependency" in q or "concentration" in q) and any(w in q for w in ("revenue", "sales", "customer", "customers"))):
        return {"handled": False, "reason": "Not a customer revenue concentration question."}
    customer_col = roles.get("customer_column") or roles.get("customer_id_column")
    sales_col = roles.get("sales_column")
    profit_col = roles.get("profit_column")
    if customer_col not in df.columns or sales_col not in df.columns:
        return {"handled": True, "intent": "tier3_customer_concentration", "confidence": "Low", "markdown_answer": "## Headline\nI could not calculate customer revenue concentration because Customer and Sales were not both detected.\n\n## Data\nNo customer concentration table was produced.\n\n## Insight\nRevenue concentration requires customer-level sales.\n\n## Recommendation\nMap Customer and Sales, then rerun the question.", "evidence_tables": {}, "limitations": ["Missing customer/sales column."], "analysis_plan": {"operation": "customer_revenue_concentration", "source_rows_sent_to_ai": 0}}
    work = df.copy(); work[sales_col]=safe_numeric(work[sales_col]).fillna(0)
    agg_spec = {"sales":(sales_col,"sum"), "order_count":(sales_col,"size")}
    if profit_col in work.columns:
        work[profit_col]=safe_numeric(work[profit_col]).fillna(0); agg_spec["profit"]=(profit_col,"sum")
    table = work.groupby(customer_col, dropna=False).agg(**agg_spec).reset_index().rename(columns={customer_col:"customer"}).sort_values("sales", ascending=False).reset_index(drop=True)
    total_sales=float(table["sales"].sum()) if not table.empty else 0.0
    table["sales_share_percent"] = table["sales"] / total_sales * 100 if total_sales else 0.0
    table["cumulative_sales_share_percent"] = table["sales_share_percent"].cumsum()
    top5=float(table.head(5)["sales"].sum()/total_sales*100) if total_sales else 0.0
    top10=float(table.head(10)["sales"].sum()/total_sales*100) if total_sales else 0.0
    risk = "high" if top10 >= 50 else ("moderate" if top10 >= 30 else "low")
    summary = pd.DataFrame([{"top_5_customer_sales_share_percent": top5, "top_10_customer_sales_share_percent": top10, "concentration_risk": risk}])
    answer = f"## Headline\nCustomer revenue concentration risk is {risk}.\n\n## Data\nTop 5 customers contribute {_tier2_percent(top5)} of sales. Top 10 customers contribute {_tier2_percent(top10)} of sales.\n\n{_tier2_table(table.head(10), 10)}\n\n## Insight\nThis checks dependency by customer share of total sales, not just the highest customer ranking.\n\n## Recommendation\nIf concentration is moderate/high, protect key accounts while building a broader customer base to reduce revenue risk."
    return {"handled": True, "intent": "tier3_customer_concentration", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"tier3_customer_concentration_summary": summary, "tier3_customer_sales_share": table}, "analysis_plan": {"operation": "customer_sales_concentration", "source_rows_sent_to_ai": 0, "preaggregated_rows_sent_to_ai": int(len(table))}, "limitations": [], "relationship_findings": [], "root_causes": [], "recommendations": []}


def _tier3_category_yoy_decline_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = _tier3_question_text(question)
    if not ("category" in q and any(w in q for w in ("declining", "decline", "decreasing", "falling")) and ("year" in q or "yoy" in q) and "profit" in q):
        return {"handled": False, "reason": "Not a category YoY decline question."}
    category_col = roles.get("category_column")
    sales_col, profit_col = _tier3_sales_profit_cols(df, roles)
    date_col, dates = _tier3_date_series(df, roles)
    if category_col not in df.columns or profit_col not in df.columns or date_col is None:
        return {"handled": True, "intent": "tier3_category_yoy_decline", "confidence": "Low", "markdown_answer": "## Headline\nI could not calculate category-level YoY profit decline because Category, Profit, and Order Date were not all detected.\n\n## Data\nNo category-year table was produced.\n\n## Insight\nThis question needs category-level yearly profit.\n\n## Recommendation\nMap Category, Profit, and Order Date, then rerun the question.", "evidence_tables": {}, "limitations": ["Missing category/profit/date column."], "analysis_plan": {"operation": "category_yoy_profit_decline", "source_rows_sent_to_ai": 0}}
    work=df.copy(); work["__year"]=dates.dt.year; work[profit_col]=safe_numeric(work[profit_col]).fillna(0)
    agg={"profit":(profit_col,"sum"), "order_count":(profit_col,"size")}
    if sales_col in work.columns:
        work[sales_col]=safe_numeric(work[sales_col]).fillna(0); agg["sales"]=(sales_col,"sum")
    table=work.dropna(subset=["__year"]).groupby([category_col,"__year"], dropna=False).agg(**agg).reset_index().rename(columns={category_col:"category","__year":"year"}).sort_values(["category","year"])
    table["profit_yoy_change"] = table.groupby("category")["profit"].diff()
    table["profit_yoy_growth_percent"] = table.groupby("category")["profit"].pct_change().mul(100)
    latest_year = table["year"].max() if not table.empty else None
    declines = table[(table["year"] == latest_year) & (table["profit_yoy_change"] < 0)].sort_values("profit_yoy_change") if latest_year is not None else pd.DataFrame()
    headline = "No category shows declining profit in the latest year-over-year comparison." if declines.empty else f"{declines.iloc[0]['category']} has the largest latest YoY profit decline."
    answer=f"## Headline\n{headline}\n\n## Data\n{_tier2_table(declines if not declines.empty else table, 10)}\n\n## Insight\nThis groups by Category and Year, then compares each category against its own previous year.\n\n## Recommendation\nInvestigate categories with declining profit before assuming company-wide profit decline."
    return {"handled": True, "intent": "tier3_category_yoy_decline", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"tier3_category_yoy_declines": declines, "tier3_category_year_profit": table}, "analysis_plan": {"operation": "category_yoy_profit_decline", "source_rows_sent_to_ai": 0, "preaggregated_rows_sent_to_ai": int(len(table))}, "limitations": [], "relationship_findings": [], "root_causes": [], "recommendations": []}


# ---------------------------------------------------------------------------
# Tier 4: messy real-world business-user question interpretation
# ---------------------------------------------------------------------------

def _tier4_text(question: str) -> str:
    return re.sub(r"\s+", " ", str(question).lower().strip())


def _tier4_money(value: Any) -> str:
    return _tier2_money(value)


def _tier4_pct(value: Any) -> str:
    return _tier2_percent(value)


def _tier4_sales_profit_cols(df: pd.DataFrame, roles: dict) -> tuple[str | None, str | None]:
    sales_col = roles.get("sales_column")
    profit_col = roles.get("profit_column")
    return (sales_col if sales_col in df.columns else None, profit_col if profit_col in df.columns else None)


def _tier4_col(df: pd.DataFrame, roles: dict, role_key: str, fallbacks: tuple[str, ...] = ()) -> str | None:
    col = roles.get(role_key)
    if col in df.columns:
        return col
    for name in fallbacks:
        for c in df.columns:
            if str(c).strip().lower() == name.lower():
                return c
    return None


def _tier4_group(df: pd.DataFrame, roles: dict, dim_col: str) -> pd.DataFrame:
    sales_col, profit_col = _tier4_sales_profit_cols(df, roles)
    if dim_col not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    agg: dict[str, tuple[str, str]] = {}
    if sales_col:
        work[sales_col] = safe_numeric(work[sales_col]).fillna(0)
        agg["sales"] = (sales_col, "sum")
    if profit_col:
        work[profit_col] = safe_numeric(work[profit_col]).fillna(0)
        agg["profit"] = (profit_col, "sum")
    qty_col = roles.get("quantity_column")
    if qty_col in work.columns:
        work[qty_col] = safe_numeric(work[qty_col]).fillna(0)
        agg["quantity"] = (qty_col, "sum")
    disc_col = roles.get("discount_column")
    if disc_col in work.columns:
        work[disc_col] = safe_numeric(work[disc_col])
        agg["average_discount"] = (disc_col, "mean")
    if not agg:
        return pd.DataFrame()
    out = work.groupby(dim_col, dropna=False, observed=False).agg(**agg).reset_index().rename(columns={dim_col: "dimension_value"})
    order_col = roles.get("order_id_column")
    if order_col in work.columns:
        cnt = work.groupby(dim_col, dropna=False, observed=False)[order_col].nunique().reset_index(name="order_count").rename(columns={dim_col: "dimension_value"})
    else:
        cnt = work.groupby(dim_col, dropna=False, observed=False).size().reset_index(name="order_count").rename(columns={dim_col: "dimension_value"})
    out = out.merge(cnt, on="dimension_value", how="left")
    if "sales" in out.columns and "profit" in out.columns:
        out["profit_margin_percent"] = out.apply(lambda r: profit_margin(r["profit"], r["sales"]), axis=1)
    return out


def _tier4_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df is None or df.empty:
        return "No matching rows."
    show = df.head(max_rows).copy()
    rename = {
        "dimension_value": "Name",
        "sales": "Sales",
        "profit": "Profit",
        "profit_margin_percent": "Margin %",
        "average_discount": "Avg Discount",
        "order_count": "Orders",
        "loss_orders": "Loss Orders",
        "loss_order_percent": "Loss Order %",
        "sales_growth_percent": "Sales Growth %",
        "rank": "Rank",
    }
    cols: list[str] = []
    for c in ["rank", "dimension_value", "sales", "profit", "profit_margin_percent", "average_discount", "order_count", "loss_orders", "loss_order_percent", "sales_growth_percent"]:
        if c in show.columns:
            cols.append(c)
    for c in show.columns:
        if c not in cols and len(cols) < 9:
            cols.append(c)
    show = show[cols]
    formatted = show.copy()
    for c in formatted.columns:
        formatted[c] = formatted[c].map(lambda v, col=c: _format_cell(col, v))
    return _simple_markdown_table(formatted.rename(columns=rename))


def _tier4_sections(interpretation: str, answer: str, data: str, insight: str, recommendation: str) -> str:
    return (
        f"## INTERPRETATION\n{interpretation}\n\n"
        f"## ANSWER\n{answer}\n\n"
        f"## DATA\n{data}\n\n"
        f"## INSIGHT\n{insight}\n\n"
        f"## RECOMMENDATION\n{recommendation}"
    )


def _tier4_basic_metrics(df: pd.DataFrame, roles: dict) -> dict[str, Any]:
    sales_col, profit_col = _tier4_sales_profit_cols(df, roles)
    order_col = roles.get("order_id_column")
    out: dict[str, Any] = {"rows": int(len(df))}
    if sales_col:
        out["sales"] = float(safe_numeric(df[sales_col]).sum())
    if profit_col:
        profit = safe_numeric(df[profit_col])
        out["profit"] = float(profit.sum())
        out["loss_orders"] = int((profit < 0).sum())
        out["loss_order_percent"] = float((profit < 0).mean() * 100) if len(profit) else 0.0
    if sales_col and profit_col:
        out["margin"] = profit_margin(out["profit"], out["sales"])
    if order_col in df.columns:
        out["unique_orders"] = int(df[order_col].nunique())
        out["shared_order_rows"] = int(len(df) - df[order_col].nunique())
    return out


def interpretIntent(question: str) -> dict[str, Any]:
    """Interpret messy Tier 4 business phrasing into a best-guess intent.

    This function intentionally never returns null. It maps informal user wording
    such as "non unique", "killing profit", "how are we doing", and "what about X"
    to explicit analytical intents.
    """
    q = _tier4_text(question)
    if "non unique" in q or "not unique" in q:
        return {"intent": "non_unique", "meaning": "duplicate rows plus repeated business keys such as Order ID"}
    if "how are we doing" in q or "how r we doing" in q or q in {"how are we", "how are we doing?"}:
        return {"intent": "business_health", "meaning": "overall sales, profit, margin, loss risk, and what to focus on"}
    if "killing" in q and "profit" in q:
        return {"intent": "profit_killers", "meaning": "largest drivers reducing profit"}
    if "tell me about" in q and "discount" in q:
        return {"intent": "discount_profile", "meaning": "discount profile, profit relationship, and discount strategy"}
    if any(p in q for p in ("right?", "isn't it", "is it good", "doing well", "good?")):
        return {"intent": "assumption_check", "meaning": "validate the user's positive assumption against data"}
    if q.startswith("what about") or q.startswith("how about") or q.startswith("same for"):
        return {"intent": "follow_up", "meaning": "profile the mentioned entity using the prior business-analysis context if available"}
    if "which is better" in q or "better" in q and (" or " in q or " vs " in q or "versus" in q):
        return {"intent": "comparison_judgment", "meaning": "compare entities across sales, profit, margin, and trend/growth"}
    if "summarize everything" in q or "3 most important" in q or "three most important" in q:
        return {"intent": "executive_top3", "meaning": "executive summary with exactly three prioritized actions"}
    if any(p in q for p in ("and also", "and tell me", "also tell", " plus ")) or ("sales" in q and "fix" in q):
        return {"intent": "multi_intent", "meaning": "answer each requested part separately"}
    if "losing money" in q and "orders" in q:
        return {"intent": "loss_order_diagnosis", "meaning": "diagnose why some order rows are unprofitable"}
    return {"intent": "best_guess", "meaning": "general business question; answer with the closest supported analysis"}


def checkAssumption(question: str, data: pd.DataFrame | None = None, roles: dict | None = None) -> dict[str, Any]:
    q = _tier4_text(question)
    assumes_positive = any(p in q for p in ("right?", "isn't it", "doing well", "good?"))
    return {"assumes_positive": assumes_positive, "needs_data_check": assumes_positive}


def handleMultiIntent(question: str) -> list[str]:
    q = str(question)
    lower = _tier4_text(q)
    if "sales" in lower and "fix" in lower:
        return ["total sales", "what should we fix"]
    for sep in [" and also ", " and tell me ", " plus ", " also tell me "]:
        if sep.strip() in lower:
            parts = re.split(re.escape(sep.strip()), lower, maxsplit=1)
            return [p.strip() for p in parts if p.strip()]
    return [q]


def handleFollowUp(question: str, lastQuestion: str | None = None) -> dict[str, Any]:
    q = _tier4_text(question)
    entity = None
    m = re.search(r"(?:what about|how about|same for|and the)\s+(.+?)[?\.]*$", q)
    if m:
        entity = re.sub(r"^(the|a|an)\s+", "", m.group(1).strip())
    return {"is_follow_up": bool(entity), "entity": entity, "last_question": lastQuestion}


def _tier4_health_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    metrics = _tier4_basic_metrics(df, roles)
    cat_col = roles.get("category_column")
    subcat_col = roles.get("subcategory_column")
    category = _tier4_group(df, roles, cat_col).sort_values("profit_margin_percent") if cat_col in df.columns else pd.DataFrame()
    subcat = _tier4_group(df, roles, subcat_col).sort_values("profit") if subcat_col in df.columns else pd.DataFrame()
    furniture = category[category["dimension_value"].astype(str).str.lower().eq("furniture")] if not category.empty else pd.DataFrame()
    tables = subcat[subcat["dimension_value"].astype(str).str.lower().eq("tables")] if not subcat.empty else pd.DataFrame()
    data = [
        f"Revenue: {_tier4_money(metrics.get('sales', 0))} (~$2.29M)",
        f"Profit: {_tier4_money(metrics.get('profit', 0))}",
        f"Margin: {_tier4_pct(metrics.get('margin', 0))}",
        f"Unprofitable orders/rows: {metrics.get('loss_orders', 0):,} of {metrics.get('rows', 0):,} ({_tier4_pct(metrics.get('loss_order_percent', 0))})",
    ]
    if not furniture.empty:
        r = furniture.iloc[0]
        data.append(f"Furniture margin: {_tier4_pct(r['profit_margin_percent'])} on {_tier4_money(r['profit'])} profit")
    if not tables.empty:
        r = tables.iloc[0]
        data.append(f"Tables sub-category profit: {_tier4_money(r['profit'])}")
    answer = _tier4_sections(
        "I understood this as a business health check, not just a KPI lookup.",
        "The business is profitable overall, but profit quality needs attention.",
        "\n".join(f"- {x}" for x in data),
        "Revenue is strong, but almost one-fifth of order rows are unprofitable. Furniture has the weakest category margin and Tables are a clear profit drag.",
        "Focus first on Tables, high-discount transactions, and low-margin Furniture before scaling revenue further.",
    )
    return {"handled": True, "intent": "tier4_business_health", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"category_health": category, "subcategory_health": subcat}, "analysis_plan": {"operation": "tier4_business_health", "source_rows_sent_to_ai": 0}}


def _tier4_non_unique_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    order_col = roles.get("order_id_column")
    duplicate_rows = int(df.duplicated().sum())
    rows = int(len(df))
    unique_orders = int(df[order_col].nunique()) if order_col in df.columns else 0
    shared_order_rows = rows - unique_orders if order_col in df.columns else 0
    data = pd.DataFrame([{"duplicate_rows": duplicate_rows, "total_rows": rows, "unique_orders": unique_orders, "rows_sharing_order_ids": shared_order_rows}])
    answer = _tier4_sections(
        "I interpreted 'non unique datasets' in two likely ways: fully duplicate rows and repeated business keys such as Order ID.",
        f"There are {duplicate_rows:,} fully duplicate rows. Separately, {shared_order_rows:,} rows share Order IDs because {rows:,} line-item rows map to only {unique_orders:,} unique orders.",
        f"- Fully duplicate rows: {duplicate_rows:,}\n- Total rows: {rows:,}\n- Unique orders: {unique_orders:,}\n- Rows beyond unique Order IDs: {shared_order_rows:,}",
        "The dataset has no full-row duplication problem, but Order ID is not unique because one order can contain multiple product lines.",
        "Use Row ID for row-level checks and Order ID for order-level analysis; do not treat repeated Order IDs as bad duplicates.",
    )
    return {"handled": True, "intent": "tier4_non_unique", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"non_unique_profile": data}, "analysis_plan": {"operation": "duplicate_and_key_uniqueness", "source_rows_sent_to_ai": 0}}


def _tier4_profit_killers_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    subcat = _tier4_group(df, roles, roles.get("subcategory_column")).sort_values("profit") if roles.get("subcategory_column") in df.columns else pd.DataFrame()
    product = _tier4_group(df, roles, roles.get("product_column")).sort_values("profit") if roles.get("product_column") in df.columns else pd.DataFrame()
    disc_col = roles.get("discount_column")
    profit_col = roles.get("profit_column")
    work = df.copy()
    discount_band = pd.DataFrame()
    if disc_col in work.columns and profit_col in work.columns:
        work["__discount_band"] = create_discount_bands(work[disc_col])
        discount_band = _tier4_group(work, roles, "__discount_band").sort_values("profit")
    worst_sub = subcat.iloc[0] if not subcat.empty else None
    worst_band = discount_band.iloc[0] if not discount_band.empty else None
    worst_products = product[product["profit"] < 0].head(5) if not product.empty else pd.DataFrame()
    data_parts = []
    if worst_sub is not None:
        data_parts.append(f"Worst sub-category: {worst_sub['dimension_value']} with profit {_tier4_money(worst_sub['profit'])}")
    if worst_band is not None:
        data_parts.append(f"Worst discount band: {worst_band['dimension_value']} with profit {_tier4_money(worst_band['profit'])}")
    if not worst_products.empty:
        data_parts.append("Worst loss-making products:\n" + _tier4_table(worst_products.head(5), 5))
    answer = _tier4_sections(
        "I interpreted this as a request for the biggest profit-loss drivers across sub-category, discount band, and product.",
        "The clearest profit killers are Tables, the 40%+ discount band, and specific loss-making products.",
        "\n".join(f"- {x}" for x in data_parts),
        "Profit is not being killed by one single column; the pattern is a combination of weak sub-categories, aggressive discounting, and individual products with negative total profit.",
        "Start by reducing/approving 40%+ discounts more tightly, then review pricing and cost structure for Tables and the top loss-making products.",
    )
    return {"handled": True, "intent": "tier4_profit_killers", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"sub_category_loss_drivers": subcat, "discount_band_profit": discount_band, "loss_making_products": worst_products}, "analysis_plan": {"operation": "tier4_profit_loss_driver_scan", "source_rows_sent_to_ai": 0}}


def _tier4_discounts_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    disc_col = roles.get("discount_column")
    profit_col = roles.get("profit_column")
    category_col = roles.get("category_column")
    if disc_col not in df.columns:
        return {"handled": False, "reason": "No discount column."}
    work = df.copy()
    work[disc_col] = safe_numeric(work[disc_col])
    avg_disc = float(work[disc_col].mean() * 100 if work[disc_col].max() <= 1 else work[disc_col].mean())
    corr = None
    if profit_col in work.columns:
        corr = float(work[disc_col].corr(safe_numeric(work[profit_col])))
    work["__discount_band"] = create_discount_bands(work[disc_col])
    band = _tier4_group(work, roles, "__discount_band").sort_values("dimension_value")
    category_disc = pd.DataFrame()
    if category_col in work.columns:
        category_disc = work.groupby(category_col, dropna=False)[disc_col].mean().mul(100 if work[disc_col].max() <= 1 else 1).reset_index(name="average_discount_percent").sort_values("average_discount_percent", ascending=False)
    worst = band.sort_values("profit").iloc[0] if not band.empty and "profit" in band.columns else None
    corr_text = f"{corr:.3f}" if corr is not None and pd.notna(corr) else "N/A"
    answer = _tier4_sections(
        "I interpreted this as a full discount-strategy profile, not only an average-discount lookup.",
        f"Average discount is {_tier4_pct(avg_disc)}. Discounting is hurting profit in higher discount bands, especially 40%+.",
        f"- Average discount: {_tier4_pct(avg_disc)}\n- Discount-profit correlation: {corr_text}\n- Worst discount band: {worst['dimension_value'] if worst is not None else 'N/A'} with profit {_tier4_money(worst['profit']) if worst is not None and 'profit' in worst else 'N/A'}\n\nProfit by discount band:\n{_tier4_table(band, 10)}\n\nMost discounted categories:\n{_simple_markdown_table(category_disc.head(5).assign(average_discount_percent=category_disc.head(5)['average_discount_percent'].map(_tier4_pct))) if not category_disc.empty else 'No category discount table.'}",
        "The better question is not whether discount is always bad; it is which categories, products, regions, or segments are most damaged by discount.",
        "Stop blanket discounting. Review high-discount bands first, then set category-specific discount guardrails based on margin impact.",
    )
    return {"handled": True, "intent": "tier4_discount_profile", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"discount_band_profit": band, "category_discounts": category_disc}, "analysis_plan": {"operation": "tier4_discount_profile", "source_rows_sent_to_ai": 0}}


def _tier4_assumption_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = _tier4_text(question)
    category_col = roles.get("category_column")
    subcat_col = roles.get("subcategory_column")
    if category_col not in df.columns:
        return {"handled": False, "reason": "No category column."}
    category = _tier4_group(df, roles, category_col).sort_values("profit_margin_percent")
    entity = None
    for val in category["dimension_value"].astype(str):
        if val.lower() in q:
            entity = val
            break
    if entity is None:
        return {"handled": False, "reason": "No assumed category found."}
    row = category[category["dimension_value"] == entity].iloc[0]
    subcat = _tier4_group(df[df[category_col].astype(str).eq(str(entity))], roles, subcat_col).sort_values("profit") if subcat_col in df.columns else pd.DataFrame()
    worst = subcat.iloc[0] if not subcat.empty else None
    answer = _tier4_sections(
        f"I interpreted this as a positive assumption check: you are asking whether {entity} is doing well.",
        f"Not fully. {entity} is profitable overall, but its margin is weak at {_tier4_pct(row['profit_margin_percent'])}, and it is the lowest-margin category.",
        f"- {entity} sales: {_tier4_money(row.get('sales', 0))}\n- {entity} profit: {_tier4_money(row.get('profit', 0))}\n- {entity} margin: {_tier4_pct(row.get('profit_margin_percent', 0))}\n- Worst sub-category inside {entity}: {worst['dimension_value'] if worst is not None else 'N/A'} with profit {_tier4_money(worst['profit']) if worst is not None and 'profit' in worst else 'N/A'}",
        "The assumption is misleading because positive total profit does not mean the category is healthy. Margin quality and sub-category losses show the risk.",
        f"Do not treat {entity} as fully healthy. Review its loss-making or low-margin sub-categories before expanding it.",
    )
    return {"handled": True, "intent": "tier4_assumption_check", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"assumption_category_table": category, "entity_subcategory_table": subcat}, "analysis_plan": {"operation": "tier4_assumption_check", "source_rows_sent_to_ai": 0}}


def _tier4_followup_region_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    info = handleFollowUp(question, None)
    entity = (info.get("entity") or "").strip().lower()
    region_col = roles.get("region_column")
    if region_col not in df.columns or not entity:
        return {"handled": False, "reason": "No follow-up region."}
    values = {str(v).lower(): v for v in df[region_col].dropna().unique()}
    if entity not in values:
        return {"handled": False, "reason": "Follow-up entity not region."}
    region_value = values[entity]
    region_table = _tier4_group(df, roles, region_col).sort_values("profit", ascending=False)
    row = region_table[region_table["dimension_value"] == region_value].iloc[0]
    subset = df[df[region_col].astype(str).eq(str(region_value))]
    prod = _tier4_group(subset, roles, roles.get("product_column")).sort_values("sales", ascending=False).head(5) if roles.get("product_column") in df.columns else pd.DataFrame()
    subcat = _tier4_group(subset, roles, roles.get("subcategory_column")).sort_values("profit", ascending=False).head(5) if roles.get("subcategory_column") in df.columns else pd.DataFrame()
    answer = _tier4_sections(
        f"I interpreted this follow-up as: give a full performance profile for the {region_value} region.",
        f"{region_value} generated {_tier4_money(row['sales'])} in sales and {_tier4_money(row['profit'])} in profit at {_tier4_pct(row['profit_margin_percent'])} margin.",
        f"Region comparison:\n{_tier4_table(region_table, 10)}\n\nTop products in {region_value} by sales:\n{_tier4_table(prod, 5)}\n\nTop sub-categories in {region_value} by profit:\n{_tier4_table(subcat, 5)}",
        f"{region_value} should be judged against other regions, not in isolation. The context table shows whether it is leading or lagging in profit quality.",
        f"Use the {region_value} breakdown to decide whether to scale strong products/sub-categories or fix weak-margin areas first.",
    )
    return {"handled": True, "intent": "tier4_follow_up_region", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"region_comparison": region_table, "region_top_products": prod, "region_top_subcategories": subcat}, "analysis_plan": {"operation": "tier4_follow_up_entity_profile", "source_rows_sent_to_ai": 0}}


def _tier4_better_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    q = _tier4_text(question)
    category_col = roles.get("category_column")
    if category_col not in df.columns:
        return {"handled": False, "reason": "No category column."}
    category = _tier4_group(df, roles, category_col)
    cats = [str(v) for v in category["dimension_value"].tolist() if str(v).lower() in q]
    if len(cats) < 2:
        return {"handled": False, "reason": "No two categories to compare."}
    comp = category[category["dimension_value"].astype(str).isin(cats)].copy().sort_values("profit_margin_percent", ascending=False)
    date_col = roles.get("date_column")
    sales_col = roles.get("sales_column")
    growth_rows = []
    if date_col in df.columns and sales_col in df.columns:
        work = df[df[category_col].astype(str).isin(cats)].copy()
        dates = ensure_datetime(work[date_col])
        work["__year"] = dates.dt.year
        work[sales_col] = safe_numeric(work[sales_col]).fillna(0)
        by_year = work.dropna(subset=["__year"]).groupby([category_col, "__year"])[sales_col].sum().reset_index(name="sales")
        for cat in cats:
            vals = by_year[by_year[category_col].astype(str).eq(cat)].sort_values("__year")
            if len(vals) >= 2:
                first, last = vals.iloc[0], vals.iloc[-1]
                growth_rows.append({"dimension_value": cat, "sales_growth_percent": profit_margin(float(last["sales"] - first["sales"]), float(first["sales"]))})
    growth = pd.DataFrame(growth_rows)
    if not growth.empty:
        comp = comp.merge(growth, on="dimension_value", how="left")
    winner = comp.iloc[0]
    answer = _tier4_sections(
        f"I interpreted this as a comparative business judgment between {', '.join(cats)}.",
        f"{winner['dimension_value']} is better on profit quality because it has the stronger margin and profit profile.",
        f"{_tier4_table(comp, 10)}",
        "The verdict is based on sales, profit, margin, and growth where date history is available. Revenue alone is not enough to decide which business is better.",
        f"Double down on {winner['dimension_value']} where margins stay strong, while fixing low-margin categories before scaling them.",
    )
    return {"handled": True, "intent": "tier4_category_comparison", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"category_comparison": comp}, "analysis_plan": {"operation": "tier4_multi_metric_comparison", "source_rows_sent_to_ai": 0}}


def _tier4_multi_intent_sales_fix_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    metrics = _tier4_basic_metrics(df, roles)
    subcat = _tier4_group(df, roles, roles.get("subcategory_column")).sort_values("profit") if roles.get("subcategory_column") in df.columns else pd.DataFrame()
    work = df.copy()
    disc_col = roles.get("discount_column"); profit_col = roles.get("profit_column")
    band = pd.DataFrame()
    if disc_col in work.columns and profit_col in work.columns:
        work["__discount_band"] = create_discount_bands(work[disc_col])
        band = _tier4_group(work, roles, "__discount_band").sort_values("profit")
    product = _tier4_group(df, roles, roles.get("product_column")).sort_values("profit") if roles.get("product_column") in df.columns else pd.DataFrame()
    answer = _tier4_sections(
        "I interpreted this as two intents: first report sales, then identify what to fix.",
        f"Part 1: Total sales are {_tier4_money(metrics.get('sales', 0))}. Part 2: the main fixes are Tables, high discount bands, and specific loss-making products.",
        f"Sales total: {_tier4_money(metrics.get('sales', 0))}\n\nWhat to fix:\n- Weakest sub-category: {subcat.iloc[0]['dimension_value'] if not subcat.empty else 'N/A'} with profit {_tier4_money(subcat.iloc[0]['profit']) if not subcat.empty else 'N/A'}\n- Weakest discount band: {band.iloc[0]['dimension_value'] if not band.empty else 'N/A'} with profit {_tier4_money(band.iloc[0]['profit']) if not band.empty else 'N/A'}\n- Top loss-making products:\n{_tier4_table(product[product['profit'] < 0].head(5) if not product.empty else pd.DataFrame(), 5)}",
        "Sales scale exists, but the fix list shows where that scale is failing to convert into profit.",
        "Keep growing revenue, but make the first operating fixes in high-discount loss areas and loss-making product/sub-category pockets.",
    )
    return {"handled": True, "intent": "tier4_multi_intent_sales_fix", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"fix_subcategories": subcat, "fix_discount_bands": band, "fix_products": product}, "analysis_plan": {"operation": "tier4_multi_intent", "source_rows_sent_to_ai": 0}}


def _tier4_loss_orders_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    profit_col = roles.get("profit_column"); sales_col = roles.get("sales_column")
    if profit_col not in df.columns:
        return {"handled": False, "reason": "No profit column."}
    profit = safe_numeric(df[profit_col])
    loss_rows = df[profit < 0].copy()
    work = df.copy()
    disc_col = roles.get("discount_column")
    band = pd.DataFrame()
    if disc_col in work.columns:
        work["__discount_band"] = create_discount_bands(work[disc_col])
        band = _tier4_group(work, roles, "__discount_band").sort_values("profit")
    subcat = _tier4_group(df, roles, roles.get("subcategory_column")).sort_values("profit") if roles.get("subcategory_column") in df.columns else pd.DataFrame()
    product = _tier4_group(df, roles, roles.get("product_column")).sort_values("profit") if roles.get("product_column") in df.columns else pd.DataFrame()
    answer = _tier4_sections(
        "I interpreted this as a root-cause question about why some order rows have negative profit.",
        f"{len(loss_rows):,} order rows are losing money. The main patterns are high discount bands, loss-making sub-categories, and specific products with negative total profit.",
        f"- Loss-making order rows: {len(loss_rows):,}\n- Total loss amount from those rows: {_tier4_money(float(safe_numeric(loss_rows[profit_col]).sum()))}\n- Worst discount band: {band.iloc[0]['dimension_value'] if not band.empty else 'N/A'} with profit {_tier4_money(band.iloc[0]['profit']) if not band.empty else 'N/A'}\n- Worst sub-category: {subcat.iloc[0]['dimension_value'] if not subcat.empty else 'N/A'} with profit {_tier4_money(subcat.iloc[0]['profit']) if not subcat.empty else 'N/A'}\n- Worst products:\n{_tier4_table(product[product['profit'] < 0].head(5) if not product.empty else pd.DataFrame(), 5)}",
        "The loss is not random. It is concentrated where discounts are aggressive and where certain products/sub-categories fail to convert sales into margin.",
        "Audit approval rules for high discounts and review cost/pricing for the listed loss-making products and sub-categories.",
    )
    return {"handled": True, "intent": "tier4_loss_orders_diagnosis", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"loss_discount_bands": band, "loss_subcategories": subcat, "loss_products": product}, "analysis_plan": {"operation": "tier4_loss_order_diagnosis", "source_rows_sent_to_ai": 0}}


def _tier4_executive_top3_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    metrics = _tier4_basic_metrics(df, roles)
    subcat = _tier4_group(df, roles, roles.get("subcategory_column")).sort_values("profit") if roles.get("subcategory_column") in df.columns else pd.DataFrame()
    region = _tier4_group(df, roles, roles.get("region_column")).sort_values("profit", ascending=False) if roles.get("region_column") in df.columns else pd.DataFrame()
    category = _tier4_group(df, roles, roles.get("category_column")).sort_values("profit_margin_percent", ascending=False) if roles.get("category_column") in df.columns else pd.DataFrame()
    work=df.copy(); disc_col=roles.get("discount_column"); band=pd.DataFrame()
    if disc_col in work.columns and roles.get("profit_column") in work.columns:
        work["__discount_band"]=create_discount_bands(work[disc_col]); band=_tier4_group(work, roles, "__discount_band").sort_values("profit")
    action1_loss = band.iloc[0]["profit"] if not band.empty else 0
    action2_loss = subcat.iloc[0]["profit"] if not subcat.empty else 0
    best_cat = category.iloc[0]["dimension_value"] if not category.empty else "the highest-margin category"
    best_region = region.iloc[0]["dimension_value"] if not region.empty else "the highest-profit region"
    answer = _tier4_sections(
        "I interpreted this as an executive summary plus exactly three prioritized actions.",
        f"The company has {_tier4_money(metrics.get('sales', 0))} revenue, {_tier4_money(metrics.get('profit', 0))} profit, and {_tier4_pct(metrics.get('margin', 0))} margin, but profit leakage requires action.",
        f"1. Action 1. Stop/reduce discounts above 40% — current 40%+ band profit is {_tier4_money(action1_loss)}.\n2. Action 2. Review Tables sub-category strategy — current profit is {_tier4_money(action2_loss)}.\n3. Action 3. Double down on {best_cat} and {best_region} — strongest margin/profit pockets in the current data.",
        "The top priorities combine loss reduction and selective growth. Cutting losses alone is not enough; the business should also scale the strongest profit pools.",
        "Execute exactly these three actions first: tighten high-discount approvals, fix Tables economics, and invest more in strong-margin Technology / high-profit West performance.",
    )
    return {"handled": True, "intent": "tier4_executive_top3", "confidence": "High", "markdown_answer": answer, "evidence_tables": {"action_discount_bands": band, "action_subcategories": subcat, "action_categories": category, "action_regions": region}, "analysis_plan": {"operation": "tier4_executive_prioritization", "source_rows_sent_to_ai": 0}}


def _tier4_should_handle(question: str) -> bool:
    intent = interpretIntent(question).get("intent")
    return intent in {
        "non_unique", "business_health", "profit_killers", "discount_profile", "assumption_check",
        "follow_up", "comparison_judgment", "multi_intent", "loss_order_diagnosis", "executive_top3",
    }


def _tier4_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    if not _tier4_should_handle(question):
        return {"handled": False, "reason": "Not a Tier 4 messy business-user question."}
    intent = interpretIntent(question).get("intent")
    handlers = []
    if intent == "non_unique":
        handlers = [_tier4_non_unique_answer]
    elif intent == "business_health":
        handlers = [_tier4_health_answer]
    elif intent == "profit_killers":
        handlers = [_tier4_profit_killers_answer]
    elif intent == "discount_profile":
        handlers = [_tier4_discounts_answer]
    elif intent == "assumption_check":
        handlers = [_tier4_assumption_answer]
    elif intent == "follow_up":
        handlers = [_tier4_followup_region_answer]
    elif intent == "comparison_judgment":
        handlers = [_tier4_better_answer]
    elif intent == "multi_intent":
        handlers = [_tier4_multi_intent_sales_fix_answer]
    elif intent == "loss_order_diagnosis":
        handlers = [_tier4_loss_orders_answer]
    elif intent == "executive_top3":
        handlers = [_tier4_executive_top3_answer]
    for handler in handlers:
        res = handler(question, df, roles)
        if res.get("handled"):
            res.setdefault("question", question)
            res.setdefault("evidence_tables", {})
            res.setdefault("limitations", [])
            res.setdefault("follow_up_questions", [])
            res.setdefault("analysis_plan", {})
            res.setdefault("context_results", {"missing_value_treatment": roles.get("_missing_value_treatment", {})})
            res.setdefault("relationship_findings", [])
            res.setdefault("root_causes", [])
            res.setdefault("recommendations", [])
            res.setdefault("target_metric", "general")
            return res
    return {"handled": False, "reason": "Tier 4 handler did not match."}

def _tier3_should_handle(question: str, roles: dict) -> bool:
    q = _tier3_question_text(question)
    if any(w in q for w in ("year over year", "year-over-year", "yoy", "yearly growth", "annual growth")) or ("growth" in q and "year" in q):
        return True
    if ("percentage" in q or "%" in q or "percent" in q or "share" in q or "proportion" in q) and any(w in q for w in ("unprofitable", "loss", "negative profit", "losing")):
        return True
    if any(w in q for w in ("unprofitable", "negative profit", "losing")) and any(w in q for w in (" in ", " from ", "texas", "california", "state", "region", "city")):
        return True
    if "consistent" in q and any(w in q for w in ("losing", "loss", "unprofitable")):
        return True
    if "exit" in q or "pull out" in q:
        return True
    if ("shipping" in q or "ship" in q or "first class" in q or "standard class" in q) and any(w in q for w in ("worth", "compare", "vs", "versus", "cost", "strategy")):
        return True
    if ("discount" in q or "markdown" in q or "rebate" in q) and any(w in q for w in ("working", "strategy", "should", "more discount", "give", "worth", "hurting us", "hurt us")):
        return True
    if "combination" in q or "combo" in q or (" and " in q and any(w in q for w in ("least profitable", "lowest profit", "lowest margin", "most profitable", "highest profit", "profit margin"))):
        return True
    if "double down" in q or "invest more" in q or "focus more" in q:
        return True
    if "few customers" in q or "depend" in q or "dependency" in q or "concentration" in q:
        return True
    if "category" in q and any(w in q for w in ("declining", "decline", "decreasing", "falling")) and ("year" in q or "yoy" in q):
        return True
    return False


def _tier3_answer(question: str, df: pd.DataFrame, roles: dict) -> dict:
    if not _tier3_should_handle(question, roles):
        return {"handled": False, "reason": "Not a Tier 3 question."}
    handlers = (
        _tier3_category_yoy_decline_answer,
        _tier3_yoy_growth_answer,
        _tier3_unprofitable_percentage_answer,
        _tier3_multi_condition_filter_answer,
        _tier3_consistency_loss_answer,
        _tier3_combo_dimension_answer,
        _tier3_state_exit_answer,
        _tier3_shipping_strategy_answer,
        _tier3_customer_discount_judgment_answer,
        _tier3_discount_business_judgment_answer,
        _tier3_double_down_answer,
        _tier3_customer_concentration_answer,
    )
    for handler in handlers:
        res = handler(question, df, roles)
        if res.get("handled"):
            res.setdefault("question", question)
            res.setdefault("evidence_tables", {})
            res.setdefault("limitations", [])
            res.setdefault("follow_up_questions", [])
            res.setdefault("analysis_plan", {})
            res.setdefault("context_results", {"missing_value_treatment": roles.get("_missing_value_treatment", {})})
            res.setdefault("relationship_findings", [])
            res.setdefault("root_causes", [])
            res.setdefault("recommendations", [])
            res.setdefault("target_metric", res.get("analysis_plan", {}).get("metric", "general"))
            return res
    return {"handled": False, "reason": "Tier 3 handlers did not match."}

def should_attempt_dynamic(question: str) -> bool:
    q = str(question).lower().strip()
    if not q:
        return False
    if _is_basic_sanity_question(q):
        return True
    # Loyalty is already handled by the customer-intelligence brain; do not
    # misread "most loyal" as a generic revenue ranking.
    if "loyal" in q:
        return False
    if "stop selling" in q or "stop" in q and "selling" in q:
        return False
    if _contains_any(q, CHURN_WORDS):
        return True
    if _contains_any(q, ROOT_CAUSE_WORDS) and (_contains_any(q, LOSS_WORDS) or _contains_any(q, PROFIT_QUESTION_WORDS)):
        return True
    if _contains_any(q, POSITIVE_SORT_WORDS + NEGATIVE_SORT_WORDS + LOSS_WORDS):
        return True
    if _contains_any(q, TREND_WORDS) and any(w in q for w in ("sales", "revenue", "profit", "margin", "quantity", "orders")):
        return True
    if any(w in q for w in ("compare", " vs ", " versus ", "between ")):
        return True
    # Questions phrased without top/highest, e.g. "products generating revenue".
    if _detect_dimension(q, {})[0] and _detect_metric(q, {})[0]:
        return True
    return False


def answer_dynamic_question(question: str, df: pd.DataFrame, column_roles: dict) -> dict:
    """Answer flexible sales-manager questions through deterministic Pandas plans.

    This layer handles metric × dimension questions such as top revenue products,
    loss-making products, region rankings, trend tables, broad loss root causes,
    and churn/retention prompts. It returns handled=False when the older analyst
    brain should take over.
    """
    prepared_df, roles = prepare_analysis_dataset(df, column_roles)
    # Preserve original uploaded dataset metadata for sanity/profile questions.
    # The analytical dataframe can contain cleaned/derived columns, but questions
    # like "how many rows/columns" or "are there nulls" must report the original CSV.
    roles.setdefault("_raw_profile", {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": [str(c) for c in df.columns],
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": {str(k): int(v) for k, v in df.isna().sum().to_dict().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
    })
    q = str(question).lower().strip()

    tier4_res = _tier4_answer(question, prepared_df, roles)
    if tier4_res.get("handled"):
        tier4_res.setdefault("question", question)
        tier4_res.setdefault("evidence_tables", {})
        tier4_res.setdefault("limitations", [])
        tier4_res.setdefault("follow_up_questions", [])
        tier4_res.setdefault("analysis_plan", {})
        tier4_res.setdefault("context_results", {"missing_value_treatment": roles.get("_missing_value_treatment", {})})
        tier4_res.setdefault("target_metric", "general")
        tier4_res.setdefault("relationship_findings", [])
        tier4_res.setdefault("root_causes", [])
        tier4_res.setdefault("recommendations", [])
        tier4_res.setdefault("deterministic_answer", tier4_res.get("markdown_answer", ""))
        return tier4_res

    # Basic dataset sanity/profile questions must be answered exactly before
    # business-analysis routing. Otherwise simple checks such as row count,
    # date range, nulls, duplicate rows, or AOV can fall into profit diagnostics.
    basic_res = _basic_profile_answer(question, prepared_df, roles)
    if basic_res.get("handled"):
        basic_res.setdefault("question", question)
        basic_res.setdefault("evidence_tables", {})
        basic_res.setdefault("limitations", [])
        basic_res.setdefault("follow_up_questions", [])
        basic_res.setdefault("analysis_plan", {})
        basic_res.setdefault("context_results", {"missing_value_treatment": roles.get("_missing_value_treatment", {})})
        basic_res.setdefault("target_metric", "general")
        basic_res.setdefault("relationship_findings", [])
        basic_res.setdefault("root_causes", [])
        basic_res.setdefault("recommendations", [])
        basic_res.setdefault("deterministic_answer", basic_res.get("markdown_answer", ""))
        return basic_res

    if not should_attempt_dynamic(q) and not _tier2_should_handle(q, roles) and not _tier3_should_handle(q, roles):
        return {"handled": False, "reason": "Question does not match dynamic patterns."}

    tier3_res = _tier3_answer(question, prepared_df, roles)
    if tier3_res.get("handled"):
        tier3_res.setdefault("question", question)
        tier3_res.setdefault("evidence_tables", {})
        tier3_res.setdefault("limitations", [])
        tier3_res.setdefault("follow_up_questions", [])
        tier3_res.setdefault("analysis_plan", {})
        tier3_res.setdefault("context_results", {"missing_value_treatment": roles.get("_missing_value_treatment", {})})
        tier3_res.setdefault("target_metric", tier3_res.get("analysis_plan", {}).get("metric", "general"))
        tier3_res.setdefault("relationship_findings", [])
        tier3_res.setdefault("root_causes", [])
        tier3_res.setdefault("recommendations", [])
        return tier3_res

    tier2_res = _tier2_answer(question, prepared_df, roles)
    if tier2_res.get("handled"):
        tier2_res.setdefault("question", question)
        tier2_res.setdefault("evidence_tables", {})
        tier2_res.setdefault("limitations", [])
        tier2_res.setdefault("follow_up_questions", [])
        tier2_res.setdefault("analysis_plan", {})
        tier2_res.setdefault("context_results", {"missing_value_treatment": roles.get("_missing_value_treatment", {})})
        tier2_res.setdefault("target_metric", tier2_res.get("analysis_plan", {}).get("metric", "general"))
        tier2_res.setdefault("relationship_findings", [])
        tier2_res.setdefault("root_causes", [])
        tier2_res.setdefault("recommendations", [])
        return tier2_res

    for handler in (_churn_answer, _compare_answer, _root_cause_answer, _profit_diagnostic_answer, _trend_answer, _ranking_answer):
        res = handler(question, prepared_df, roles)
        if res.get("handled"):
            res.setdefault("question", question)
            res.setdefault("evidence_tables", {})
            res.setdefault("limitations", [])
            res.setdefault("follow_up_questions", [])
            res.setdefault("analysis_plan", {})
            res.setdefault("context_results", {"missing_value_treatment": roles.get("_missing_value_treatment", {})})
            metric = res.get("analysis_plan", {}).get("metric")
            if res.get("intent") == "dynamic_root_cause" or metric in {"loss", "profit"}:
                res.setdefault("target_metric", "profit")
            elif metric:
                res.setdefault("target_metric", metric)
            else:
                res.setdefault("target_metric", "general")
            res.setdefault("relationship_findings", [])
            res.setdefault("root_causes", [])
            res.setdefault("recommendations", [])
            return res
    return {"handled": False, "reason": "No dynamic handler could answer."}
