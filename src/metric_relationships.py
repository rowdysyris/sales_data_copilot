from __future__ import annotations
import pandas as pd
import numpy as np
from src.utils import safe_numeric, profit_margin, format_currency, format_percent, prepare_analysis_dataset, safe_correlation, ensure_datetime
from src.kpi_engine import performance_by_category, performance_by_subcategory, performance_by_product, performance_by_region, performance_by_customer_segment


def _roles(roles): return roles.get("roles", roles) if isinstance(roles, dict) else {}


def create_discount_bands(series: pd.Series) -> pd.Series:
    s = safe_numeric(series).fillna(0)
    # Handle both 0-1 and 0-100 discount formats
    if s.max() <= 1.0:
        pct = s * 100
    else:
        pct = s
    return pd.cut(pct, bins=[-0.001, 0, 10, 20, 30, 40, float("inf")], labels=["0%", "0-10%", "10-20%", "20-30%", "30-40%", "40%+"])


def calculate_margin_by_dimension(df, roles, dim_col):
    from src.kpi_engine import _group_performance
    return _group_performance(df, roles, dim_col)


def find_high_sales_low_profit_groups(df, roles, dim_col=None):
    roles = _roles(roles)
    dim_col = dim_col or roles.get("category_column")
    g = calculate_margin_by_dimension(df, roles, dim_col)
    if g.empty or "sales" not in g or "profit_margin_percent" not in g:
        return pd.DataFrame()
    sales_threshold = g["sales"].quantile(0.60)
    margin_threshold = g["profit_margin_percent"].median()
    return g[(g["sales"] >= sales_threshold) & (g["profit_margin_percent"] <= margin_threshold)].sort_values(["sales","profit_margin_percent"], ascending=[False, True])


def calculate_profit_contribution(df, roles, dim_col):
    return calculate_margin_by_dimension(df, roles, dim_col)


def detect_profit_leakage_points(df, roles):
    roles = _roles(roles)
    frames = []
    for dim in ["category_column","subcategory_column","product_column","region_column","segment_column"]:
        col = roles.get(dim)
        g = calculate_margin_by_dimension(df, roles, col)
        if not g.empty and "profit" in g:
            bad = g[(g["profit"] < 0) | (g.get("profit_margin_percent", 100) < 5)].copy()
            bad["dimension"] = col
            frames.append(bad)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def analyze_metric_relationships(df, column_roles, target_metric="profit"):
    """Analyze relationships between profit and sales, discount, quantity, dimensions, and time."""
    df, roles = prepare_analysis_dataset(df, column_roles)
    sales_col, profit_col = roles.get("sales_column"), roles.get("profit_column")
    if profit_col not in df.columns:
        return {"target_metric": target_metric, "relationship_summary": ["Profit column missing."], "limitations": ["No profit column detected."]}
    profit = safe_numeric(df[profit_col])
    sales = safe_numeric(df[sales_col]) if sales_col in df.columns else pd.Series(dtype=float)
    overall_profit = float(profit.sum())
    overall_sales = float(sales.sum()) if len(sales) else None
    summary = []
    correlations = []
    for role, col in roles.items():
        if not isinstance(col, str) or role.startswith("_"):
            continue
        if col in df.columns and safe_numeric(df[col]).notna().mean() > 0.8:
            s = safe_numeric(df[col])
            corr = safe_correlation(s, profit)
            if corr is not None:
                correlations.append({"metric": col, "correlation_with_profit": corr})
    corr_df = pd.DataFrame(correlations).sort_values("correlation_with_profit") if correlations else pd.DataFrame()

    by_discount = pd.DataFrame()
    high_discount_low_profit = pd.DataFrame()
    disc_col = roles.get("discount_column")
    if disc_col in df.columns:
        tmp = df.copy()
        tmp["discount_band"] = create_discount_bands(tmp[disc_col])
        by_discount = tmp.groupby("discount_band", observed=False).agg(
            sales=(sales_col, "sum") if sales_col in tmp.columns else (profit_col, "size"),
            profit=(profit_col, "sum"),
            order_count=(profit_col, "size"),
        ).reset_index()
        by_discount["margin_percent"] = by_discount.apply(lambda r: profit_margin(r["profit"], r["sales"]) if "sales" in r else 0, axis=1)
        loss = by_discount[by_discount["profit"] < 0]
        if not loss.empty:
            worst = loss.sort_values("profit").iloc[0]
            summary.append(f"The {worst['discount_band']} discount band generated profit/loss of {format_currency(worst['profit'])}.")
        high_discount_low_profit = tmp[(safe_numeric(tmp[disc_col]) >= (0.3 if safe_numeric(tmp[disc_col]).max() <= 1 else 30)) & (safe_numeric(tmp[profit_col]) < 0)]

    dimension_relationships = {}
    for name, role in [("category","category_column"),("subcategory","subcategory_column"),("product","product_column"),("region","region_column"),("segment","segment_column"),("customer","customer_column")]:
        col = roles.get(role)
        g = calculate_margin_by_dimension(df, roles, col) if col else pd.DataFrame()
        if not g.empty:
            dimension_relationships[name] = g
            if "profit" in g:
                worst = g.sort_values("profit").head(1)
                if not worst.empty:
                    summary.append(f"Worst {name}: {worst.iloc[0]['dimension_value']} with profit of {format_currency(worst.iloc[0]['profit'])}.")

    high_sales_low_profit = pd.DataFrame()
    for role in ["category_column","subcategory_column","product_column","region_column"]:
        col = roles.get(role)
        hs = find_high_sales_low_profit_groups(df, roles, col) if col else pd.DataFrame()
        if not hs.empty:
            high_sales_low_profit = hs
            top = hs.iloc[0]
            summary.append(f"{top['dimension_value']} has high sales but weak margin at {format_percent(top.get('profit_margin_percent', 0))}.")
            break

    loss_concentration = pd.DataFrame()
    product_col = roles.get("product_column") or roles.get("subcategory_column") or roles.get("category_column")
    if product_col:
        loss_concentration = calculate_margin_by_dimension(df, roles, product_col)
        if not loss_concentration.empty and "loss_contribution_percent" in loss_concentration:
            loss_concentration = loss_concentration.sort_values("loss_contribution_percent", ascending=False)

    date_col = roles.get("date_column")
    time_relationships = {}
    if date_col in df.columns:
        tmp = df.copy()
        tmp[date_col] = ensure_datetime(tmp[date_col])
        tmp = tmp.dropna(subset=[date_col])
        if not tmp.empty:
            tmp["period"] = tmp[date_col].dt.to_period("M").astype(str)
            tmp[profit_col] = safe_numeric(tmp[profit_col]).fillna(0)
            time_relationships["monthly_profit"] = tmp.groupby("period")[profit_col].sum().reset_index(name="profit")

    if not summary:
        summary.append("No strong profit relationship could be identified from available columns.")

    return {
        "target_metric": target_metric,
        "overall_profit": overall_profit,
        "overall_margin": profit_margin(overall_profit, overall_sales) if overall_sales else None,
        "correlations": corr_df,
        "profit_by_discount_band": by_discount,
        "profit_by_sales_band": pd.DataFrame(),
        "high_sales_low_profit": high_sales_low_profit,
        "high_discount_low_profit": high_discount_low_profit,
        "loss_concentration": loss_concentration,
        "dimension_relationships": dimension_relationships,
        "time_relationships": time_relationships,
        "relationship_summary": summary[:8],
    }
