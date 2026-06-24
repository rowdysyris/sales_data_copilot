from __future__ import annotations
import pandas as pd
from src.utils import get_col, safe_numeric, profit_margin, prepare_analysis_dataset, ensure_datetime


def _roles(roles): return roles.get("roles", roles) if isinstance(roles, dict) else {}


def calculate_kpis(df: pd.DataFrame, column_roles: dict) -> dict:
    """Calculate executive sales KPIs."""
    df, roles = prepare_analysis_dataset(df, column_roles)
    sales_col, profit_col, qty_col = roles.get("sales_column"), roles.get("profit_column"), roles.get("quantity_column")
    discount_col, cust_col, order_col = roles.get("discount_column"), roles.get("customer_column") or roles.get("customer_id_column"), roles.get("order_id_column")
    category_col, region_col, product_col = roles.get("category_column"), roles.get("region_column"), roles.get("product_column")

    sales = safe_numeric(df[sales_col]) if sales_col in df.columns else pd.Series(dtype=float)
    profit = safe_numeric(df[profit_col]) if profit_col in df.columns else pd.Series(dtype=float)
    qty = safe_numeric(df[qty_col]) if qty_col in df.columns else pd.Series(dtype=float)

    total_sales = float(sales.sum()) if len(sales) else None
    total_profit = float(profit.sum()) if len(profit) else None
    total_order_rows = int(len(df))
    unique_orders = int(df[order_col].nunique()) if order_col in df.columns else total_order_rows
    grain = roles.get("_dataset_grain", {}) if isinstance(roles, dict) else {}
    order_count_basis = "row_count" if order_col in df.columns else "row_count_proxy_no_order_id"
    loss_mask = profit < 0 if len(profit) else pd.Series(False, index=df.index)

    out = {
        "total_sales": total_sales,
        "total_profit": total_profit,
        "total_quantity": float(qty.sum()) if len(qty) else None,
        # In this app, total_orders is intentionally the number of order rows/line items.
        # Distinct business orders are exposed separately as unique_orders.
        "total_orders": total_order_rows,
        "order_rows": total_order_rows,
        "unique_orders": unique_orders,
        "unique_order_count": unique_orders,
        "order_count_basis": order_count_basis,
        "average_order_value": float(sales.mean()) if len(sales) else None,
        "median_order_value": float(sales.median()) if len(sales) else None,
        "unique_order_average_value": total_sales / unique_orders if total_sales is not None and unique_orders else None,
        "profit_margin_percent": profit_margin(total_profit, total_sales) if total_profit is not None and total_sales else None,
        "average_discount": float(safe_numeric(df[discount_col]).mean()) if discount_col in df.columns else None,
        "unique_customers": int(df[cust_col].nunique()) if cust_col in df.columns else None,
        "loss_making_orders_count": int(loss_mask.sum()) if len(profit) else None,
        "loss_making_orders_percent": float(loss_mask.mean() * 100) if len(profit) else None,
        "total_loss_amount": float(profit[profit < 0].sum()) if len(profit) else None,
        "profitable_orders_count": int((profit > 0).sum()) if len(profit) else None,
        "profitable_orders_percent": float((profit > 0).mean() * 100) if len(profit) else None,
        "unresolved_sales_rows": int(df[sales_col].isna().sum()) if sales_col in df.columns else 0,
        "unresolved_profit_rows": int(df[profit_col].isna().sum()) if profit_col in df.columns else 0,
        "dataset_grain": grain.get("grain", "unknown"),
        "count_label": grain.get("count_label", "Rows"),
    }
    if out["unique_customers"]:
        out["sales_per_customer"] = total_sales / out["unique_customers"] if total_sales is not None else None
        out["profit_per_customer"] = total_profit / out["unique_customers"] if total_profit is not None else None
    else:
        out["sales_per_customer"] = out["profit_per_customer"] = None

    def best_worst(dim, metric, best=True):
        if dim in df.columns and metric in df.columns:
            g = df.groupby(dim, dropna=False)[metric].sum().sort_values(ascending=not best)
            return str(g.index[0]) if not g.empty else None
        return None

    if sales_col:
        out["best_category_by_sales"] = best_worst(category_col, sales_col, True) if category_col else None
        out["best_region_by_sales"] = best_worst(region_col, sales_col, True) if region_col else None
    if profit_col:
        out["best_category_by_profit"] = best_worst(category_col, profit_col, True) if category_col else None
        out["worst_category_by_profit"] = best_worst(category_col, profit_col, False) if category_col else None
        out["worst_region_by_profit"] = best_worst(region_col, profit_col, False) if region_col else None
        out["best_product_by_profit"] = best_worst(product_col, profit_col, True) if product_col else None
        out["worst_product_by_profit"] = best_worst(product_col, profit_col, False) if product_col else None
    return out


def _group_performance(df, roles, dim_col):
    df, roles = prepare_analysis_dataset(df, roles)
    if not dim_col or dim_col not in df.columns:
        return pd.DataFrame()
    sales_col, profit_col, qty_col = roles.get("sales_column"), roles.get("profit_column"), roles.get("quantity_column")
    disc_col, order_col = roles.get("discount_column"), roles.get("order_id_column")
    agg = {}
    work = df.copy()
    if sales_col in work.columns:
        work[sales_col] = safe_numeric(work[sales_col]).fillna(0)
        agg[sales_col] = "sum"
    if profit_col in work.columns:
        work[profit_col] = safe_numeric(work[profit_col]).fillna(0)
        agg[profit_col] = "sum"
    if qty_col in work.columns:
        work[qty_col] = safe_numeric(work[qty_col]).fillna(0)
        agg[qty_col] = "sum"
    if disc_col in work.columns:
        work[disc_col] = safe_numeric(work[disc_col])
        agg[disc_col] = "mean"
    if not agg: return pd.DataFrame()
    g = work.groupby(dim_col, dropna=False, observed=False).agg(agg).reset_index()
    rename = {dim_col: "dimension_value"}
    if sales_col: rename[sales_col] = "sales"
    if profit_col: rename[profit_col] = "profit"
    if qty_col: rename[qty_col] = "quantity"
    if disc_col: rename[disc_col] = "average_discount"
    g = g.rename(columns=rename)
    if order_col in df.columns:
        orders = work.groupby(dim_col, dropna=False, observed=False)[order_col].nunique().reset_index(name="order_count").rename(columns={dim_col:"dimension_value"})
    else:
        orders = work.groupby(dim_col, dropna=False, observed=False).size().reset_index(name="order_count").rename(columns={dim_col:"dimension_value"})
    g = g.merge(orders, on="dimension_value", how="left")
    grain = roles.get("_dataset_grain", {}) if isinstance(roles, dict) else {}
    g["order_count_basis"] = "distinct_order_id" if order_col in work.columns else grain.get("count_label", "Rows")
    if sales_col in work.columns:
        unresolved_sales = work.groupby(dim_col, dropna=False, observed=False)[sales_col].apply(lambda s: int(s.isna().sum())).reset_index(name="unresolved_sales_rows").rename(columns={dim_col: "dimension_value"})
        g = g.merge(unresolved_sales, on="dimension_value", how="left")
    if profit_col in work.columns:
        unresolved_profit = work.groupby(dim_col, dropna=False, observed=False)[profit_col].apply(lambda s: int(s.isna().sum())).reset_index(name="unresolved_profit_rows").rename(columns={dim_col: "dimension_value"})
        g = g.merge(unresolved_profit, on="dimension_value", how="left")
    if "sales" in g.columns and "profit" in g.columns:
        g["profit_margin_percent"] = g.apply(lambda r: profit_margin(r["profit"], r["sales"]), axis=1)
        total_sales, total_profit = g["sales"].sum(), g["profit"].sum()
        total_loss = abs(g.loc[g["profit"] < 0, "profit"].sum())
        g["sales_contribution_percent"] = g["sales"] / total_sales * 100 if total_sales else 0
        g["profit_contribution_percent"] = g["profit"] / total_profit * 100 if total_profit else 0
        g["loss_amount"] = g["profit"].where(g["profit"] < 0, 0)
        g["loss_contribution_percent"] = abs(g["loss_amount"]) / total_loss * 100 if total_loss else 0
    return g.sort_values("profit" if "profit" in g.columns else "sales", ascending=False).reset_index(drop=True)


def performance_by_category(df, roles): return _group_performance(df, roles, _roles(roles).get("category_column"))
def performance_by_subcategory(df, roles): return _group_performance(df, roles, _roles(roles).get("subcategory_column"))
def performance_by_product(df, roles): return _group_performance(df, roles, _roles(roles).get("product_column"))
def performance_by_region(df, roles): return _group_performance(df, roles, _roles(roles).get("region_column"))
def performance_by_customer_segment(df, roles): return _group_performance(df, roles, _roles(roles).get("segment_column"))


def performance_over_time(df, roles, frequency="M"):
    roles = _roles(roles)
    date_col, sales_col, profit_col = roles.get("date_column"), roles.get("sales_column"), roles.get("profit_column")
    if date_col not in df.columns or sales_col not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    tmp[date_col] = ensure_datetime(tmp[date_col])
    tmp = tmp.dropna(subset=[date_col])
    if tmp.empty: return pd.DataFrame()
    tmp["period"] = tmp[date_col].dt.to_period(frequency).astype(str)
    tmp[sales_col] = safe_numeric(tmp[sales_col]).fillna(0)
    agg = {sales_col: "sum"}
    if profit_col in tmp.columns:
        tmp[profit_col] = safe_numeric(tmp[profit_col]).fillna(0)
        agg[profit_col] = "sum"
    g = tmp.groupby("period").agg(agg).reset_index()
    g = g.rename(columns={sales_col: "sales", profit_col: "profit"} if profit_col else {sales_col: "sales"})
    if "sales" in g: g["sales_growth_percent"] = g["sales"].pct_change() * 100
    if "profit" in g: g["profit_growth_percent"] = g["profit"].pct_change() * 100
    return g
