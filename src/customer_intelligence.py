from __future__ import annotations
import pandas as pd
from src.utils import safe_numeric, prepare_analysis_dataset, normalize_0_100, profit_margin, ensure_datetime


def _roles(roles): return roles.get("roles", roles) if isinstance(roles, dict) else {}


def analyze_customer_loyalty(df, column_roles):
    """Analyze loyalty, profitability, and risk for customers."""
    df, roles = prepare_analysis_dataset(df, column_roles)
    cust_col = roles.get("customer_column") or roles.get("customer_id_column")
    sales_col, profit_col, date_col, order_col, disc_col = roles.get("sales_column"), roles.get("profit_column"), roles.get("date_column"), roles.get("order_id_column"), roles.get("discount_column")
    if cust_col not in df.columns:
        return {"limitations": ["No customer column detected."], "top_loyal_customers": pd.DataFrame(), "customer_summary": {}}

    tmp = df.copy()
    if date_col in tmp.columns:
        tmp[date_col] = ensure_datetime(tmp[date_col])

    agg = {}
    if sales_col in tmp.columns:
        tmp[sales_col] = safe_numeric(tmp[sales_col]).fillna(0)
        agg[sales_col] = "sum"
    if profit_col in tmp.columns:
        tmp[profit_col] = safe_numeric(tmp[profit_col]).fillna(0)
        agg[profit_col] = "sum"
    if disc_col in tmp.columns:
        tmp[disc_col] = safe_numeric(tmp[disc_col])
        agg[disc_col] = "mean"
    g = tmp.groupby(cust_col, dropna=False).agg(agg).reset_index()
    g = g.rename(columns={cust_col:"customer", sales_col:"total_sales", profit_col:"total_profit", disc_col:"average_discount"})
    if order_col in tmp.columns:
        orders = tmp.groupby(cust_col)[order_col].nunique().reset_index(name="order_count").rename(columns={cust_col:"customer"})
    else:
        orders = tmp.groupby(cust_col).size().reset_index(name="order_count").rename(columns={cust_col:"customer"})
    g = g.merge(orders, on="customer", how="left")
    if "total_sales" in g:
        g["average_order_value"] = g["total_sales"] / g["order_count"].replace(0, pd.NA)
    if "total_profit" in g and "total_sales" in g:
        g["profit_margin_percent"] = g.apply(lambda r: profit_margin(r["total_profit"], r["total_sales"]), axis=1)

    if date_col in tmp.columns:
        dates = tmp.groupby(cust_col)[date_col].agg(["min","max"]).reset_index().rename(columns={cust_col:"customer","min":"first_order_date","max":"last_order_date"})
        g = g.merge(dates, on="customer", how="left")
        max_date = tmp[date_col].max()
        g["customer_lifespan_days"] = (g["last_order_date"] - g["first_order_date"]).dt.days.fillna(0)
        g["days_since_last_order"] = (max_date - g["last_order_date"]).dt.days.fillna(9999)
        tmp["month"] = tmp[date_col].dt.to_period("M").astype(str)
        active = tmp.groupby(cust_col)["month"].nunique().reset_index(name="active_month_count").rename(columns={cust_col:"customer"})
        g = g.merge(active, on="customer", how="left")
    else:
        g["customer_lifespan_days"] = 0
        g["days_since_last_order"] = 0
        g["active_month_count"] = 0

    if profit_col in tmp.columns:
        loss = tmp[safe_numeric(tmp[profit_col]) < 0].groupby(cust_col).size().reset_index(name="loss_order_count").rename(columns={cust_col:"customer"})
        g = g.merge(loss, on="customer", how="left")
        g["loss_order_count"] = g["loss_order_count"].fillna(0).astype(int)
        g["loss_order_percent"] = g["loss_order_count"] / g["order_count"].replace(0, pd.NA) * 100
    else:
        g["loss_order_count"] = 0
        g["loss_order_percent"] = 0

    score = (
        normalize_0_100(g["order_count"]) * 0.30 +
        normalize_0_100(g.get("total_sales", pd.Series(0, index=g.index))) * 0.20 +
        normalize_0_100(g.get("total_profit", pd.Series(0, index=g.index))) * 0.20 +
        normalize_0_100(g["customer_lifespan_days"]) * 0.15 +
        normalize_0_100(g["days_since_last_order"], invert=True) * 0.15
    )
    g["loyalty_score"] = score.round(2)

    def segment(row):
        profit = row.get("total_profit", 0)
        orders = row.get("order_count", 0)
        score = row.get("loyalty_score", 0)
        inactive = row.get("days_since_last_order", 0) > max(90, g["days_since_last_order"].quantile(0.75))
        if profit < 0:
            return "Unprofitable Customer" if score < 70 else "Loyal but Low Profit"
        if score >= 70 and profit > 0:
            return "Loyal and Profitable"
        if inactive and score >= 55:
            return "At-Risk Loyal Customer"
        if orders <= 2 and row.get("average_order_value", 0) >= g.get("average_order_value", pd.Series([0])).median():
            return "New Promising Customer"
        if row.get("total_sales", 0) >= g.get("total_sales", pd.Series([0])).quantile(0.75) and orders <= g["order_count"].median():
            return "High Value but Infrequent"
        return "Regular Customer"
    g["customer_segment"] = g.apply(segment, axis=1)

    top = g.sort_values("loyalty_score", ascending=False).reset_index(drop=True)
    return {
        "top_loyal_customers": top.head(20),
        "loyal_profitable_customers": top[top["customer_segment"] == "Loyal and Profitable"],
        "loyal_low_profit_customers": top[top["customer_segment"] == "Loyal but Low Profit"],
        "at_risk_customers": top[top["customer_segment"] == "At-Risk Loyal Customer"],
        "unprofitable_customers": top[top["customer_segment"].isin(["Unprofitable Customer","Loyal but Low Profit"])],
        "customer_summary": {
            "total_customers": int(g["customer"].nunique()),
            "average_orders_per_customer": float(g["order_count"].mean()),
            "top_customer": str(top.iloc[0]["customer"]) if not top.empty else None,
        },
        "insights": [
            f"Top loyal customer: {top.iloc[0]['customer']} with score {top.iloc[0]['loyalty_score']:.2f}." if not top.empty else "No customer insight available."
        ],
    }
