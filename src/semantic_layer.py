from __future__ import annotations
from src.utils import get_col


def build_semantic_model(df, column_roles):
    """Build a semantic model describing the dataset."""
    roles = column_roles.get("roles", column_roles) if isinstance(column_roles, dict) else {}
    metrics_map = {
        "sales": "sales_column",
        "revenue": "revenue_column",
        "profit": "profit_column",
        "cost": "cost_column",
        "discount": "discount_column",
        "quantity": "quantity_column",
        "unit_price": "unit_price_column",
        "margin_percent": "margin_percent_column",
    }
    dims_map = {
        "category": "category_column",
        "subcategory": "subcategory_column",
        "product": "product_column",
        "region": "region_column",
        "state": "state_column",
        "city": "city_column",
        "country": "country_column",
        "customer": "customer_column",
        "customer_id": "customer_id_column",
        "segment": "segment_column",
        "order_id": "order_id_column",
    }
    metrics = {k: roles.get(v) for k, v in metrics_map.items() if roles.get(v)}
    dimensions = {k: roles.get(v) for k, v in dims_map.items() if roles.get(v)}
    time_columns = [roles[r] for r in ("date_column", "ship_date_column") if roles.get(r)]

    if roles.get("order_id_column") and roles.get("sales_column"):
        grain = "order_level"
    elif roles.get("customer_column") and not roles.get("order_id_column"):
        grain = "customer_level"
    elif roles.get("product_column") and not roles.get("order_id_column"):
        grain = "product_level"
    elif roles.get("date_column") and roles.get("sales_column"):
        grain = "daily_or_monthly_summary_level"
    else:
        grain = "unknown"

    if roles.get("sales_column") and (roles.get("category_column") or roles.get("product_column")):
        dataset_type = "sales_transaction_dataset"
    elif roles.get("customer_column"):
        dataset_type = "customer_dataset"
    elif roles.get("product_column"):
        dataset_type = "product_dataset"
    elif roles.get("profit_column") or roles.get("cost_column"):
        dataset_type = "finance_dataset"
    else:
        dataset_type = "unknown_dataset"

    available = []
    missing = []
    if roles.get("sales_column") and roles.get("profit_column"):
        available += ["profitability_analysis", "loss_driver_analysis", "scenario_simulation"]
    elif roles.get("sales_column") and roles.get("cost_column"):
        available += ["derived_profitability_analysis", "loss_driver_analysis", "scenario_simulation"]
    elif roles.get("sales_column") and roles.get("margin_percent_column"):
        available += ["estimated_profitability_analysis", "scenario_simulation"]
    else:
        missing += ["Profit column for margin/loss analysis"] if not roles.get("profit_column") else []
    if roles.get("discount_column") and roles.get("profit_column"):
        available.append("discount_impact_analysis")
    if (roles.get("customer_column") or roles.get("customer_id_column")) and roles.get("order_id_column") and roles.get("sales_column") and roles.get("date_column"):
        available.append("customer_loyalty_analysis")
    if (roles.get("customer_column") or roles.get("customer_id_column")) and roles.get("profit_column"):
        available.append("customer_profitability_analysis")
    if roles.get("product_column") and roles.get("profit_column"):
        available.append("product_portfolio_analysis")
    if roles.get("region_column") and roles.get("profit_column"):
        available.append("regional_performance_analysis")
    if roles.get("date_column") and roles.get("sales_column"):
        available += ["time_trend_analysis", "forecasting", "anomaly_detection"]

    return {
        "dataset_type": dataset_type,
        "grain": grain,
        "metrics": metrics,
        "dimensions": dimensions,
        "time_columns": time_columns,
        "customer_columns": [c for c in [roles.get("customer_column"), roles.get("customer_id_column")] if c],
        "product_columns": [c for c in [roles.get("category_column"), roles.get("subcategory_column"), roles.get("product_column")] if c],
        "geo_columns": [c for c in [roles.get("country_column"), roles.get("state_column"), roles.get("city_column"), roles.get("region_column")] if c],
        "financial_columns": [c for c in [roles.get("sales_column"), roles.get("profit_column"), roles.get("cost_column"), roles.get("discount_column"), roles.get("unit_price_column"), roles.get("margin_percent_column")] if c],
        "available_analyses": sorted(set(available)),
        "missing_but_useful_columns": missing,
    }
