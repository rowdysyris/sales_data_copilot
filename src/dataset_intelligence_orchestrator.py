from __future__ import annotations

from typing import Any
import pandas as pd


def _table_size(value: Any) -> dict:
    if isinstance(value, pd.DataFrame):
        return {"rows": int(len(value)), "columns": int(value.shape[1])}
    return {"rows": 0, "columns": 0}


def build_dataset_intelligence(context_results: dict, roles: dict) -> dict:
    """Create a cached cross-module intelligence summary.

    This orchestrator does not replace existing modules. It summarizes all module
    outputs already computed by ``app._run_core`` so Q&A/reporting can use one
    consistent full-dataset context without re-running or breaking older logic.
    """
    ctx = context_results or {}
    roles = roles or {}
    kpis = ctx.get("kpis", {}) or {}
    pi = ctx.get("profit_intelligence", {}) or {}
    ci = ctx.get("customer_intelligence", {}) or {}
    prod = ctx.get("product_intelligence", {}) or {}
    reg = ctx.get("region_intelligence", {}) or {}
    rel = ctx.get("metric_relationships", {}) or {}
    hidden = ctx.get("hidden_insights", []) or []
    forecast = ctx.get("forecasting", {}) or {}
    anomalies = ctx.get("anomaly_detection", {}) or {}
    missing = ctx.get("missing_value_treatment", {}) or {}

    available_modules = []
    for name, obj in [
        ("kpis", kpis),
        ("data_quality", ctx.get("data_quality")),
        ("profit_intelligence", pi),
        ("metric_relationships", rel),
        ("customer_intelligence", ci),
        ("product_intelligence", prod),
        ("region_intelligence", reg),
        ("hidden_insights", hidden),
        ("forecasting", forecast),
        ("anomaly_detection", anomalies),
    ]:
        if obj is not None:
            available_modules.append(name)

    headline_flags: list[str] = []
    if missing.get("cleaned_dataset_used"):
        headline_flags.append("cleaned_dataset_used")
    if kpis.get("unresolved_sales_rows"):
        headline_flags.append("unresolved_sales_rows")
    if kpis.get("unresolved_profit_rows"):
        headline_flags.append("unresolved_profit_rows")
    if kpis.get("total_profit") is not None and kpis.get("total_profit") < 0:
        headline_flags.append("company_profit_negative")
    if kpis.get("loss_making_orders_count"):
        headline_flags.append("loss_making_records_present")
    if forecast.get("limitations"):
        headline_flags.append("forecast_limitations_present")

    table_inventory = {
        "profit_by_discount_band": _table_size(rel.get("profit_by_discount_band")),
        "high_sales_low_profit": _table_size(rel.get("high_sales_low_profit")),
        "top_loyal_customers": _table_size(ci.get("top_loyal_customers")),
        "problem_products": _table_size(prod.get("problem_products")),
        "weak_regions": _table_size(reg.get("weak_regions")),
        "forecast": _table_size(forecast.get("forecast")),
        "unusual_rows": _table_size(anomalies.get("unusual_rows")),
    }

    return {
        "available_modules": available_modules,
        "headline_flags": headline_flags,
        "table_inventory": table_inventory,
        "dataset_grain": roles.get("_dataset_grain", {}),
        "detected_roles": {k: v for k, v in roles.items() if not str(k).startswith("_")},
        "manager_notes": [
            "All numeric answers are calculated from Pandas outputs, not guessed by the LLM.",
            "Q&A can use direct dynamic calculations and this cached full-dataset intelligence context.",
        ],
    }
