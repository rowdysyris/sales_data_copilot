from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.profit_intelligence import analyze_profit_intelligence
from src.customer_intelligence import analyze_customer_loyalty
from src.product_intelligence import analyze_product_portfolio
from src.region_intelligence import analyze_region_segment_performance


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _label_from_dimension(raw: Any) -> str:
    if raw is None or pd.isna(raw):
        return "Business Area"
    text = str(raw).replace("_column", "").replace("_", " ").strip()
    return text.title() if text else "Business Area"


def _dedupe_key(title: str, evidence: list[str]) -> str:
    joined = " ".join([title] + evidence).lower()
    joined = re.sub(r"[^a-z0-9]+", " ", joined).strip()
    return joined


def _add_insight(insights: list[dict], seen: set[str], insight: dict) -> None:
    """Add an insight only once, preventing repeated manager cards."""
    evidence = [str(x) for x in insight.get("evidence", [])]
    key = _dedupe_key(str(insight.get("title", "")), evidence)
    if key and key not in seen:
        seen.add(key)
        insights.append(insight)


def _build_profit_leakage_insights(df: pd.DataFrame, column_roles: dict) -> list[dict]:
    profit = analyze_profit_intelligence(df, column_roles)
    leakage = profit.get("profit_leakage_points", pd.DataFrame())
    insights: list[dict] = []

    if isinstance(leakage, pd.DataFrame) and not leakage.empty:
        view = leakage.copy()
        if "profit" in view.columns:
            view = view.sort_values("profit", ascending=True)
        for _, row in view.head(3).iterrows():
            value = row.get("dimension_value", row.get("value", "Unknown"))
            dim = _label_from_dimension(row.get("dimension"))
            sales = _safe_float(row.get("sales"))
            profit_value = _safe_float(row.get("profit"))
            margin = _safe_float(row.get("profit_margin_percent"))
            orders = int(_safe_float(row.get("order_count", row.get("orders", 0))))
            insights.append({
                "title": f"Profit leakage: {dim} = {value}",
                "severity": "High" if profit_value < 0 else "Medium",
                "business_area": "Profit",
                "evidence": [
                    f"Sales: {sales:,.2f}",
                    f"Profit: {profit_value:,.2f}",
                    f"Margin: {margin:.2f}%",
                    f"Orders: {orders:,}",
                ],
                "why_it_matters": "This area is reducing margin and can hide behind healthy top-line sales.",
                "recommended_action": f"Review pricing, discounting, cost, and product mix for {value}.",
                "confidence": "High",
            })
        return insights

    # Fallback when leakage table is unavailable.
    for cause in profit.get("root_cause_summary", [])[:3]:
        insights.append({
            "title": "Profit leakage detected",
            "severity": "High",
            "business_area": "Profit",
            "evidence": [str(cause)],
            "why_it_matters": "Loss concentration reduces overall margin.",
            "recommended_action": "Review pricing, discounting, and cost for the affected area.",
            "confidence": "Medium",
        })
    return insights


def generate_hidden_insights(df, column_roles):
    """Scan a dataset and surface non-duplicate senior-analyst insights.

    The function intentionally avoids showing the same generic card multiple
    times. Each insight must have a distinct title/evidence pair so the Hidden
    Insights tab is manager-demo clean.
    """
    insights: list[dict] = []
    seen: set[str] = set()

    for insight in _build_profit_leakage_insights(df, column_roles):
        _add_insight(insights, seen, insight)

    prod = analyze_product_portfolio(df, column_roles)
    prob = prod.get("problem_products", pd.DataFrame())
    if isinstance(prob, pd.DataFrame) and not prob.empty:
        row = prob.sort_values("profit", ascending=True).iloc[0] if "profit" in prob.columns else prob.iloc[0]
        product_name = row.get("dimension_value", "Unknown product")
        _add_insight(insights, seen, {
            "title": f"Problem product: {product_name}",
            "severity": "High",
            "business_area": "Product",
            "evidence": [
                f"Sales: {_safe_float(row.get('sales')):,.2f}",
                f"Profit: {_safe_float(row.get('profit')):,.2f}",
                f"Margin: {_safe_float(row.get('profit_margin_percent')):.2f}%",
            ],
            "why_it_matters": "Products with negative profit can hide behind top-line sales.",
            "recommended_action": "Audit cost, discounting, selling price, and whether this product should stay in the portfolio.",
            "confidence": "High",
        })

    cust = analyze_customer_loyalty(df, column_roles)
    unprof = cust.get("unprofitable_customers", pd.DataFrame())
    if isinstance(unprof, pd.DataFrame) and not unprof.empty:
        row = unprof.sort_values("total_profit", ascending=True).iloc[0] if "total_profit" in unprof.columns else unprof.iloc[0]
        customer_name = row.get("customer", "Unknown customer")
        _add_insight(insights, seen, {
            "title": f"Unprofitable customer risk: {customer_name}",
            "severity": "Medium",
            "business_area": "Customer",
            "evidence": [
                f"Sales: {_safe_float(row.get('total_sales')):,.2f}",
                f"Profit: {_safe_float(row.get('total_profit')):,.2f}",
                f"Orders: {int(_safe_float(row.get('order_count'))):,}",
            ],
            "why_it_matters": "Repeat customers are not automatically good customers if they create losses.",
            "recommended_action": "Review discounting, service cost, and retention strategy for this customer.",
            "confidence": "Medium",
        })

    reg = analyze_region_segment_performance(df, column_roles)
    weak = reg.get("weak_regions", pd.DataFrame())
    if isinstance(weak, pd.DataFrame) and not weak.empty:
        row = weak.sort_values("profit", ascending=True).iloc[0] if "profit" in weak.columns else weak.iloc[0]
        region_name = row.get("dimension_value", "Unknown region")
        _add_insight(insights, seen, {
            "title": f"Weak region: {region_name}",
            "severity": "Medium",
            "business_area": "Region",
            "evidence": [
                f"Sales: {_safe_float(row.get('sales')):,.2f}",
                f"Profit: {_safe_float(row.get('profit')):,.2f}",
                f"Margin: {_safe_float(row.get('profit_margin_percent')):.2f}%",
            ],
            "why_it_matters": "Regional underperformance may require local pricing, discount, or product-mix action.",
            "recommended_action": "Audit regional discounts, product mix, and sales strategy.",
            "confidence": "Medium",
        })

    if not insights:
        _add_insight(insights, seen, {
            "title": "No major hidden issue detected",
            "severity": "Low",
            "business_area": "General",
            "evidence": ["Available columns do not show major risk concentration."],
            "why_it_matters": "More columns may reveal deeper drivers.",
            "recommended_action": "Add cost, discount, customer, and date fields for stronger diagnostics.",
            "confidence": "Medium",
        })

    return insights[:10]
