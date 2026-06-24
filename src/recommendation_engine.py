from __future__ import annotations
import html
import pandas as pd
from src.profit_intelligence import analyze_profit_intelligence
from src.customer_intelligence import analyze_customer_loyalty
from src.product_intelligence import analyze_product_portfolio
from src.region_intelligence import analyze_region_segment_performance
from src.kpi_engine import calculate_kpis
from src.utils import format_currency, format_percent


SUMMARY_LABELS = {
    "business_performance": "Business Performance",
    "key_problem": "Key Problem",
    "root_cause": "Root Cause",
    "financial_impact": "Financial Impact",
    "recommended_action": "Recommended Action",
    "risk": "Risk",
    "next_step": "Next Step",
}


def _unique_text(items: list[str]) -> list[str]:
    """Return non-empty unique strings while preserving order."""
    seen = set()
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        key = text.lower().rstrip(".")
        if key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _first_nonempty(*values, default: str = "Not available from detected columns.") -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def _clean_manager_sentence(text: str) -> str:
    """Normalize generated findings so they read like manager-facing sentences."""
    value = str(text or "").strip()
    value = value.replace(";;", ";")
    while "  " in value:
        value = value.replace("  ", " ")
    return value.rstrip(" .;") + "." if value else ""


def _manager_numbered_list(items: list[str], limit: int = 3) -> str:
    """Return a readable numbered sentence list instead of semicolon-joined text."""
    clean = [_clean_manager_sentence(item) for item in _unique_text(items) if str(item).strip()]
    clean = clean[:limit]
    if not clean:
        return ""
    parts = [f"{idx}. {item}" for idx, item in enumerate(clean, start=1)]
    return "The likely drivers are: " + " ".join(parts)


def generate_recommendations(df, column_roles, context_results=None):
    """Generate prioritized business recommendations."""
    ctx = context_results or {}
    profit = ctx.get("profit_intelligence") or analyze_profit_intelligence(df, column_roles)
    prod = ctx.get("product_intelligence") or analyze_product_portfolio(df, column_roles)
    cust = ctx.get("customer_intelligence") or analyze_customer_loyalty(df, column_roles)
    reg = ctx.get("region_intelligence") or analyze_region_segment_performance(df, column_roles)
    recs = []

    for cause in profit.get("root_cause_summary", [])[:3]:
        recs.append({
            "title": "Fix profit leakage",
            "priority": "High",
            "issue_detected": cause,
            "evidence_from_data": cause,
            "recommended_action": "Review pricing, discounting, and cost for the affected segment.",
            "expected_impact": "Improves margin and reduces loss-making orders.",
            "confidence": "High",
            "related_columns": ["profit", "sales", "discount"],
            "related_segments": [],
        })

    prob = prod.get("problem_products", pd.DataFrame())
    if isinstance(prob, pd.DataFrame) and not prob.empty:
        row = prob.iloc[0]
        recs.append({
            "title": f"Review loss-making product: {row['dimension_value']}",
            "priority": "High",
            "issue_detected": f"Product profit is {row.get('profit',0):.2f}.",
            "evidence_from_data": f"{row['dimension_value']} has margin {row.get('profit_margin_percent',0):.2f}%.",
            "recommended_action": "Review cost, pricing, discounts, or consider removing from focus.",
            "expected_impact": "Reduces recurring product-level losses.",
            "confidence": "High",
            "related_columns": ["product", "profit"],
            "related_segments": [str(row["dimension_value"])],
        })

    unprof = cust.get("unprofitable_customers", pd.DataFrame())
    if isinstance(unprof, pd.DataFrame) and not unprof.empty:
        row = unprof.iloc[0]
        recs.append({
            "title": "Review loyal but unprofitable customers",
            "priority": "Medium",
            "issue_detected": f"{row['customer']} has weak or negative profit.",
            "evidence_from_data": f"Profit: {row.get('total_profit',0):.2f}, loyalty score: {row.get('loyalty_score',0):.2f}.",
            "recommended_action": "Check discounting and service cost for these customers.",
            "expected_impact": "Improves customer profitability without losing loyal customers.",
            "confidence": "Medium",
            "related_columns": ["customer", "profit"],
            "related_segments": [str(row["customer"])],
        })

    weak = reg.get("weak_regions", pd.DataFrame())
    if isinstance(weak, pd.DataFrame) and not weak.empty:
        row = weak.iloc[0]
        recs.append({
            "title": f"Fix weak region: {row['dimension_value']}",
            "priority": "Medium",
            "issue_detected": f"Region profit is {row.get('profit',0):.2f}.",
            "evidence_from_data": f"Margin: {row.get('profit_margin_percent',0):.2f}%.",
            "recommended_action": "Audit regional pricing, discounts, and product mix.",
            "expected_impact": "Improves geographic margin.",
            "confidence": "Medium",
            "related_columns": ["region", "profit"],
            "related_segments": [str(row["dimension_value"])],
        })

    if not recs:
        recs.append({
            "title": "Continue monitoring KPIs",
            "priority": "Low",
            "issue_detected": "No major issue detected from available columns.",
            "evidence_from_data": "Available data did not show clear loss concentration.",
            "recommended_action": "Track sales, profit, margin, and anomalies monthly.",
            "expected_impact": "Maintains visibility.",
            "confidence": "Medium",
            "related_columns": [],
            "related_segments": [],
        })
    priority = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(recs, key=lambda r: priority.get(r["priority"], 9))


def generate_executive_summary(context_results):
    """Create a manager-readable executive summary with distinct problem and root-cause fields.

    Key Problem = the main observed business issue.
    Root Cause = the driver(s) behind that issue. This intentionally excludes the exact
    key-problem sentence so the two fields do not repeat the same text.
    """
    kpis = context_results.get("kpis", {}) or {}
    profit = context_results.get("profit_intelligence", {}) or {}
    recs = context_results.get("recommendations", []) or []
    rel = context_results.get("metric_relationships", {}) or profit.get("profit_relationships", {}) or {}

    root_items = _unique_text(profit.get("root_cause_summary", []) if isinstance(profit, dict) else [])
    # Relationship findings can include unrelated global watch areas. Use them only when
    # profit intelligence could not isolate root-cause drivers for the key problem.
    relationship_items = _unique_text(rel.get("relationship_summary", []) if isinstance(rel, dict) else [])

    total_sales = kpis.get("total_sales")
    total_profit = kpis.get("total_profit")
    margin = kpis.get("profit_margin_percent")
    total_loss = kpis.get("total_loss_amount")
    loss_orders = kpis.get("loss_making_orders_count")
    loss_order_pct = kpis.get("loss_making_orders_percent")

    business_performance = (
        f"Total sales are {format_currency(total_sales)}, total profit is {format_currency(total_profit)}, "
        f"and profit margin is {format_percent(margin)}."
    )

    key_problem = _clean_manager_sentence(root_items[0]) if root_items else "No single major profit leakage point was detected from the available columns."

    # Root cause must be different from key problem. Prefer drivers generated inside
    # the same key-problem context. Only fall back to broad relationship findings when
    # there are no scoped drivers, otherwise unrelated groups can be mixed together.
    scoped_causes = _unique_text(root_items[1:])
    key_norm = key_problem.lower().rstrip(".")
    scoped_causes = [c for c in scoped_causes if c.lower().rstrip(".") != key_norm]
    fallback_causes = [c for c in relationship_items if c.lower().rstrip(".") != key_norm]
    candidate_causes = scoped_causes or fallback_causes

    if candidate_causes:
        root_cause = _manager_numbered_list(candidate_causes, limit=3)
    elif total_loss is not None and float(total_loss or 0) < 0:
        root_cause = "Loss-making orders exist, but the exact driver needs discount, product, region, customer, and cost context."
    else:
        root_cause = "No separate root-cause driver was strong enough to isolate from the available columns."

    financial_bits = []
    if total_loss is not None:
        financial_bits.append(f"Total loss amount is {format_currency(total_loss)}.")
    if loss_orders is not None:
        financial_bits.append(f"Loss-making orders are {loss_orders:,} ({format_percent(loss_order_pct)}).")
    financial_impact = " ".join(financial_bits) if financial_bits else "Financial impact could not be calculated from the detected columns."

    recommended_action = _first_nonempty(
        recs[0].get("recommended_action") if recs else None,
        (profit.get("recommended_actions", []) or [None])[0] if isinstance(profit, dict) else None,
        default="Review pricing, discounting, cost, and product mix for the affected area.",
    )

    return {
        "business_performance": business_performance,
        "key_problem": key_problem,
        "root_cause": root_cause,
        "financial_impact": financial_impact,
        "recommended_action": recommended_action,
        "risk": "Recommendations depend on detected columns, data completeness, and business context such as cost policy, returns, and pricing rules.",
        "next_step": "Open Profit Intelligence, validate the top loss driver with business owners, and test a discount/pricing change on the affected segment.",
    }


def executive_summary_to_markdown(summary: dict) -> str:
    """Render executive summary as readable markdown instead of a raw Python dict."""
    if not isinstance(summary, dict) or not summary:
        return "No executive summary is available yet."
    lines = []
    for key, label in SUMMARY_LABELS.items():
        value = summary.get(key)
        if value:
            lines.append(f"**{label}:** {value}")
    return "\n\n".join(lines)


def executive_summary_to_html(summary: dict) -> str:
    """Render executive summary as safe HTML for the app's modern card component."""
    if not isinstance(summary, dict) or not summary:
        return "No executive summary is available yet."
    parts = []
    for key, label in SUMMARY_LABELS.items():
        value = summary.get(key)
        if value:
            parts.append(f"<p><strong>{html.escape(label)}:</strong> {html.escape(str(value))}</p>")
    return "".join(parts)
