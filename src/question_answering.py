from __future__ import annotations

from typing import Any
import pandas as pd

from src.analyst_brain import answer_with_analyst_brain
from src.llm_explainer import beautify_answer_with_llm
from src.dynamic_question_engine import answer_dynamic_question
from src.utils import format_currency, format_percent


def _first_nonempty_table(*tables: Any) -> pd.DataFrame:
    for table in tables:
        if isinstance(table, pd.DataFrame) and not table.empty:
            return table
    return pd.DataFrame()


def _ctx_table(ctx: dict, dotted_key: str) -> pd.DataFrame:
    obj: Any = ctx
    for part in dotted_key.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return pd.DataFrame()
    return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame()


def _table_from_context(ctx: dict, q: str) -> pd.DataFrame:
    ql = q.lower()
    if "loyal" in ql or "customer" in ql:
        ci = ctx.get("customer_intelligence", {})
        if "unprofitable" in ql or "loss" in ql:
            return ci.get("unprofitable_customers", pd.DataFrame())
        if "risk" in ql or "churn" in ql:
            return ci.get("at_risk_customers", pd.DataFrame())
        return ci.get("top_loyal_customers", pd.DataFrame())
    if "stop selling" in ql or "remove" in ql or "product" in ql:
        prod = ctx.get("product_intelligence", {})
        return _first_nonempty_table(prod.get("problem_products"), prod.get("discount_damaged_products"), prod.get("product_portfolio"))
    if "region" in ql or "city" in ql or "state" in ql:
        reg = ctx.get("region_intelligence", {})
        return _first_nonempty_table(reg.get("weak_regions"), reg.get("high_sales_low_margin_regions"), reg.get("region_summary"))
    if "discount" in ql:
        return _ctx_table(ctx, "metric_relationships.profit_by_discount_band")
    if "category" in ql or "subcategory" in ql:
        return _first_nonempty_table(ctx.get("category_performance"), _ctx_table(ctx, "product_intelligence.category_summary"), _ctx_table(ctx, "product_intelligence.subcategory_summary"))
    if "forecast" in ql or "predict" in ql:
        return _ctx_table(ctx, "forecasting.forecast")
    return pd.DataFrame()


def _format_value(value: Any, kind: str = "number") -> str:
    if value is None:
        return "N/A"
    if kind == "currency":
        return format_currency(value)
    if kind == "percent":
        return format_percent(value)
    try:
        if isinstance(value, float):
            return f"{value:,.2f}"
        return f"{value:,}"
    except Exception:
        return str(value)


def _direct_answer(question: str, brain_res: dict) -> str:
    ctx = brain_res["context_results"]
    q = question.lower()
    k = ctx.get("kpis", {})
    if "loyal" in q:
        top = ctx.get("customer_intelligence", {}).get("top_loyal_customers", pd.DataFrame())
        if not top.empty:
            r = top.iloc[0]
            return f"The most loyal customer is {r['customer']}. The ranking uses repeat orders, recency, relationship duration, sales, and profit — not sales alone."
        return "Customer loyalty cannot be fully analyzed because the needed customer/order/date fields are missing or weak."
    if "at risk" in q or "churn" in q:
        t = ctx.get("customer_intelligence", {}).get("at_risk_customers", pd.DataFrame())
        return f"There are {len(t)} at-risk loyal customers based on recency and prior activity." if isinstance(t, pd.DataFrame) else "At-risk customers cannot be analyzed from this dataset."
    if "discount" in q and any(w in q for w in ["hurt", "affect", "profit", "margin"]):
        rel = ctx.get("metric_relationships", {})
        band = rel.get("profit_by_discount_band", pd.DataFrame())
        if isinstance(band, pd.DataFrame) and not band.empty:
            worst = band.sort_values("profit").iloc[0]
            return f"Discounting appears to affect profit. The weakest discount band is {worst.get('discount_band')} with profit {format_currency(worst.get('profit'))}."
        return "Discount impact cannot be fully calculated because a discount column was not detected."
    if "sales high" in q or "high sales" in q:
        hs = _ctx_table(ctx, "metric_relationships.high_sales_low_profit")
        if not hs.empty:
            r = hs.iloc[0]
            return f"Sales are not fully converting into profit. {r['dimension_value']} has high sales but weak margin at {format_percent(r.get('profit_margin_percent'))}."
    if any(w in q for w in ["profit", "loss", "margin", "losing money"]):
        roots = brain_res.get("root_causes") or []
        if roots:
            return roots[0]
        return f"Profit was analyzed using the available fields. Total profit is {format_currency(k.get('total_profit'))} and margin is {format_percent(k.get('profit_margin_percent'))}."
    if "board" in q or "management" in q or "summary" in q:
        summ = ctx.get("executive_summary", {})
        return str(summ.get("key_problem", "The dataset has been analyzed for performance, root causes, risks, and actions."))
    if any(w in q for w in ["wrong", "problem", "hidden", "insight", "focus"]):
        insights = ctx.get("hidden_insights", [])
        if insights:
            return f"The main issue is: {insights[0]['title']}. {insights[0]['why_it_matters']}"
        return "No strong hidden issue was detected from the available fields."
    return "I analyzed the dataset using the detected business columns and generated evidence-backed findings."


def generate_business_answer(question: str, brain_res: dict) -> str:
    ctx = brain_res["context_results"]
    k = ctx.get("kpis", {})
    q = question.lower()
    lines: list[str] = []

    lines.append("## Direct Answer")
    lines.append(_direct_answer(question, brain_res))

    missing_meta = ctx.get("missing_value_treatment", {}) if isinstance(ctx, dict) else {}
    if isinstance(missing_meta, dict) and missing_meta.get("cleaned_dataset_used"):
        lines.append("\n## Data Reliability Note")
        lines.append(f"- {missing_meta.get('manager_warning', 'Calculations are based on a cleaned dataset because missing values were found in the original dataset.')}")
        if missing_meta.get("summary_text"):
            lines.append(f"- {missing_meta['summary_text']}")

    lines.append("\n## Evidence From Data")
    if k.get("total_sales") is not None:
        lines.append(f"- Total Sales: {format_currency(k.get('total_sales'))}")
    if k.get("total_profit") is not None:
        lines.append(f"- Total Profit: {format_currency(k.get('total_profit'))}")
    if k.get("profit_margin_percent") is not None:
        lines.append(f"- Profit Margin: {format_percent(k.get('profit_margin_percent'))}")
    if k.get("loss_making_orders_count") is not None:
        lines.append(f"- Loss-making Orders: {_format_value(k.get('loss_making_orders_count'))} ({format_percent(k.get('loss_making_orders_percent'))})")
    if k.get("average_discount") is not None:
        lines.append(f"- Average Discount: {_format_value(k.get('average_discount'))}")

    table = _table_from_context(ctx, q)
    if isinstance(table, pd.DataFrame) and not table.empty:
        first = table.head(1).replace({pd.NA: None}).to_dict("records")[0]
        compact_parts = []
        for col in list(first)[:6]:
            val = first[col]
            compact_parts.append(f"{str(col).replace('_', ' ').title()}: {_format_value(val) if isinstance(val, (int, float)) else str(val)}")
        lines.append(f"- Top evidence row: {' | '.join(compact_parts)}")

    lines.append("\n## Relationships / Drivers")
    rel = brain_res.get("relationship_findings", [])
    if rel:
        for r in rel[:7]:
            lines.append(f"- {r}")
    else:
        lines.append("- No strong driver relationship could be proven from the available columns.")

    if any(w in q for w in ["profit", "loss", "margin", "discount", "sales high", "high sales"]):
        pi = ctx.get("profit_intelligence", {})
        ps = pi.get("profit_summary", {}) if isinstance(pi, dict) else {}
        if ps:
            lines.append(f"- Profit conversion check: sales {format_currency(ps.get('total_sales'))} generated profit {format_currency(ps.get('total_profit'))}, giving margin {format_percent(ps.get('overall_margin'))}.")

    lines.append("\n## Root Cause")
    roots = brain_res.get("root_causes", [])
    if roots:
        for r in roots[:4]:
            lines.append(f"- {r}")
    else:
        lines.append("- Root cause requires more detailed columns such as profit, discount, product, customer, region, and date.")

    lines.append("\n## Recommended Action")
    recs = brain_res.get("recommendations", [])
    if recs:
        for rec in recs[:4]:
            title = rec.get("title", "Recommendation")
            action = rec.get("recommended_action", "Review this area.")
            evidence = rec.get("evidence_from_data", "")
            lines.append(f"- **{title}:** {action} Evidence: {evidence}")
    else:
        lines.append("- Keep monitoring core KPIs and improve dataset detail for deeper diagnosis.")

    lines.append("\n## Confidence")
    lines.append(f"{brain_res.get('confidence', 'Medium')} — based on detected columns, entity matching, and computed evidence.")

    lines.append("\n## Limitations")
    limitations = brain_res.get("limitations") or []
    if limitations:
        for lim in limitations[:8]:
            lines.append(f"- {lim}")
    else:
        lines.append("- Results depend on uploaded data quality and available columns.")

    lines.append("\n## Suggested Follow-up Questions")
    for f in brain_res.get("follow_up_questions", [])[:3]:
        lines.append(f"- {f}")
    return "\n".join(lines)


def answer_business_question(question, df, column_roles, context_results: dict | None = None):
    dynamic = answer_dynamic_question(question, df, column_roles)
    if isinstance(dynamic, dict) and dynamic.get("handled"):
        if isinstance(context_results, dict):
            dynamic["context_results"] = {**context_results, **dynamic.get("context_results", {})}
        deterministic = dynamic.get("markdown_answer", "")
        final_answer, llm_meta = beautify_answer_with_llm(deterministic, dynamic)
        dynamic["markdown_answer"] = final_answer
        dynamic["deterministic_answer"] = deterministic
        dynamic["llm_meta"] = llm_meta
        return dynamic

    brain = answer_with_analyst_brain(question, df, column_roles)
    if isinstance(context_results, dict):
        brain["context_results"] = {**brain.get("context_results", {}), **context_results}
    deterministic = generate_business_answer(question, brain)
    final_answer, llm_meta = beautify_answer_with_llm(deterministic, brain)
    brain["markdown_answer"] = final_answer
    brain["deterministic_answer"] = deterministic
    brain["llm_meta"] = llm_meta
    return brain
