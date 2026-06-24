from __future__ import annotations

from typing import Any
import pandas as pd

from src.semantic_layer import build_semantic_model
from src.entity_resolver import resolve_entities, filter_dataframe_by_entities
from src.analysis_planner import create_analysis_plan
from src.kpi_engine import calculate_kpis, performance_by_category, performance_by_region, performance_over_time
from src.metric_relationships import analyze_metric_relationships
from src.profit_intelligence import analyze_profit_intelligence
from src.root_cause import find_loss_drivers, root_cause_for_selection, explain_metric_drop
from src.customer_intelligence import analyze_customer_loyalty
from src.product_intelligence import analyze_product_portfolio
from src.region_intelligence import analyze_region_segment_performance
from src.hidden_insights import generate_hidden_insights
from src.recommendation_engine import generate_recommendations, generate_executive_summary
from src.forecasting import forecast_sales_profit
from src.anomaly_detection import detect_business_anomalies
from src.utils import prepare_analysis_dataset


PROFIT_WORDS = [
    "profit", "loss", "margin", "profitability", "losing money", "low profit",
    "negative profit", "revenue leakage", "bleeding money", "bleeding",
    "bad contribution", "contribution", "margin leakage", "leakage",
    "profit drain", "money drain", "underperforming", "unprofitable", "loss-making"
]
CUSTOMER_WORDS = ["customer", "loyal", "retention", "repeat", "churn", "buyer", "client", "account"]
PRODUCT_WORDS = ["product", "category", "subcategory", "sub-category", "sku", "item", "stop selling", "portfolio", "assortment"]
REGION_WORDS = ["region", "city", "state", "country", "zone", "territory", "segment", "market", "area"]
TIME_WORDS = ["month", "year", "trend", "dropped", "increased", "decreased", "growth", "latest", "previous", "forecast", "predict"]
VAGUE_WORDS = ["wrong", "problem", "hidden", "insight", "focus", "summary", "board", "management", "analyze", "risks"]


def _roles(roles: dict) -> dict:
    return roles.get("roles", roles) if isinstance(roles, dict) else {}


def _detect_intent(question: str) -> str:
    q = question.lower()
    if "board" in q:
        return "board_summary"
    if any(w in q for w in PROFIT_WORDS):
        return "profit_analysis"
    if any(w in q for w in CUSTOMER_WORDS):
        return "customer_analysis"
    if any(w in q for w in PRODUCT_WORDS):
        return "product_analysis"
    if any(w in q for w in REGION_WORDS):
        return "region_analysis"
    if any(w in q for w in TIME_WORDS):
        return "time_trend_analysis"
    if any(w in q for w in VAGUE_WORDS):
        return "hidden_insights"
    return "unknown_general_dataset_question"


def _collect_tables(obj: Any, prefix: str = "") -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    if isinstance(obj, pd.DataFrame):
        if not obj.empty:
            tables[prefix or "table"] = obj
        return tables
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_name = f"{prefix}.{key}" if prefix else str(key)
            tables.update(_collect_tables(value, child_name))
    return tables


def _column_limitations(roles: dict, semantic: dict) -> list[str]:
    limitations: list[str] = []
    missing = semantic.get("missing_but_useful_columns", []) if isinstance(semantic, dict) else []
    for m in missing[:6]:
        limitations.append(f"Useful column missing or not detected: {m}.")
    if not roles.get("profit_column"):
        limitations.append("Profit column was not detected, so margin/loss analysis is limited.")
    if not roles.get("date_column"):
        limitations.append("Date column was not detected, so trend, recency, and forecasting analysis are limited.")
    if not roles.get("discount_column"):
        limitations.append("Discount column was not detected, so discount-impact analysis is limited.")
    if roles.get("_profit_derivation_note"):
        limitations.append(roles["_profit_derivation_note"] + " Treat profit findings as derived from detected columns.")
    if roles.get("_profit_is_estimated"):
        limitations.append("Profit was estimated from margin percentage, so loss analysis is directional rather than accounting-grade.")
    missing_meta = roles.get("_missing_value_treatment", {}) if isinstance(roles, dict) else {}
    if missing_meta.get("cleaned_dataset_used"):
        limitations.append(missing_meta.get("manager_warning", "Calculations are based on a cleaned dataset because missing values were found."))
    if semantic.get("grain") in {"daily_or_monthly_summary_level", "monthly_summary_level"}:
        limitations.append("Dataset appears aggregated, so order count, AOV, and customer loyalty metrics may represent records rather than individual transactions.")
    return limitations


def answer_with_analyst_brain(question, df, column_roles):
    """Central deterministic reasoning layer for manager Q&A.

    It runs a semantic model, entity resolution, intent planning, computation modules, and then
    returns evidence tables plus findings. LLMs are not used here, so all numbers remain grounded.
    """
    df, roles = prepare_analysis_dataset(df, column_roles)
    semantic = build_semantic_model(df, roles)
    entities = resolve_entities(question, df, roles)
    plan = create_analysis_plan(question, semantic, roles, entities)
    q = str(question).lower()
    intent = _detect_intent(question)
    target_df = filter_dataframe_by_entities(df, entities) if entities else df
    if target_df is None or target_df.empty:
        target_df = df
    ctx: dict[str, Any] = {}
    limitations = _column_limitations(roles, semantic)

    # Always calculate the core executive layer.
    ctx["semantic_model"] = semantic
    ctx["missing_value_treatment"] = roles.get("_missing_value_treatment", {})
    ctx["kpis"] = calculate_kpis(target_df, roles)

    needs_profit = any(w in q for w in PROFIT_WORDS) or intent in {"profit_analysis", "hidden_insights", "board_summary"}
    needs_customer = any(w in q for w in CUSTOMER_WORDS) or intent in {"customer_analysis", "board_summary", "hidden_insights"}
    needs_product = any(w in q for w in PRODUCT_WORDS) or intent in {"product_analysis", "board_summary", "hidden_insights"}
    needs_region = any(w in q for w in REGION_WORDS) or intent in {"region_analysis", "board_summary", "hidden_insights"}
    needs_time = any(w in q for w in TIME_WORDS) or intent in {"time_trend_analysis", "board_summary"}
    needs_broad = intent in {"hidden_insights", "unknown_general_dataset_question", "board_summary"}

    if needs_profit or needs_broad:
        ctx["profit_intelligence"] = analyze_profit_intelligence(target_df, roles, entities)
        ctx["metric_relationships"] = analyze_metric_relationships(target_df, roles)
        ctx["loss_drivers"] = find_loss_drivers(target_df, roles)
        try:
            ctx["metric_drop"] = explain_metric_drop(target_df, roles, metric="profit")
        except Exception as exc:
            ctx["metric_drop"] = {"limitations": [str(exc)]}

    if entities:
        entity_details = []
        for ent in entities:
            col, val = ent.get("matched_column"), ent.get("matched_value")
            if col in df.columns:
                try:
                    entity_details.append(root_cause_for_selection(df, roles, col, val))
                except Exception as exc:
                    entity_details.append({"target": val, "limitations": [str(exc)]})
        ctx["entity_root_causes"] = entity_details

    if needs_customer or needs_broad:
        ctx["customer_intelligence"] = analyze_customer_loyalty(target_df, roles)
    if needs_product or needs_broad:
        ctx["product_intelligence"] = analyze_product_portfolio(target_df, roles)
    if needs_region or needs_broad:
        ctx["region_intelligence"] = analyze_region_segment_performance(target_df, roles)
    if needs_time or needs_broad:
        ctx["time_performance"] = performance_over_time(target_df, roles)
        ctx["forecasting"] = forecast_sales_profit(target_df, roles)
        ctx["anomaly_detection"] = detect_business_anomalies(target_df, roles)
    if needs_broad or any(w in q for w in VAGUE_WORDS):
        ctx["hidden_insights"] = generate_hidden_insights(target_df, roles)

    # For common ranking questions, add ready tables even if the broad modules were not triggered.
    if "category" in q:
        ctx["category_performance"] = performance_by_category(target_df, roles)
    if "region" in q:
        ctx["region_performance"] = performance_by_region(target_df, roles)

    ctx["recommendations"] = generate_recommendations(target_df, roles, ctx)
    ctx["executive_summary"] = generate_executive_summary(ctx)

    relationship_findings = []
    if isinstance(ctx.get("metric_relationships"), dict):
        relationship_findings.extend(ctx["metric_relationships"].get("relationship_summary", []))
    if not relationship_findings and needs_profit:
        relationship_findings.append("Profit relationships could not be fully calculated because one or more driver columns were missing.")

    root_causes = []
    if isinstance(ctx.get("profit_intelligence"), dict):
        root_causes.extend(ctx["profit_intelligence"].get("root_cause_summary", []))
    if ctx.get("entity_root_causes"):
        for erc in ctx["entity_root_causes"][:3]:
            if isinstance(erc, dict):
                for rc in erc.get("likely_root_causes", [])[:2]:
                    root_causes.append(rc)

    if not root_causes and isinstance(ctx.get("hidden_insights"), list) and ctx["hidden_insights"]:
        root_causes.append(ctx["hidden_insights"][0].get("why_it_matters", "Hidden insight detected."))

    follow = _followups(intent, roles)
    confidence = "High"
    if limitations:
        confidence = "Medium"
    if not roles.get("sales_column") and not roles.get("profit_column"):
        confidence = "Low"

    return {
        "question": question,
        "detected_intent": intent,
        "target_metric": "profit" if needs_profit else ("customer" if needs_customer else "general"),
        "entities": entities,
        "analysis_plan": plan,
        "answer": "",
        "evidence_tables": _collect_tables(ctx),
        "relationship_findings": relationship_findings[:10],
        "root_causes": root_causes[:10],
        "recommendations": ctx["recommendations"],
        "confidence": confidence,
        "limitations": limitations,
        "follow_up_questions": follow,
        "context_results": ctx,
    }


def _followups(intent: str, roles: dict) -> list[str]:
    if intent == "customer_analysis":
        return ["Which loyal customers are unprofitable?", "Which customers are at risk?", "Which customers should we retain first?"]
    if intent == "product_analysis":
        return ["Which products should we stop selling?", "Which products have high sales but low profit?", "Which products are margin winners?"]
    if intent == "region_analysis":
        return ["Which region should we fix first?", "Which region has high sales but low profit?", "Is discounting hurting any region?"]
    if intent == "board_summary":
        return ["What are the top 5 problems?", "What action has the highest expected impact?", "What risks should management monitor?" ]
    return ["Which category is hurting profit the most?", "Is discounting damaging margin?", "Which customer or product should we review first?"]
