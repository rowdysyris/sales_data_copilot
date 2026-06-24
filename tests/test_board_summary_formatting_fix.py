from __future__ import annotations
import pandas as pd

from src.column_detection import detect_columns
from src.question_answering import answer_business_question
from src.recommendation_engine import generate_executive_summary, executive_summary_to_html, executive_summary_to_markdown
from src.kpi_engine import calculate_kpis
from src.profit_intelligence import analyze_profit_intelligence


def board_df():
    return pd.DataFrame([
        {"Order ID":"1", "Order Date":"2024-01-01", "Category":"Furniture", "Sub-Category":"Tables", "Region":"South", "Product Name":"Table A", "Sales":1000, "Profit":-500, "Discount":0.45, "Quantity":2},
        {"Order ID":"2", "Order Date":"2024-01-02", "Category":"Furniture", "Sub-Category":"Tables", "Region":"South", "Product Name":"Table A", "Sales":1200, "Profit":-650, "Discount":0.50, "Quantity":3},
        {"Order ID":"3", "Order Date":"2024-01-03", "Category":"Technology", "Sub-Category":"Phones", "Region":"West", "Product Name":"Phone A", "Sales":2200, "Profit":350, "Discount":0.05, "Quantity":1},
        {"Order ID":"4", "Order Date":"2024-01-04", "Category":"Office Supplies", "Sub-Category":"Paper", "Region":"North", "Product Name":"Paper A", "Sales":300, "Profit":90, "Discount":0.00, "Quantity":5},
    ])


def test_executive_summary_fields_are_not_duplicate():
    df = board_df()
    roles = detect_columns(df)["roles"]
    ctx = {"kpis": calculate_kpis(df, roles), "profit_intelligence": analyze_profit_intelligence(df, roles)}
    ctx["recommendations"] = []
    summary = generate_executive_summary(ctx)

    assert isinstance(summary, dict)
    assert summary["key_problem"]
    assert summary["root_cause"]
    assert summary["key_problem"] != summary["root_cause"]
    assert "Sales are 2297200" not in summary["business_performance"]
    assert "$" in summary["business_performance"]


def test_executive_summary_rendering_is_not_raw_dict():
    summary = {
        "business_performance": "Total sales are $1,000.00.",
        "key_problem": "Tables are loss-making.",
        "root_cause": "High discount is damaging margin.",
        "financial_impact": "Total loss amount is -$1,150.00.",
        "recommended_action": "Cap discounts.",
        "risk": "Depends on data completeness.",
        "next_step": "Validate with business owners.",
    }
    html = executive_summary_to_html(summary)
    markdown = executive_summary_to_markdown(summary)

    assert "{'business_performance'" not in html
    assert "Business Performance" in html
    assert "Key Problem" in html
    assert "Root Cause" in html
    assert "**Business Performance:**" in markdown
    assert "{'" not in markdown


def test_board_level_answer_does_not_show_python_dict():
    df = board_df()
    roles = detect_columns(df)["roles"]
    ans = answer_business_question("Give me a board-level summary", df, roles)
    md = ans["markdown_answer"]
    summary = ans["context_results"]["executive_summary"]

    assert "{'business_performance'" not in md
    assert "## Direct Answer" in md
    assert summary["key_problem"] != summary["root_cause"]
