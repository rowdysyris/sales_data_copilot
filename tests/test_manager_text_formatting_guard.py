import pandas as pd

from src.column_detection import detect_columns
from src.kpi_engine import calculate_kpis
from src.profit_intelligence import analyze_profit_intelligence
from src.recommendation_engine import generate_executive_summary
from src.question_answering import answer_business_question


def _sample_sales_df():
    rows = []
    for i in range(12):
        rows.append({
            "Order ID": f"T-{i}",
            "Order Date": f"2024-01-{i+1:02d}",
            "Category": "Furniture",
            "Sub-Category": "Tables",
            "Product Name": "Table Model A",
            "Region": "South",
            "Segment": "Consumer",
            "Customer Name": f"Customer {i%3}",
            "Sales": 1000,
            "Quantity": 2,
            "Discount": 0.45,
            "Profit": -250,
        })
    for i in range(8):
        rows.append({
            "Order ID": f"P-{i}",
            "Order Date": f"2024-02-{i+1:02d}",
            "Category": "Technology",
            "Sub-Category": "Phones",
            "Product Name": "Phone Model B",
            "Region": "West",
            "Segment": "Corporate",
            "Customer Name": f"Customer {i+3}",
            "Sales": 1500,
            "Quantity": 1,
            "Discount": 0.05,
            "Profit": 80,
        })
    return pd.DataFrame(rows)


def test_profit_root_causes_use_manager_friendly_currency_not_bare_decimals():
    df = _sample_sales_df()
    roles = detect_columns(df)["roles"]
    pi = analyze_profit_intelligence(df, roles)
    root_text = " ".join(pi.get("root_cause_summary", []))

    assert "profit of" in root_text
    assert "$" in root_text
    assert "profit -" not in root_text
    assert "low margin (" not in root_text


def test_executive_summary_root_cause_not_semicolon_chain():
    df = _sample_sales_df()
    roles = detect_columns(df)["roles"]
    pi = analyze_profit_intelligence(df, roles)
    ctx = {"kpis": calculate_kpis(df, roles), "profit_intelligence": pi, "recommendations": []}
    summary = generate_executive_summary(ctx)

    assert summary["key_problem"] != summary["root_cause"]
    assert ";" not in summary["root_cause"]
    assert "The likely drivers are:" in summary["root_cause"] or "Loss-making orders exist" in summary["root_cause"]
    assert ";" not in summary["financial_impact"]


def test_qa_root_cause_section_does_not_collapse_drivers_into_one_semicolon_sentence():
    df = _sample_sales_df()
    roles = detect_columns(df)["roles"]
    ans = answer_business_question("Give me a board-level summary", df, roles)
    text = ans["markdown_answer"]

    assert "## Root Cause" in text
    assert "; Discount band" not in text
    assert "profit -" not in text
