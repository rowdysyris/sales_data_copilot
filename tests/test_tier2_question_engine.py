import pandas as pd

from src.column_detection import detect_columns
from src.question_answering import answer_business_question


def tier2_df():
    return pd.DataFrame({
        "Order ID": [f"O{i}" for i in range(1, 13)],
        "Category": ["Furniture", "Furniture", "Technology", "Technology", "Office Supplies", "Office Supplies", "Furniture", "Technology", "Technology", "Office Supplies", "Furniture", "Technology"],
        "Region": ["East", "West", "East", "West", "East", "West", "South", "South", "Central", "Central", "East", "West"],
        "Segment": ["Consumer", "Corporate", "Consumer", "Corporate", "Home Office", "Consumer", "Corporate", "Consumer", "Home Office", "Corporate", "Consumer", "Corporate"],
        "Product Name": ["Table A", "Chair B", "Phone A", "Laptop B", "Paper A", "Binder B", "Table C", "Phone C", "Copier D", "Storage E", "Chair F", "Machine G"],
        "Sales": [1000, 800, 2000, 3000, 200, 400, 500, 2500, 3500, 700, 900, 4500],
        "Discount": [40, 10, 5, 0, 0, 30, 50, 10, 5, 20, 15, 0],
        "Profit": [-300, 120, 500, 900, 50, -80, -250, 450, 1100, 100, 90, 1300],
        "Quantity": [2, 1, 2, 3, 4, 5, 1, 2, 1, 2, 1, 3],
    })


def roles():
    return detect_columns(tier2_df())["roles"]


def assert_tier2_structure(ans):
    text = ans["markdown_answer"]
    assert "## Headline" in text
    assert "## Data" in text
    assert "## Insight" in text
    assert "## Recommendation" in text
    assert "₹" not in text
    assert "$" in text or "discount" in text.lower()
    assert ans["analysis_plan"].get("source_rows_sent_to_ai") == 0


def test_most_profitable_category_is_preaggregated_and_uses_profit_metric():
    ans = answer_business_question("Which category is the most profitable?", tier2_df(), roles())
    assert ans["intent"] == "dynamic_ranking"
    assert ans["analysis_plan"]["metric"] == "profit"
    table = ans["evidence_tables"]["tier2_result"]
    assert table.iloc[0]["dimension_value"] == "Technology"
    assert table.iloc[0]["profit"] == 4250
    assert_tier2_structure(ans)


def test_discount_affect_profit_uses_correlation_and_band_table():
    ans = answer_business_question("How does discount affect profit?", tier2_df(), roles())
    assert ans["intent"] == "tier2_correlation"
    assert ans["analysis_plan"]["operation"] == "correlate_and_group_by_band"
    assert ans["analysis_plan"]["preaggregated_rows_sent_to_ai"] < len(tier2_df())
    assert "tier2_discount_profit_by_band" in ans["evidence_tables"]
    assert "correlation" in ans["markdown_answer"].lower()
    assert_tier2_structure(ans)


def test_top_5_products_by_sales_uses_product_sales_ranking():
    ans = answer_business_question("What are the top 5 products by sales?", tier2_df(), roles())
    assert ans["intent"] == "dynamic_ranking"
    assert ans["analysis_plan"]["dimension"] == "product"
    assert ans["analysis_plan"]["metric"] == "sales"
    assert ans["analysis_plan"]["limit"] == 5
    table = ans["evidence_tables"]["tier2_result"]
    assert len(table) == 5
    assert table.iloc[0]["dimension_value"] == "Machine G"
    assert table.iloc[0]["sales"] == 4500
    assert_tier2_structure(ans)
