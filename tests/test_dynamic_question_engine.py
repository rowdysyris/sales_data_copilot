import pandas as pd

from src.column_detection import detect_columns
from src.question_answering import answer_business_question


def _df():
    return pd.DataFrame({
        "Order ID": [str(i) for i in range(1, 13)],
        "Order Date": pd.to_datetime([
            "2024-01-01", "2024-01-03", "2024-02-01", "2024-02-05",
            "2024-03-01", "2024-03-02", "2024-03-07", "2024-03-10",
            "2024-04-01", "2024-04-05", "2024-05-01", "2024-05-10",
        ]),
        "Customer Name": ["A", "A", "A", "B", "C", "C", "D", "D", "E", "F", "G", "H"],
        "Region": ["South", "South", "North", "South", "West", "West", "South", "East", "North", "East", "South", "West"],
        "Category": ["Furniture", "Furniture", "Technology", "Furniture", "Office Supplies", "Technology", "Furniture", "Technology", "Technology", "Furniture", "Office Supplies", "Technology"],
        "Sub-Category": ["Tables", "Tables", "Phones", "Tables", "Paper", "Copiers", "Chairs", "Phones", "Machines", "Bookcases", "Binders", "Phones"],
        "Product Name": ["Table X", "Table X", "Phone A", "Table Y", "Paper A", "Copier A", "Chair A", "Phone B", "Machine A", "Bookcase A", "Binder A", "Phone C"],
        "Sales": [1000, 1200, 2000, 1500, 200, 3000, 700, 2500, 5000, 1000, 300, 6000],
        "Quantity": [2, 2, 1, 3, 4, 1, 1, 2, 3, 2, 4, 3],
        "Discount": [40, 50, 10, 40, 0, 10, 20, 5, 30, 45, 60, 5],
        "Profit": [-300, -500, 400, -600, 60, 900, 80, 500, -1000, -200, -50, 1000],
    })


def _roles():
    return detect_columns(_df())["roles"]


def test_top_loss_generating_products_uses_product_level_table():
    ans = answer_business_question("what are the top 5 loss generating products?", _df(), _roles())
    assert ans["intent"] == "dynamic_ranking"
    table = ans["evidence_tables"]["dynamic_ranking_result"]
    assert table.iloc[0]["dimension_value"] == "Machine A"
    assert table.iloc[0]["profit"] == -1000
    assert "Loss-Generating Products" in ans["markdown_answer"]


def test_highest_revenue_products_and_regions_are_dynamic_rankings():
    product_ans = answer_business_question("what are the products generating highest revenue", _df(), _roles())
    region_ans = answer_business_question("what are the regions generating highest revenue", _df(), _roles())
    assert product_ans["evidence_tables"]["dynamic_ranking_result"].iloc[0]["dimension_value"] == "Phone C"
    assert region_ans["evidence_tables"]["dynamic_ranking_result"].iloc[0]["dimension_value"] == "West"
    assert "Revenue alone does not prove business quality" in product_ans["deterministic_answer"]


def test_broad_product_loss_question_explains_drivers_without_mixing_specific_entity_fallback():
    broad = answer_business_question("why are products making loss", _df(), _roles())
    assert broad["intent"] == "dynamic_root_cause"
    assert "Root-Cause Drivers" in broad["markdown_answer"]
    assert "Machine A" in broad["markdown_answer"]

    specific = answer_business_question("why are Tables losing money", _df(), _roles())
    assert specific.get("detected_intent") == "profit_analysis"
    assert specific.get("entities")


def test_churn_control_question_uses_customer_date_history_or_limits():
    ans = answer_business_question("how should I control churn", _df(), _roles())
    assert ans["intent"] == "churn_analysis"
    assert "churn" in ans["markdown_answer"].lower()
    assert "churn_risk_customers" in ans["evidence_tables"]


def test_monthly_highest_sales_question_uses_trend_engine():
    ans = answer_business_question("which month had highest sales", _df(), _roles())
    assert ans["intent"] == "dynamic_trend"
    assert "2024-03" in ans["markdown_answer"]


def test_dynamic_ranking_does_not_require_tabulate(monkeypatch):
    def fail_to_markdown(self, *args, **kwargs):
        raise ImportError("tabulate missing")

    monkeypatch.setattr(pd.DataFrame, "to_markdown", fail_to_markdown, raising=False)
    ans = answer_business_question("what are the top 5 loss generating products?", _df(), _roles())
    assert ans["intent"] == "dynamic_ranking"
    assert "| Rank | Name |" in ans["markdown_answer"]
    assert "Machine A" in ans["markdown_answer"]


def test_broad_profit_low_scans_all_dimensions_not_single_table_only():
    ans = answer_business_question("Why is profit low?", _df(), _roles())
    assert ans["intent"] == "dynamic_profit_diagnostic"
    text = ans["deterministic_answer"]
    assert "Full Profit Diagnostic" in text
    assert "all available dimensions" in text
    assert "product" in ans["analysis_plan"]["dimensions_scanned"]
    assert "category" in ans["analysis_plan"]["dimensions_scanned"]
    assert "profit_diagnostic_all_dimensions" in ans["evidence_tables"]
    table = ans["evidence_tables"]["profit_diagnostic_all_dimensions"]
    assert set(["dimension_type", "dimension_value", "profit"]).issubset(table.columns)
    assert table["dimension_type"].nunique() >= 3
