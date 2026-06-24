import os
import pandas as pd
from src.column_detection import detect_columns
from src.question_answering import answer_business_question
from src.forecasting import forecast_sales_profit
from src.anomaly_detection import detect_business_anomalies
from src.report_generator import generate_excel_report, generate_pdf_report
from src.scenario_simulator import simulate_discount_cap, simulate_price_increase
from src.llm_explainer import beautify_answer_with_llm


def advanced_df():
    rows = []
    for m in range(1, 15):
        rows.extend([
            {
                "Order ID": f"T{m}a", "Order Date": pd.Timestamp(2024, min(m, 12), 5),
                "Customer Name": "Alpha", "Region": "South", "Segment": "Consumer",
                "Category": "Furniture", "Sub-Category": "Tables", "Product Name": "Table X",
                "Sales": 1000 + m * 20, "Quantity": 2, "Discount": 0.45, "Profit": -350 - m * 5,
            },
            {
                "Order ID": f"P{m}a", "Order Date": pd.Timestamp(2024, min(m, 12), 10),
                "Customer Name": "Beta", "Region": "West", "Segment": "Corporate",
                "Category": "Technology", "Sub-Category": "Phones", "Product Name": "Phone A",
                "Sales": 2200 + m * 30, "Quantity": 1, "Discount": 0.05, "Profit": 600 + m * 10,
            },
            {
                "Order ID": f"C{m}a", "Order Date": pd.Timestamp(2024, min(m, 12), 15),
                "Customer Name": "Gamma", "Region": "North", "Segment": "Home Office",
                "Category": "Office Supplies", "Sub-Category": "Paper", "Product Name": "Paper A",
                "Sales": 200 + m * 5, "Quantity": 5, "Discount": 0.0, "Profit": 50 + m,
            },
        ])
    # A stale high-value customer and an outlier.
    rows.append({"Order ID":"OLD1", "Order Date": pd.Timestamp(2023, 1, 1), "Customer Name":"Old Loyal", "Region":"East", "Segment":"Consumer", "Category":"Technology", "Sub-Category":"Copiers", "Product Name":"Copier Z", "Sales":8000, "Quantity":1, "Discount":0.1, "Profit":1200})
    rows.append({"Order ID":"OUT1", "Order Date": pd.Timestamp(2024, 12, 20), "Customer Name":"Outlier", "Region":"South", "Segment":"Consumer", "Category":"Furniture", "Sub-Category":"Tables", "Product Name":"Table Mega", "Sales":50000, "Quantity":20, "Discount":0.7, "Profit":-25000})
    return pd.DataFrame(rows)


def roles():
    return detect_columns(advanced_df())["roles"]


def test_profit_question_is_relationship_first():
    ans = answer_business_question("Why is profit low?", advanced_df(), roles())
    md = ans["markdown_answer"]
    assert "Relationships / Drivers" in md
    assert "Discount" in md or "discount" in md
    assert "Total Profit" in md
    assert ans["relationship_findings"]


def test_entity_specific_question_tables():
    ans = answer_business_question("Why are Tables losing money?", advanced_df(), roles())
    assert any(e["matched_value"] == "Tables" for e in ans["entities"])
    assert "Tables" in ans["markdown_answer"] or "Table" in ans["markdown_answer"]


def test_customer_loyalty_question_not_sales_only():
    ans = answer_business_question("Who are our most loyal customers?", advanced_df(), roles())
    table = ans["context_results"]["customer_intelligence"]["top_loyal_customers"]
    assert "loyalty_score" in table.columns
    assert "order_count" in table.columns
    assert "days_since_last_order" in table.columns
    assert not table.empty


def test_forecasting_has_backtest_and_limits():
    out = forecast_sales_profit(advanced_df(), roles(), periods=3)
    assert not out["forecast"].empty
    assert "backtest" in out
    assert out["confidence"] in {"High", "Medium", "Low"}
    assert out["limitations"]


def test_anomaly_detection_returns_dict_and_rows():
    out = detect_business_anomalies(advanced_df(), roles())
    assert "anomalies" in out
    assert isinstance(out["anomalies"], list)
    assert len(out["anomalies"]) >= 1


def test_reports_generate_bytes():
    df = advanced_df()
    r = roles()
    ans = answer_business_question("Give me board-level summary", df, r)
    ctx = ans["context_results"]
    xlsx = generate_excel_report(df, r, ctx)
    pdf = generate_pdf_report(ctx)
    assert len(xlsx) > 5000
    assert len(pdf) > 1000


def test_scenarios_work():
    df = advanced_df()
    r = roles()
    cap = simulate_discount_cap(df, r, 30)
    price = simulate_price_increase(df, r, 5)
    assert "estimated_profit_change" in cap
    assert price["simulated_profit"] >= price["baseline_profit"]


def test_llm_fallback_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    answer, meta = beautify_answer_with_llm("## Direct Answer\nComputed answer only.", {"confidence":"High"})
    assert answer.startswith("## Direct Answer")
    assert meta["fallback_used"] is True


def test_dataset_without_profit_does_not_crash():
    df = advanced_df().drop(columns=["Profit"])
    r = detect_columns(df)["roles"]
    ans = answer_business_question("Why is profit low?", df, r)
    assert "Profit column" in ans["markdown_answer"] or "profit" in ans["markdown_answer"].lower()
    assert ans["confidence"] in {"Medium", "Low"}
