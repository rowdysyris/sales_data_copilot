import pandas as pd

from src.column_detection import detect_columns
from src.utils import prepare_analysis_dataset
from src.data_quality import analyze_data_quality
from src.kpi_engine import calculate_kpis
from src.profit_intelligence import analyze_profit_intelligence
from src.question_answering import answer_business_question
from src.forecasting import forecast_sales_profit
from src.anomaly_detection import detect_business_anomalies
from src.powerbi_exporter import build_powerbi_star_schema
from src.hidden_insights import generate_hidden_insights


def _run_core_smoke(df, roles, question="Which SKU is bleeding money?"):
    prepared_df, prepared_roles = prepare_analysis_dataset(df, roles)
    assert isinstance(prepared_df, pd.DataFrame)
    assert isinstance(prepared_roles, dict)
    assert isinstance(calculate_kpis(prepared_df, prepared_roles), dict)
    assert isinstance(analyze_data_quality(prepared_df, prepared_roles), dict)
    assert isinstance(analyze_profit_intelligence(prepared_df, prepared_roles), dict)
    assert isinstance(answer_business_question(question, prepared_df, prepared_roles), dict)
    assert isinstance(forecast_sales_profit(prepared_df, prepared_roles), dict)
    assert isinstance(detect_business_anomalies(prepared_df, prepared_roles), dict)
    assert isinstance(build_powerbi_star_schema(prepared_df, prepared_roles), dict)
    assert isinstance(generate_hidden_insights(prepared_df, prepared_roles), list)
    return prepared_df, prepared_roles


def test_sales_cost_dataset_derives_profit_and_handles_returns():
    df = pd.DataFrame({
        "Invoice Date": ["01/02/2024", "15/02/2024", "01/03/2024", "15/03/2024"],
        "SKU": ["A", "B", "A", "C"],
        "Product Category": ["Cat1", "Cat2", "Cat1", "Cat3"],
        "Region": ["South", "North", "South", "West"],
        "Customer": ["X", "Y", "X", "Z"],
        "Order ID": ["1", "2", "3", "4"],
        "Sales": [1000, 2000, -500, 1200],
        "Total Cost": [800, 2300, -400, 900],
        "Qty": [2, 3, -1, 1],
        "Transaction Type": ["Sale", "Sale", "Return", "Sale"],
    })
    roles = detect_columns(df)["roles"]
    assert roles["sales_column"] == "Sales"
    assert roles["cost_column"] == "Total Cost"
    assert "profit_column" not in roles
    prepared_df, prepared_roles = _run_core_smoke(df, roles)
    assert prepared_roles["profit_column"] == "__Derived Profit"
    assert "Profit was derived" in prepared_roles["_profit_derivation_note"]
    dq = analyze_data_quality(prepared_df, prepared_roles)
    assert any("returns" in s.lower() or "refund" in s.lower() for s in dq.get("business_signals", []))


def test_margin_percent_dataset_does_not_treat_margin_as_absolute_profit():
    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-02-01"],
        "Product": ["P1", "P2", "P3"],
        "Revenue": [1000, 500, 800],
        "Margin %": ["12%", "-5%", "20%"],
        "Discount": ["5%", "10%", "0%"],
    })
    roles = detect_columns(df)["roles"]
    assert roles["margin_percent_column"] == "Margin %"
    assert roles.get("profit_column") != "Margin %"
    prepared_df, prepared_roles = _run_core_smoke(df, roles, "Where is margin leakage?")
    assert prepared_roles["profit_column"] == "__Estimated Profit From Margin"
    assert prepared_df["__Estimated Profit From Margin"].round(2).tolist() == [120.0, -25.0, 160.0]


def test_aggregated_monthly_dataset_works_with_row_count_proxy_warning():
    df = pd.DataFrame({
        "Month": ["2024-01", "2024-02", "2024-03"],
        "Category": ["A", "A", "B"],
        "Sales": [1000, 1200, 900],
        "Cost": [600, 900, 950],
    })
    roles = detect_columns(df)["roles"]
    assert roles["date_column"] == "Month"
    assert roles["category_column"] == "Category"
    prepared_df, prepared_roles = _run_core_smoke(df, roles, "What is underperforming?")
    kpis = calculate_kpis(prepared_df, prepared_roles)
    assert kpis["order_count_basis"] == "row_count_proxy_no_order_id"
    answer = answer_business_question("Give me a management summary", prepared_df, prepared_roles)
    assert any("aggregated" in lim.lower() or "records" in lim.lower() for lim in answer.get("limitations", []))


def test_transliterated_hindi_sales_columns_are_detected_and_work():
    df = pd.DataFrame({
        "tarikh": ["2024-01-01", "2024-01-02"],
        "vikray": [100, 200],
        "lagat": [80, 250],
        "grahak": ["Ram", "Shyam"],
        "utpad": ["A", "B"],
        "chhoot": ["10%", "20%"],
    })
    roles = detect_columns(df)["roles"]
    assert roles["date_column"] == "tarikh"
    assert roles["sales_column"] == "vikray"
    assert roles["cost_column"] == "lagat"
    assert roles["customer_column"] == "grahak"
    assert roles["product_column"] == "utpad"
    prepared_df, prepared_roles = _run_core_smoke(df, roles, "Which product is bleeding money?")
    assert prepared_roles["profit_column"] == "__Derived Profit"


def test_profit_synonyms_trigger_profit_analysis():
    df = pd.DataFrame({
        "Order Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Product": ["A", "B", "C"],
        "Sales": [100, 200, 300],
        "Cost": [80, 250, 180],
        "Discount": [0.1, 0.4, 0.0],
    })
    roles = detect_columns(df)["roles"]
    answer = answer_business_question("Which item is bleeding money and has bad contribution?", df, roles)
    assert answer["target_metric"] == "profit"
    assert "markdown_answer" in answer


def test_constant_numeric_columns_do_not_emit_correlation_warnings():
    import warnings
    import pandas as pd
    from src.column_detection import detect_columns
    from src.metric_relationships import analyze_metric_relationships
    from src.eda_engine import generate_eda_summary

    df = pd.DataFrame({
        "Order Date": pd.date_range("2024-01-01", periods=6),
        "Category": ["A", "A", "B", "B", "C", "C"],
        "Sub-Category": ["X", "X", "Y", "Y", "Z", "Z"],
        "Product Name": ["P1", "P1", "P2", "P2", "P3", "P3"],
        "Sales": [100, 100, 100, 100, 100, 100],
        "Discount": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
        "Profit": [-10, -20, 10, 20, 30, 40],
    })
    roles = detect_columns(df)["roles"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        analyze_metric_relationships(df, roles)
        generate_eda_summary(df, roles)
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []
