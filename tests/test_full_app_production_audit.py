from __future__ import annotations

from io import BytesIO
import zipfile

import pandas as pd

from src.ingestion import load_dataset
from src.column_detection import detect_columns
from src.utils import prepare_analysis_dataset
from src.data_quality import analyze_data_quality
from src.kpi_engine import calculate_kpis
from src.eda_engine import generate_eda_summary
from src.profit_intelligence import analyze_profit_intelligence
from src.metric_relationships import analyze_metric_relationships
from src.customer_intelligence import analyze_customer_loyalty
from src.product_intelligence import analyze_product_portfolio
from src.region_intelligence import analyze_region_segment_performance
from src.hidden_insights import generate_hidden_insights
from src.forecasting import forecast_sales_profit
from src.anomaly_detection import detect_business_anomalies
from src.recommendation_engine import generate_recommendations, generate_executive_summary
from src.report_generator import generate_pdf_report, generate_excel_report
from src.powerbi_exporter import generate_powerbi_export_pack
from src.embedded_bi_dashboard import build_bi_dashboard_model, prepare_bi_dataframe, apply_dashboard_filters
from src.scenario_simulator import simulate_discount_reduction, simulate_price_increase, simulate_removing_loss_making_products
from src.missing_value_treatment import apply_missing_value_treatment
from src.llm_explainer import beautify_answer_with_llm
from src.question_answering import answer_business_question


class NamedBytesIO(BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def _superstore_bytes() -> bytes:
    with open("tests/fixtures/sample_superstore.csv", "rb") as f:
        return f.read()


def _load_superstore():
    df, metadata = load_dataset(NamedBytesIO(_superstore_bytes(), "Sample Superstore.csv"))
    roles = detect_columns(df)["roles"]
    prepared, prepared_roles = prepare_analysis_dataset(df, roles)
    return df, metadata, prepared, prepared_roles


def _context(df, roles):
    kpis = calculate_kpis(df, roles)
    dq = analyze_data_quality(df, roles)
    pi = analyze_profit_intelligence(df, roles)
    ci = analyze_customer_loyalty(df, roles)
    prod = analyze_product_portfolio(df, roles)
    reg = analyze_region_segment_performance(df, roles)
    hidden = generate_hidden_insights(df, roles)
    metric_relationships = analyze_metric_relationships(df, roles)
    forecast = forecast_sales_profit(df, roles)
    anomalies = detect_business_anomalies(df, roles)
    ctx = {
        "column_roles": roles,
        "kpis": kpis,
        "data_quality": dq,
        "profit_intelligence": pi,
        "customer_intelligence": ci,
        "product_intelligence": prod,
        "region_intelligence": reg,
        "hidden_insights": hidden,
        "metric_relationships": metric_relationships,
        "forecasting": forecast,
        "anomaly_detection": anomalies,
        "missing_value_treatment": roles.get("_missing_value_treatment", {}),
    }
    ctx["recommendations"] = generate_recommendations(df, roles, ctx)
    ctx["executive_summary"] = generate_executive_summary(ctx)
    return ctx


def test_section_1_upload_handler_behaviour():
    df, metadata = load_dataset(NamedBytesIO(_superstore_bytes(), "sample.csv"))
    assert df.shape == (9994, 21)
    assert metadata["file_type"] == "csv"

    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False)
    xlsx_df, xlsx_meta = load_dataset(NamedBytesIO(excel_buffer.getvalue(), "sample.xlsx"))
    assert xlsx_df.shape == (9994, 21)
    assert xlsx_meta["file_type"] == "excel"

    try:
        load_dataset(NamedBytesIO(b"not a dataset", "sample.txt"))
    except ValueError as exc:
        assert "Unsupported file type" in str(exc)
    else:
        raise AssertionError("Unsupported non-CSV/non-Excel file did not raise a clear error.")


def test_sections_2_3_7_profile_column_detection_and_kpis():
    df, metadata, prepared, roles = _load_superstore()
    assert metadata["row_count"] == 9994
    assert metadata["column_count"] == 21
    expected_columns = [
        "Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID", "Customer Name", "Segment", "Country", "City", "State", "Postal Code", "Region", "Product ID", "Category", "Sub-Category", "Product Name", "Sales", "Quantity", "Discount", "Profit"
    ]
    assert metadata["columns"] == expected_columns
    for col in ["Sales", "Profit", "Discount", "Quantity", "Postal Code"]:
        assert col in df.select_dtypes(include="number").columns
    for role, column in {
        "sales_column": "Sales",
        "profit_column": "Profit",
        "discount_column": "Discount",
        "quantity_column": "Quantity",
        "date_column": "Order Date",
        "ship_date_column": "Ship Date",
        "ship_mode_column": "Ship Mode",
        "customer_column": "Customer Name",
        "customer_id_column": "Customer ID",
        "segment_column": "Segment",
        "region_column": "Region",
        "category_column": "Category",
        "subcategory_column": "Sub-Category",
        "product_column": "Product Name",
        "state_column": "State",
        "city_column": "City",
        "order_id_column": "Order ID",
    }.items():
        assert roles.get(role) == column

    kpis = calculate_kpis(prepared, roles)
    assert round(kpis["total_sales"], 2) == 2297200.86
    assert round(kpis["total_profit"], 2) == 286397.02
    assert round(kpis["profit_margin_percent"], 2) == 12.47
    assert kpis["total_orders"] == 9994
    assert kpis["unique_orders"] == 5009
    assert kpis["unique_customers"] == 793
    assert round(kpis["average_order_value"], 2) == 229.86
    assert round(kpis["median_order_value"], 2) == 54.49
    assert round(kpis["average_discount"], 2) == 15.62
    assert kpis["loss_making_orders_count"] == 1871
    assert round(kpis["loss_making_orders_percent"], 2) == 18.72
    assert round(kpis["total_loss_amount"], 2) == -156131.29


def test_section_3_alternate_column_names():
    df = pd.DataFrame({
        "Total Amount": [100, 200],
        "Total Cost": [60, 180],
        "Margin %": [40, 10],
        "Shipping Mode": ["Standard", "First"],
        "Subcategory": ["A", "B"],
        "Revenue": [100, 200],
        "Order Date": ["2024-01-01", "2024-01-02"],
    })
    roles = detect_columns(df)["roles"]
    assert roles["sales_column"] in {"Revenue", "Total Amount"}
    assert roles["cost_column"] == "Total Cost"
    assert roles["margin_percent_column"] == "Margin %"
    assert roles["ship_mode_column"] == "Shipping Mode"
    assert roles["subcategory_column"] == "Subcategory"


def test_sections_5_6_8_data_quality_missing_treatment_and_eda():
    df, _, prepared, roles = _load_superstore()
    dq = analyze_data_quality(prepared, roles)
    assert dq["basic_checks"]["missing_cells_total"] == 0
    assert dq["basic_checks"]["duplicate_rows_count"] == 0
    assert dq["business_signals"]
    assert "negative profit" not in " ".join(dq.get("warnings", [])).lower()

    missing = pd.DataFrame({
        "Sales": [100.0, None, 300.0],
        "Cost": [50.0, 80.0, 200.0],
        "Profit": [50.0, None, None],
        "Quantity": [2, 4, None],
        "Unit Price": [50.0, 20.0, 100.0],
        "Discount": [0.0, None, 0.2],
        "Category": ["A", None, "B"],
        "Order Date": ["2024-01-01", None, "2024-01-03"],
    })
    roles2 = detect_columns(missing)["roles"]
    cleaned, treated_roles = apply_missing_value_treatment(missing, roles2)
    meta = treated_roles["_missing_value_treatment"]
    assert meta["cleaned_dataset_used"] is True
    assert meta["treatment_log"]
    assert cleaned["Category"].astype(str).str.contains("Unknown").any()
    assert cleaned["Order Date"].isna().sum() == 1

    eda = generate_eda_summary(prepared, roles)
    assert eda["dataset_overview"]["row_count"] == 9994
    assert eda["numeric_summary"] is not None
    assert eda["categorical_summary"] is not None
    assert "correlation_matrix" in eda


def test_sections_9_to_12_all_39_questions_have_required_grounding():
    df, _, prepared, roles = _load_superstore()
    questions = [
        "How many rows and columns does this dataset have?",
        "What are all the columns in this dataset?",
        "What is the total sales revenue?",
        "What is the total profit?",
        "Are there any missing or null values in this dataset?",
        "What is the date range of orders in this dataset?",
        "How many unique customers are there?",
        "How many unique orders are there?",
        "What categories of products does this dataset cover?",
        "What customer segments are in this data?",
        "What regions are covered in this dataset?",
        "What shipping modes are available?",
        "What is the average order value?",
        "Are there any duplicate records in this dataset?",
        "Give me a quick data profile of this dataset",
        "Which category is the most profitable?",
        "Which region generates the highest profit?",
        "What is the profit margin by category?",
        "Which customer segment drives the most revenue?",
        "What are the top 5 best selling products by revenue?",
        "Which sub-category is losing us the most money?",
        "How does discount affect profit?",
        "Show me year over year sales growth from 2014 to 2017",
        "What percentage of our orders are unprofitable?",
        "Which products are we selling at a loss consistently?",
        "Which state should we consider exiting?",
        "Is First Class shipping worth it vs Standard Class?",
        "Is our discount strategy working?",
        "Should we give Sean Miller more discounts?",
        "How many non unique datasets are there?",
        "How are we doing?",
        "What is killing our profit?",
        "Tell me about discounts",
        "Our furniture business is doing well right?",
        "What about the west?",
        "Which is better — technology or furniture?",
        "Give me sales and also tell me what to fix",
        "Why are we losing money in some orders?",
        "Summarize everything and tell me the 3 most important things I should act on right now",
    ]
    required = [
        "9,994", "Row ID", "$2,297,200.86", "$286,397.02", "0 missing", "January 3 2014", "793", "5,009", "Furniture", "Consumer", "West", "Standard Class", "$229.86", "No duplicate", "Quick Dataset Profile", "Technology", "West", "17.40%", "Consumer", "Canon imageCLASS", "Tables", "40%+", "2014", "18.72%", "loss order", "Texas", "First Class", "misleading", "Sean Miller", "4,985", "$2.29M", "Tables", "15.62%", "2.49%", "West", "Technology", "$2,297,200.86", "40%+", "Action 1"
    ]
    for idx, (question, marker) in enumerate(zip(questions, required), start=1):
        answer = answer_business_question(question, prepared, roles)["markdown_answer"]
        assert "₹" not in answer, f"Q{idx} used rupee symbol"
        assert "Recommendation" in answer or "RECOMMENDATION" in answer, f"Q{idx} missing recommendation"
        assert marker in answer, f"Q{idx} missing expected marker {marker!r}\n{answer[:500]}"


def test_section_13_advanced_features_and_scenarios():
    _, _, prepared, roles = _load_superstore()
    pi = analyze_profit_intelligence(prepared, roles)
    assert not pi["profit_leakage_points"].empty
    assert "profit_by_discount_band" in pi["profit_relationships"]
    assert pi["all_loss_driver_tables"]

    mr = analyze_metric_relationships(prepared, roles)
    assert not mr["profit_by_discount_band"].empty
    assert "40%+" in set(mr["profit_by_discount_band"]["discount_band"].astype(str))

    ci = analyze_customer_loyalty(prepared, roles)
    assert not ci["top_loyal_customers"].empty
    assert not ci["unprofitable_customers"].empty

    prod = analyze_product_portfolio(prepared, roles)
    assert prod["product_summary"].get("total_products", 0) == 1850
    assert not prod["problem_products"].empty

    reg = analyze_region_segment_performance(prepared, roles)
    assert not reg["region_summary"].empty
    assert not reg["state_summary"].empty

    hidden = generate_hidden_insights(prepared, roles)
    assert hidden

    forecast = forecast_sales_profit(prepared, roles)
    assert "history" in forecast and "forecast" in forecast

    anomalies = detect_business_anomalies(prepared, roles)
    assert "anomalies" in anomalies

    assert simulate_discount_reduction(prepared, roles, 5)
    assert simulate_price_increase(prepared, roles, 5)
    assert simulate_removing_loss_making_products(prepared, roles)


def test_sections_14_15_dashboard_and_export_features(tmp_path):
    _, _, prepared, roles = _load_superstore()
    model = build_bi_dashboard_model(prepared, roles)
    for page in ["Executive Overview", "Profit Intelligence", "Product Portfolio", "Customer Intelligence", "Regional Performance", "Discount & Margin Control"]:
        assert page in model["pages"]
    bi_df = prepare_bi_dataframe(prepared, roles)
    filtered = apply_dashboard_filters(bi_df, {"column_filters": {"Category": ["Furniture"], "Region": ["West"]}, "loss_only": True})
    assert set(filtered["Category"].unique()) <= {"Furniture"}
    assert set(filtered["Region"].unique()) <= {"West"}
    assert (filtered["Profit"] < 0).all()

    ctx = _context(prepared, roles)
    pdf_bytes = generate_pdf_report(ctx)
    assert isinstance(pdf_bytes, (bytes, bytearray)) and len(pdf_bytes) > 1000
    excel_bytes = generate_excel_report(prepared, roles, ctx)
    xlsx = pd.ExcelFile(BytesIO(excel_bytes))
    expected_sheets = {"Executive Summary", "KPIs", "Recommendations", "Profit Intelligence", "Customer Intelligence", "Product Intelligence", "Region Intelligence", "Forecasting", "Anomalies", "Missing Value Treatment"}
    assert expected_sheets.issubset(set(xlsx.sheet_names))

    powerbi_bytes = generate_powerbi_export_pack(prepared, roles)
    with zipfile.ZipFile(BytesIO(powerbi_bytes)) as zf:
        names = set(zf.namelist())
    for name in ["FactSales.csv", "DimDate.csv", "DimCustomer.csv", "DimProduct.csv", "DimGeography.csv", "dax_measures.md", "theme_sales_copilot.json"]:
        assert name in names


def test_section_16_no_api_mode_and_llm_fallback(monkeypatch):
    _, _, prepared, roles = _load_superstore()
    monkeypatch.setenv("LLM_PROVIDER", "none")
    answer = answer_business_question("Which category is the most profitable?", prepared, roles)["markdown_answer"]
    assert "Technology" in answer
    polished, meta = beautify_answer_with_llm(answer, {"confidence": "High"})
    assert polished == answer
    assert meta["provider"] == "none"
    assert meta["used"] is False

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    polished2, meta2 = beautify_answer_with_llm(answer, {"confidence": "High"})
    assert polished2 == answer
    assert meta2["fallback_used"] is True
