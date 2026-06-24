import pandas as pd

from src.column_detection import detect_columns
from src.utils import prepare_analysis_dataset
from src.data_quality import analyze_data_quality
from src.kpi_engine import calculate_kpis
from src.question_answering import answer_business_question
from src.report_generator import generate_excel_report, generate_pdf_report


def test_business_aware_missing_value_treatment_log_and_kpis():
    df = pd.DataFrame({
        "Order Date": ["2024-01-01", None, "2024-01-03"],
        "Order ID": ["O1", "O2", "O3"],
        "Customer Name": ["A", "B", None],
        "Category": ["Furniture", None, "Technology"],
        "Product Name": ["Chair", "Table", "Phone"],
        "Sales": [100.0, None, 300.0],
        "Quantity": [2, 5, None],
        "Unit Price": [50, 20, 100],
        "Cost": [60, 80, None],
        "Profit": [40, None, 90],
        "Discount": [None, "10%", 0],
    })
    roles = detect_columns(df)["roles"]
    assert roles["unit_price_column"] == "Unit Price"

    cleaned, cleaned_roles = prepare_analysis_dataset(df, roles)
    meta = cleaned_roles["_missing_value_treatment"]

    assert meta["cleaned_dataset_used"] is True
    assert meta["original_missing_cells_total"] == 8
    assert "Missing Value Treatment Log" not in meta["summary_text"]  # summary remains compact
    assert cleaned.loc[1, "Sales"] == 100.0  # Quantity * Unit Price
    assert cleaned.loc[1, "Profit"] == 20.0  # Sales - Cost after Sales derivation
    assert cleaned.loc[2, "Quantity"] == 3.0  # Sales / Unit Price
    assert cleaned.loc[0, "Discount"] == 0.0
    assert cleaned.loc[1, "Category"] == "Unknown Category"
    assert cleaned.loc[2, "Customer Name"] == "Unknown Customer"

    kpis = calculate_kpis(cleaned, cleaned_roles)
    assert kpis["total_sales"] == 500.0
    assert kpis["total_profit"] == 150.0

    dq = analyze_data_quality(cleaned, cleaned_roles)
    assert dq["basic_checks"]["missing_cells_total"] == 8
    assert dq["basic_checks"]["cleaned_missing_cells_total"] == 1
    assert dq["missing_value_treatment"]["cleaned_dataset_used"] is True
    assert any("cleaned dataset" in w.lower() for w in dq["warnings"])


def test_core_sales_profit_are_not_mean_filled_when_no_safe_formula_exists():
    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Product": ["A", "B", "C"],
        "Sales": [100.0, None, 300.0],
        "Profit": [20.0, None, -50.0],
    })
    roles = detect_columns(df)["roles"]
    cleaned, cleaned_roles = prepare_analysis_dataset(df, roles)
    meta = cleaned_roles["_missing_value_treatment"]

    assert pd.isna(cleaned.loc[1, "Sales"])
    assert pd.isna(cleaned.loc[1, "Profit"])
    log = pd.DataFrame(meta["treatment_log"])
    sales_row = log[log["column"] == "Sales"].iloc[0]
    profit_row = log[log["column"] == "Profit"].iloc[0]
    assert "core KPI" in sales_row["treatment"]
    assert sales_row["values_still_missing"] == 1
    assert profit_row["values_still_missing"] == 1


def test_qa_and_reports_include_cleaned_dataset_warning():
    df = pd.DataFrame({
        "Date": ["2024-01-01", None, "2024-01-03"],
        "Product": ["A", "B", "C"],
        "Sales": [100, 200, 300],
        "Cost": [80, None, 250],
        "Discount": [0, None, "20%"],
    })
    roles = detect_columns(df)["roles"]
    cleaned, cleaned_roles = prepare_analysis_dataset(df, roles)
    ans = answer_business_question("Where is margin leakage?", cleaned, cleaned_roles)

    assert "Data Reliability Note" in ans["markdown_answer"]
    assert "cleaned dataset" in ans["markdown_answer"].lower()

    ctx = ans["context_results"]
    ctx["data_quality"] = analyze_data_quality(cleaned, cleaned_roles)
    excel = generate_excel_report(cleaned, cleaned_roles, ctx)
    pdf = generate_pdf_report(ctx)
    assert len(excel) > 5000
    assert len(pdf) > 5000
