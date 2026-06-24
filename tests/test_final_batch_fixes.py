import pandas as pd

from src.column_detection import detect_columns
from src.dynamic_question_engine import answer_dynamic_question
from src.utils import ensure_datetime, prepare_analysis_dataset
from src.dataset_intelligence_orchestrator import build_dataset_intelligence
from src.kpi_engine import calculate_kpis, performance_by_category
from src.metric_relationships import analyze_metric_relationships


def test_discount_amount_question_routes_to_discount_not_sales():
    df = pd.DataFrame({
        "Category": ["A", "A", "B", "B"],
        "Sales": [1000, 1200, 900, 800],
        "Profit": [100, 80, 50, 40],
        "Discount": [5, 10, 40, 45],
    })
    roles = detect_columns(df)["roles"]
    ans = answer_dynamic_question("which category has highest discount amount", df, roles)
    assert ans["handled"] is True
    assert ans["analysis_plan"]["metric"] == "discount"
    assert "average discount" in ans["markdown_answer"].lower()


def test_explicit_churn_uses_unique_customer_rate():
    df = pd.DataFrame({
        "Customer Name": ["A", "A", "A", "B", "C"],
        "Order Date": ["01/01/2024", "02/01/2024", "03/01/2024", "04/01/2024", "05/01/2024"],
        "Status": ["Active", "Active", "Churned", "Active", "Active"],
        "Sales": [100, 100, 100, 200, 300],
        "Profit": [10, 10, 10, 20, 30],
    })
    roles = detect_columns(df)["roles"]
    ans = answer_dynamic_question("how should I control churn", df, roles)
    assert ans["handled"] is True
    assert "Unique-customer churn" in ans["markdown_answer"]
    assert "33.33%" in ans["markdown_answer"]
    table = ans["evidence_tables"].get("customer_level_churn_status")
    assert table is not None and len(table) == 3


def test_dayfirst_date_parser_handles_indian_dates():
    dates = ensure_datetime(pd.Series(["31/01/2024", "15/02/2024", "01/03/2024"]))
    assert dates.iloc[0].month == 1
    assert dates.iloc[1].month == 2


def test_aggregate_dataset_keeps_record_count_label_and_unresolved_flags():
    df = pd.DataFrame({
        "Month": ["2024-01", "2024-02", "2024-03"],
        "Category": ["A", "A", "B"],
        "Sales": [1000, None, 900],
        "Cost": [600, 900, 950],
    })
    roles = detect_columns(df)["roles"]
    prepared, prepared_roles = prepare_analysis_dataset(df, roles)
    kpis = calculate_kpis(prepared, prepared_roles)
    assert kpis["order_count_basis"] == "row_count_proxy_no_order_id"
    assert kpis["count_label"] in {"Records", "Rows"}
    perf = performance_by_category(prepared, prepared_roles)
    assert "unresolved_sales_rows" in perf.columns
    assert "order_count_basis" in perf.columns


def test_orchestrator_summarizes_cached_context():
    df = pd.DataFrame({"Category": ["A", "B"], "Sales": [100, 200], "Profit": [10, -5], "Discount": [5, 40]})
    roles = detect_columns(df)["roles"]
    prepared, prepared_roles = prepare_analysis_dataset(df, roles)
    ctx = {
        "kpis": calculate_kpis(prepared, prepared_roles),
        "metric_relationships": analyze_metric_relationships(prepared, prepared_roles),
        "missing_value_treatment": prepared_roles.get("_missing_value_treatment", {}),
        "hidden_insights": [],
    }
    intel = build_dataset_intelligence(ctx, prepared_roles)
    assert "kpis" in intel["available_modules"]
    assert "metric_relationships" in intel["available_modules"]
    assert "detected_roles" in intel
