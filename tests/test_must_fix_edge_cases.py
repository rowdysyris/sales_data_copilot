from io import BytesIO
import pandas as pd

from src.column_detection import detect_columns
from src.ingestion import load_dataset
from src.forecasting import forecast_sales_profit
from src.anomaly_detection import detect_business_anomalies
from src.profit_intelligence import analyze_profit_intelligence
from src.root_cause import find_loss_drivers
from src.region_intelligence import analyze_region_segment_performance
from src.hidden_insights import generate_hidden_insights
from src.question_answering import answer_business_question
from src.powerbi_exporter import build_powerbi_star_schema
from src.kpi_engine import performance_by_category


def test_numeric_sales_is_not_detected_as_date_and_time_modules_do_not_crash():
    df = pd.DataFrame({"Revenue": [1000, 2000, 1500], "Product Category": ["A", "B", "A"]})
    roles = detect_columns(df)["roles"]
    assert roles.get("sales_column") == "Revenue"
    assert "date_column" not in roles

    forecast = forecast_sales_profit(df, roles)
    assert forecast["forecast"].empty
    anomalies = detect_business_anomalies(df, roles)
    assert isinstance(anomalies["anomalies"], list)


def test_subcategory_is_not_misclassified_as_category():
    df = pd.DataFrame({"Subcategory": ["Tables", "Binders"], "Sales": [100, 200], "Profit": [-10, 20]})
    roles = detect_columns(df)["roles"]
    assert roles.get("subcategory_column") == "Subcategory"
    assert roles.get("category_column") != "Subcategory"


def test_total_cost_is_not_detected_as_sales_when_total_amount_exists():
    df = pd.DataFrame({
        "Total Cost": [80, 120, 150],
        "Total Amount": [100, 200, 300],
        "Profit": [20, 80, 150],
        "Order Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
    })
    roles = detect_columns(df)["roles"]
    assert roles.get("cost_column") == "Total Cost"
    assert roles.get("sales_column") == "Total Amount"
    assert roles.get("date_column") == "Order Date"


def test_percentage_discount_strings_do_not_break_core_modules():
    df = pd.DataFrame({
        "Order ID": ["O1", "O2", "O3", "O4"],
        "Order Date": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15"],
        "Category": ["Furniture", "Furniture", "Technology", "Office Supplies"],
        "Sub-Category": ["Tables", "Chairs", "Phones", "Binders"],
        "Product Name": ["Table A", "Chair A", "Phone A", "Binder A"],
        "Region": ["South", "North", "West", "South"],
        "Segment": ["Consumer", "Corporate", "Consumer", "Home Office"],
        "Customer Name": ["A", "B", "A", "C"],
        "Sales": [1000, 500, 1200, 300],
        "Profit": [-300, 100, 50, -60],
        "Quantity": [2, 1, 1, 3],
        "Discount": ["40%", "10%", "5%", "50%"],
    })
    roles = detect_columns(df)["roles"]
    assert roles.get("discount_column") == "Discount"

    assert not performance_by_category(df, roles).empty
    pi = analyze_profit_intelligence(df, roles)
    assert pi["profit_summary"]["total_profit"] == -210
    rc = find_loss_drivers(df, roles)
    assert rc["total_loss_orders"] == 2
    region = analyze_region_segment_performance(df, roles)
    assert "region_summary" in region
    insights = generate_hidden_insights(df, roles)
    assert isinstance(insights, list)
    answer = answer_business_question("Is discount hurting profit?", df, roles)
    assert isinstance(answer, dict)
    anomalies = detect_business_anomalies(df, roles)
    assert isinstance(anomalies["anomalies"], list)


def test_duplicate_column_names_are_deduped_during_ingestion():
    raw = b"Sales,Profit,Profit\n100,20,21\n200,30,31\n"
    f = BytesIO(raw)
    f.name = "duplicate_columns.csv"
    df, meta = load_dataset(f)
    assert list(df.columns) == ["Sales", "Profit", "Profit__2"]
    assert any("Duplicate column" in warning for warning in meta["load_warnings"])
    roles = detect_columns(df)["roles"]
    assert roles.get("sales_column") == "Sales"
    assert roles.get("profit_column") == "Profit"


def test_powerbi_export_handles_tiny_customer_dataset():
    df = pd.DataFrame({
        "Order ID": ["O1", "O2"],
        "Order Date": ["2024-01-01", "2024-01-02"],
        "Customer Name": ["Only Customer", "Only Customer"],
        "Product Name": ["P1", "P2"],
        "Region": ["South", "South"],
        "Sales": [100, 200],
        "Profit": [10, -20],
        "Discount": ["0%", "40%"],
    })
    roles = detect_columns(df)["roles"]
    schema = build_powerbi_star_schema(df, roles)
    assert "DimCustomer" in schema
    assert "Customer Value Segment" in schema["DimCustomer"].columns
    assert not schema["FactSales"].empty
