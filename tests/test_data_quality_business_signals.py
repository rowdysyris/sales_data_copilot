import pandas as pd

from src.data_quality import analyze_data_quality


def test_negative_profit_is_business_signal_not_quality_warning():
    df = pd.DataFrame({
        "Sales": [100, 200, 300, 400],
        "Profit": [10, -20, 30, -40],
        "Quantity": [1, 2, 3, 4],
        "Order Date": pd.date_range("2024-01-01", periods=4),
    })
    roles = {
        "sales_column": "Sales",
        "profit_column": "Profit",
        "quantity_column": "Quantity",
        "date_column": "Order Date",
    }
    dq = analyze_data_quality(df, roles)
    assert dq["quality_score"] >= 95
    assert any("loss-making records" in s for s in dq["business_signals"])
    assert not any("Profit has" in w and "negative" in w for w in dq["warnings"])


def test_negative_sales_remains_data_quality_warning():
    df = pd.DataFrame({"Sales": [100, -200, 300], "Profit": [10, 20, 30]})
    roles = {"sales_column": "Sales", "profit_column": "Profit"}
    dq = analyze_data_quality(df, roles)
    assert any("Sales has" in w and "negative" in w for w in dq["warnings"])
    assert dq["quality_score"] < 100
