import pandas as pd

from src.column_detection import detect_columns
from src.dynamic_question_engine import answer_dynamic_question


def _sample():
    df = pd.read_csv("sample_data/sample_sales_data.csv")
    roles = detect_columns(df)["roles"]
    return df, roles


def test_high_sales_low_profit_category_prioritizes_weak_margin_category():
    df, roles = _sample()
    res = answer_dynamic_question("Which category has high sales but low profit?", df, roles)
    assert res["handled"] is True
    assert res["intent"] == "tier2_high_sales_low_profit"
    assert "Furniture" in res["markdown_answer"]
    assert "Technology has high sales but relatively weak profit margin" not in res["markdown_answer"]


def test_compare_missing_value_reports_available_values_instead_of_sales_missing():
    df, roles = _sample()
    res = answer_dynamic_question("Compare Central and West regions", df, roles)
    assert res["handled"] is True
    assert res["intent"] == "dynamic_comparison_value_not_found"
    text = res["markdown_answer"]
    assert "fewer than two region values" in text
    assert "West" in text
    assert "Sales column was not detected" not in text


def test_consistently_losing_respects_requested_dimension():
    df, roles = _sample()
    res = answer_dynamic_question("Which region is consistently losing money?", df, roles)
    assert res["handled"] is True
    assert res["intent"] == "tier3_consistency_loss"
    assert "regions" in res["markdown_answer"]
    assert "products/groups" not in res["markdown_answer"]


def test_worst_month_margin_sorts_by_margin_ascending():
    df, roles = _sample()
    res = answer_dynamic_question("Which month has the worst profit margin?", df, roles)
    assert res["handled"] is True
    assert res["intent"] == "dynamic_trend"
    table = res["evidence_tables"]["dynamic_trend"]
    assert table.iloc[0]["profit_margin_percent"] == table["profit_margin_percent"].min()


def test_shipping_strategy_aov_is_currency_formatted_when_ship_mode_exists():
    df = pd.DataFrame({
        "Order ID": ["O1", "O2", "O3", "O4"],
        "Order Date": ["2017-01-01", "2017-01-02", "2017-01-03", "2017-01-04"],
        "Ship Mode": ["First Class", "Standard Class", "First Class", "Standard Class"],
        "Sales": [100, 200, 120, 250],
        "Profit": [5, 60, 4, 80],
        "Discount": [0.2, 0.0, 0.3, 0.0],
    })
    roles = detect_columns(df)["roles"]
    res = answer_dynamic_question("Is First Class shipping worth the cost compared to Standard?", df, roles)
    assert res["handled"] is True
    assert res["intent"] == "tier3_shipping_strategy"
    assert "AOV" in res["markdown_answer"]
    assert "$" in res["markdown_answer"]


def test_unprofitable_order_percentage_expected_superstore_style_number():
    rows = []
    for i in range(9994):
        rows.append({
            "Order ID": f"O{i}",
            "Order Date": f"2017-01-{(i % 28) + 1:02d}",
            "Sales": 100,
            "Profit": -1 if i < 1871 else 1,
        })
    df = pd.DataFrame(rows)
    roles = detect_columns(df)["roles"]
    res = answer_dynamic_question("What percentage of our orders are unprofitable?", df, roles)
    assert res["handled"] is True
    assert "18.72%" in res["markdown_answer"]
    assert "1,871 out of 9,994" in res["markdown_answer"]
