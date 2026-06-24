import numpy as np
import pandas as pd

from src.column_detection import detect_columns
from src.dynamic_question_engine import answer_dynamic_question


def _superstore_like_df(force_negative_texas: bool = False):
    n = 9994
    rng = np.random.default_rng(42 if not force_negative_texas else 43)
    years = np.tile(np.arange(2014, 2018), int(np.ceil(n / 4)))[:n]
    dates = pd.to_datetime([f"{y}-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i, y in enumerate(years)])
    states = rng.choice(["California", "New York", "Texas", "Ohio", "Florida"], n, p=[0.30, 0.20, 0.20, 0.15, 0.15])
    sales = rng.gamma(2, 200, n) + 50
    discount = rng.choice([0, 0.1, 0.2, 0.3, 0.4], n, p=[0.35, 0.25, 0.20, 0.12, 0.08])
    profit = sales * (0.18 - discount * 0.7) + rng.normal(0, 40, n)

    # Force the exact expected count for the percentage test.
    idx = np.argsort(profit)
    loss_idx = idx[:1871]
    good_idx = idx[1871:]
    profit[loss_idx] = -np.abs(profit[loss_idx]) - 10
    profit[good_idx] = np.abs(profit[good_idx]) + 10

    df = pd.DataFrame({
        "Order ID": [f"CA-{i}" for i in range(n)],
        "Order Date": dates.strftime("%m/%d/%Y"),
        "Ship Mode": rng.choice(["First Class", "Standard Class", "Second Class", "Same Day"], n, p=[0.15, 0.55, 0.25, 0.05]),
        "Customer Name": rng.choice([f"Customer {i}" for i in range(700)], n),
        "Segment": rng.choice(["Consumer", "Corporate", "Home Office"], n),
        "State": states,
        "Region": rng.choice(["West", "East", "Central", "South"], n),
        "Category": rng.choice(["Furniture", "Office Supplies", "Technology"], n),
        "Sub-Category": rng.choice(["Tables", "Phones", "Chairs", "Binders", "Storage"], n),
        "Product Name": [f"Product {i % 50}" for i in range(n)],
        "Sales": sales.round(2),
        "Profit": profit.round(2),
        "Discount": discount,
        "Quantity": rng.integers(1, 8, n),
    })
    if force_negative_texas:
        texas = df["State"].eq("Texas")
        df.loc[texas, "Profit"] = df.loc[texas, "Profit"] - 100
    return df


def _answer(question, df):
    roles = detect_columns(df)
    return answer_dynamic_question(question, df, roles)


def test_tier3_yoy_growth_uses_years_and_growth_percent():
    df = _superstore_like_df()
    ans = _answer("Show me year over year sales growth from 2014 to 2017", df)
    text = ans["markdown_answer"]
    assert ans["intent"] == "tier3_yoy_growth"
    assert "2014" in text and "2015" in text and "2016" in text and "2017" in text
    assert "sales_yoy_growth_percent" in text
    assert "$" in text and "₹" not in text
    assert ans["analysis_plan"]["source_rows_sent_to_ai"] == 0


def test_tier3_unprofitable_percentage_matches_expected_count():
    df = _superstore_like_df()
    ans = _answer("What percentage of our orders are unprofitable?", df)
    text = ans["markdown_answer"]
    assert ans["intent"] == "tier3_unprofitable_percentage"
    assert "18.72%" in text
    assert "1,871 out of 9,994" in text
    assert ans["analysis_plan"]["source_rows_sent_to_ai"] == 0


def test_tier3_state_exit_gives_business_recommendation_and_negative_state():
    df = _superstore_like_df(force_negative_texas=True)
    ans = _answer("Which state should we consider exiting based on profitability?", df)
    text = ans["markdown_answer"]
    assert ans["intent"] == "tier3_state_exit"
    assert "Texas" in text
    assert "Do not exit purely because profit is negative" in text
    assert "## Recommendation" in text
    assert "$" in text and "₹" not in text


def test_tier3_shipping_strategy_compares_first_and_standard_with_verdict():
    df = _superstore_like_df()
    ans = _answer("Is First Class shipping worth the cost compared to Standard?", df)
    text = ans["markdown_answer"]
    assert ans["intent"] == "tier3_shipping_strategy"
    assert "First Class" in text and "Standard Class" in text
    assert "## Headline" in text and "## Recommendation" in text
    assert "worth" in text.lower() or "verdict" in text.lower()
    assert "$" in text and "₹" not in text


def test_tier3_multi_condition_filter_and_consistency_detection():
    df = _superstore_like_df(force_negative_texas=True)
    ans = _answer("Show me unprofitable orders in Texas", df)
    text = ans["markdown_answer"]
    assert ans["intent"] == "tier3_multi_condition_filter"
    assert "State = Texas" in text
    assert "Profit < 0" in text
    assert "Total loss" in text

    cons = _answer("Which products are consistently losing money?", df)
    assert cons["intent"] == "tier3_consistency_loss"
    assert "loss order % above 60%" in cons["markdown_answer"]


def test_tier3_discount_pushback_on_bad_question():
    df = _superstore_like_df()
    ans = _answer("Is discount hurting us?", df)
    text = ans["markdown_answer"]
    assert ans["intent"] == "tier3_discount_business_judgment"
    assert "## Question Pushback" in text
    assert "## Better Question" in text
    assert "which categories" in text.lower()
