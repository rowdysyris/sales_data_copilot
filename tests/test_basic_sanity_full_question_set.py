from __future__ import annotations

import pandas as pd

from src.column_detection import detect_columns
from src.missing_value_treatment import apply_missing_value_treatment
from src.question_answering import answer_business_question


QUESTIONS_EXPECTED_INTENTS = [
    ("How many rows and columns does this dataset have?", "dataset_shape"),
    ("What are all the columns in this dataset?", "dataset_columns"),
    ("What is the total sales revenue?", "total_sales_check"),
    ("What is the total profit?", "total_profit_check"),
    ("Are there any missing or null values in this dataset?", "missing_values_check"),
    ("What is the date range of orders in this dataset?", "order_date_range_check"),
    ("How many unique customers are there?", "unique_customers_check"),
    ("How many unique orders are there?", "unique_orders_check"),
    ("What categories of products does this dataset cover?", "product_categories_check"),
    ("What customer segments are in this data?", "customer_segments_check"),
    ("What regions are covered in this dataset?", "regions_covered_check"),
    ("What shipping modes are available?", "shipping_modes_check"),
    ("What is the average order value?", "average_order_value_check"),
    ("Are there any duplicate records in this dataset?", "duplicate_rows_check"),
    ("Give me a quick data profile and overview of this dataset", "dataset_profile"),
]


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Order ID": ["O-1", "O-2", "O-2"],
            "Order Date": ["01/02/2024", "05/02/2024", "05/02/2024"],
            "Customer ID": ["C-1", "C-2", "C-2"],
            "Customer Name": ["A", "B", "B"],
            "Segment": ["Consumer", "Corporate", "Corporate"],
            "Region": ["South", "West", "West"],
            "Category": ["Furniture", "Technology", "Technology"],
            "Product Name": ["Chair", "Phone", "Phone"],
            "Ship Mode": ["First Class", "Standard Class", "Standard Class"],
            "Sales": [100.0, 200.0, 200.0],
            "Profit": [10.0, 20.0, 20.0],
            "Quantity": [1, 2, 2],
            "Discount": [0.0, 0.1, 0.1],
        }
    )


def test_basic_sanity_question_set_routes_to_exact_handlers():
    df = _sample_df()
    roles = detect_columns(df)["roles"]
    cleaned_df, _ = apply_missing_value_treatment(df, roles)
    for question, expected_intent in QUESTIONS_EXPECTED_INTENTS:
        answer = answer_business_question(question, cleaned_df, roles)
        intent = answer.get("intent") or answer.get("detected_intent")
        assert answer.get("handled") is True, question
        assert intent == expected_intent, question
        text = answer.get("markdown_answer", "")
        assert "Profit leakage" not in text, question
        assert "Worst category" not in text, question


def test_missing_values_clean_file_answer_is_no():
    df = _sample_df()
    roles = detect_columns(df)["roles"]
    cleaned_df, _ = apply_missing_value_treatment(df, roles)
    answer = answer_business_question("Are there any missing or null values in this dataset?", cleaned_df, roles)
    assert "**No.**" in answer["markdown_answer"]
    assert "0 missing/null values" in answer["markdown_answer"]


def test_duplicate_records_do_not_route_to_row_count():
    df = _sample_df()
    roles = detect_columns(df)["roles"]
    cleaned_df, _ = apply_missing_value_treatment(df, roles)
    answer = answer_business_question("Are there any duplicate records in this dataset?", cleaned_df, roles)
    assert answer["intent"] == "duplicate_rows_check"
    assert "duplicate rows" in answer["markdown_answer"].lower()
    assert "dataset size" not in answer["markdown_answer"].lower()
