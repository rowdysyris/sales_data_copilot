from __future__ import annotations

import pandas as pd

from src.column_detection import detect_columns
from src.question_answering import answer_business_question
from src.utils import prepare_analysis_dataset


def _broad_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Order ID": ["O1", "O2", "O3", "O4"],
            "Order Date": ["01/02/2024", "02/02/2024", "03/02/2024", "04/02/2024"],
            "Ship Date": ["02/02/2024", "03/02/2024", "04/02/2024", "05/02/2024"],
            "Ship Mode": ["First Class", "Standard Class", "Second Class", "Standard Class"],
            "Customer ID": ["C1", "C2", "C2", "C3"],
            "Customer Name": ["Ann", "Bob", "Bob", "Chin"],
            "Segment": ["Consumer", "Corporate", "Corporate", "Home Office"],
            "Country": ["India", "India", "India", "India"],
            "City": ["Bhopal", "Indore", "Indore", "Pune"],
            "State": ["MP", "MP", "MP", "MH"],
            "Region": ["Central", "West", "West", "South"],
            "Category": ["Furniture", "Technology", "Office Supplies", "Furniture"],
            "Sub-Category": ["Chairs", "Phones", "Binders", "Tables"],
            "Product Name": ["Chair A", "Phone B", "Binder C", "Table D"],
            "Sales": [100.0, 200.0, 300.0, 400.0],
            "Profit": [10.0, 20.0, -5.0, 30.0],
            "Quantity": [1, 2, 3, 4],
            "Discount": [0, 0.1, 0.2, 0.3],
        }
    )


BASIC_QUESTIONS = {
    "Can you read this CSV?": "csv_parse_check",
    "Can you profile this CSV?": "dataset_profile",
    "What is dataset shape?": "dataset_shape",
    "rows cols": "dataset_shape",
    "Number of rows?": "dataset_shape",
    "Number of columns?": "dataset_shape",
    "list all columns": "dataset_columns",
    "show schema": "dataset_schema",
    "show dtypes": "dataset_schema",
    "what are numeric columns": "numeric_columns_check",
    "what are categorical columns": "categorical_columns_check",
    "null detection": "missing_values_check",
    "missing count by column": "missing_values_check",
    "Do we have blanks?": "missing_values_check",
    "are there duplicate records": "duplicate_rows_check",
    "duplicate row count": "duplicate_rows_check",
    "basic aggregation": "basic_aggregation_check",
    "sum of sales": "total_sales_check",
    "what is revenue sum": "total_sales_check",
    "sum profit": "total_profit_check",
    "how much profit": "total_profit_check",
    "total quantity": "total_quantity_check",
    "average discount": "average_discount_check",
    "minimum sales": "minimum_sales_check",
    "maximum sales": "maximum_sales_check",
    "average sales": "average_sales_check",
    "how many unique customer ids": "unique_customers_check",
    "count distinct customers": "unique_customers_check",
    "how many unique products": "product_values_check",
    "how many product categories": "product_categories_check",
    "what sub categories are present": "subcategory_values_check",
    "what cities are present": "city_values_check",
    "what states are covered": "state_values_check",
    "what countries are covered": "country_values_check",
    "what ship modes are available": "shipping_modes_check",
    "date range": "order_date_range_check",
    "min and max order date": "order_date_range_check",
    "first order date and last order date": "order_date_range_check",
    "average order value": "average_order_value_check",
    "how many unique orders": "unique_orders_check",
    "show first 5 rows": "dataset_preview",
    "preview dataset": "dataset_preview",
    "sample rows": "dataset_preview",
}


def test_broad_basic_sanity_questions_never_fall_back_to_profit_diagnostics():
    df = _broad_df()
    roles = detect_columns(df)["roles"]
    prepared_df, prepared_roles = prepare_analysis_dataset(df, roles)
    for question, expected_intent in BASIC_QUESTIONS.items():
        answer = answer_business_question(question, prepared_df, prepared_roles)
        text = answer.get("deterministic_answer") or answer.get("markdown_answer", "")
        assert answer.get("intent") == expected_intent, question
        assert "I analyzed the dataset using the detected business columns" not in text, question
        assert "Profit leakage" not in text, question
        assert "Worst category" not in text, question


def test_sanity_questions_can_report_original_metadata_when_app_passes_raw_profile():
    df = pd.DataFrame({"Order ID": ["O1", "O2"], "Sales": [100, 200], "Cost": [60, 100]})
    roles = detect_columns(df)["roles"]
    prepared_df, prepared_roles = prepare_analysis_dataset(df, roles)
    prepared_roles["_raw_profile"] = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": [str(c) for c in df.columns],
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": {str(k): int(v) for k, v in df.isna().sum().to_dict().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
    }
    cols_answer = answer_business_question("list all columns", prepared_df, prepared_roles)
    assert cols_answer["intent"] == "dataset_columns"
    assert "__Derived Profit" not in cols_answer["deterministic_answer"]
    assert "3 columns" in cols_answer["deterministic_answer"]
