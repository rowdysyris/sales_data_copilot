import pandas as pd

from src.column_detection import detect_columns
from src.question_answering import answer_business_question


def _sanity_df():
    return pd.DataFrame({
        "Order ID": ["O1", "O2", "O3"],
        "Order Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Product Name": ["A", "B", "C"],
        "Category": ["Cat1", "Cat2", "Cat1"],
        "Region": ["South", "West", "South"],
        "Sales": [100.0, 200.0, 300.0],
        "Profit": [10.0, -5.0, 40.0],
        "Discount": [0.0, 10.0, 20.0],
    })


def _roles(df):
    return detect_columns(df)["roles"]


def test_shape_question_answers_exact_rows_and_columns_not_generic_profit():
    df = _sanity_df()
    ans = answer_business_question("How many rows and columns does this dataset have?", df, _roles(df))
    text = ans["deterministic_answer"]
    assert ans["intent"] == "dataset_shape"
    assert "3 rows" in text
    assert "8 columns" in text
    assert "Profit leakage" not in text


def test_columns_question_lists_uploaded_columns():
    df = _sanity_df()
    ans = answer_business_question("What columns are in this dataset?", df, _roles(df))
    text = ans["deterministic_answer"]
    assert ans["intent"] == "dataset_columns"
    assert "Order ID" in text
    assert "Product Name" in text
    assert "Sales" in text
    assert "Profit" in text


def test_total_sales_and_profit_questions_return_exact_aggregation():
    df = _sanity_df()
    roles = _roles(df)
    sales_ans = answer_business_question("What is the total sales revenue?", df, roles)
    profit_ans = answer_business_question("What is the total profit?", df, roles)
    assert sales_ans["intent"] == "total_sales_check"
    assert "$600.00" in sales_ans["deterministic_answer"]
    assert profit_ans["intent"] == "total_profit_check"
    assert "$45.00" in profit_ans["deterministic_answer"]


def test_missing_values_question_says_no_for_clean_file():
    df = _sanity_df()
    ans = answer_business_question("Are there any missing values in this dataset?", df, _roles(df))
    text = ans["deterministic_answer"]
    assert ans["intent"] == "missing_values_check"
    assert "No." in text
    assert "0 missing/null values" in text


def test_missing_values_question_reports_columns_for_dirty_file():
    df = _sanity_df()
    df.loc[1, "Sales"] = None
    df.loc[2, "Region"] = None
    ans = answer_business_question("null detection", df, _roles(df))
    text = ans["deterministic_answer"]
    assert ans["intent"] == "missing_values_check"
    assert "Yes." in text
    assert "2 missing/null values" in text
    table = ans["evidence_tables"]["missing_values_by_column"]
    assert set(table["Column"]) == {"Sales", "Region"}


def test_column_detection_and_basic_aggregation_sanity_questions():
    df = _sanity_df()
    roles = _roles(df)
    col_ans = answer_business_question("Column detection", df, roles)
    agg_ans = answer_business_question("basic aggregation", df, roles)
    assert col_ans["intent"] == "column_detection_check"
    assert "Sales | Sales" in col_ans["deterministic_answer"]
    assert "Profit | Profit" in col_ans["deterministic_answer"]
    assert agg_ans["intent"] == "basic_aggregation_check"
    assert "Total Sales/Revenue" in agg_ans["deterministic_answer"]
    assert "$600.00" in agg_ans["deterministic_answer"]


def test_basic_sanity_questions_use_original_uploaded_metadata_even_if_cleaning_derives_columns():
    df = pd.DataFrame({
        "Order ID": ["O1", "O2"],
        "Order Date": ["2024-01-01", "2024-01-02"],
        "Sales": [100.0, 200.0],
        "Cost": [70.0, 120.0],
    })
    roles = _roles(df)
    shape_ans = answer_business_question("How many rows and columns does this dataset have?", df, roles)
    assert "2 rows" in shape_ans["deterministic_answer"]
    assert "4 columns" in shape_ans["deterministic_answer"]
    cols_ans = answer_business_question("What columns are in this dataset?", df, roles)
    assert "__Derived Profit" not in cols_ans["deterministic_answer"]
