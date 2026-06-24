from __future__ import annotations

import pandas as pd

from src.anomaly_detection import detect_business_anomalies
from src.column_detection import detect_columns
from src.customer_intelligence import analyze_customer_loyalty
from src.data_quality import analyze_data_quality
from src.dataset_intelligence_orchestrator import build_dataset_intelligence
from src.forecasting import forecast_sales_profit
from src.hidden_insights import generate_hidden_insights
from src.kpi_engine import calculate_kpis
from src.metric_relationships import analyze_metric_relationships
from src.product_intelligence import analyze_product_portfolio
from src.profit_intelligence import analyze_profit_intelligence
from src.question_answering import answer_business_question
from src.recommendation_engine import generate_executive_summary, generate_recommendations
from src.region_intelligence import analyze_region_segment_performance
from src.semantic_layer import build_semantic_model
from src.utils import prepare_analysis_dataset


QUESTIONS = [
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


def _raw_question_roles(df: pd.DataFrame, roles: dict) -> dict:
    raw_roles = dict(roles)
    raw_roles["_raw_profile"] = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": [str(c) for c in df.columns],
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column": {str(k): int(v) for k, v in df.isna().sum().to_dict().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
    }
    return raw_roles


def _context(df: pd.DataFrame, roles: dict) -> tuple[pd.DataFrame, dict, dict]:
    df_analysis, prepared_roles = prepare_analysis_dataset(df, roles)
    ctx = {
        "semantic_model": build_semantic_model(df_analysis, prepared_roles),
        "missing_value_treatment": prepared_roles.get("_missing_value_treatment", {}),
        "column_roles": prepared_roles,
        "kpis": calculate_kpis(df_analysis, prepared_roles),
        "data_quality": analyze_data_quality(df_analysis, prepared_roles),
        "profit_intelligence": analyze_profit_intelligence(df_analysis, prepared_roles),
        "customer_intelligence": analyze_customer_loyalty(df_analysis, prepared_roles),
        "product_intelligence": analyze_product_portfolio(df_analysis, prepared_roles),
        "region_intelligence": analyze_region_segment_performance(df_analysis, prepared_roles),
        "hidden_insights": generate_hidden_insights(df_analysis, prepared_roles),
        "metric_relationships": analyze_metric_relationships(df_analysis, prepared_roles),
        "forecasting": forecast_sales_profit(df_analysis, prepared_roles),
        "anomaly_detection": detect_business_anomalies(df_analysis, prepared_roles),
    }
    ctx["recommendations"] = generate_recommendations(df_analysis, prepared_roles, ctx)
    ctx["executive_summary"] = generate_executive_summary(ctx)
    ctx["dataset_intelligence"] = build_dataset_intelligence(ctx, prepared_roles)
    return df_analysis, prepared_roles, ctx


def _answers() -> dict[int, str]:
    df = pd.read_csv("tests/fixtures/sample_superstore.csv", encoding="latin1")
    detected = detect_columns(df)["roles"]
    raw_roles = _raw_question_roles(df, detected)
    _, prepared_roles, ctx = _context(df, detected)
    raw_roles.update({k: v for k, v in prepared_roles.items() if str(k).startswith("_")})
    raw_roles["_raw_profile"] = _raw_question_roles(df, detected)["_raw_profile"]
    return {i: answer_business_question(q, df, raw_roles, ctx)["markdown_answer"] for i, q in enumerate(QUESTIONS, 1)}


def _has_all(text: str, *needles: str) -> bool:
    return all(n in text for n in needles)


def test_superstore_q1_to_q39_end_to_end_contract():
    answers = _answers()

    for idx, answer in answers.items():
        assert "₹" not in answer, f"Q{idx} used rupee currency"
        assert "## Recommendation" in answer or "## RECOMMENDATION" in answer, f"Q{idx} lacked a recommendation"

    assert _has_all(answers[1], "9,994 rows", "21 columns")
    assert all(col in answers[2] for col in ["Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID", "Customer Name", "Segment", "Country", "City", "State", "Postal Code", "Region", "Product ID", "Category", "Sub-Category", "Product Name", "Sales", "Quantity", "Discount", "Profit"])
    assert "$2,297,200.86" in answers[3]
    assert "$286,397.02" in answers[4]
    assert _has_all(answers[5], "No", "0 missing/null values")
    assert _has_all(answers[6], "January 3 2014", "December 30 2017")
    assert "793 unique customers" in answers[7]
    assert "5,009 unique orders" in answers[8]
    assert all(v in answers[9] for v in ["Furniture", "Office Supplies", "Technology"])
    assert all(v in answers[10] for v in ["Consumer", "Corporate", "Home Office"])
    assert all(v in answers[11] for v in ["South", "West", "Central", "East"])
    assert all(v in answers[12] for v in ["Second Class", "Standard Class", "First Class", "Same Day"])
    assert _has_all(answers[13], "$229.86", "$54.49")
    assert "No duplicate rows" in answers[14]
    assert all(v in answers[15] for v in ["9,994", "21", "January 3 2014", "December 30 2017", "$2,297,200.86", "$286,397.02", "Furniture", "West", "Consumer"])

    assert _has_all(answers[16], "Technology", "$145,454.95", "Office Supplies", "Furniture")
    assert _has_all(answers[17], "West", "$108,418.45", "East", "South", "Central")
    assert _has_all(answers[18], "17.40%", "17.04%", "2.49%")
    assert _has_all(answers[19], "Consumer", "$1,161,401.34", "Corporate", "Home Office")
    assert _has_all(answers[20], "Canon imageCLASS 2200 Advanced Copier", "$61,599.82")
    assert _has_all(answers[21], "Tables", "-$17,725.48", "Bookcases", "Supplies")
    assert _has_all(answers[22], "40%+", "-$99,558.59", "higher discount", "lower profit")

    assert _has_all(answers[23], "2014", "2015", "2016", "2017", "-2.83%", "29.47%", "20.36%")
    assert _has_all(answers[24], "18.72%", "1,871", "9,994")
    assert _has_all(answers[25], "loss order % above 60%", "multiple orders", "100.00%")
    assert _has_all(answers[26], "Texas", "-$25,729.36", "Recommendation")
    assert _has_all(answers[27], "First Class", "Standard Class", "$48,969.84", "$164,088.79")
    assert _has_all(answers[28], "Question Pushback", "Better Question", "40%+", "-$99,558.59")
    assert _has_all(answers[29], "Sean Miller", "-7.91%", "Do not give more discounts")

    assert _has_all(answers[30], "0 fully duplicate rows", "4,985", "5,009 unique orders")
    assert _has_all(answers[31], "$2,297,200.86", "$286,397.02", "12.47%", "18.72%", "Tables")
    assert _has_all(answers[32], "Tables", "-$17,725.48", "40%+", "-$99,558.59", "Cubify")
    assert _has_all(answers[33], "Average discount", "15.62%", "Discount-profit correlation", "Furniture")
    assert _has_all(answers[34], "Furniture", "2.49%", "lowest-margin", "Tables")
    assert _has_all(answers[35], "West", "$725,457.82", "$108,418.45", "14.94%", "Region comparison")
    assert _has_all(answers[36], "Technology", "Furniture", "17.40%", "2.49%", "Technology is better")
    assert _has_all(answers[37], "$2,297,200.86", "Tables", "40%+", "loss-making products")
    assert _has_all(answers[38], "1,871", "-$156,131.29", "40%+", "Tables", "Cubify")
    assert answers[39].count(". Stop/reduce discounts above 40%") == 1
    assert answers[39].count(". Review Tables sub-category strategy") == 1
    assert answers[39].count(". Double down on Technology and West") == 1
    assert _has_all(answers[39], "$2,297,200.86", "$286,397.02", "12.47%")
