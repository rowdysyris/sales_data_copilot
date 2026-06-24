import pandas as pd

from src.column_detection import detect_columns
from src.question_answering import answer_business_question


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
]

REQUIRED_SNIPPETS = {
    1: ["9,994", "21 columns"],
    2: ["Row ID", "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID", "Customer Name", "Segment", "Country", "City", "State", "Postal Code", "Region", "Product ID", "Category", "Sub-Category", "Product Name", "Sales", "Quantity", "Discount", "Profit"],
    3: ["$2,297,200.86"],
    4: ["$286,397.02"],
    5: ["No", "0 missing/null values"],
    6: ["January 3 2014", "December 30 2017"],
    7: ["793 unique customers"],
    8: ["5,009 unique orders"],
    9: ["Furniture", "Office Supplies", "Technology"],
    10: ["Consumer", "Corporate", "Home Office"],
    11: ["South", "West", "Central", "East"],
    12: ["Second Class", "Standard Class", "First Class", "Same Day"],
    13: ["$229.86", "$54.49"],
    14: ["No duplicate rows"],
    15: ["9,994", "21", "January 3 2014", "December 30 2017", "Furniture", "Office Supplies", "Technology", "South", "West", "Central", "East", "Consumer", "Corporate", "Home Office", "$2,297,200.86", "$286,397.02", "Missing Values | 0"],
    16: ["Technology", "$145,454.95", "Office Supplies", "Furniture"],
    17: ["West", "$108,418.45", "East", "South", "Central"],
    18: ["17.40%", "17.04%", "2.49%"],
    19: ["Consumer", "$1,161,401.34", "Corporate", "Home Office"],
    20: ["Canon imageCLASS 2200 Advanced Copier", "$61,599.82"],
    21: ["Tables", "-$17,725.48", "Bookcases", "Supplies"],
    22: ["40%+", "-$99,558.59"],
    23: ["2014", "2015", "2016", "2017", "sales_yoy_growth_percent"],
    24: ["18.72%", "1,871", "9,994"],
    25: ["loss order % above 60%", "total_orders", "loss_orders"],
    26: ["Texas", "negative total profit", "Recommendation"],
    27: ["First Class", "Standard Class", "$48,969.84", "$164,088.79"],
    28: ["discount", "-0.219", "40%+", "Better Question"],
    29: ["Sean Miller", "-7.91%", "discount", "Recommendation"],
}


def _answers():
    df = pd.read_csv("tests/fixtures/sample_superstore.csv", encoding="latin1")
    roles = detect_columns(df)
    return [answer_business_question(q, df, roles)["markdown_answer"] for q in QUESTIONS]


def test_superstore_29_questions_pass_exact_business_expectations():
    answers = _answers()
    assert len(answers) == 29
    for idx, answer in enumerate(answers, start=1):
        assert "₹" not in answer, f"Q{idx} used rupee symbol"
        assert "## Recommendation" in answer, f"Q{idx} missing recommendation section"
        assert answer.strip().split("## Recommendation")[-1].strip(), f"Q{idx} empty recommendation"
        for snippet in REQUIRED_SNIPPETS[idx]:
            assert snippet in answer, f"Q{idx} missing expected snippet: {snippet}\nANSWER:\n{answer}"
