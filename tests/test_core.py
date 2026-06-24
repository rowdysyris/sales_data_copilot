import pandas as pd
from io import StringIO
from src.column_detection import detect_columns
from src.semantic_layer import build_semantic_model
from src.kpi_engine import calculate_kpis
from src.profit_intelligence import analyze_profit_intelligence
from src.customer_intelligence import analyze_customer_loyalty
from src.question_answering import answer_business_question
from src.entity_resolver import resolve_entities
from src.hidden_insights import generate_hidden_insights

def sample_df():
    return pd.DataFrame({
        "Order ID": ["1","2","3","4","5","6","7","8"],
        "Order Date": pd.to_datetime(["2024-01-01","2024-01-03","2024-02-01","2024-02-05","2024-03-01","2024-03-02","2024-03-07","2024-03-10"]),
        "Customer Name": ["A","A","A","B","C","C","D","D"],
        "Region": ["South","South","North","South","West","West","South","East"],
        "Category": ["Furniture","Furniture","Technology","Furniture","Office Supplies","Technology","Furniture","Technology"],
        "Sub-Category": ["Tables","Tables","Phones","Tables","Paper","Copiers","Chairs","Phones"],
        "Product Name": ["Table X","Table X","Phone A","Table Y","Paper A","Copier A","Chair A","Phone B"],
        "Sales": [1000,1200,2000,1500,200,3000,700,2500],
        "Quantity": [2,2,1,3,4,1,1,2],
        "Discount": [0.4,0.5,0.1,0.4,0,0.1,0.2,0.05],
        "Profit": [-300,-500,400,-600,60,900,80,500],
    })

def roles():
    return detect_columns(sample_df())["roles"]

def test_column_detection():
    r = roles()
    assert r["sales_column"] == "Sales"
    assert r["profit_column"] == "Profit"
    assert r["subcategory_column"] == "Sub-Category"

def test_semantic():
    s = build_semantic_model(sample_df(), roles())
    assert "profitability_analysis" in s["available_analyses"]
    assert s["dataset_type"] == "sales_transaction_dataset"

def test_kpis():
    k = calculate_kpis(sample_df(), roles())
    assert k["total_sales"] == 12100
    assert k["total_profit"] == 540
    assert k["loss_making_orders_count"] == 3

def test_profit_intelligence_mentions_loss():
    pi = analyze_profit_intelligence(sample_df(), roles())
    assert pi["profit_summary"]["total_profit"] == 540
    assert len(pi["root_cause_summary"]) >= 1

def test_customer_loyalty_not_empty():
    ci = analyze_customer_loyalty(sample_df(), roles())
    assert not ci["top_loyal_customers"].empty
    assert "loyalty_score" in ci["top_loyal_customers"].columns

def test_entity_resolver_tables():
    ents = resolve_entities("Why are tables losing money?", sample_df(), roles())
    assert any(e["matched_value"] == "Tables" for e in ents)

def test_question_profit_has_relationships():
    ans = answer_business_question("Why is profit low?", sample_df(), roles())
    md = ans["markdown_answer"]
    assert "Relationships / Drivers" in md
    assert "Total Profit" in md

def test_loyal_customers_question():
    ans = answer_business_question("Who are our most loyal customers?", sample_df(), roles())
    assert "most loyal customer" in ans["markdown_answer"].lower()

def test_hidden_insights():
    insights = generate_hidden_insights(sample_df(), roles())
    assert len(insights) >= 1
