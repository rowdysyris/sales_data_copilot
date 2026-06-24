import pandas as pd

from src.column_detection import detect_columns
from src.profit_intelligence import analyze_profit_intelligence
from src.recommendation_engine import generate_executive_summary
from src.kpi_engine import calculate_kpis
from src.question_answering import answer_business_question
from src.entity_resolver import resolve_entities


def _mixed_loss_df():
    rows = []
    # Biggest loss is Tables, caused by high discounts and South region.
    for i in range(15):
        rows.append({
            "Order ID": f"T-{i}",
            "Order Date": f"2024-01-{(i % 28) + 1:02d}",
            "Category": "Furniture",
            "Sub-Category": "Tables",
            "Product Name": "Executive Table",
            "Region": "South",
            "Segment": "Consumer",
            "Customer Name": f"Customer {i % 4}",
            "Sales": 1000,
            "Quantity": 2,
            "Discount": 0.45,
            "Profit": -300,
        })
    # Phones is a separate weak-margin watch area; it should not be a root cause of Tables.
    for i in range(10):
        rows.append({
            "Order ID": f"P-{i}",
            "Order Date": f"2024-02-{(i % 28) + 1:02d}",
            "Category": "Technology",
            "Sub-Category": "Phones",
            "Product Name": "Mobile Phone",
            "Region": "West",
            "Segment": "Corporate",
            "Customer Name": f"Phone Customer {i}",
            "Sales": 4000,
            "Quantity": 1,
            "Discount": 0.05,
            "Profit": 100,
        })
    # Another loss area to prove the model scans beyond Tables/Furniture.
    for i in range(6):
        rows.append({
            "Order ID": f"B-{i}",
            "Order Date": f"2024-03-{(i % 28) + 1:02d}",
            "Category": "Office Supplies",
            "Sub-Category": "Binders",
            "Product Name": "Heavy Binder",
            "Region": "East",
            "Segment": "Home Office",
            "Customer Name": f"Binder Customer {i}",
            "Sales": 500,
            "Quantity": 3,
            "Discount": 0.35,
            "Profit": -120,
        })
    return pd.DataFrame(rows)


def test_global_root_cause_does_not_mix_unrelated_subcategories_as_same_causal_chain():
    df = _mixed_loss_df()
    roles = detect_columns(df)["roles"]
    pi = analyze_profit_intelligence(df, roles)
    summary = generate_executive_summary({
        "kpis": calculate_kpis(df, roles),
        "profit_intelligence": pi,
        "metric_relationships": pi.get("profit_relationships", {}),
        "recommendations": [],
    })

    assert "Tables" in summary["key_problem"]
    assert "Phones" not in summary["root_cause"]
    assert "Within this same area" in summary["root_cause"] or "discount band" in summary["root_cause"]


def test_profit_intelligence_scans_all_loss_making_subcategories_not_only_tables():
    df = _mixed_loss_df()
    roles = detect_columns(df)["roles"]
    pi = analyze_profit_intelligence(df, roles)

    other = pi.get("other_loss_areas")
    assert isinstance(other, pd.DataFrame)
    assert not other.empty
    assert "Binders" in set(other["Area"].astype(str))

    subcat_table = pi.get("all_loss_driver_tables", {}).get("subcategories")
    assert isinstance(subcat_table, pd.DataFrame)
    assert {"Tables", "Binders"}.issubset(set(subcat_table["dimension_value"].astype(str)))


def test_entity_specific_tables_question_stays_inside_tables_context():
    df = _mixed_loss_df()
    roles = detect_columns(df)["roles"]
    entities = resolve_entities("Why are Tables losing money?", df, roles)
    pi = analyze_profit_intelligence(df, roles, entities)
    root_text = " ".join(pi.get("root_cause_summary", []))

    assert "Tables" in root_text
    assert "Phones" not in root_text
    assert "Within this same area" in root_text


def test_qa_for_tables_does_not_report_phones_as_root_cause():
    df = _mixed_loss_df()
    roles = detect_columns(df)["roles"]
    ans = answer_business_question("Why are Tables losing money?", df, roles)
    text = ans["markdown_answer"]

    root_section = text.split("## Root Cause", 1)[1].split("## Recommended Action", 1)[0]
    assert "Tables" in text
    assert "Phones" not in root_section
