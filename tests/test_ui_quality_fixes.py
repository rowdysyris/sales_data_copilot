import pandas as pd

from src.column_detection import detect_columns
from src.hidden_insights import generate_hidden_insights
from src.root_cause import find_loss_drivers


def manager_demo_df():
    return pd.DataFrame({
        "Order ID": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
        "Order Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01", "2024-02-03", "2024-03-01", "2024-03-03", "2024-04-01", "2024-04-03", "2024-05-01", "2024-05-03"]),
        "Customer Name": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
        "Region": ["South", "South", "West", "West", "North", "North", "East", "East", "South", "South"],
        "Segment": ["Consumer", "Consumer", "Corporate", "Corporate", "Home Office", "Home Office", "Consumer", "Consumer", "Corporate", "Corporate"],
        "Category": ["Furniture", "Furniture", "Office Supplies", "Office Supplies", "Technology", "Technology", "Furniture", "Furniture", "Office Supplies", "Office Supplies"],
        "Sub-Category": ["Tables", "Tables", "Binders", "Binders", "Machines", "Machines", "Chairs", "Chairs", "Storage", "Storage"],
        "Product Name": ["Table X", "Table X", "Binder A", "Binder A", "Machine Z", "Machine Z", "Chair A", "Chair A", "Storage A", "Storage A"],
        "Sales": [1000, 1100, 500, 600, 2500, 2600, 900, 950, 700, 750],
        "Quantity": [2, 2, 5, 6, 1, 1, 2, 2, 3, 3],
        "Discount": [0.45, 0.50, 0.30, 0.35, 0.40, 0.42, 0.10, 0.10, 0.20, 0.22],
        "Profit": [-400, -450, -300, -350, -900, -1000, 120, 140, -100, -120],
    })


def roles():
    return detect_columns(manager_demo_df())["roles"]


def test_hidden_insights_do_not_repeat_generic_cards():
    insights = generate_hidden_insights(manager_demo_df(), roles())
    titles = [i["title"] for i in insights]
    assert len(titles) == len(set(titles))
    assert not (titles.count("Profit leakage detected") > 1)
    assert any("Profit leakage:" in title for title in titles)
    for insight in insights:
        assert insight.get("evidence")


def test_biggest_loss_combinations_are_manager_clean():
    loss = find_loss_drivers(manager_demo_df(), roles())
    combos = loss["biggest_loss_combinations"]
    assert not combos.empty
    assert "Combination Type" in combos.columns
    manager_columns = ["Region", "Segment", "Product Name"]
    existing = [c for c in manager_columns if c in combos.columns]
    assert existing
    for col in existing:
        assert combos[col].isna().sum() == 0
        assert not combos[col].astype(str).str.lower().eq("none").any()
    assert any(combos[col].astype(str).str.startswith("All ").any() for col in existing)
