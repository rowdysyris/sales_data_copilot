import pandas as pd

from src.embedded_bi_dashboard import build_bi_dashboard_model, get_available_dashboard_pages, powerbi_embed_status


def _df():
    return pd.DataFrame({
        "Order ID": ["O1", "O2", "O3", "O4"],
        "Order Date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-02-01", "2024-02-05"]),
        "Customer": ["A", "A", "B", "C"],
        "Category": ["Furniture", "Furniture", "Technology", "Office Supplies"],
        "Product": ["Tables", "Chairs", "Phones", "Binders"],
        "Region": ["South", "South", "West", "East"],
        "Segment": ["Consumer", "Consumer", "Corporate", "Home Office"],
        "Sales": [1000, 500, 2000, 300],
        "Profit": [-200, 80, 600, -50],
        "Quantity": [4, 2, 3, 10],
        "Discount": [0.4, 0.1, 0.0, 0.3],
    })


def _roles():
    return {
        "order_id_column": "Order ID",
        "date_column": "Order Date",
        "customer_column": "Customer",
        "category_column": "Category",
        "product_column": "Product",
        "region_column": "Region",
        "segment_column": "Segment",
        "sales_column": "Sales",
        "profit_column": "Profit",
        "quantity_column": "Quantity",
        "discount_column": "Discount",
    }


def test_embedded_bi_model_has_dashboard_tables():
    model = build_bi_dashboard_model(_df(), _roles())
    assert model["summary"]["total_sales"] == 3800
    assert model["summary"]["total_profit"] == 430
    assert "category" in model["tables"]
    assert not model["tables"]["category"].empty
    assert "discount_band" in model["tables"]
    assert "Executive Overview" in model["pages"]
    assert "Profit Intelligence" in model["pages"]


def test_available_dashboard_pages_are_column_aware():
    pages = get_available_dashboard_pages(_df(), _roles())
    assert "Customer Intelligence" in pages
    assert "Regional Performance" in pages
    assert "Discount & Margin Control" in pages


def test_powerbi_embed_status_defaults_to_in_app(monkeypatch):
    monkeypatch.delenv("POWERBI_EMBED_URL", raising=False)
    status = powerbi_embed_status()
    assert status["manual_powerbi_desktop_required"] is False
    assert status["mode"] == "streamlit_in_app_bi_dashboard"

from src.embedded_bi_dashboard import prepare_bi_dataframe, apply_dashboard_filters


def test_interactive_dashboard_filters_reduce_records():
    work = prepare_bi_dataframe(_df(), _roles())
    filtered = apply_dashboard_filters(work, {"column_filters": {"Region": ["South"]}})
    assert len(filtered) == 2
    assert set(filtered["Region"]) == {"South"}


def test_interactive_dashboard_loss_only_filter():
    work = prepare_bi_dataframe(_df(), _roles())
    filtered = apply_dashboard_filters(work, {"loss_only": True})
    assert len(filtered) == 2
    assert filtered["BI Is Loss Order"].all()
