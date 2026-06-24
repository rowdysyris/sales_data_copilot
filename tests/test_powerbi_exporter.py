import zipfile
from io import BytesIO

from src.column_detection import detect_columns
from src.powerbi_exporter import build_powerbi_star_schema, generate_powerbi_export_pack, powerbi_pack_summary
from tests.test_core import sample_df


def test_powerbi_star_schema_contains_required_tables():
    df = sample_df()
    roles = detect_columns(df)["roles"]
    tables = build_powerbi_star_schema(df, roles)
    assert set(["FactSales", "DimDate", "DimCustomer", "DimProduct", "DimGeography", "DimSegment"]).issubset(tables)
    assert len(tables["FactSales"]) == len(df)
    assert "Profit Margin %" in tables["FactSales"].columns
    assert "Discount Band" in tables["FactSales"].columns
    assert not tables["DimProduct"].empty


def test_powerbi_pack_zip_contains_dashboard_assets():
    df = sample_df()
    roles = detect_columns(df)["roles"]
    payload = generate_powerbi_export_pack(df, roles)
    assert len(payload) > 3000
    with zipfile.ZipFile(BytesIO(payload)) as zf:
        names = set(zf.namelist())
    assert "data_model/FactSales.csv" in names
    assert "data_model/DimDate.csv" in names
    assert "dax_measures.md" in names
    assert "power_query_template.pq" in names
    assert "dashboard_blueprint.md" in names
    assert "theme_sales_copilot.json" in names
    assert "README_POWER_BI.md" in names


def test_powerbi_summary_lists_senior_dashboard_pages():
    df = sample_df()
    roles = detect_columns(df)["roles"]
    summary = powerbi_pack_summary(df, roles)
    assert summary["tables"]["FactSales"]["rows"] == len(df)
    assert "Profit Intelligence" in summary["report_pages"]
    assert "Discount & Margin Control" in summary["report_pages"]
