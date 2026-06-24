from __future__ import annotations

from io import BytesIO
import json
import re
import zipfile
from typing import Any

import pandas as pd

from src.utils import safe_numeric, profit_margin, prepare_analysis_dataset, ensure_datetime


def _roles(column_roles: dict) -> dict:
    return column_roles.get("roles", column_roles) if isinstance(column_roles, dict) else {}


def _safe_text(series: pd.Series, fallback: str = "Unknown") -> pd.Series:
    return series.astype("string").fillna(fallback).replace({"": fallback})


def _clean_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "file"


def _discount_band(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return "Unknown"
    if pd.isna(v):
        return "Unknown"
    # Handle datasets that store discount as 0.3 or 30.
    if v <= 1:
        v *= 100
    if v <= 0:
        return "0%"
    if v <= 10:
        return "0-10%"
    if v <= 20:
        return "10-20%"
    if v <= 30:
        return "20-30%"
    if v <= 40:
        return "30-40%"
    return "40%+"


def _date_key(series: pd.Series) -> pd.Series:
    dt = ensure_datetime(series)
    return dt.dt.strftime("%Y%m%d").fillna("Unknown")


def _make_key(df: pd.DataFrame, cols: list[str], prefix: str) -> pd.Series:
    available = [c for c in cols if c and c in df.columns]
    if not available:
        return pd.Series([f"{prefix}-Unknown"] * len(df), index=df.index)
    raw = df[available].astype("string").fillna("Unknown").agg("|".join, axis=1)
    return prefix + "-" + pd.factorize(raw, sort=True)[0].astype(str)


def _dimension_from_keys(df: pd.DataFrame, key_col: str, attrs: list[str]) -> pd.DataFrame:
    cols = [key_col] + [c for c in attrs if c in df.columns]
    out = df[cols].drop_duplicates().copy()
    for c in out.columns:
        out[c] = _safe_text(out[c]) if c != key_col else out[c]
    return out.sort_values(key_col).reset_index(drop=True)


def build_powerbi_star_schema(df: pd.DataFrame, column_roles: dict) -> dict[str, pd.DataFrame]:
    """Build senior-analyst-friendly Power BI star schema tables from the uploaded sales dataset.

    Returns CSV-ready tables. The schema is intentionally simple and robust:
    FactSales connects to DimDate, DimCustomer, DimProduct, DimGeography, and DimSegment.
    """
    df, r = prepare_analysis_dataset(df, column_roles)
    if df is None or df.empty:
        return {"FactSales": pd.DataFrame()}

    sales_col = r.get("sales_column")
    profit_col = r.get("profit_column")
    qty_col = r.get("quantity_column")
    discount_col = r.get("discount_column")
    date_col = r.get("date_column")
    ship_date_col = r.get("ship_date_column")
    order_col = r.get("order_id_column")
    customer_col = r.get("customer_column") or r.get("customer_id_column")
    product_col = r.get("product_column")
    category_col = r.get("category_column")
    subcategory_col = r.get("subcategory_column")
    region_col = r.get("region_column")
    country_col = r.get("country_column")
    state_col = r.get("state_column")
    city_col = r.get("city_column")
    segment_col = r.get("segment_column")
    cost_col = r.get("cost_column")

    work = df.copy().reset_index(drop=True)
    work["Row ID"] = range(1, len(work) + 1)
    work["Order Key"] = _safe_text(work[order_col]) if order_col in work.columns else "ORD-" + work["Row ID"].astype(str)
    work["Order Date Key"] = _date_key(work[date_col]) if date_col in work.columns else "Unknown"
    work["Ship Date Key"] = _date_key(work[ship_date_col]) if ship_date_col in work.columns else "Unknown"
    work["Customer Key"] = _make_key(work, [customer_col], "CUST")
    work["Product Key"] = _make_key(work, [product_col, category_col, subcategory_col], "PROD")
    work["Geography Key"] = _make_key(work, [country_col, state_col, city_col, region_col], "GEO")
    work["Segment Key"] = _make_key(work, [segment_col], "SEG")
    work["Sales"] = safe_numeric(work[sales_col]) if sales_col in work.columns else 0.0
    work["Profit"] = safe_numeric(work[profit_col]) if profit_col in work.columns else 0.0
    work["Quantity"] = safe_numeric(work[qty_col]) if qty_col in work.columns else 0.0
    work["Discount"] = safe_numeric(work[discount_col]) if discount_col in work.columns else 0.0
    work["Cost"] = safe_numeric(work[cost_col]) if cost_col in work.columns else (work["Sales"] - work["Profit"])
    work["Profit Margin %"] = work.apply(lambda x: profit_margin(x["Profit"], x["Sales"]), axis=1)
    work["Is Loss Order"] = work["Profit"] < 0
    work["Loss Amount"] = work["Profit"].where(work["Profit"] < 0, 0.0)
    work["Discount Band"] = work["Discount"].apply(_discount_band)

    fact_cols = [
        "Row ID", "Order Key", "Order Date Key", "Ship Date Key", "Customer Key", "Product Key",
        "Geography Key", "Segment Key", "Sales", "Profit", "Cost", "Quantity", "Discount",
        "Discount Band", "Profit Margin %", "Is Loss Order", "Loss Amount",
    ]
    fact = work[fact_cols].copy()

    product_attrs = []
    if product_col in work.columns:
        work["Product Name"] = _safe_text(work[product_col])
        product_attrs.append("Product Name")
    if category_col in work.columns:
        work["Category"] = _safe_text(work[category_col])
        product_attrs.append("Category")
    if subcategory_col in work.columns:
        work["Sub-Category"] = _safe_text(work[subcategory_col])
        product_attrs.append("Sub-Category")
    dim_product = _dimension_from_keys(work, "Product Key", product_attrs)

    customer_attrs = []
    if customer_col in work.columns:
        work["Customer"] = _safe_text(work[customer_col])
        customer_attrs.append("Customer")
    dim_customer = _dimension_from_keys(work, "Customer Key", customer_attrs)
    if not dim_customer.empty:
        cust_metrics = fact.groupby("Customer Key", dropna=False).agg(
            Total_Sales=("Sales", "sum"), Total_Profit=("Profit", "sum"), Order_Count=("Order Key", "nunique")
        ).reset_index()
        dim_customer = dim_customer.merge(cust_metrics, on="Customer Key", how="left")
        # Robust segmentation for small or low-cardinality customer datasets.
        # pd.cut can fail when bin edges collapse; percentile ranks are stable.
        n_customers = len(dim_customer)
        if n_customers <= 1:
            dim_customer["Customer Value Segment"] = "Strategic"
        else:
            pct_rank = dim_customer["Total_Sales"].rank(method="first", pct=True)
            dim_customer["Customer Value Segment"] = pd.cut(
                pct_rank,
                bins=[0, 0.50, 0.80, 1.00],
                labels=["Core", "High Value", "Strategic"],
                include_lowest=True,
            ).astype(str)

    geo_attrs = []
    for src, label in [(country_col, "Country"), (state_col, "State"), (city_col, "City"), (region_col, "Region")]:
        if src in work.columns:
            work[label] = _safe_text(work[src])
            geo_attrs.append(label)
    dim_geography = _dimension_from_keys(work, "Geography Key", geo_attrs)

    if segment_col in work.columns:
        work["Segment"] = _safe_text(work[segment_col])
    else:
        work["Segment"] = "All Segments"
    dim_segment = _dimension_from_keys(work, "Segment Key", ["Segment"])

    date_values = []
    for c in [date_col, ship_date_col]:
        if c in work.columns:
            date_values.append(ensure_datetime(work[c]))
    if date_values:
        dates = pd.concat(date_values).dropna().drop_duplicates().sort_values()
        dim_date = pd.DataFrame({"Date": dates})
        dim_date["Date Key"] = dim_date["Date"].dt.strftime("%Y%m%d")
        dim_date["Year"] = dim_date["Date"].dt.year
        dim_date["Quarter"] = "Q" + dim_date["Date"].dt.quarter.astype(str)
        dim_date["Month Number"] = dim_date["Date"].dt.month
        dim_date["Month Name"] = dim_date["Date"].dt.month_name()
        dim_date["Year Month"] = dim_date["Date"].dt.strftime("%Y-%m")
        dim_date["Weekday"] = dim_date["Date"].dt.day_name()
        dim_date["Date"] = dim_date["Date"].dt.strftime("%Y-%m-%d")
        dim_date = dim_date[["Date Key", "Date", "Year", "Quarter", "Month Number", "Month Name", "Year Month", "Weekday"]]
    else:
        dim_date = pd.DataFrame(columns=["Date Key", "Date", "Year", "Quarter", "Month Number", "Month Name", "Year Month", "Weekday"])

    return {
        "FactSales": fact,
        "DimDate": dim_date,
        "DimCustomer": dim_customer,
        "DimProduct": dim_product,
        "DimGeography": dim_geography,
        "DimSegment": dim_segment,
    }


def generate_dax_measures() -> str:
    """Return a senior sales analyst DAX measure catalog for the generated star schema."""
    return """# Power BI DAX Measures — AI Sales Analyst Copilot

Create these measures in the FactSales table after loading the exported CSV model.

```DAX
Total Sales = SUM(FactSales[Sales])
Total Profit = SUM(FactSales[Profit])
Total Cost = SUM(FactSales[Cost])
Total Quantity = SUM(FactSales[Quantity])
Total Orders = DISTINCTCOUNT(FactSales[Order Key])
Average Order Value = DIVIDE([Total Sales], [Total Orders])
Profit Margin % = DIVIDE([Total Profit], [Total Sales])
Loss Amount = SUM(FactSales[Loss Amount])
Loss Orders = CALCULATE([Total Orders], FactSales[Is Loss Order] = TRUE())
Loss Order % = DIVIDE([Loss Orders], [Total Orders])
Average Discount = AVERAGE(FactSales[Discount])
Profit per Order = DIVIDE([Total Profit], [Total Orders])
Sales Contribution % = DIVIDE([Total Sales], CALCULATE([Total Sales], ALL(DimProduct)))
Profit Contribution % = DIVIDE([Total Profit], CALCULATE([Total Profit], ALL(DimProduct)))
Loss Contribution % = DIVIDE(ABS([Loss Amount]), CALCULATE(ABS([Loss Amount]), ALL(DimProduct)))

Sales Last Month = CALCULATE([Total Sales], DATEADD(DimDate[Date], -1, MONTH))
Profit Last Month = CALCULATE([Total Profit], DATEADD(DimDate[Date], -1, MONTH))
Sales MoM % = DIVIDE([Total Sales] - [Sales Last Month], [Sales Last Month])
Profit MoM % = DIVIDE([Total Profit] - [Profit Last Month], [Profit Last Month])

High Discount Sales = CALCULATE([Total Sales], FactSales[Discount Band] IN {"30-40%", "40%+"})
High Discount Profit = CALCULATE([Total Profit], FactSales[Discount Band] IN {"30-40%", "40%+"})
High Discount Margin % = DIVIDE([High Discount Profit], [High Discount Sales])

Customer Count = DISTINCTCOUNT(FactSales[Customer Key])
Sales per Customer = DIVIDE([Total Sales], [Customer Count])
Profit per Customer = DIVIDE([Total Profit], [Customer Count])

Problem Product Flag = IF([Total Profit] < 0 || [Profit Margin %] < 0.05, 1, 0)
```

Recommended formatting:
- Currency: Total Sales, Total Profit, Total Cost, Average Order Value, Loss Amount
- Percentage: Profit Margin %, Loss Order %, Sales MoM %, Profit MoM %, Average Discount
"""


def generate_power_query_template() -> str:
    """Return Power Query M template for loading exported star-schema CSVs from a local folder."""
    return """// Power Query M Template — AI Sales Analyst Copilot
// In Power BI Desktop: Get Data → Blank Query → Advanced Editor.
// Replace FolderPath with the folder where you extracted the exported Power BI CSV pack.

let
    FolderPath = "C:\\Path\\To\\power_bi_export\\",
    LoadCsv = (FileName as text) =>
        let
            Source = Csv.Document(File.Contents(FolderPath & FileName), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
            PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true])
        in
            PromotedHeaders,
    FactSales = LoadCsv("FactSales.csv"),
    DimDate = LoadCsv("DimDate.csv"),
    DimCustomer = LoadCsv("DimCustomer.csv"),
    DimProduct = LoadCsv("DimProduct.csv"),
    DimGeography = LoadCsv("DimGeography.csv"),
    DimSegment = LoadCsv("DimSegment.csv")
in
    FactSales
"""


def generate_dashboard_blueprint() -> str:
    """Return senior-analyst Power BI report blueprint."""
    return """# Power BI Dashboard Blueprint — Senior Sales Analyst View

This dashboard is designed as a decision-making layer, not just a chart collection.

## Data Model
Use a star schema:
- FactSales → central fact table
- DimDate → connect FactSales[Order Date Key] to DimDate[Date Key]
- DimCustomer → connect FactSales[Customer Key]
- DimProduct → connect FactSales[Product Key]
- DimGeography → connect FactSales[Geography Key]
- DimSegment → connect FactSales[Segment Key]

Set cross-filter direction to single direction from dimensions to fact unless a specific use case requires otherwise.

## Page 1 — Executive Overview
Purpose: Management-level health check.

Visuals:
- KPI Cards: Total Sales, Total Profit, Profit Margin %, Total Orders, Loss Orders %
- Line chart: Sales and Profit by Year Month
- Bar chart: Profit by Category
- Bar chart: Profit by Region
- Matrix: Category → Sub-Category with Sales, Profit, Margin %, Loss Amount
- Slicers: Date, Region, Category, Segment

Senior analyst question answered:
"Is the business growing profitably, and where is profit leaking?"

## Page 2 — Profit Intelligence
Purpose: Explain profit, not just report it.

Visuals:
- Scatter: Sales vs Profit by Product, size = Quantity, color = Category
- Column chart: Profit by Discount Band
- Matrix: Product / Region / Segment with Sales, Profit, Margin %, Discount, Loss Contribution %
- Top N table: Worst products by Loss Amount
- Decomposition Tree: Profit explained by Category → Sub-Category → Product → Region → Segment

Senior analyst question answered:
"Why is profit low and what is driving losses?"

## Page 3 — Product Portfolio
Purpose: Decide what to promote, fix, or remove.

Visuals:
- Product portfolio matrix: Sales vs Margin %, color by Category
- Table: Stars, Volume Drivers, Problem Products, Niche Winners
- Bar chart: Sales Contribution % vs Profit Contribution % by Product
- Slicer: Category / Sub-Category

Senior analyst question answered:
"Which products deserve focus, review, or removal?"

## Page 4 — Customer Intelligence
Purpose: Retention and profitability.

Visuals:
- Table: Top customers by Sales, Profit, Order Count, Profit per Customer
- Bar chart: Profit by Customer Value Segment
- Matrix: Segment → Customer with Sales, Profit, Margin %
- Scatter: Sales per Customer vs Profit per Customer

Senior analyst question answered:
"Who are our best customers and which customers are not profitable?"

## Page 5 — Regional Performance
Purpose: Find geography-level problems.

Visuals:
- Filled map or bar chart by Region/State/City
- Matrix: Region → State → City with Sales, Profit, Margin %, Discount
- Bar chart: High Discount Profit by Region
- Table: High sales, low margin geographies

Senior analyst question answered:
"Which region should management fix first?"

## Page 6 — Discount & Margin Control
Purpose: Control revenue leakage.

Visuals:
- Profit by Discount Band
- Average Discount by Category/Product/Region
- Margin % by Discount Band
- Table: High discount, negative profit orders/products

Senior analyst question answered:
"Is discounting hurting profit?"

## Recommended Bookmarks
- Executive view
- Loss investigation view
- Discount problem view
- Product removal shortlist
- Customer retention shortlist

## Demo Flow
1. Start on Executive Overview.
2. Show profit problem using Profit Margin % and loss order %.
3. Drill into Profit Intelligence.
4. Show discount band impact.
5. Open Product Portfolio to show problem products.
6. End with Customer/Region action list.
"""


def generate_theme_json() -> dict[str, Any]:
    """Return a clean corporate Power BI theme."""
    return {
        "name": "AI Sales Analyst Copilot",
        "dataColors": ["#1F4E78", "#70AD47", "#FFC000", "#C00000", "#5B9BD5", "#A5A5A5", "#ED7D31", "#4472C4"],
        "background": "#FFFFFF",
        "foreground": "#1F2937",
        "tableAccent": "#1F4E78",
        "visualStyles": {
            "*": {
                "*": {
                    "title": [{"fontSize": 12, "fontFamily": "Segoe UI", "color": {"solid": {"color": "#1F2937"}}}],
                    "labels": [{"fontSize": 10, "fontFamily": "Segoe UI"}],
                }
            }
        },
    }


def powerbi_readme() -> str:
    return """# Power BI Dashboard Pack

This folder is the Power BI companion layer for the AI Sales Analyst Copilot.

It does not include a `.pbix` file because `.pbix` is a proprietary binary format that should be created in Power BI Desktop. Instead, this project generates the exact assets a senior analyst needs to build the dashboard quickly and safely:

- star-schema CSV tables
- DAX measure catalog
- Power Query M template
- corporate theme JSON
- dashboard blueprint
- data model instructions

## How to use

1. Run the Streamlit app.
2. Load your sales dataset or the sample dataset.
3. Open the **Power BI Dashboard** tab.
4. Download the **Power BI Dashboard Pack**.
5. Extract the ZIP locally.
6. Open Power BI Desktop.
7. Load the CSV tables from the extracted folder.
8. Create relationships:
   - FactSales[Order Date Key] → DimDate[Date Key]
   - FactSales[Customer Key] → DimCustomer[Customer Key]
   - FactSales[Product Key] → DimProduct[Product Key]
   - FactSales[Geography Key] → DimGeography[Geography Key]
   - FactSales[Segment Key] → DimSegment[Segment Key]
9. Add the DAX measures from `dax_measures.md`.
10. Import `theme_sales_copilot.json` as the report theme.
11. Build pages using `dashboard_blueprint.md`.

## Why this is senior-analyst ready

The dashboard is not only for showing sales. It is structured to answer:

- Why is profit low?
- Which products are damaging margin?
- Is discounting hurting profit?
- Which regions need management attention?
- Which customers are valuable or unprofitable?
- What should leadership fix first?
"""


def generate_powerbi_export_pack(df: pd.DataFrame, column_roles: dict) -> bytes:
    """Generate a downloadable Power BI dashboard starter pack as a ZIP file."""
    tables = build_powerbi_star_schema(df, column_roles)
    output = BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, table in tables.items():
            csv_bytes = table.to_csv(index=False).encode("utf-8-sig")
            clean_name = f"{_clean_filename(name)}.csv"
            # Keep both root-level files for simple manager downloads and a data_model/
            # folder for users who prefer a star-schema directory layout.
            zf.writestr(clean_name, csv_bytes)
            zf.writestr(f"data_model/{clean_name}", csv_bytes)
        zf.writestr("dax_measures.md", generate_dax_measures())
        zf.writestr("power_query_template.pq", generate_power_query_template())
        zf.writestr("dashboard_blueprint.md", generate_dashboard_blueprint())
        zf.writestr("theme_sales_copilot.json", json.dumps(generate_theme_json(), indent=2))
        zf.writestr("README_POWER_BI.md", powerbi_readme())
    output.seek(0)
    return output.getvalue()


def powerbi_pack_summary(df: pd.DataFrame, column_roles: dict) -> dict[str, Any]:
    """Return a small summary of the generated Power BI model."""
    tables = build_powerbi_star_schema(df, column_roles)
    return {
        "tables": {name: {"rows": int(len(table)), "columns": int(table.shape[1])} for name, table in tables.items()},
        "relationship_count": 5,
        "report_pages": [
            "Executive Overview",
            "Profit Intelligence",
            "Product Portfolio",
            "Customer Intelligence",
            "Regional Performance",
            "Discount & Margin Control",
        ],
        "assets": ["CSV star schema", "DAX measures", "Power Query template", "Theme JSON", "Dashboard blueprint"],
    }
