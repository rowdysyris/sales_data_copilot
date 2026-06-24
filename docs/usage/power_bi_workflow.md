# Power BI Dashboard Workflow

This project does not generate a `.pbix` binary because Power BI Desktop creates that file. The app generates a Power BI-ready dashboard pack.

## Steps

1. Run the Streamlit app:

```powershell
streamlit run app.py
```

2. Upload your dataset.
3. Open the `Power BI Dashboard` tab.
4. Click `Download Power BI Dashboard Pack`.
5. Extract the downloaded ZIP.
6. Open Power BI Desktop.
7. Import these CSV tables from `data_model/`:
   - `FactSales.csv`
   - `DimDate.csv`
   - `DimCustomer.csv`
   - `DimProduct.csv`
   - `DimGeography.csv`
   - `DimSegment.csv`
8. Create relationships as described in `README_POWER_BI.md` inside the exported pack.
9. Paste DAX measures from `dax_measures.md`.
10. Apply `theme_sales_copilot.json` from Power BI Desktop:

`View → Themes → Browse for themes`

## Senior analyst dashboard pages

Build these pages:

1. Executive Overview
2. Profit Intelligence
3. Product Portfolio
4. Customer Intelligence
5. Regional Performance
6. Discount & Margin Control

## Suggested visual order

- Top row: Sales, Profit, Margin, Orders, Loss Amount
- Trend: Sales and Profit by Month
- Category: Sales vs Profit by Category/Sub-Category
- Profit leakage: Loss by Product/Region/Discount Band
- Customer: Loyal, at-risk, and unprofitable customers
- Region: Region/state/city margin map/table
