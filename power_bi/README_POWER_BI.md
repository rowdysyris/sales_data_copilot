# Power BI Dashboard Pack

This project includes a Power BI companion layer for senior sales analyst dashboards.

The project does not ship a `.pbix` file because `.pbix` is a proprietary Power BI Desktop binary. Instead, the Streamlit app generates a Power BI-ready dashboard pack containing:

- star-schema CSV model
- DAX measure catalog
- Power Query M template
- corporate theme JSON
- senior analyst dashboard blueprint

## Dashboard pages included in the blueprint

1. Executive Overview
2. Profit Intelligence
3. Product Portfolio
4. Customer Intelligence
5. Regional Performance
6. Discount & Margin Control

## How to generate

Run the app:

```bash
streamlit run app.py
```

Open the **Power BI Dashboard** tab and click **Download Power BI Dashboard Pack**.

## Why this matters

This gives the manager both options:

- Streamlit AI analyst app for automated questioning and insight generation
- Power BI dashboard layer for formal business reporting and leadership dashboards
