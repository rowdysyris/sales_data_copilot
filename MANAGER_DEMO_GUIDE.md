# Manager Demo Guide — AI Sales Analyst Copilot

This guide is the exact demo script for presenting the project to a manager, mentor, interviewer, or stakeholder.

## Demo objective

Show that the app can reduce routine senior data analyst workload by turning a raw sales dataset into:

- trusted data profile
- executive KPIs
- profit relationship analysis
- root-cause diagnosis
- customer/product/region intelligence
- manager-ready recommendations
- downloadable PDF/Excel reports
- natural-language business answers grounded in calculations

## Demo setup

```bash
cd ai_sales_analyst_copilot
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python sample_data/generate_sample_sales_data.py
streamlit run app.py
```

Open the URL shown by Streamlit, normally `http://localhost:8501`.

## 8-minute manager demo flow

### 1. Open with the problem statement — 45 seconds

> Current sales analysis often depends on manual Excel/Pandas/Power BI work. Managers ask questions such as why profit is low, which products are loss-making, which customers are loyal, and what action to take. This tool automates that routine analyst workflow from a CSV or Excel file.

### 2. Load sample dataset — 30 seconds

Use **Load Sample Sales Dataset** from the sidebar.

Say:

> The app can also accept any CSV/Excel sales-like dataset. The sample dataset is intentionally built with realistic issues: loss-making Tables, high discounts, weak South region margin, loyal customers, inactive customers, and outlier orders.

### 3. Upload & Overview tab — 45 seconds

Show:

- row count
- column detection
- semantic dataset type
- available analyses

Say:

> Before answering anything, the system detects business columns and builds a semantic model. It does not depend on hardcoded column names.

### 4. Data Quality tab — 45 seconds

Show:

- data quality score
- warnings
- outlier summary

Say:

> A senior analyst first checks whether the data is trustworthy. This module surfaces missing values, duplicates, outliers, and suspicious values before analysis.

### 5. Executive Dashboard tab — 60 seconds

Show:

- total sales
- total profit
- margin
- sales/profit charts

Say:

> This replaces routine KPI preparation. It gives a manager quick performance context before deeper diagnosis.

### 6. Profit Intelligence tab — 90 seconds

Show:

- profit summary
- root-cause summary
- profit leakage points
- discount band analysis

Say:

> This is the core senior analyst layer. Profit questions automatically trigger relationship analysis with sales, discount, category, product, region, customer, segment, quantity, and time wherever those fields exist.

### 7. Ask AI Analyst tab — 2 minutes

Ask these questions:

1. `Why is profit low?`
2. `Why are Tables losing money?`
3. `Is discount hurting profit?`
4. `Who are our most loyal customers?`
5. `Which region should we fix first?`
6. `Give me a board-level summary.`

Say:

> The answer format is evidence-first: direct answer, computed evidence, relationships, root cause, recommendation, confidence, and limitations. The LLM, if enabled, only improves wording. It does not calculate or invent numbers.

### 8. Reports tab — 45 seconds

Download Excel and PDF report.

Say:

> The same computed analysis can be exported as manager-ready reports.

## Best demo questions

Use these in the live demo:

```text
Why is profit low?
Why are Tables losing money?
Is discount hurting profit?
Why are sales high but profit low?
Who are our most loyal customers?
Which customers are loyal but unprofitable?
Which products should we stop selling?
Which region should we fix first?
What is going wrong?
Give me a board-level summary.
What happens if we cap discounts above 30%?
```

## What to emphasize

- Numbers come from Pandas calculations, not LLM guesses.
- The app works without paid API keys.
- It handles missing columns gracefully.
- It is modular and test-covered.
- It is not just a dashboard; it has root-cause and recommendation logic.

## Honest limitations to mention

- This is an MVP, not a deployed enterprise SaaS platform.
- Forecasts/scenarios are estimates and include assumptions.
- It currently analyzes one dataset at a time.
- SQL connectors, user login, scheduled reports, and multi-file modeling are future upgrades.

## Demo close

> This project shows how an AI-assisted analyst system can reduce repetitive analysis work and help one manager or analyst answer business questions faster, with evidence and recommendations.

## Optional Power BI Demo Segment

After showing the Streamlit AI analyst workflow, open the **Power BI Dashboard** tab and explain:

> “For leadership dashboards, this project also exports a Power BI-ready model. It creates a clean star schema, DAX measures, Power Query template, corporate theme, and a senior analyst dashboard blueprint. So the same analysis can be consumed in Streamlit for AI Q&A and in Power BI for formal reporting.”

Recommended demonstration:

1. Click **Download Power BI Dashboard Pack**.
2. Show that it contains FactSales, dimension tables, DAX measures, Power Query, theme, and dashboard blueprint.
3. Explain the six Power BI pages: Executive Overview, Profit Intelligence, Product Portfolio, Customer Intelligence, Regional Performance, Discount & Margin Control.
