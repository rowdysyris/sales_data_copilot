# Project Report — AI Sales Analyst Copilot

## Problem statement

Sales managers need fast answers to business questions such as why profit is low, which product is causing losses, whether discounting is hurting margins, which customers are loyal, and what actions should be taken. Traditional analysis requires manual dataset cleaning, KPI preparation, dashboarding, root-cause analysis, and report writing.

## Proposed solution

AI Sales Analyst Copilot is a Streamlit application that reads a sales-like CSV/Excel dataset and automates the routine workflow of a senior data analyst. It combines deterministic Pandas calculations with optional LLM wording support. The LLM is never responsible for numeric calculations.

## Architecture

```text
Dataset Upload
→ Ingestion and cleanup
→ Column detection
→ Semantic model
→ Data quality checks
→ KPI engine
→ EDA and correlations
→ Profit intelligence
→ Root-cause analysis
→ Customer/product/region intelligence
→ Hidden insights
→ Manager Q&A
→ Recommendations
→ Forecasting and anomaly detection
→ Scenario simulator
→ Excel/PDF reports
```

## Key modules

| Module | Purpose |
|---|---|
| `ingestion.py` | Loads and cleans CSV/Excel files |
| `column_detection.py` | Detects business columns such as Sales, Profit, Discount, Customer, Product, Region |
| `semantic_layer.py` | Identifies dataset type, grain, metrics, dimensions, and available analyses |
| `data_quality.py` | Checks missing values, duplicates, outliers, invalid values, and quality score |
| `kpi_engine.py` | Calculates executive KPIs and grouped performance |
| `metric_relationships.py` | Finds relationships between profit and drivers |
| `profit_intelligence.py` | Diagnoses profit, margin, loss, leakage, and root causes |
| `customer_intelligence.py` | Finds loyal, profitable, at-risk, and unprofitable customers |
| `product_intelligence.py` | Classifies products into stars, volume drivers, niche winners, and problem products |
| `region_intelligence.py` | Finds strong and weak geographies/segments |
| `analyst_brain.py` | Central reasoning pipeline for manager questions |
| `question_answering.py` | Evidence-first business answer generator |
| `forecasting.py` | Conservative forecast using moving average, smoothing, trend, and seasonality |
| `anomaly_detection.py` | Flags unusual drops, spikes, and outlier rows |
| `scenario_simulator.py` | Estimates impact of discount, price, quantity, and margin scenarios |
| `report_generator.py` | Exports Excel and PDF manager reports |

## Evidence-first Q&A design

Every answer follows this structure:

1. Direct Answer
2. Evidence From Data
3. Relationships / Drivers
4. Root Cause
5. Recommended Action
6. Confidence
7. Limitations
8. Suggested Follow-up Questions

## Profit-intelligence rule

Any question containing profit, loss, margin, profitability, low profit, losing money, or revenue leakage automatically triggers profit intelligence and metric relationship analysis.

## Business impact

The system reduces analyst time spent on:

- repeated KPI calculations
- standard EDA
- profit/loss breakdowns
- customer segmentation
- product/region performance checks
- report drafting
- answering repetitive manager questions

## Validation

The project includes automated tests and a validation script. Current validation confirms compilation, test suite, sample data generation, report export, and major manager Q&A flows.

## Limitations

- MVP-level app, not a deployed enterprise SaaS platform.
- Single-dataset workflow only.
- Forecasts and simulations are estimates.
- Database connectors and user authentication are not included.
- LLM support is optional and wording-only.

## Future scope

- SQL/database connectors
- multi-file relationship modeling
- role-based login
- scheduled reports
- cloud deployment
- PPT report generation
- richer UI/UX
