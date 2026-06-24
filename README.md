# AI Sales Analyst Copilot

A manager-ready AI sales analyst app that reads a sales CSV/Excel file, profiles it, answers Tier 1–Tier 4 business questions, finds profit drivers, and exports BI-ready reports. All numbers are calculated with Pandas; optional OpenAI/Ollama only polishes wording.

## Live Project Links

Live Demo: https://salesdatacopilot.streamlit.app  
GitHub Repository: https://github.com/rowdysyris/sales_data_copilot

## Install in 3 steps

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

## No-API mode

The default `.env.example` setting is:

```env
LLM_PROVIDER=none
```

The app works fully without an API key.

## Optional API mode

Copy `.env.example` to `.env`, then set:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o-mini
```

or for Ollama:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
```

## Core features

- CSV/Excel upload and safe file validation
- Dataset profiling: rows, columns, dtypes, nulls, duplicates, date range, unique counts
- Smart column detection and manual column override
- Data quality analysis and business-signal separation
- Business-aware missing-value treatment log
- KPI engine: sales, profit, margin, orders, AOV, discounts, loss order rate
- Tier 1 sanity questions
- Tier 2 analyst questions: ranking, grouping, margin, correlation, filtering, comparison
- Tier 3 hard questions: YoY growth, multi-condition filters, subset %, consistency, judgment calls
- Tier 4 messy real-world questions: vague phrasing, assumptions, multi-intent, follow-ups
- Profit, customer, product, region, discount, shipping, trend, and scenario intelligence
- Interactive BI dashboard with filters and drilldowns
- PDF, Excel, and Power BI export pack
- No-API, OpenAI, and Ollama modes
- GitHub Actions CI and regression tests

## Demo

Live app: https://salesdatacopilot.streamlit.app

The application can also be run locally using Streamlit:

```bash
streamlit run app.py

## Validation

Run:

```bash
python scripts/validate_project.py
```

This compiles the project and runs the complete regression suite in stable chunks.

## License

MIT
