# Contributing

## Local development

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python sample_data/generate_sample_sales_data.py
python scripts/validate_project.py
streamlit run app.py
```

## Rules for analytics logic

- Do not let an LLM calculate metrics.
- All numbers must come from deterministic Python/Pandas functions.
- Missing columns must return limitations, not crashes.
- Add tests for every new analysis function.
- Keep manager answers evidence-first.

## Test before committing

```bash
python scripts/validate_project.py
```
