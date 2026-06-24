# Demo Checklist

Before the demo:

- [ ] Extract ZIP
- [ ] Create virtual environment
- [ ] Install dependencies
- [ ] Generate sample data
- [ ] Run tests
- [ ] Launch Streamlit
- [ ] Load sample dataset
- [ ] Test one Q&A question
- [ ] Download Excel report
- [ ] Download PDF report

Commands:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python sample_data/generate_sample_sales_data.py
python scripts/validate_project.py
streamlit run app.py
```

Questions to ask in demo:

1. Why is profit low?
2. Why are Tables losing money?
3. Is discount hurting profit?
4. Who are our most loyal customers?
5. Which region should we fix first?
6. Which products should we stop selling?
7. Give me a board-level summary.

Pass criteria:

- [ ] App loads without crash
- [ ] Column detection appears
- [ ] KPI dashboard appears
- [ ] Profit Intelligence has findings
- [ ] Ask AI Analyst answers with evidence
- [ ] Reports download
