@echo off
cd /d %~dp0\..
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate
pip install -r requirements.txt
python sample_data\generate_sample_sales_data.py
streamlit run app.py
