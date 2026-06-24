#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
if [ ! -d .venv ]; then
  python -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt
python sample_data/generate_sample_sales_data.py
streamlit run app.py
