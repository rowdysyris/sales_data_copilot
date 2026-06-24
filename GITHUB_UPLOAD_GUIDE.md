# GitHub Upload Guide

This project is ready to upload to GitHub.

## Option A — Upload with GitHub Desktop

1. Extract the ZIP.
2. Open GitHub Desktop.
3. Click **File → Add local repository**.
4. Select the `ai_sales_analyst_copilot` folder.
5. If GitHub Desktop says it is not a repository, click **create a repository**.
6. Repository name: `ai-sales-analyst-copilot`.
7. Commit message: `Initial manager-demo release`.
8. Click **Publish repository**.
9. Keep it public if you want it visible on your resume.

## Option B — Upload with terminal

From inside the extracted project folder:

```bash
git init
git add .
git commit -m "Initial manager-demo release: AI Sales Analyst Copilot"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ai-sales-analyst-copilot.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

## Option C — Create repo with GitHub CLI

```bash
gh repo create ai-sales-analyst-copilot --public --source=. --remote=origin --push
```

## Before uploading

Run:

```bash
python sample_data/generate_sample_sales_data.py
python -m compileall .
python scripts/validate_project.py
```

Expected result:

```text
compileall: pass
validation: all tests pass with warnings treated as errors
```

## Recommended GitHub repository description

```text
Evidence-first AI Sales Analyst Copilot that turns CSV/Excel sales datasets into KPIs, profit intelligence, root-cause analysis, customer/product insights, manager Q&A, scenario simulation, and reports.
```

## Recommended resume bullet

```text
Built an AI Sales Analyst Copilot using Python, Streamlit, Pandas, Plotly, and ReportLab to automate sales KPI analysis, profit relationship diagnosis, root-cause analysis, customer loyalty segmentation, scenario simulation, and manager-ready report generation from CSV/Excel datasets.
```
