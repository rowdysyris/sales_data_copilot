"""Manager-demo and GitHub-readiness tests."""
from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd

from src.column_detection import detect_columns
from src.data_quality import analyze_data_quality
from src.kpi_engine import calculate_kpis
from src.question_answering import answer_business_question
from src.report_generator import generate_excel_report, generate_pdf_report
from src.recommendation_engine import generate_recommendations
from src.semantic_layer import build_semantic_model


ROOT = Path(__file__).resolve().parents[1]


def demo_df() -> pd.DataFrame:
    rows = []
    for month in range(1, 7):
        rows.extend([
            {
                "Order ID": f"T-{month}-1",
                "Order Date": pd.Timestamp(2025, month, 5),
                "Customer Name": "Alpha Retail",
                "Region": "South",
                "Segment": "Consumer",
                "Category": "Furniture",
                "Sub-Category": "Tables",
                "Product Name": "Executive Table",
                "Sales": 1800 + month * 20,
                "Quantity": 2,
                "Discount": 0.45,
                "Profit": -500 - month * 25,
            },
            {
                "Order ID": f"P-{month}-1",
                "Order Date": pd.Timestamp(2025, month, 8),
                "Customer Name": "Beta Corp",
                "Region": "West",
                "Segment": "Corporate",
                "Category": "Technology",
                "Sub-Category": "Phones",
                "Product Name": "Pro Phone",
                "Sales": 3000 + month * 50,
                "Quantity": 1,
                "Discount": 0.05,
                "Profit": 900 + month * 20,
            },
            {
                "Order ID": f"C-{month}-1",
                "Order Date": pd.Timestamp(2025, month, 12),
                "Customer Name": "Gamma Office",
                "Region": "North",
                "Segment": "Home Office",
                "Category": "Office Supplies",
                "Sub-Category": "Paper",
                "Product Name": "Copy Paper",
                "Sales": 250 + month * 10,
                "Quantity": 5,
                "Discount": 0.0,
                "Profit": 60 + month * 2,
            },
        ])
    rows.append({
        "Order ID": "OUT-1",
        "Order Date": pd.Timestamp(2025, 6, 20),
        "Customer Name": "Discount Heavy",
        "Region": "South",
        "Segment": "Consumer",
        "Category": "Furniture",
        "Sub-Category": "Tables",
        "Product Name": "Mega Table",
        "Sales": 25000,
        "Quantity": 12,
        "Discount": 0.70,
        "Profit": -12000,
    })
    return pd.DataFrame(rows)


def demo_roles() -> dict:
    return detect_columns(demo_df())["roles"]


def test_github_and_manager_docs_exist():
    required = [
        "README.md",
        "MANAGER_DEMO_GUIDE.md",
        "GITHUB_UPLOAD_GUIDE.md",
        "PROJECT_REPORT.md",
        "docs/ARCHITECTURE.md",
        "docs/FEATURE_MATRIX.md",
        "docs/DEMO_CHECKLIST.md",
        "demo/demo_questions.md",
        "demo/demo_script.md",
        ".github/workflows/ci.yml",
        "LICENSE",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    assert not missing, f"Missing readiness files: {missing}"


def test_all_src_modules_importable():
    for path in (ROOT / "src").glob("*.py"):
        if path.name == "__init__.py":
            continue
        importlib.import_module(f"src.{path.stem}")


def test_manager_questions_are_evidence_first():
    df = demo_df()
    roles = demo_roles()
    questions = [
        "Why is profit low?",
        "Why are Tables losing money?",
        "Is discount hurting profit?",
        "Who are our most loyal customers?",
        "Which region should we fix first?",
        "Which products should we stop selling?",
        "Give me a board-level summary.",
    ]
    for question in questions:
        answer = answer_business_question(question, df, roles)
        markdown = answer["markdown_answer"]
        assert "Direct Answer" in markdown
        assert "Evidence From Data" in markdown
        assert "Recommended Action" in markdown
        assert answer["confidence"] in {"High", "Medium", "Low"}


def test_profit_questions_include_relationships():
    df = demo_df()
    roles = demo_roles()
    for question in ["Why is profit low?", "Is discount hurting profit?", "Why are sales high but profit low?"]:
        answer = answer_business_question(question, df, roles)
        markdown = answer["markdown_answer"]
        assert "Relationships / Drivers" in markdown
        assert answer["relationship_findings"]


def test_core_analysis_outputs_are_not_empty():
    df = demo_df()
    roles = demo_roles()
    semantic = build_semantic_model(df, roles)
    dq = analyze_data_quality(df, roles)
    kpis = calculate_kpis(df, roles)
    recommendations = generate_recommendations(df, roles, {"kpis": kpis, "data_quality": dq})
    assert semantic["dataset_type"] == "sales_transaction_dataset"
    assert 0 <= dq["quality_score"] <= 100
    assert kpis["total_sales"] > 0
    assert recommendations


def test_reports_generate_manager_ready_files():
    df = demo_df()
    roles = demo_roles()
    ctx = answer_business_question("Give me board-level summary", df, roles)["context_results"]
    excel_bytes = generate_excel_report(df, roles, ctx)
    pdf_bytes = generate_pdf_report(ctx)
    assert excel_bytes[:2] == b"PK"
    assert pdf_bytes[:4] == b"%PDF"
    assert len(excel_bytes) > 5000
    assert len(pdf_bytes) > 1000
