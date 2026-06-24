from __future__ import annotations

import os
import html
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.ingestion import load_dataset
from src.column_detection import detect_columns
from src.semantic_layer import build_semantic_model
from src.data_quality import analyze_data_quality
from src.kpi_engine import calculate_kpis, performance_by_category, performance_by_region, performance_over_time
from src.eda_engine import generate_eda_summary
from src.profit_intelligence import analyze_profit_intelligence
from src.root_cause import find_loss_drivers
from src.customer_intelligence import analyze_customer_loyalty
from src.product_intelligence import analyze_product_portfolio
from src.region_intelligence import analyze_region_segment_performance
from src.hidden_insights import generate_hidden_insights
from src.metric_relationships import analyze_metric_relationships
from src.dataset_intelligence_orchestrator import build_dataset_intelligence
from src.question_answering import answer_business_question
from src.forecasting import forecast_sales_profit
from src.anomaly_detection import detect_business_anomalies
from src.recommendation_engine import generate_recommendations, generate_executive_summary, executive_summary_to_html
from src.report_generator import generate_excel_report, generate_pdf_report
from src.powerbi_exporter import generate_powerbi_export_pack, powerbi_pack_summary
from src.embedded_bi_dashboard import render_embedded_bi_dashboard, powerbi_embed_status
from src.scenario_simulator import (
    simulate_discount_reduction,
    simulate_price_increase,
    simulate_removing_loss_making_products,
    simulate_quantity_change,
    simulate_discount_cap,
    simulate_margin_improvement,
)
from src.ui_theme import apply_maven_style_theme, render_hero, section_header, modern_card, insight_card
from src.utils import format_currency, format_percent, prepare_analysis_dataset


load_dotenv()

st.set_page_config(
    page_title="AI Sales Analyst Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.markdown("### Appearance")
theme_mode = st.sidebar.radio(
    "Theme",
    ["Light", "Dark"],
    index=0,
    horizontal=True,
    key="theme_mode",
)
apply_maven_style_theme(theme_mode)


@st.cache_data(show_spinner=False)
def _load_sample() -> pd.DataFrame:
    path = Path("sample_data/sample_sales_data.csv")
    if not path.exists():
        from sample_data.generate_sample_sales_data import generate_sample_sales_data

        path.parent.mkdir(exist_ok=True)
        generate_sample_sales_data().to_csv(path, index=False)
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _cached_detect(df: pd.DataFrame) -> dict:
    return detect_columns(df)


@st.cache_data(show_spinner="Running senior analyst engines...")
def _run_core(df: pd.DataFrame, roles: dict) -> dict:
    df, roles = prepare_analysis_dataset(df, roles)
    semantic = build_semantic_model(df, roles)
    kpis = calculate_kpis(df, roles)
    dq = analyze_data_quality(df, roles)
    pi = analyze_profit_intelligence(df, roles)
    ci = analyze_customer_loyalty(df, roles)
    prod = analyze_product_portfolio(df, roles)
    reg = analyze_region_segment_performance(df, roles)
    hidden = generate_hidden_insights(df, roles)
    metric_relationships = analyze_metric_relationships(df, roles)
    forecast = forecast_sales_profit(df, roles)
    anomalies = detect_business_anomalies(df, roles)
    ctx = {
        "semantic_model": semantic,
        "missing_value_treatment": roles.get("_missing_value_treatment", {}),
        "column_roles": roles,
        "kpis": kpis,
        "data_quality": dq,
        "profit_intelligence": pi,
        "customer_intelligence": ci,
        "product_intelligence": prod,
        "region_intelligence": reg,
        "hidden_insights": hidden,
        "metric_relationships": metric_relationships,
        "forecasting": forecast,
        "anomaly_detection": anomalies,
    }
    ctx["recommendations"] = generate_recommendations(df, roles, ctx)
    ctx["executive_summary"] = generate_executive_summary(ctx)
    ctx["dataset_intelligence"] = build_dataset_intelligence(ctx, roles)
    return ctx


def _safe_df(value) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _dataframe_section(title: str, frame: pd.DataFrame, rows: int = 50) -> None:
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        st.subheader(title)
        st.dataframe(frame.head(rows), use_container_width=True)


def _pretty_label(key: str) -> str:
    """Convert snake_case/camel-ish keys into manager-readable labels."""
    text = str(key).replace("_", " ").replace("-", " ").strip()
    return text.title() if text else "Value"


def _format_manager_value(value) -> str:
    """Render values without Python dict/list syntax."""
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, (list, tuple, set)):
        items = [str(v) for v in value if v not in (None, "")]
        return ", ".join(items) if items else "Not available"
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            if isinstance(v, (dict, list, tuple, set)):
                v = _format_manager_value(v)
            parts.append(f"{_pretty_label(k)}: {v}")
        return " | ".join(parts) if parts else "Not available"
    return str(value)


def _dict_to_display_frame(data: dict) -> pd.DataFrame:
    rows = []
    for key, value in (data or {}).items():
        if isinstance(value, pd.DataFrame):
            continue
        rows.append({"Field": _pretty_label(key), "Value": _format_manager_value(value)})
    return pd.DataFrame(rows)


def _render_key_value_summary(title: str, data: dict | None, *, empty_message: str = "No summary available.") -> None:
    """Show dict summaries as clean two-column manager tables instead of raw Python objects."""
    if not isinstance(data, dict) or not data:
        st.info(empty_message)
        return
    frame = _dict_to_display_frame(data)
    if frame.empty:
        st.info(empty_message)
        return
    st.subheader(title)
    st.dataframe(frame, use_container_width=True, hide_index=True)


def _format_backtest_number(value, *, currency: bool = False, percent: bool = False) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if currency:
        return format_currency(value)
    if percent:
        return format_percent(value)
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return str(value)


def _backtest_quality_label(mape_value, *, target: str) -> tuple[str, str]:
    if mape_value is None or pd.isna(mape_value):
        return "Not enough history", "More historical periods are required for a reliable accuracy check."
    try:
        mape = float(mape_value)
    except Exception:
        return "Not enough history", "The accuracy value could not be interpreted."

    if target == "profit":
        if mape < 20:
            return "Good", "Profit prediction error is low for the available history."
        if mape < 50:
            return "Moderate", "Profit prediction is usable for trend discussion, but not final decisions."
        return "Weak", "Profit prediction error is high. Profit can be difficult to predict when values are small, negative, or highly variable."

    if mape < 10:
        return "Strong", "Sales prediction error is low for the available history."
    if mape < 25:
        return "Acceptable", "Sales prediction is usable, but accuracy can still be improved."
    return "Weak", "Sales prediction error is high. Treat the forecast as a rough trend estimate, not a decision-grade prediction."


def _render_backtest_summary(backtest: dict | None) -> None:
    """Render forecast backtest results as business-friendly metric cards."""
    if not isinstance(backtest, dict) or not backtest:
        st.info("Backtest summary is unavailable. More dated history may be required.")
        return

    st.subheader("Backtest Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sales MAE", _format_backtest_number(backtest.get("sales_mae"), currency=True), help="Average sales prediction error. Lower is better.")
    c2.metric("Sales MAPE", _format_backtest_number(backtest.get("sales_mape_percent"), percent=True), help="Average sales prediction error in percentage terms. Lower is better.")
    c3.metric("Profit MAE", _format_backtest_number(backtest.get("profit_mae"), currency=True), help="Average profit prediction error. Lower is better.")
    c4.metric("Profit MAPE", _format_backtest_number(backtest.get("profit_mape_percent"), percent=True), help="Average profit prediction error in percentage terms. Can be unstable when profit is small or negative.")

    st.info(
        "Backtesting tests the forecasting method on past data. MAE shows the average size of the error. "
        "MAPE shows the average percentage error. Lower values mean better forecasting accuracy."
    )

    sales_label, sales_text = _backtest_quality_label(backtest.get("sales_mape_percent"), target="sales")
    profit_label, profit_text = _backtest_quality_label(backtest.get("profit_mape_percent"), target="profit")
    interpretation = pd.DataFrame(
        [
            {"Metric": "Sales Forecast", "Accuracy": sales_label, "Meaning": sales_text},
            {"Metric": "Profit Forecast", "Accuracy": profit_label, "Meaning": profit_text},
        ]
    )
    st.markdown("**Model Accuracy Interpretation**")
    st.dataframe(interpretation, use_container_width=True, hide_index=True)


def _render_status_table(title: str, data: dict | None) -> None:
    if isinstance(data, dict) and data:
        _render_key_value_summary(title, data)
    else:
        st.info("No status details available.")


def _missing_treatment_meta(roles: dict) -> dict:
    return roles.get("_missing_value_treatment", {}) if isinstance(roles, dict) else {}


def _cleaned_dataset_used(roles: dict) -> bool:
    return bool(_missing_treatment_meta(roles).get("cleaned_dataset_used"))


def _render_cleaned_data_warning(roles: dict, *, compact: bool = True) -> None:
    meta = _missing_treatment_meta(roles)
    if not meta.get("cleaned_dataset_used"):
        return
    text = meta.get("manager_warning") or (
        "Some values were missing in the original dataset. Calculations below are based on a cleaned dataset. "
        "Review the Missing Value Treatment Log before making business decisions."
    )
    if compact:
        st.warning(text)
    else:
        st.warning(text)
        if meta.get("summary_text"):
            st.caption(meta["summary_text"])


def _missing_treatment_frame(roles: dict) -> pd.DataFrame:
    meta = _missing_treatment_meta(roles)
    rows = meta.get("treatment_log", []) if isinstance(meta, dict) else []
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _render_scenario_result(result: dict) -> None:
    """Render scenario simulator output cleanly without st.json/raw dicts."""
    if not isinstance(result, dict) or not result:
        st.info("No scenario result available.")
        return

    summary_keys = [
        "baseline_sales", "baseline_profit", "baseline_margin",
        "simulated_sales", "simulated_profit", "simulated_margin",
        "estimated_profit_change", "estimated_sales_change",
    ]
    summary = {k: result.get(k) for k in summary_keys if k in result}
    if summary:
        _render_key_value_summary("Scenario Summary", summary)

    assumptions = result.get("assumptions")
    if assumptions:
        st.subheader("Assumptions")
        for item in assumptions if isinstance(assumptions, list) else [assumptions]:
            st.caption(f"• {item}")

    limitations = result.get("limitations")
    if limitations:
        st.subheader("Limitations")
        for item in limitations if isinstance(limitations, list) else [limitations]:
            st.caption(f"• {item}")

    remaining = {k: v for k, v in result.items() if k not in set(summary_keys + ["assumptions", "limitations"])}
    if remaining:
        with st.expander("Additional scenario details"):
            _render_key_value_summary("Additional Details", remaining)


def _sidebar_status(provider: str) -> None:
    st.sidebar.markdown("### AI Analyst Setup")
    if provider == "openai":
        if os.getenv("OPENAI_API_KEY", "").strip():
            st.sidebar.success("OpenAI wording layer enabled")
        else:
            st.sidebar.warning("LLM_PROVIDER=openai but OPENAI_API_KEY is missing")
    elif provider == "ollama":
        st.sidebar.info("Ollama wording layer selected")
    else:
        st.sidebar.info("Deterministic mode: no API key needed")


provider = os.getenv("LLM_PROVIDER", "none").strip().lower()
_sidebar_status(provider)

st.sidebar.markdown("### Dataset")
uploaded = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"])
use_sample = st.sidebar.button("Load Sample Sales Dataset", use_container_width=True)

if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.metadata = {}

if uploaded:
    try:
        st.session_state.df, st.session_state.metadata = load_dataset(uploaded)
    except Exception as exc:
        st.session_state.df = None
        st.session_state.metadata = {}
        st.sidebar.error(f"Could not load dataset: {exc}")
elif use_sample:
    st.session_state.df = _load_sample()
    st.session_state.metadata = {
        "row_count": len(st.session_state.df),
        "column_count": st.session_state.df.shape[1],
        "columns": list(st.session_state.df.columns),
        "dtypes": {c: str(t) for c, t in st.session_state.df.dtypes.items()},
        "file_type": "sample",
        "load_warnings": [],
    }

df = st.session_state.df
render_hero(dataset_loaded=df is not None, provider=provider)

tab_names = [
    "Upload & Overview",
    "Data Quality",
    "Executive Dashboard",
    "Auto EDA",
    "Profit Intelligence",
    "Root Cause Analysis",
    "Customer Intelligence",
    "Product Intelligence",
    "Hidden Insights",
    "Ask AI Analyst",
    "Forecasts & Alerts",
    "Scenario Simulator",
    "Reports",
    "BI Dashboard",
]
tabs = st.tabs(tab_names)

if df is None:
    with tabs[0]:
        st.info("Please upload a dataset first, or click Load Sample Sales Dataset in the sidebar.")
        section_header("Start with your sales dataset", "Upload your company CSV/Excel file or use the sample dataset for a safe demo.")
        c1, c2, c3 = st.columns(3)
        with c1:
            modern_card("1. Upload data", "Use the sidebar uploader for CSV, XLSX, or XLS files. The app detects sales, profit, customer, product, and date fields.", "Step 1")
        with c2:
            modern_card("2. Review AI findings", "The system calculates KPIs, profit drivers, root causes, loyalty, risks, and recommendations using Pandas evidence.", "Step 2")
        with c3:
            modern_card("3. Export reports", "Download Excel/PDF reports and a Power BI-ready star schema pack for formal dashboarding.", "Step 3")
        st.markdown("### Manager demo questions")
        st.code("""Why is profit low?\nWhy are Tables losing money?\nIs discount hurting profit?\nWho are our most loyal customers?\nWhich region should we fix first?\nGive me a board-level summary.""")
    st.stop()

# Column detection and optional manual override.
det = _cached_detect(df)
roles = det["roles"].copy()
IMPORTANT_ROLE_ORDER = [
    "sales_column",
    "profit_column",
    "cost_column",
    "margin_percent_column",
    "date_column",
    "ship_date_column",
    "order_id_column",
    "customer_column",
    "customer_id_column",
    "product_column",
    "category_column",
    "subcategory_column",
    "region_column",
    "state_column",
    "city_column",
    "country_column",
    "segment_column",
    "discount_column",
    "quantity_column",
    "unit_price_column",
]
with st.sidebar.expander("Detected columns / override", expanded=False):
    st.caption("Set or correct any business role manually. Leave blank if the dataset does not have that field.")
    available = [None] + list(df.columns)
    all_roles = IMPORTANT_ROLE_ORDER + sorted(r for r in roles if r not in IMPORTANT_ROLE_ORDER and not str(r).startswith("_"))
    for role in all_roles:
        default_col = roles.get(role)
        idx = available.index(default_col) if default_col in available else 0
        selected = st.selectbox(role, available, index=idx, key=f"role_{role}")
        if selected:
            roles[role] = selected
        else:
            roles.pop(role, None)

# Preserve original uploaded-file metadata for Ask AI Analyst sanity checks.
# The analysis dataframe can contain cleaned/derived columns, but questions like
# row count, original columns, null values, and duplicates must be answered from
# the uploaded dataframe, not guessed from business diagnostics.
raw_question_roles = dict(roles)
raw_question_roles["_raw_profile"] = {
    "row_count": int(len(df)),
    "column_count": int(len(df.columns)),
    "columns": [str(c) for c in df.columns],
    "missing_total": int(df.isna().sum().sum()),
    "missing_by_column": {str(k): int(v) for k, v in df.isna().sum().to_dict().items()},
    "duplicate_rows": int(df.duplicated().sum()),
    "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
}

df_analysis, roles = prepare_analysis_dataset(df, roles)
# Keep raw uploaded metadata attached to prepared roles for reports and Q&A.
roles["_raw_profile"] = raw_question_roles["_raw_profile"]
ctx = _run_core(df_analysis, roles)
semantic = ctx["semantic_model"]
kpis = ctx["kpis"]

st.sidebar.success("Dataset loaded")
st.sidebar.write(f"Rows: {len(df):,}")
st.sidebar.write(f"Columns: {df.shape[1]:,}")
st.sidebar.write(f"Dataset type: {semantic['dataset_type']}")
st.sidebar.write(f"Available analyses: {len(semantic.get('available_analyses', []))}")
if roles.get("_profit_derivation_note"):
    st.sidebar.info(roles["_profit_derivation_note"])
if isinstance(roles.get("_dataset_grain"), dict):
    st.sidebar.caption(f"Count basis: {roles['_dataset_grain'].get('count_label', 'Rows')} ({roles['_dataset_grain'].get('reason', 'detected')})")
if _cleaned_dataset_used(roles):
    st.sidebar.warning("Cleaned dataset used: missing values were treated before analysis.")
if kpis.get("total_sales") is not None:
    st.sidebar.metric("Sales", format_currency(kpis["total_sales"]))
if kpis.get("total_profit") is not None:
    st.sidebar.metric("Profit", format_currency(kpis["total_profit"]))
if kpis.get("profit_margin_percent") is not None:
    st.sidebar.metric("Margin", format_percent(kpis["profit_margin_percent"]))

with tabs[0]:
    section_header("Executive Summary", "A fast read of what the uploaded dataset can support.")
    _render_cleaned_data_warning(roles, compact=False)
    summary = ctx["executive_summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Detected Type", semantic["dataset_type"].replace("_", " ").title())
    c4.metric("Data Quality", ctx["data_quality"].get("quality_score"))
    modern_card("Manager-ready summary", executive_summary_to_html(summary), "Executive Readout")
    section_header("Dataset Preview")
    st.caption("Original uploaded dataset preview.")
    st.dataframe(df.head(50), use_container_width=True)
    if _cleaned_dataset_used(roles):
        with st.expander("Cleaned dataset preview used for calculations", expanded=False):
            st.dataframe(df_analysis.head(50), use_container_width=True)
    section_header("Detected Columns")
    det_rows = [
        {"role": k, "column": v, "confidence": det["confidence"].get(k, {}).get("confidence")}
        for k, v in roles.items() if not str(k).startswith("_")
    ]
    st.dataframe(pd.DataFrame(det_rows), use_container_width=True)
    section_header("Available Analyses")
    st.write(", ".join(semantic.get("available_analyses", [])) or "No advanced analyses detected.")

with tabs[1]:
    section_header("Data Quality", "Technical trust checks only: missing values, duplicates, invalid dates, schema issues, and suspicious values.")
    _render_cleaned_data_warning(roles, compact=False)
    dq = ctx["data_quality"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Quality Score", dq.get("quality_score"))
    c2.metric("Duplicate Rows", dq.get("basic_checks", {}).get("duplicate_rows_count"))
    c3.metric("Missing Cells %", format_percent(dq.get("basic_checks", {}).get("missing_cells_percent")))

    st.caption("Negative profit is a business-performance signal, not a data-quality error. Loss-making records are routed to Profit Intelligence.")

    for w in dq.get("warnings", []):
        if "No major" in str(w):
            st.success(w)
        else:
            st.warning(w)

    business_signals = dq.get("business_signals", [])
    if business_signals:
        st.markdown("### Business Signals Routed to Profit Intelligence")
        for signal in business_signals:
            st.info(signal)
        pi_summary = ctx.get("profit_intelligence", {}).get("profit_summary", {})
        if pi_summary:
            b1, b2, b3 = st.columns(3)
            b1.metric("Loss-Making Orders", pi_summary.get("loss_making_orders"))
            b2.metric("Total Loss Amount", format_currency(pi_summary.get("total_loss_amount")))
            b3.metric("Overall Margin", format_percent(pi_summary.get("overall_margin")))
            st.caption("Open the Profit Intelligence tab for loss drivers, discount impact, leakage points, and recommended actions.")

    treatment_frame = _missing_treatment_frame(roles)
    if not treatment_frame.empty:
        st.markdown("### Missing Value Treatment Applied")
        st.caption("This log explains what was filled, derived, excluded, or left missing before analysis.")
        st.dataframe(treatment_frame, use_container_width=True, hide_index=True)

    with st.expander("Why is the quality score this number?", expanded=False):
        explanation = dq.get("score_explanation", {})
        st.write(explanation.get("note", ""))
        st.dataframe(pd.DataFrame(explanation.items(), columns=["Score Component", "Value"]), use_container_width=True, hide_index=True)

    _dataframe_section("Basic Checks", pd.DataFrame(dq["basic_checks"].items(), columns=["Check", "Value"]), 200)
    _dataframe_section("Technical Outliers", dq.get("outlier_summary", pd.DataFrame()), 100)
    _dataframe_section("Business Metric Outliers — reviewed in EDA/Profit Intelligence", dq.get("business_outlier_summary", pd.DataFrame()), 100)

with tabs[2]:
    section_header("Executive Dashboard", "Core sales, profit, margin, category, region, and trend performance.")
    _render_cleaned_data_warning(roles)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales", format_currency(kpis.get("total_sales")))
    c2.metric("Total Profit", format_currency(kpis.get("total_profit")))
    c3.metric("Margin", format_percent(kpis.get("profit_margin_percent")))
    c4.metric("Orders", kpis.get("total_orders"))
    cat = performance_by_category(df_analysis, roles)
    if not cat.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(cat, x="dimension_value", y="sales", title="Sales by Category"), use_container_width=True)
        if "profit" in cat:
            with c2:
                st.plotly_chart(px.bar(cat, x="dimension_value", y="profit", title="Profit by Category"), use_container_width=True)
    reg = performance_by_region(df_analysis, roles)
    if not reg.empty:
        st.plotly_chart(px.bar(reg, x="dimension_value", y="profit" if "profit" in reg else "sales", title="Region Performance"), use_container_width=True)
    trend = performance_over_time(df_analysis, roles)
    if not trend.empty:
        y_cols = [c for c in ["sales", "profit"] if c in trend.columns]
        st.plotly_chart(px.line(trend, x="period", y=y_cols, title="Sales/Profit Trend"), use_container_width=True)

with tabs[3]:
    section_header("Auto EDA", "Automated profiling, correlations, and observations.")
    _render_cleaned_data_warning(roles)
    eda = generate_eda_summary(df_analysis, roles)
    _render_key_value_summary("Dataset Overview", eda.get("dataset_overview", {}))
    _dataframe_section("Numeric Summary", eda["numeric_summary"], 100)
    _dataframe_section("Categorical Summary", eda["categorical_summary"], 100)
    if not eda["correlation_matrix"].empty:
        st.plotly_chart(px.imshow(eda["correlation_matrix"], text_auto=True, title="Correlation Matrix"), use_container_width=True)
    for obs in eda.get("observations", []):
        st.info(obs)

with tabs[4]:
    section_header("Profit Intelligence", "Profit conversion, leakage, discount impact, and deterministic root causes.")
    _render_cleaned_data_warning(roles)
    pi = ctx["profit_intelligence"]
    ps = pi.get("profit_summary", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sales", format_currency(ps.get("total_sales")))
    c2.metric("Profit", format_currency(ps.get("total_profit")))
    c3.metric("Margin", format_percent(ps.get("overall_margin")))
    c4.metric("Loss Orders", ps.get("loss_making_orders"))
    for cause in pi.get("root_cause_summary", []):
        st.warning(cause)
    for action in pi.get("recommended_actions", []):
        st.success(action)
    _dataframe_section("Profit Leakage Points", pi.get("profit_leakage_points"), 100)
    rel = pi.get("profit_relationships", {})
    band = rel.get("profit_by_discount_band") if isinstance(rel, dict) else None
    _dataframe_section("Profit by Discount Band", band, 100)

with tabs[5]:
    section_header("Root Cause Analysis", "Where the loss is concentrated and which combinations need action.")
    _render_cleaned_data_warning(roles)
    loss = find_loss_drivers(df_analysis, roles)
    c1, c2 = st.columns(2)
    c1.metric("Total Loss Amount", format_currency(loss.get("total_loss_amount")))
    c2.metric("Loss Orders", loss.get("total_loss_orders"))
    for key, val in loss.items():
        if isinstance(val, pd.DataFrame) and not val.empty:
            _dataframe_section(key.replace("_", " ").title(), val, 50)

with tabs[6]:
    section_header("Customer Intelligence", "Loyalty, profitability, at-risk customers, and retention priorities.")
    _render_cleaned_data_warning(roles)
    ci = ctx["customer_intelligence"]
    _render_key_value_summary("Customer Summary", ci.get("customer_summary", {}))
    for key in ["top_loyal_customers", "loyal_profitable_customers", "loyal_low_profit_customers", "at_risk_customers", "unprofitable_customers"]:
        _dataframe_section(key.replace("_", " ").title(), ci.get(key), 30)

with tabs[7]:
    section_header("Product Intelligence", "Portfolio groups: stars, volume drivers, niche winners, and problem products.")
    _render_cleaned_data_warning(roles)
    prod = ctx["product_intelligence"]
    for key in ["stars", "volume_drivers", "niche_winners", "problem_products", "discount_damaged_products", "category_summary", "subcategory_summary"]:
        _dataframe_section(key.replace("_", " ").title(), prod.get(key), 50)

with tabs[8]:
    section_header("Hidden Insights", "Deduplicated senior analyst observations with evidence and actions.")
    _render_cleaned_data_warning(roles)
    if not ctx["hidden_insights"]:
        st.info("No major hidden insights detected from the available fields.")
    for ins in ctx["hidden_insights"]:
        insight_card(
            ins.get("severity", "Medium"),
            ins.get("title", "Insight"),
            ins.get("evidence", []),
            ins.get("why_it_matters", "Review this area."),
            ins.get("recommended_action", "Investigate with business owners."),
        )

with tabs[9]:
    section_header("Ask AI Analyst", "Questions are answered from computed evidence first. OpenAI/Ollama can only improve wording.")
    _render_cleaned_data_warning(roles)
    st.markdown('<div class="qa-shell">', unsafe_allow_html=True)
    q = st.text_input("Ask a manager question", value="Why is profit low?")
    c1, c2, c3 = st.columns([1, 1, 2])
    ask_clicked = c1.button("Ask AI Analyst", use_container_width=True)
    c2.caption(f"LLM Provider: {provider or 'none'}")
    st.markdown('</div>', unsafe_allow_html=True)
    if ask_clicked:
        ans = answer_business_question(q, df, raw_question_roles, ctx)
        st.markdown(ans["markdown_answer"])
        first_evidence = next((t for t in ans.get("evidence_tables", {}).values() if isinstance(t, pd.DataFrame) and not t.empty), None)
        if first_evidence is not None and str(ans.get("intent", "")).startswith("dynamic"):
            st.subheader("Interactive Evidence Table")
            st.dataframe(first_evidence.head(100), use_container_width=True)
        with st.expander("LLM status"):
            _render_status_table("LLM Status", ans.get("llm_meta", {}))
        for name in list(ans["evidence_tables"].keys())[:12]:
            table = ans["evidence_tables"][name]
            if isinstance(table, pd.DataFrame) and not table.empty:
                with st.expander(name):
                    st.dataframe(table.head(50), use_container_width=True)
    with st.expander("Suggested demo questions"):
        st.code("""Why is profit low?\nWhy are Tables losing money?\nIs discount hurting profit?\nWho are our most loyal customers?\nWhich products should we stop selling?\nWhich region should we fix first?\nGive me a board-level summary.""")

with tabs[10]:
    section_header("Forecasts & Alerts", "Conservative forecasts and anomaly checks.")
    _render_cleaned_data_warning(roles)
    forecast = ctx["forecasting"]
    fdf = forecast.get("forecast", pd.DataFrame())
    if isinstance(fdf, pd.DataFrame) and not fdf.empty:
        _dataframe_section("Forecast", fdf, 50)
        _render_backtest_summary(forecast.get("backtest", {}))
    else:
        st.warning("Forecast unavailable: " + " | ".join(forecast.get("limitations", [])))
    anomalies = ctx["anomaly_detection"]
    st.subheader("Anomalies")
    anomaly_items = anomalies.get("anomalies", []) if isinstance(anomalies, dict) else []
    if not anomaly_items:
        st.info("No major anomalies were detected in the selected data.")
    for a in anomaly_items:
        st.warning(f"**{a['severity']} — {a['title']}** — {a['explanation']} Recommended check: {a['recommended_check']}")
    _dataframe_section("Unusual Rows", anomalies.get("unusual_rows", pd.DataFrame()), 100)

with tabs[11]:
    section_header("Scenario Simulator", "Estimate impact. These are assumptions, not guaranteed forecasts.")
    _render_cleaned_data_warning(roles)
    scenario = st.selectbox("Choose scenario", ["Discount reduction", "Price increase", "Remove loss-making records", "Quantity change", "Discount cap", "Margin improvement"])
    result = None
    if scenario == "Discount reduction":
        pct = st.slider("Discount reduction percent", 1, 50, 10)
        result = simulate_discount_reduction(df_analysis, roles, pct)
    elif scenario == "Price increase":
        pct = st.slider("Price increase percent", 1, 30, 5)
        result = simulate_price_increase(df_analysis, roles, pct)
    elif scenario == "Remove loss-making records":
        result = simulate_removing_loss_making_products(df_analysis, roles)
    elif scenario == "Quantity change":
        pct = st.slider("Quantity change percent", -50, 50, 10)
        result = simulate_quantity_change(df_analysis, roles, pct)
    elif scenario == "Discount cap":
        cap = st.slider("Max discount percent", 0, 80, 30)
        result = simulate_discount_cap(df_analysis, roles, cap)
    elif scenario == "Margin improvement":
        dim = st.selectbox("Dimension", list(df.columns))
        values = list(df[dim].dropna().astype(str).unique())[:500]
        val = st.selectbox("Value", values) if values else None
        pct = st.slider("Margin improvement percentage points", 1, 30, 5)
        if val is not None:
            result = simulate_margin_improvement(df_analysis, roles, dim, val, pct)
    if result:
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline Profit", format_currency(result.get("baseline_profit")))
        c2.metric("Simulated Profit", format_currency(result.get("simulated_profit")))
        c3.metric("Estimated Profit Change", format_currency(result.get("estimated_profit_change")))
        _render_scenario_result(result)

with tabs[12]:
    section_header("Reports", "Download manager-ready Excel and PDF reports.")
    _render_cleaned_data_warning(roles, compact=False)
    modern_card("Executive summary preview", executive_summary_to_html(ctx["executive_summary"]), "Report Preview")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download Excel Report", generate_excel_report(df_analysis, roles, ctx), "ai_sales_report.xlsx", use_container_width=True)
    with c2:
        st.download_button("Download PDF Report", generate_pdf_report(ctx), "ai_sales_report.pdf", use_container_width=True)

with tabs[13]:
    section_header("BI Dashboard", "A Power BI-style leadership dashboard built directly inside this project.")
    _render_cleaned_data_warning(roles)
    embed_status = powerbi_embed_status()
    c1, c2, c3 = st.columns(3)
    c1.metric("Dashboard Mode", "In-App" if not embed_status["powerbi_embed_configured"] else "Power BI Embed + In-App")
    c2.metric("Manual Setup", "Not Required")
    c3.metric("Power BI Pack", "Available")

    render_embedded_bi_dashboard(df_analysis, roles, theme_mode)

    with st.expander("Optional: export formal Power BI Desktop pack"):
        st.caption("The dashboard above works directly inside this app. This export is only for users who also want to build a separate .pbix file.")
        summary = powerbi_pack_summary(df_analysis, roles)
        st.markdown('<div class="powerbi-card">', unsafe_allow_html=True)
        p1, p2, p3 = st.columns(3)
        p1.metric("Model Tables", len(summary.get("tables", {})))
        p2.metric("Relationships", summary.get("relationship_count"))
        p3.metric("Dashboard Pages", len(summary.get("report_pages", [])))
        st.markdown('</div>', unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame([
                {"table": name, "rows": meta["rows"], "columns": meta["columns"]}
                for name, meta in summary.get("tables", {}).items()
            ]),
            use_container_width=True,
        )
        st.download_button(
            "Download Optional Power BI Desktop Pack",
            generate_powerbi_export_pack(df_analysis, roles),
            "power_bi_dashboard_pack.zip",
            mime="application/zip",
            use_container_width=True,
        )
