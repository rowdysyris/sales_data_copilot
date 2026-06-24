from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets"


def _light_theme_overrides() -> str:
    return """
    .theme-chip::after { content: 'Light'; }
    """


def _dark_theme_overrides() -> str:
    return """
    :root {
        --navy: #f8fbff;
        --navy-2: #dbeafe;
        --ink: #eaf2ff;
        --muted: #9fb1c9;
        --line: rgba(148,163,184,.24);
        --panel: #0f2138;
        --soft: #091a2e;
        --sky: #0e2943;
        --accent: #35e2cf;
        --accent-2: #ffbf5a;
        --blue: #60a5fa;
        --danger: #fb7185;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(53,226,207,.13), transparent 32rem),
            radial-gradient(circle at top right, rgba(96,165,250,.10), transparent 30rem),
            linear-gradient(180deg, #061225 0%, #08172a 46%, #071322 100%) !important;
        color: var(--ink) !important;
    }

    .block-container { color: var(--ink) !important; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #030b18 0%, #06162a 100%) !important;
        border-right: 1px solid rgba(148,163,184,.16) !important;
    }

    .app-hero {
        background: linear-gradient(135deg, #020a16 0%, #06213c 47%, #0b4b5c 100%) !important;
        box-shadow: 0 28px 70px rgba(0, 0, 0, 0.36) !important;
    }

    .modern-card, .insight-card, .qa-shell, div[data-testid="stMetric"] {
        background: rgba(15,33,56,.92) !important;
        border-color: rgba(148,163,184,.24) !important;
        box-shadow: 0 18px 44px rgba(0, 0, 0, .22) !important;
        color: var(--ink) !important;
    }

    .powerbi-card {
        background: linear-gradient(135deg, rgba(80,57,15,.85), rgba(15,33,56,.94) 52%, rgba(7,39,49,.9)) !important;
        border-color: rgba(255,191,90,.25) !important;
        color: var(--ink) !important;
    }

    .section-title h2, .section-title h3, .insight-title, .card-title, [data-testid="stMetricValue"] {
        color: #f8fbff !important;
    }

    .small-muted, .insight-meta, [data-testid="stMetricLabel"] {
        color: #9fb1c9 !important;
    }

    .card-badge {
        color: #ccfff6 !important;
        background: rgba(53,226,207,.12) !important;
        border-color: rgba(53,226,207,.30) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        border-bottom-color: rgba(148,163,184,.24) !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #cbd5e1 !important;
    }

    .stTabs [aria-selected="true"] {
        background: #35e2cf !important;
        color: #04111f !important;
    }

    div[data-testid="stDataFrame"], .stDataFrame {
        border: 1px solid rgba(148,163,184,.20) !important;
    }

    .stMarkdown, .stText, p, li, span, label {
        color: inherit;
    }

    .theme-chip::after { content: 'Dark'; }
    """


def asset_data_uri(path: str) -> str:
    file_path = ROOT / path
    if not file_path.exists():
        return ""
    mime = "image/svg+xml" if file_path.suffix.lower() == ".svg" else "image/png"
    encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def apply_maven_style_theme(mode: str = "Light") -> None:
    """Apply a modern analytics-academy-inspired theme to the Streamlit app.

    The theme is controlled inside the app so users can switch between a
    bright manager-demo view and a dark executive dashboard view without
    editing .streamlit/config.toml or restarting the app.
    """
    mode_normalized = (mode or "Light").strip().lower()
    is_dark = mode_normalized == "dark"
    theme_css = _dark_theme_overrides() if is_dark else _light_theme_overrides()
    st.markdown(
        ("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    :root {
        --navy: #06152b;
        --navy-2: #0b1f3a;
        --ink: #0f172a;
        --muted: #64748b;
        --line: #e2e8f0;
        --panel: #ffffff;
        --soft: #f8fafc;
        --sky: #e8f4ff;
        --accent: #1db7a6;
        --accent-2: #ffb23f;
        --blue: #2563eb;
        --danger: #ef4444;
        --radius: 22px;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(29,183,166,.16), transparent 32rem),
            radial-gradient(circle at top right, rgba(37,99,235,.12), transparent 30rem),
            linear-gradient(180deg, #f8fbff 0%, #ffffff 42%, #f8fafc 100%);
        color: var(--ink);
    }

    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071a33 0%, #0b2140 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }

    section[data-testid="stSidebar"] * {
        color: #eaf2ff !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, var(--accent), #48d3c4);
        color: #04111f !important;
        border: 0;
        font-weight: 800;
        border-radius: 999px;
        box-shadow: 0 12px 28px rgba(29,183,166,.22);
    }

    .app-hero {
        position: relative;
        overflow: hidden;
        border-radius: 30px;
        background: linear-gradient(135deg, #071a33 0%, #0a2a4f 47%, #0d3f5e 100%);
        color: white;
        padding: 34px 36px;
        box-shadow: 0 28px 70px rgba(6, 21, 43, 0.24);
        border: 1px solid rgba(255,255,255,.12);
        margin: .3rem 0 1.3rem 0;
    }

    .app-hero::before {
        content: "";
        position: absolute;
        inset: -30% -10% auto auto;
        width: 520px;
        height: 520px;
        background: radial-gradient(circle, rgba(29,183,166,.45), rgba(29,183,166,.04) 58%, transparent 68%);
        transform: rotate(20deg);
    }

    .hero-grid {
        position: relative;
        z-index: 2;
        display: grid;
        grid-template-columns: minmax(0, 1.15fr) minmax(360px, .85fr);
        gap: 28px;
        align-items: center;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,.11);
        color: #dffcf7;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        border: 1px solid rgba(255,255,255,.14);
    }

    .hero-title {
        margin: 18px 0 10px 0;
        font-size: clamp(2.25rem, 4.2vw, 4.8rem);
        line-height: .96;
        letter-spacing: -.065em;
        font-weight: 900;
        color: #ffffff;
    }

    .hero-title span {
        color: #72f5df;
    }

    .hero-copy {
        color: #cbdaf0;
        max-width: 740px;
        font-size: 1.05rem;
        line-height: 1.7;
        margin: 0 0 22px 0;
    }

    .hero-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
    }

    .hero-pill {
        padding: 9px 13px;
        border: 1px solid rgba(255,255,255,.16);
        background: rgba(255,255,255,.10);
        border-radius: 999px;
        color: #edf9ff;
        font-weight: 700;
        font-size: .86rem;
    }

    .hero-visual {
        position: relative;
        min-height: 265px;
        border-radius: 26px;
        background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.14);
        backdrop-filter: blur(8px);
        padding: 18px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.16);
    }

    .hero-visual img {
        width: 100%;
        height: auto;
        display: block;
        border-radius: 22px;
    }

    .section-title {
        display: flex;
        align-items: center;
        gap: .75rem;
        margin: 1.2rem 0 .85rem 0;
    }

    .section-kicker {
        height: 12px;
        width: 12px;
        border-radius: 99px;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        box-shadow: 0 0 0 7px rgba(29,183,166,.13);
    }

    .section-title h2, .section-title h3 {
        margin: 0;
        color: var(--navy);
        letter-spacing: -.035em;
        font-weight: 900;
    }

    .modern-card {
        background: rgba(255,255,255,.92);
        border: 1px solid rgba(226,232,240,.95);
        border-radius: var(--radius);
        padding: 1.15rem 1.25rem;
        box-shadow: 0 16px 42px rgba(15, 23, 42, 0.07);
        margin: .75rem 0;
    }

    .card-badge {
        display: inline-flex;
        padding: 8px 12px;
        border-radius: 999px;
        background: #e7fbf7;
        color: #0b3b34;
        border: 1px solid #bbefe4;
        margin-bottom: .8rem;
        font-size: .78rem;
        font-weight: 900;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .card-title {
        font-weight: 900;
        color: var(--navy);
        font-size: 1.1rem;
        margin-bottom: .35rem;
    }

    .insight-card {
        border-radius: 22px;
        padding: 1.1rem 1.25rem;
        background: #ffffff;
        border: 1px solid #e5edf6;
        box-shadow: 0 14px 34px rgba(15,23,42,.07);
        margin-bottom: 1rem;
    }

    .insight-card.high { border-left: 6px solid #ef4444; }
    .insight-card.medium { border-left: 6px solid #ffb23f; }
    .insight-card.low { border-left: 6px solid #1db7a6; }

    .insight-title {
        color: var(--navy);
        font-weight: 900;
        font-size: 1.05rem;
        margin-bottom: .65rem;
    }

    .insight-meta {
        color: var(--muted);
        font-size: .9rem;
        line-height: 1.55;
    }

    .qa-shell {
        border-radius: 28px;
        padding: 1.25rem;
        background: linear-gradient(180deg, #ffffff, #f8fbff);
        border: 1px solid #e2e8f0;
        box-shadow: 0 20px 55px rgba(15,23,42,.08);
    }

    .powerbi-card {
        border-radius: 26px;
        padding: 1.2rem;
        background: linear-gradient(135deg, #fff7e6, #ffffff 48%, #effbff);
        border: 1px solid #f2dfb9;
        box-shadow: 0 18px 45px rgba(121, 82, 17, .08);
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 14px 16px;
        box-shadow: 0 12px 30px rgba(15,23,42,.06);
    }

    [data-testid="stMetricLabel"] {
        color: #64748b;
        font-weight: 800;
    }

    [data-testid="stMetricValue"] {
        color: #06152b;
        font-size: 1.52rem;
        font-weight: 900;
        letter-spacing: -.04em;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: .45rem;
        border-bottom: 1px solid #e2e8f0;
        overflow-x: auto;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 999px 999px 0 0;
        padding: 0 16px;
        font-weight: 800;
        color: #334155;
    }

    .stTabs [aria-selected="true"] {
        background: #06152b !important;
        color: #ffffff !important;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 999px;
        border: 0;
        background: linear-gradient(135deg, #1db7a6, #41d5c1);
        color: #04111f;
        font-weight: 900;
        padding: .68rem 1.1rem;
        box-shadow: 0 14px 28px rgba(29,183,166,.20);
    }

    .stDataFrame, div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
    }

    .small-muted {
        color: #64748b;
        font-size: .92rem;
        line-height: 1.65;
    }

    @media (max-width: 980px) {
        .hero-grid { grid-template-columns: 1fr; }
        .hero-visual { min-height: auto; }
    }

    {theme_css}
</style>
        """).replace("{theme_css}", theme_css),
        unsafe_allow_html=True,
    )


def render_hero(dataset_loaded: bool, provider: str, hero_asset: str = "assets/hero_sales_copilot.svg") -> None:
    hero_uri = asset_data_uri(hero_asset)
    status = "Dataset loaded" if dataset_loaded else "Upload your sales dataset"
    provider_text = provider.upper() if provider and provider.lower() != "none" else "Deterministic AI mode"
    img_html = f'<img src="{hero_uri}" alt="AI Sales Analyst dashboard illustration" />' if hero_uri else ""
    st.markdown(
        f"""
<section class="app-hero">
  <div class="hero-grid">
    <div>
      <div class="eyebrow">Senior Analyst Copilot • Manager Demo Ready</div>
      <h1 class="hero-title">Turn sales data into <span>board-ready decisions.</span></h1>
      <p class="hero-copy">
        Upload a CSV or Excel dataset and get KPI diagnostics, profit drivers, loss root causes,
        customer loyalty, product portfolio insights, Power BI exports, and evidence-first Q&A.
      </p>
      <div class="hero-pill-row">
        <div class="hero-pill">{status}</div>
        <div class="hero-pill">{provider_text}</div>
        <div class="hero-pill">Pandas evidence engine</div>
        <div class="hero-pill">Power BI-ready star schema</div>
      </div>
    </div>
    <div class="hero-visual">{img_html}</div>
  </div>
</section>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str | None = None) -> None:
    sub = f'<div class="small-muted">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
<div class="section-title">
  <div class="section-kicker"></div>
  <div><h2>{title}</h2>{sub}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def modern_card(title: str, body: str, badge: str | None = None) -> None:
    badge_html = f'<div class="card-badge">{badge}</div>' if badge else ""
    st.markdown(
        f"""
<div class="modern-card">
  {badge_html}
  <div class="card-title">{title}</div>
  <div class="small-muted">{body}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(severity: str, title: str, evidence: list[Any] | None, why: str, action: str) -> None:
    sev = (severity or "medium").lower()
    evidence_items = "".join(f"<li>{str(item)}</li>" for item in (evidence or ["No supporting evidence available."]))
    st.markdown(
        f"""
<div class="insight-card {sev}">
  <div class="insight-title">{severity} — {title}</div>
  <div class="insight-meta"><strong>Evidence</strong><ul>{evidence_items}</ul></div>
  <div class="insight-meta"><strong>Why it matters:</strong> {why}</div>
  <div class="insight-meta"><strong>Action:</strong> {action}</div>
</div>
        """,
        unsafe_allow_html=True,
    )
