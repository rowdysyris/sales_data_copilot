from pathlib import Path

from src.ui_theme import asset_data_uri
from src.llm_explainer import beautify_answer_with_llm


def test_hero_asset_exists_and_encodes():
    assert Path("assets/hero_sales_copilot.svg").exists()
    uri = asset_data_uri("assets/hero_sales_copilot.svg")
    assert uri.startswith("data:image/svg+xml;base64,")


def test_llm_none_fallback(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "none")
    answer, meta = beautify_answer_with_llm("## Direct Answer\nProfit is low.", {"confidence": "High"})
    assert "Profit is low" in answer
    assert meta["fallback_used"] is True
    assert meta["provider"] == "none"


def test_theme_function_accepts_light_and_dark(monkeypatch):
    import src.ui_theme as ui_theme

    calls = []
    monkeypatch.setattr(ui_theme.st, "markdown", lambda content, unsafe_allow_html=False: calls.append(content))
    ui_theme.apply_maven_style_theme("Light")
    ui_theme.apply_maven_style_theme("Dark")
    joined = "\n".join(calls)
    assert ".theme-chip::after { content: 'Light'; }" in joined
    assert ".theme-chip::after { content: 'Dark'; }" in joined
    assert "--panel: #0f2138" in joined


def test_app_has_theme_toggle():
    app_source = Path("app.py").read_text(encoding="utf-8")
    assert "Appearance" in app_source
    assert "st.sidebar.radio" in app_source
    assert "apply_maven_style_theme(theme_mode)" in app_source
