from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.utils import get_col, safe_numeric, profit_margin, format_currency, format_percent, ensure_datetime


DASHBOARD_PAGES = [
    "Executive Overview",
    "Profit Intelligence",
    "Product Portfolio",
    "Customer Intelligence",
    "Regional Performance",
    "Discount & Margin Control",
]


def _existing_role(column_roles: dict, role: str, df: pd.DataFrame) -> str | None:
    col = get_col(column_roles, role)
    return col if col and col in df.columns else None


def _discount_band(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return "Unknown"
    if pd.isna(v):
        return "Unknown"
    if v <= 1:
        v *= 100
    if v <= 0:
        return "0%"
    if v <= 10:
        return "0-10%"
    if v <= 20:
        return "10-20%"
    if v <= 30:
        return "20-30%"
    if v <= 40:
        return "30-40%"
    return "40%+"


def _plot_template(theme_mode: str = "Light") -> str:
    return "plotly_dark" if str(theme_mode).lower() == "dark" else "plotly_white"


def _chart(fig: go.Figure, theme_mode: str = "Light") -> go.Figure:
    """Apply consistent dashboard styling to Plotly figures."""
    is_dark = str(theme_mode).lower() == "dark"
    fig.update_layout(
        template=_plot_template(theme_mode),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=55, b=25),
        height=420,
        font=dict(size=13, color="#EAF2FF" if is_dark else "#10233F"),
        title=dict(font=dict(size=18)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.22)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.22)")
    return fig


def _top_values(df: pd.DataFrame, col: str | None, limit: int = 60) -> list[str]:
    if not col or col not in df.columns:
        return []
    values = df[col].dropna().astype(str).value_counts().head(limit).index.tolist()
    return values


def get_available_dashboard_pages(df: pd.DataFrame, column_roles: dict) -> list[str]:
    """Return BI pages that can be rendered with the available columns."""
    if df is None or df.empty:
        return []
    pages = ["Executive Overview"]
    if _existing_role(column_roles, "profit_column", df):
        pages.append("Profit Intelligence")
    if _existing_role(column_roles, "product_column", df) or _existing_role(column_roles, "category_column", df):
        pages.append("Product Portfolio")
    if _existing_role(column_roles, "customer_column", df) or _existing_role(column_roles, "customer_id_column", df):
        pages.append("Customer Intelligence")
    if _existing_role(column_roles, "region_column", df) or _existing_role(column_roles, "state_column", df) or _existing_role(column_roles, "city_column", df):
        pages.append("Regional Performance")
    if _existing_role(column_roles, "discount_column", df) and _existing_role(column_roles, "profit_column", df):
        pages.append("Discount & Margin Control")
    return [p for p in DASHBOARD_PAGES if p in pages]


def prepare_bi_dataframe(df: pd.DataFrame, column_roles: dict) -> pd.DataFrame:
    """Create a normalized BI-ready dataframe for in-app dashboard visuals."""
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()
    sales_col = _existing_role(column_roles, "sales_column", work)
    profit_col = _existing_role(column_roles, "profit_column", work)
    qty_col = _existing_role(column_roles, "quantity_column", work)
    discount_col = _existing_role(column_roles, "discount_column", work)
    date_col = _existing_role(column_roles, "date_column", work)
    order_col = _existing_role(column_roles, "order_id_column", work)

    work["BI Sales"] = safe_numeric(work[sales_col]) if sales_col else 0.0
    work["BI Profit"] = safe_numeric(work[profit_col]) if profit_col else 0.0
    work["BI Quantity"] = safe_numeric(work[qty_col]) if qty_col else 0.0
    work["BI Discount"] = safe_numeric(work[discount_col]) if discount_col else 0.0
    work["BI Cost"] = work["BI Sales"] - work["BI Profit"]
    work["BI Margin %"] = work.apply(lambda r: profit_margin(r["BI Profit"], r["BI Sales"]), axis=1)
    work["BI Loss Amount"] = work["BI Profit"].where(work["BI Profit"] < 0, 0.0)
    work["BI Is Loss Order"] = work["BI Profit"] < 0
    work["BI Discount Band"] = work["BI Discount"].apply(_discount_band)
    work["BI Order Key"] = work[order_col].astype(str) if order_col else pd.Series([f"ROW-{i+1}" for i in range(len(work))], index=work.index)

    if date_col:
        work["BI Date"] = ensure_datetime(work[date_col])
        work["BI Year Month"] = work["BI Date"].dt.to_period("M").astype(str).replace("NaT", "Unknown")
        work["BI Year"] = work["BI Date"].dt.year
        work["BI Quarter"] = "Q" + work["BI Date"].dt.quarter.astype("Int64").astype(str)
    else:
        work["BI Date"] = pd.NaT
        work["BI Year Month"] = "Unknown"
        work["BI Year"] = pd.NA
        work["BI Quarter"] = "Unknown"
    return work


def apply_dashboard_filters(work: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """Apply interactive BI filters. Extracted for testability."""
    if work is None or work.empty:
        return pd.DataFrame()
    filtered = work.copy()

    date_range = filters.get("date_range")
    if date_range and "BI Date" in filtered.columns and len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered = filtered[(filtered["BI Date"].isna()) | ((filtered["BI Date"] >= start) & (filtered["BI Date"] <= end))]

    for col, selected in filters.get("column_filters", {}).items():
        if col in filtered.columns and selected:
            selected_str = {str(v) for v in selected}
            filtered = filtered[filtered[col].astype(str).isin(selected_str)]

    if filters.get("loss_only"):
        filtered = filtered[filtered.get("BI Is Loss Order", False) == True]

    min_sales = filters.get("min_sales")
    if min_sales not in (None, 0) and "BI Sales" in filtered.columns:
        filtered = filtered[filtered["BI Sales"] >= float(min_sales)]

    return filtered


def build_bi_dashboard_model(df: pd.DataFrame, column_roles: dict) -> dict[str, Any]:
    """Build deterministic tables used by the embedded BI dashboard."""
    work = prepare_bi_dataframe(df, column_roles)
    if work.empty:
        return {"summary": {}, "tables": {}, "pages": []}

    order_count = int(work["BI Order Key"].nunique()) if "BI Order Key" in work else len(work)
    total_sales = float(work["BI Sales"].sum())
    total_profit = float(work["BI Profit"].sum())
    total_loss = float(work["BI Loss Amount"].sum())

    summary = {
        "total_sales": total_sales,
        "total_profit": total_profit,
        "total_cost": float(work["BI Cost"].sum()),
        "total_quantity": float(work["BI Quantity"].sum()),
        "total_orders": order_count,
        "average_order_value": total_sales / order_count if order_count else 0.0,
        "profit_margin_percent": profit_margin(total_profit, total_sales),
        "loss_amount": total_loss,
        "loss_orders": int(work["BI Is Loss Order"].sum()),
        "average_discount": float(work["BI Discount"].mean()) if "BI Discount" in work else 0.0,
        "record_count": int(len(work)),
    }
    summary["loss_order_percent"] = summary["loss_orders"] / order_count * 100 if order_count else 0.0

    # Period deltas for executive cards.
    trend = work.groupby("BI Year Month", dropna=False).agg(Sales=("BI Sales", "sum"), Profit=("BI Profit", "sum")).reset_index().sort_values("BI Year Month")
    if len(trend) >= 2 and trend.iloc[-1]["BI Year Month"] != "Unknown":
        current = trend.iloc[-1]
        previous = trend.iloc[-2]
        summary["latest_period"] = current["BI Year Month"]
        summary["sales_delta_vs_prev"] = float(current["Sales"] - previous["Sales"])
        summary["profit_delta_vs_prev"] = float(current["Profit"] - previous["Profit"])
    else:
        summary["latest_period"] = None
        summary["sales_delta_vs_prev"] = None
        summary["profit_delta_vs_prev"] = None

    def group_by(role: str, label: str | None = None) -> pd.DataFrame:
        col = _existing_role(column_roles, role, work)
        if not col:
            return pd.DataFrame()
        g = work.groupby(col, dropna=False).agg(
            Sales=("BI Sales", "sum"),
            Profit=("BI Profit", "sum"),
            Quantity=("BI Quantity", "sum"),
            Orders=("BI Order Key", "nunique"),
            Avg_Discount=("BI Discount", "mean"),
            Loss_Amount=("BI Loss Amount", "sum"),
        ).reset_index().rename(columns={col: label or col})
        g["Margin %"] = g.apply(lambda r: profit_margin(r["Profit"], r["Sales"]), axis=1)
        g["Sales Contribution %"] = g["Sales"] / total_sales * 100 if total_sales else 0
        g["Profit Contribution %"] = g["Profit"] / total_profit * 100 if total_profit else 0
        g["Loss Contribution %"] = g["Loss_Amount"] / total_loss * 100 if total_loss else 0
        return g.sort_values("Profit", ascending=True).reset_index(drop=True)

    discount_order = ["0%", "0-10%", "10-20%", "20-30%", "30-40%", "40%+", "Unknown"]
    band = work.groupby("BI Discount Band", dropna=False).agg(
        Sales=("BI Sales", "sum"),
        Profit=("BI Profit", "sum"),
        Orders=("BI Order Key", "nunique"),
        Avg_Discount=("BI Discount", "mean"),
        Loss_Amount=("BI Loss Amount", "sum"),
    ).reset_index()
    band["_sort"] = band["BI Discount Band"].apply(lambda x: discount_order.index(x) if x in discount_order else 999)
    band = band.sort_values("_sort").drop(columns="_sort")
    band["Margin %"] = band.apply(lambda r: profit_margin(r["Profit"], r["Sales"]), axis=1)

    tables = {
        "trend": trend,
        "category": group_by("category_column", "Category"),
        "subcategory": group_by("subcategory_column", "Sub-Category"),
        "product": group_by("product_column", "Product"),
        "region": group_by("region_column", "Region"),
        "state": group_by("state_column", "State"),
        "city": group_by("city_column", "City"),
        "segment": group_by("segment_column", "Segment"),
        "customer": group_by("customer_column", "Customer"),
        "discount_band": band,
        "loss_orders": work.loc[work["BI Is Loss Order"]].sort_values("BI Profit").head(250),
        "detail_rows": work.head(5000),
    }
    return {"summary": summary, "tables": tables, "pages": get_available_dashboard_pages(df, column_roles)}


def _metric_cards(summary: dict[str, Any]) -> None:
    """Render KPI cards with useful deltas."""
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Sales", format_currency(summary.get("total_sales")), delta=format_currency(summary.get("sales_delta_vs_prev")) if summary.get("sales_delta_vs_prev") is not None else None)
    c2.metric("Total Profit", format_currency(summary.get("total_profit")), delta=format_currency(summary.get("profit_delta_vs_prev")) if summary.get("profit_delta_vs_prev") is not None else None)
    c3.metric("Margin", format_percent(summary.get("profit_margin_percent")))
    c4.metric("Orders", f"{int(summary.get('total_orders', 0)):,}")
    c5.metric("Loss Orders", format_percent(summary.get("loss_order_percent")))
    c6.metric("Avg Discount", format_percent(summary.get("average_discount")))


def _show_table(title: str, table: pd.DataFrame, rows: int = 30) -> None:
    if isinstance(table, pd.DataFrame) and not table.empty:
        st.markdown(f"#### {title}")
        st.dataframe(table.head(rows), use_container_width=True, hide_index=True)


def _safe_top(table: pd.DataFrame, metric: str = "Profit", n: int = 20, ascending: bool = False) -> pd.DataFrame:
    if table is None or table.empty or metric not in table.columns:
        return pd.DataFrame()
    return table.sort_values(metric, ascending=ascending).head(n)


def _interactive_filters(full_df: pd.DataFrame, work: pd.DataFrame, column_roles: dict) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Render a Power BI-like filter bar and return filtered BI dataframe."""
    st.markdown("### Interactive Filters")
    st.caption("Use these filters like Power BI slicers. Charts and tables below update immediately.")

    filters: dict[str, Any] = {"column_filters": {}}
    with st.container(border=True):
        r1c1, r1c2, r1c3, r1c4 = st.columns([1.4, 1.4, 1.4, 1.1])
        date_valid = work["BI Date"].dropna() if "BI Date" in work else pd.Series(dtype="datetime64[ns]")
        if not date_valid.empty:
            min_date = date_valid.min().date()
            max_date = date_valid.max().date()
            filters["date_range"] = r1c1.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="bi_date_range")
        else:
            r1c1.info("No date field detected")

        cat_col = _existing_role(column_roles, "category_column", full_df)
        reg_col = _existing_role(column_roles, "region_column", full_df)
        seg_col = _existing_role(column_roles, "segment_column", full_df)
        prod_col = _existing_role(column_roles, "product_column", full_df)
        cust_col = _existing_role(column_roles, "customer_column", full_df)

        if cat_col:
            filters["column_filters"][cat_col] = r1c2.multiselect("Category", _top_values(full_df, cat_col), key="bi_filter_category")
        if reg_col:
            filters["column_filters"][reg_col] = r1c3.multiselect("Region", _top_values(full_df, reg_col), key="bi_filter_region")
        if seg_col:
            filters["column_filters"][seg_col] = r1c4.multiselect("Segment", _top_values(full_df, seg_col), key="bi_filter_segment")

        r2c1, r2c2, r2c3, r2c4 = st.columns([1.4, 1.4, 1.4, 1.1])
        if prod_col:
            filters["column_filters"][prod_col] = r2c1.multiselect("Product", _top_values(full_df, prod_col, 80), key="bi_filter_product")
        if cust_col:
            filters["column_filters"][cust_col] = r2c2.multiselect("Customer", _top_values(full_df, cust_col, 80), key="bi_filter_customer")
        filters["column_filters"]["BI Discount Band"] = r2c3.multiselect("Discount Band", ["0%", "0-10%", "10-20%", "20-30%", "30-40%", "40%+", "Unknown"], key="bi_filter_discount_band")
        filters["loss_only"] = r2c4.toggle("Loss orders only", key="bi_loss_only")

    filtered = apply_dashboard_filters(work, filters)
    st.caption(f"Showing {len(filtered):,} of {len(work):,} records after filters.")
    return filtered, filters


def _render_executive(model: dict[str, Any], theme_mode: str) -> None:
    _metric_cards(model["summary"])
    tables = model["tables"]
    trend = tables.get("trend", pd.DataFrame())
    cat = tables.get("category", pd.DataFrame())
    reg = tables.get("region", pd.DataFrame())
    product = tables.get("product", pd.DataFrame())

    c1, c2 = st.columns([1.4, 1.0])
    with c1:
        if not trend.empty and trend["BI Year Month"].nunique() > 1:
            fig = px.line(trend, x="BI Year Month", y=["Sales", "Profit"], markers=True, title="Sales and Profit Trend")
            st.plotly_chart(_chart(fig, theme_mode), use_container_width=True)
    with c2:
        if not cat.empty:
            fig = px.bar(cat.sort_values("Profit"), x="Profit", y="Category", orientation="h", title="Profit by Category", color="Profit", color_continuous_scale="Tealgrn")
            st.plotly_chart(_chart(fig, theme_mode), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if not reg.empty:
            fig = px.bar(reg.sort_values("Profit"), x="Profit", y="Region", orientation="h", title="Profit by Region", color="Margin %", color_continuous_scale="RdYlGn")
            st.plotly_chart(_chart(fig, theme_mode), use_container_width=True)
    with c4:
        if not product.empty:
            top_loss = product.sort_values("Profit").head(12)
            fig = px.bar(top_loss, x="Profit", y="Product", orientation="h", title="Top Loss-Making Products", color="Profit", color_continuous_scale="Reds_r")
            st.plotly_chart(_chart(fig, theme_mode), use_container_width=True)

    _show_table("Executive Performance Table", cat.sort_values("Profit") if not cat.empty else pd.DataFrame(), 30)


def _render_profit(model: dict[str, Any], theme_mode: str) -> None:
    _metric_cards(model["summary"])
    tables = model["tables"]
    product = tables.get("product", pd.DataFrame())
    band = tables.get("discount_band", pd.DataFrame())
    cat = tables.get("category", pd.DataFrame())

    c1, c2 = st.columns(2)
    with c1:
        if not product.empty:
            fig = px.scatter(product, x="Sales", y="Profit", size="Orders", color="Margin %", hover_name="Product", title="Interactive Sales vs Profit by Product", color_continuous_scale="RdYlGn")
            st.plotly_chart(_chart(fig, theme_mode), use_container_width=True)
    with c2:
        if not band.empty:
            fig = px.bar(band, x="BI Discount Band", y="Profit", title="Profit by Discount Band", color="Profit", color_continuous_scale="RdYlGn")
            st.plotly_chart(_chart(fig, theme_mode), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if not band.empty:
            fig = px.bar(band, x="BI Discount Band", y="Loss_Amount", title="Loss by Discount Band", color="Loss_Amount", color_continuous_scale="Reds")
            st.plotly_chart(_chart(fig, theme_mode), use_container_width=True)
    with c4:
        if not cat.empty:
            fig = px.treemap(cat, path=["Category"], values="Sales", color="Margin %", color_continuous_scale="RdYlGn", title="Category Revenue Size vs Margin")
            st.plotly_chart(_chart(fig, theme_mode), use_container_width=True)

    _show_table("Worst Profit Products", product.sort_values("Profit") if not product.empty else pd.DataFrame(), 30)
    _show_table("Loss Order Drilldown", tables.get("loss_orders", pd.DataFrame()), 30)


def _render_product(model: dict[str, Any], theme_mode: str) -> None:
    product = model["tables"].get("product", pd.DataFrame())
    cat = model["tables"].get("category", pd.DataFrame())
    source = product if not product.empty else cat
    if source.empty:
        st.warning("Product fields were not detected, so product portfolio analysis is unavailable.")
        return
    label = "Product" if "Product" in source.columns else "Category"
    fig = px.scatter(source, x="Sales", y="Margin %", size="Orders", color="Profit", hover_name=label, title="Product Portfolio Matrix: Sales vs Margin", color_continuous_scale="RdYlGn")
    st.plotly_chart(_chart(fig, theme_mode), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        stars = source[(source["Sales"] >= source["Sales"].median()) & (source["Profit"] > 0) & (source["Margin %"] >= source["Margin %"].median())].sort_values("Profit", ascending=False)
        _show_table("Stars: High Sales + Healthy Profit", stars, 25)
    with c2:
        problems = source[(source["Profit"] < 0) | (source["Margin %"] < 0)].sort_values("Profit")
        _show_table("Problem Products / Categories", problems, 25)


def _render_customer(model: dict[str, Any], theme_mode: str) -> None:
    cust = model["tables"].get("customer", pd.DataFrame())
    seg = model["tables"].get("segment", pd.DataFrame())
    if cust.empty and seg.empty:
        st.warning("Customer fields were not detected, so customer BI dashboarding is unavailable.")
        return

    c1, c2 = st.columns(2)
    with c1:
        if not cust.empty:
            fig = px.bar(cust.sort_values("Profit", ascending=False).head(20), x="Profit", y="Customer", orientation="h", title="Top Customers by Profit", color="Profit", color_continuous_scale="Tealgrn")
            st.plotly_chart(_chart(fig, theme_mode), use_container_width=True)
    with c2:
        if not seg.empty:
            fig = px.bar(seg.sort_values("Profit", ascending=False), x="Segment", y="Profit", title="Profit by Segment", color="Margin %", color_continuous_scale="RdYlGn")
            st.plotly_chart(_chart(fig, theme_mode), use_container_width=True)
    _show_table("Customer Profitability Drilldown", cust.sort_values("Profit", ascending=False) if not cust.empty else pd.DataFrame(), 50)


def _render_region(model: dict[str, Any], theme_mode: str) -> None:
    reg = model["tables"].get("region", pd.DataFrame())
    state = model["tables"].get("state", pd.DataFrame())
    city = model["tables"].get("city", pd.DataFrame())
    table = reg if not reg.empty else state if not state.empty else city
    if table.empty:
        st.warning("Geography fields were not detected, so regional dashboarding is unavailable.")
        return
    first_col = table.columns[0]
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(table.sort_values("Profit"), x="Profit", y=first_col, orientation="h", title=f"Profit by {first_col}", color="Margin %", color_continuous_scale="RdYlGn")
        st.plotly_chart(_chart(fig, theme_mode), use_container_width=True)
    with c2:
        fig = px.scatter(table, x="Sales", y="Profit", size="Orders", hover_name=first_col, color="Avg_Discount", title=f"Sales vs Profit by {first_col}", color_continuous_scale="Bluered")
        st.plotly_chart(_chart(fig, theme_mode), use_container_width=True)
    _show_table(f"{first_col} Performance Drilldown", table.sort_values("Profit"), 50)


def _render_discount(model: dict[str, Any], theme_mode: str) -> None:
    band = model["tables"].get("discount_band", pd.DataFrame())
    product = model["tables"].get("product", pd.DataFrame())
    if band.empty:
        st.warning("Discount/profit fields were not detected, so discount control dashboarding is unavailable.")
        return

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(band, x="BI Discount Band", y="Profit", title="Profit by Discount Band", color="Profit", color_continuous_scale="RdYlGn")
        st.plotly_chart(_chart(fig, theme_mode), use_container_width=True)
    with c2:
        fig = px.line(band, x="BI Discount Band", y="Margin %", markers=True, title="Margin by Discount Band")
        st.plotly_chart(_chart(fig, theme_mode), use_container_width=True)
    if not product.empty:
        damaged = product[(product["Avg_Discount"] > product["Avg_Discount"].median()) & (product["Profit"] < 0)].sort_values("Profit")
        _show_table("High Discount + Negative Profit Products", damaged, 40)
    _show_table("Discount Band Control Table", band, 20)


def render_embedded_bi_dashboard(df: pd.DataFrame, column_roles: dict, theme_mode: str = "Light") -> None:
    """Render an interactive Power BI-style dashboard directly inside Streamlit.

    This is a no-setup BI experience: slicer-style filters, KPI cards, drilldown tables,
    and interactive Plotly visuals are all built into the project.
    """
    embed_url = os.getenv("POWERBI_EMBED_URL", "").strip()
    if embed_url:
        st.subheader("Published Power BI Report")
        components.iframe(embed_url, height=720, scrolling=True)
        st.caption("POWERBI_EMBED_URL is configured. The built-in dashboard below remains available as a fallback.")

    full_work = prepare_bi_dataframe(df, column_roles)
    if full_work.empty:
        st.info("Upload a dataset to activate the BI dashboard.")
        return

    st.markdown("### Interactive BI Dashboard")
    st.caption("Built directly into the app: slicers, KPI cards, drilldowns, hover tooltips, sorting, and downloads. No manual Power BI Desktop setup needed.")

    filtered_work, _ = _interactive_filters(df, full_work, column_roles)
    if filtered_work.empty:
        st.warning("No records match the current filters. Clear filters to see the dashboard.")
        return

    model = build_bi_dashboard_model(filtered_work, column_roles)
    pages = model.get("pages", []) or ["Executive Overview"]

    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        page = st.selectbox("Dashboard page", pages, key="embedded_bi_page_select")
    with c2:
        top_n = st.slider("Top N", 5, 50, 20, key="embedded_bi_top_n")
    with c3:
        csv_bytes = filtered_work.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered data", csv_bytes, "filtered_bi_data.csv", "text/csv", use_container_width=True)

    st.divider()
    if page == "Executive Overview":
        _render_executive(model, theme_mode)
    elif page == "Profit Intelligence":
        _render_profit(model, theme_mode)
    elif page == "Product Portfolio":
        _render_product(model, theme_mode)
    elif page == "Customer Intelligence":
        _render_customer(model, theme_mode)
    elif page == "Regional Performance":
        _render_region(model, theme_mode)
    elif page == "Discount & Margin Control":
        _render_discount(model, theme_mode)

    # Generic Top-N drilldown controlled by manager.
    with st.expander("Manager drilldown table", expanded=False):
        tables = model.get("tables", {})
        selectable = {k: v for k, v in tables.items() if isinstance(v, pd.DataFrame) and not v.empty and k not in {"detail_rows"}}
        selected_table = st.selectbox("Choose drilldown", list(selectable.keys()), format_func=lambda x: x.replace("_", " ").title(), key="bi_drill_table")
        sort_cols = [c for c in selectable[selected_table].columns if pd.api.types.is_numeric_dtype(selectable[selected_table][c])]
        sort_by = st.selectbox("Sort by", sort_cols or list(selectable[selected_table].columns), index=sort_cols.index("Profit") if "Profit" in sort_cols else 0, key="bi_drill_sort")
        ascending = st.toggle("Ascending sort", value=False, key="bi_drill_asc")
        drill = selectable[selected_table].sort_values(sort_by, ascending=ascending).head(top_n)
        st.dataframe(drill, use_container_width=True, hide_index=True)


def powerbi_embed_status() -> dict[str, Any]:
    """Return optional true Power BI embed status."""
    url = os.getenv("POWERBI_EMBED_URL", "").strip()
    return {
        "powerbi_embed_configured": bool(url),
        "mode": "published_powerbi_embed" if url else "streamlit_in_app_bi_dashboard",
        "manual_powerbi_desktop_required": False,
    }
