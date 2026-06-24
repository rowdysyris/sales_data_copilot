from __future__ import annotations

from typing import Any
import pandas as pd

from src.utils import safe_numeric, profit_margin, format_currency, format_percent, prepare_analysis_dataset
from src.metric_relationships import (
    analyze_metric_relationships,
    detect_profit_leakage_points,
    calculate_margin_by_dimension,
    create_discount_bands,
)
from src.kpi_engine import calculate_kpis


def _roles(roles):
    return roles.get("roles", roles) if isinstance(roles, dict) else {}


def _clean_text(value: Any) -> str:
    return str(value).strip()


def _dimension_label(col: str | None, roles: dict) -> str:
    if not col:
        return "dimension"
    mapping = {
        roles.get("category_column"): "Category",
        roles.get("subcategory_column"): "Sub-Category",
        roles.get("product_column"): "Product",
        roles.get("region_column"): "Region",
        roles.get("segment_column"): "Segment",
        roles.get("customer_column"): "Customer",
        roles.get("customer_id_column"): "Customer",
    }
    return mapping.get(col, str(col).replace("_", " ").title())


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item or "").strip().rstrip(".;")
        if not text:
            continue
        key = text.lower()
        if key not in seen:
            seen.add(key)
            out.append(text + ".")
    return out


def _filter_by_resolved_entities(df: pd.DataFrame, resolved_entities: list[dict] | None) -> tuple[pd.DataFrame, list[str]]:
    """Apply question entities safely and return readable context labels."""
    if not resolved_entities:
        return df, []
    filtered = df.copy()
    labels: list[str] = []
    for ent in resolved_entities:
        col = ent.get("matched_column")
        val = ent.get("matched_value")
        if col in filtered.columns:
            mask = filtered[col].astype(str).str.lower() == str(val).lower()
            if mask.any():
                filtered = filtered[mask].copy()
                labels.append(f"{col} = {val}")
    return (filtered if not filtered.empty else df), labels


def _top_leakage_context(leakage: pd.DataFrame, roles: dict) -> dict[str, Any]:
    """Find the strongest current profit leakage point across all available dimensions.

    When several dimensions have the same loss because they describe the same rows
    (for example Furniture -> Tables -> one product -> one region), prefer the
    business diagnosis level a senior analyst normally starts with: Sub-Category,
    Product, Category, Region, Segment, then Customer. This avoids always reporting
    the broad Category just because it appears first in the concatenated table.
    """
    if not isinstance(leakage, pd.DataFrame) or leakage.empty or "profit" not in leakage.columns:
        return {}
    priority = {
        roles.get("subcategory_column"): 0,
        roles.get("product_column"): 1,
        roles.get("category_column"): 2,
        roles.get("region_column"): 3,
        roles.get("segment_column"): 4,
        roles.get("customer_column"): 5,
    }
    ranked = leakage.copy()
    ranked["_dimension_priority"] = ranked.get("dimension").map(lambda x: priority.get(x, 99))
    sort_cols = ["profit", "_dimension_priority"]
    ascending = [True, True]
    if "profit_margin_percent" in ranked.columns:
        sort_cols.append("profit_margin_percent")
        ascending.append(True)
    ranked = ranked.sort_values(sort_cols, ascending=ascending)
    row = ranked.iloc[0]
    dim_col = row.get("dimension")
    dim_value = row.get("dimension_value")
    return {
        "dimension_column": dim_col,
        "dimension_label": _dimension_label(dim_col, roles),
        "dimension_value": dim_value,
        "profit": float(row.get("profit", 0) or 0),
        "sales": float(row.get("sales", 0) or 0) if "sales" in row else None,
        "margin": float(row.get("profit_margin_percent", 0) or 0) if "profit_margin_percent" in row else None,
        "order_count": int(row.get("order_count", 0) or 0) if "order_count" in row else None,
    }


def _filter_to_context(df: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    col = context.get("dimension_column")
    val = context.get("dimension_value")
    if col in df.columns:
        mask = df[col].astype(str).str.lower() == str(val).lower()
        if mask.any():
            return df[mask].copy()
    return df.copy()


def _context_discount_driver(ctx_df: pd.DataFrame, roles: dict) -> str | None:
    profit_col = roles.get("profit_column")
    disc_col = roles.get("discount_column")
    sales_col = roles.get("sales_column")
    if profit_col not in ctx_df.columns or disc_col not in ctx_df.columns:
        return None
    tmp = ctx_df.copy()
    tmp["discount_band"] = create_discount_bands(tmp[disc_col])
    agg = {profit_col: "sum"}
    if sales_col in tmp.columns:
        agg[sales_col] = "sum"
    band = tmp.groupby("discount_band", observed=False).agg(agg).reset_index()
    rename = {profit_col: "profit"}
    if sales_col in tmp.columns:
        rename[sales_col] = "sales"
    band = band.rename(columns=rename)
    loss = band[band["profit"] < 0].sort_values("profit")
    if loss.empty:
        return None
    row = loss.iloc[0]
    return f"Within this same area, the {row['discount_band']} discount band is loss-making with profit of {format_currency(row['profit'])}"


def _context_dimension_driver(ctx_df: pd.DataFrame, roles: dict, context: dict[str, Any]) -> list[str]:
    """Find drivers inside the same problem context only, avoiding unrelated global categories."""
    profit_col = roles.get("profit_column")
    if profit_col not in ctx_df.columns:
        return []
    source_col = context.get("dimension_column")
    candidates = [
        ("region_column", "Region"),
        ("segment_column", "Segment"),
        ("product_column", "Product"),
        ("subcategory_column", "Sub-Category"),
        ("category_column", "Category"),
        ("customer_column", "Customer"),
    ]
    drivers: list[str] = []
    for role, label in candidates:
        col = roles.get(role)
        if not col or col == source_col or col not in ctx_df.columns:
            continue
        g = calculate_margin_by_dimension(ctx_df, roles, col)
        if g.empty or "profit" not in g.columns:
            continue
        loss = g[g["profit"] < 0].sort_values("profit")
        if loss.empty:
            continue
        row = loss.iloc[0]
        drivers.append(f"Within this same area, {label} '{row['dimension_value']}' contributes the largest loss with profit of {format_currency(row['profit'])}")
        if len(drivers) >= 2:
            break
    return drivers


def _build_loss_driver_tables(df: pd.DataFrame, roles: dict) -> dict[str, pd.DataFrame]:
    """Scan every available business dimension for loss-making / low-margin areas."""
    out: dict[str, pd.DataFrame] = {}
    for label, role in [
        ("categories", "category_column"),
        ("subcategories", "subcategory_column"),
        ("products", "product_column"),
        ("regions", "region_column"),
        ("segments", "segment_column"),
        ("customers", "customer_column"),
    ]:
        col = roles.get(role)
        if col in df.columns:
            g = calculate_margin_by_dimension(df, roles, col)
            if not g.empty and "profit" in g.columns:
                out[label] = g.sort_values(["profit", "profit_margin_percent" if "profit_margin_percent" in g.columns else "profit"], ascending=[True, True]).reset_index(drop=True)
    return out


def _other_loss_areas(loss_tables: dict[str, pd.DataFrame], top_context: dict[str, Any], limit: int = 8) -> pd.DataFrame:
    """Return other loss areas across all dimensions without repeating the key problem.

    This is intentionally not only the absolute top rows, because those may be the
    same issue repeated as product/region/segment/customer. A manager needs to see
    whether other categories/subcategories/products are also loss-making.
    """
    rows: list[dict[str, Any]] = []
    top_val = str(top_context.get("dimension_value", "")).lower()
    priority = {
        "subcategories": 0,
        "categories": 1,
        "products": 2,
        "regions": 3,
        "segments": 4,
        "customers": 5,
    }
    for table_name, table in loss_tables.items():
        if not isinstance(table, pd.DataFrame) or table.empty or "profit" not in table.columns:
            continue
        loss_rows = table[table["profit"] < 0].sort_values("profit").head(3)
        for _, row in loss_rows.iterrows():
            value = str(row.get("dimension_value", ""))
            if value.lower() == top_val:
                continue
            rows.append({
                "Area Type": table_name.replace("_", " ").title(),
                "Area": value,
                "Sales": row.get("sales"),
                "Profit": row.get("profit"),
                "Margin %": row.get("profit_margin_percent"),
                "Loss Contribution %": row.get("loss_contribution_percent"),
                "_Priority": priority.get(table_name, 99),
            })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["_Priority", "Profit"], ascending=[True, True]).head(limit).reset_index(drop=True)
    return out.drop(columns=["_Priority"], errors="ignore")


def analyze_profit_intelligence(df, column_roles, resolved_entities=None):
    """Senior analyst-level profit diagnosis.

    Important behavior:
    - The model scans every available category, subcategory, product, region, segment, and customer.
    - If a specific entity is asked about, root causes are calculated only within that entity's rows.
    - Global watch areas are kept separate so unrelated groups such as Tables and Phones are not presented as one causal chain.
    """
    df, roles = prepare_analysis_dataset(df, column_roles)
    scoped_df, entity_context_labels = _filter_by_resolved_entities(df, resolved_entities)
    sales_col, profit_col = roles.get("sales_column"), roles.get("profit_column")
    if profit_col not in scoped_df.columns:
        return {"profit_summary": {}, "root_cause_summary": ["Profit column missing."], "recommended_actions": [], "limitations": ["No profit column detected."]}

    kpis = calculate_kpis(scoped_df, roles)
    rel = analyze_metric_relationships(scoped_df, roles, "profit")

    margin_analysis: dict[str, pd.DataFrame] = {}
    for name, role in [("category", "category_column"), ("subcategory", "subcategory_column"), ("product", "product_column"), ("region", "region_column"), ("segment", "segment_column"), ("customer", "customer_column")]:
        col = roles.get(role)
        if col in scoped_df.columns:
            margin_analysis[name] = calculate_margin_by_dimension(scoped_df, roles, col)

    leakage = detect_profit_leakage_points(scoped_df, roles)
    loss_tables = _build_loss_driver_tables(scoped_df, roles)
    top_context = _top_leakage_context(leakage, roles)
    context_df = _filter_to_context(scoped_df, top_context) if top_context else scoped_df.copy()

    root_causes: list[str] = []
    actions: list[str] = []
    if top_context:
        root_causes.append(
            f"Profit leakage is concentrated in {top_context['dimension_value']} ({top_context['dimension_label']}) with profit of {format_currency(top_context['profit'])}"
        )
        discount_driver = _context_discount_driver(context_df, roles)
        if discount_driver:
            root_causes.append(discount_driver)
        root_causes.extend(_context_dimension_driver(context_df, roles, top_context))
        actions.append(f"Review pricing, cost, and discount strategy for {top_context['dimension_value']}.")
    else:
        root_causes.extend(rel.get("relationship_summary", [])[:3])

    other_areas = _other_loss_areas(loss_tables, top_context, limit=8)
    other_watch_areas: list[str] = []
    if not other_areas.empty:
        for _, row in other_areas.head(5).iterrows():
            other_watch_areas.append(
                f"{row['Area']} ({row['Area Type']}) is also loss-making with profit of {format_currency(row['Profit'])}"
            )

    # Keep high-sales/low-margin findings as watch areas, not root causes for the top problem.
    hs = rel.get("high_sales_low_profit")
    if isinstance(hs, pd.DataFrame) and not hs.empty:
        for _, row in hs.head(3).iterrows():
            text = f"{row['dimension_value']} has high sales but weak margin at {format_percent(row.get('profit_margin_percent', 0))}"
            if text not in other_watch_areas:
                other_watch_areas.append(text)

    profit_summary = {
        "total_sales": kpis.get("total_sales"),
        "total_profit": kpis.get("total_profit"),
        "overall_margin": kpis.get("profit_margin_percent"),
        "loss_making_orders": kpis.get("loss_making_orders_count"),
        "total_loss_amount": kpis.get("total_loss_amount"),
        "profitable_orders": kpis.get("profitable_orders_count"),
        "average_profit_per_order": (kpis.get("total_profit") / kpis.get("total_orders")) if kpis.get("total_profit") is not None and kpis.get("total_orders") else None,
    }

    return {
        "analysis_scope": entity_context_labels or ["Full dataset"],
        "profit_summary": profit_summary,
        "profit_relationships": rel,
        "profit_leakage_points": leakage,
        "top_profit_leakage_context": top_context,
        "all_loss_driver_tables": loss_tables,
        "other_loss_areas": other_areas,
        "other_profit_watch_areas": _unique(other_watch_areas),
        "loss_drivers": {"loss_concentration": rel.get("loss_concentration", pd.DataFrame())},
        "margin_analysis": margin_analysis,
        "discount_impact": {"profit_by_discount_band": rel.get("profit_by_discount_band", pd.DataFrame())},
        "sales_vs_profit_analysis": {"high_sales_low_profit": rel.get("high_sales_low_profit", pd.DataFrame())},
        "time_profit_analysis": rel.get("time_relationships", {}),
        "customer_profit_analysis": margin_analysis.get("customer", pd.DataFrame()),
        "product_profit_analysis": margin_analysis.get("product", pd.DataFrame()),
        "regional_profit_analysis": margin_analysis.get("region", pd.DataFrame()),
        "root_cause_summary": _unique(root_causes),
        "recommended_actions": actions,
    }
