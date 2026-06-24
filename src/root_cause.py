from __future__ import annotations
import pandas as pd
from src.utils import safe_numeric, profit_margin, prepare_analysis_dataset, ensure_datetime
from src.metric_relationships import create_discount_bands, calculate_margin_by_dimension


def _roles(roles): return roles.get("roles", roles) if isinstance(roles, dict) else {}


def find_loss_drivers(df, column_roles):
    """Analyze negative profit drivers."""
    df, roles = prepare_analysis_dataset(df, column_roles)
    profit_col = roles.get("profit_column")
    if profit_col not in df.columns:
        return {"limitations": ["No profit column detected."]}
    loss_df = df[safe_numeric(df[profit_col]) < 0].copy()
    out = {
        "total_loss_amount": float(safe_numeric(loss_df[profit_col]).sum()) if not loss_df.empty else 0.0,
        "total_loss_orders": int(len(loss_df)),
    }
    for label, role in [("category","category_column"),("subcategory","subcategory_column"),("product","product_column"),("region","region_column"),("customer_segment","segment_column"),("customer","customer_column")]:
        col = roles.get(role)
        out[f"loss_by_{label}"] = calculate_margin_by_dimension(loss_df, roles, col) if col in loss_df.columns else pd.DataFrame()
    disc_col = roles.get("discount_column")
    if disc_col in loss_df.columns:
        loss_df["discount_band"] = create_discount_bands(loss_df[disc_col])
        out["loss_by_discount_band"] = calculate_margin_by_dimension(loss_df, roles, "discount_band")
    else:
        out["loss_by_discount_band"] = pd.DataFrame()

    combos = []
    combo_specs = [
        ("Category + Sub-Category", "category_column", "subcategory_column"),
        ("Category + Region", "category_column", "region_column"),
        ("Sub-Category + Region", "subcategory_column", "region_column"),
        ("Sub-Category + Segment", "subcategory_column", "segment_column"),
        ("Product + Region", "product_column", "region_column"),
        ("Product + Segment", "product_column", "segment_column"),
    ]
    display_defaults = {
        roles.get("category_column"): "All Categories",
        roles.get("subcategory_column"): "All Sub-Categories",
        roles.get("product_column"): "All Products",
        roles.get("region_column"): "All Regions",
        roles.get("segment_column"): "All Segments",
    }
    display_defaults = {k: v for k, v in display_defaults.items() if k}

    for combo_name, a, b in combo_specs:
        ca, cb = roles.get(a), roles.get(b)
        if ca in loss_df.columns and cb in loss_df.columns:
            g = (
                loss_df.groupby([ca, cb], dropna=False)
                .agg(loss=(profit_col, "sum"), orders=(profit_col, "size"))
                .reset_index()
                .sort_values("loss", ascending=True)
                .head(10)
            )
            g.insert(0, "Combination Type", combo_name)
            for col, default in display_defaults.items():
                if col not in g.columns:
                    g[col] = default
                else:
                    g[col] = g[col].fillna(default)
            combos.append(g)
    if combos:
        combo_df = pd.concat(combos, ignore_index=True).sort_values("loss", ascending=True).reset_index(drop=True)
        preferred = ["Combination Type"] + [c for c in display_defaults if c in combo_df.columns] + ["loss", "orders"]
        remaining = [c for c in combo_df.columns if c not in preferred]
        out["biggest_loss_combinations"] = combo_df[preferred + remaining]
    else:
        out["biggest_loss_combinations"] = pd.DataFrame()
    return out


def analyze_low_margin_segments(df, column_roles):
    df, roles = prepare_analysis_dataset(df, column_roles)
    result = {}
    for name, role in [("category","category_column"),("subcategory","subcategory_column"),("region","region_column"),("segment","segment_column"),("product","product_column")]:
        col = roles.get(role)
        g = calculate_margin_by_dimension(df, roles, col) if col in df.columns else pd.DataFrame()
        if not g.empty and "profit_margin_percent" in g:
            result[name] = g.sort_values("profit_margin_percent")
    return result


def explain_metric_drop(df, column_roles, metric="profit", time_grain="M"):
    df, roles = prepare_analysis_dataset(df, column_roles)
    metric_col = roles.get(f"{metric}_column") or roles.get("profit_column")
    date_col = roles.get("date_column")
    if metric_col not in df.columns or date_col not in df.columns:
        return {"limitations": ["Metric or date column missing."]}
    tmp = df.copy()
    tmp[date_col] = ensure_datetime(tmp[date_col])
    tmp = tmp.dropna(subset=[date_col])
    tmp["period"] = tmp[date_col].dt.to_period(time_grain).astype(str)
    p = tmp.groupby("period")[metric_col].sum().sort_index()
    if len(p) < 2:
        return {"limitations": ["Need at least two time periods."]}
    previous, current = p.iloc[-2], p.iloc[-1]
    change = current - previous
    return {
        "current_period": p.index[-1],
        "previous_period": p.index[-2],
        "current_value": float(current),
        "previous_value": float(previous),
        "absolute_change": float(change),
        "percent_change": float(change / previous * 100) if previous else None,
    }


def root_cause_for_selection(df, column_roles, target_dimension, target_value):
    df, roles = prepare_analysis_dataset(df, column_roles)
    if target_dimension not in df.columns:
        return {"limitations": [f"{target_dimension} not found."]}
    f = df[df[target_dimension].astype(str).str.lower() == str(target_value).lower()].copy()
    if f.empty:
        return {"limitations": [f"No rows found for {target_value}."]}
    sales_col, profit_col, disc_col = roles.get("sales_column"), roles.get("profit_column"), roles.get("discount_column")
    total_sales = float(safe_numeric(f[sales_col]).sum()) if sales_col in f.columns else None
    total_profit = float(safe_numeric(f[profit_col]).sum()) if profit_col in f.columns else None
    causes = []
    if total_profit is not None and total_profit < 0:
        causes.append("The selected area is loss-making.")
    if disc_col in f.columns and safe_numeric(f[disc_col]).mean() > (0.2 if safe_numeric(f[disc_col]).max() <= 1 else 20):
        causes.append("Average discount is high, suggesting discount pressure.")
    breakdowns = {}
    for name, role in [("subcategory","subcategory_column"),("product","product_column"),("region","region_column"),("segment","segment_column")]:
        col = roles.get(role)
        if col in f.columns:
            breakdowns[name] = calculate_margin_by_dimension(f, roles, col)
    return {
        "target_dimension": target_dimension,
        "target_value": target_value,
        "sales": total_sales,
        "profit": total_profit,
        "margin": profit_margin(total_profit, total_sales) if total_profit is not None and total_sales else None,
        "loss_amount": float(safe_numeric(f[profit_col])[safe_numeric(f[profit_col]) < 0].sum()) if profit_col in f.columns else None,
        "order_count": int(len(f)),
        "average_discount": float(safe_numeric(f[disc_col]).mean()) if disc_col in f.columns else None,
        "breakdowns": breakdowns,
        "likely_root_causes": causes,
    }
