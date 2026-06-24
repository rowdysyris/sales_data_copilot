from __future__ import annotations
import pandas as pd
from src.utils import prepare_analysis_dataset
from src.kpi_engine import _group_performance


def analyze_region_segment_performance(df, column_roles):
    df, roles = prepare_analysis_dataset(df, column_roles)
    out = {}
    for key, role in [("region_summary","region_column"),("state_summary","state_column"),("city_summary","city_column"),("segment_summary","segment_column")]:
        col = roles.get(role)
        out[key] = _group_performance(df, roles, col) if col in df.columns else pd.DataFrame()
    reg = out.get("region_summary", pd.DataFrame())
    if not reg.empty and "profit" in reg:
        out["weak_regions"] = reg.sort_values(["profit","profit_margin_percent"] if "profit_margin_percent" in reg else ["profit"]).head(10)
        out["strong_regions"] = reg.sort_values("profit", ascending=False).head(10)
        if "sales" in reg and "profit_margin_percent" in reg:
            out["high_sales_low_margin_regions"] = reg[(reg["sales"] >= reg["sales"].median()) & (reg["profit_margin_percent"] <= reg["profit_margin_percent"].median())]
    else:
        out["weak_regions"] = out["strong_regions"] = out["high_sales_low_margin_regions"] = pd.DataFrame()
    out["segment_insights"] = []
    out["regional_insights"] = []
    if not out["weak_regions"].empty:
        r = out["weak_regions"].iloc[0]
        out["regional_insights"].append(f"Weakest region is {r['dimension_value']} with profit {r.get('profit',0):.2f}.")
    return out
