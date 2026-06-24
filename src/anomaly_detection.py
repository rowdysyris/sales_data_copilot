from __future__ import annotations

import pandas as pd
import numpy as np
from src.utils import safe_numeric, iqr_outlier_mask, profit_margin, prepare_analysis_dataset, ensure_datetime


def _roles(roles):
    return roles.get("roles", roles) if isinstance(roles, dict) else {}


def _severity_from_pct_drop(pct: float) -> str:
    if pct <= -0.40:
        return "High"
    if pct <= -0.25:
        return "Medium"
    return "Low"


def detect_business_anomalies(df, column_roles):
    """Detect record-level and period-level sales/profit anomalies."""
    df, roles = prepare_analysis_dataset(df, column_roles)
    anomalies = []
    unusual_rows = []
    for metric_role in ["sales_column", "profit_column", "quantity_column", "discount_column"]:
        col = roles.get(metric_role)
        if col in df.columns:
            mask = iqr_outlier_mask(df[col]).fillna(False)
            if mask.any():
                vals = safe_numeric(df.loc[mask, col])
                anomalies.append({
                    "title": f"Unusual values in {col}",
                    "severity": "Medium" if int(mask.sum()) < max(5, len(df) * 0.05) else "High",
                    "metric": col,
                    "value": int(mask.sum()),
                    "expected_range": "IQR normal range",
                    "explanation": f"{int(mask.sum())} records are outside the IQR range for {col}.",
                    "recommended_check": "Inspect these records for data-entry issues, one-off deals, refunds, or unusual business events.",
                })
                sample = df.loc[mask].head(20).copy()
                sample["anomaly_metric"] = col
                unusual_rows.append(sample)

    date_col, sales_col, profit_col, discount_col = roles.get("date_column"), roles.get("sales_column"), roles.get("profit_column"), roles.get("discount_column")
    if date_col in df.columns and (sales_col in df.columns or profit_col in df.columns):
        tmp = df.copy()
        tmp[date_col] = ensure_datetime(tmp[date_col])
        tmp = tmp.dropna(subset=[date_col])
        if not tmp.empty:
            tmp["period"] = tmp[date_col].dt.to_period("M").astype(str)
            for col in [sales_col, profit_col, discount_col]:
                if col in tmp.columns:
                    tmp[col] = safe_numeric(tmp[col])
                    agg_func = "mean" if col == discount_col else "sum"
                    s = tmp.groupby("period")[col].agg(agg_func)
                    pct = s.pct_change().replace([np.inf, -np.inf], np.nan)
                    if col == discount_col:
                        spikes = pct[pct > 0.25]
                        for period, val in spikes.items():
                            anomalies.append({
                                "title": f"Discount spike in {period}",
                                "severity": "Medium" if val < 0.50 else "High",
                                "metric": col,
                                "value": float(s.loc[period]),
                                "expected_range": "Within 25% of previous period",
                                "explanation": f"Average discount increased {val*100:.2f}% versus previous period.",
                                "recommended_check": "Check whether discounting is intentional and whether it damaged margin.",
                            })
                    else:
                        drops = pct[pct < -0.25]
                        for period, val in drops.items():
                            anomalies.append({
                                "title": f"Sudden {col} drop in {period}",
                                "severity": _severity_from_pct_drop(float(val)),
                                "metric": col,
                                "value": float(s.loc[period]),
                                "expected_range": "Within 25% of previous period",
                                "explanation": f"{col} dropped {val*100:.2f}% versus previous period.",
                                "recommended_check": "Break down the drop by category, product, region, and segment.",
                            })

    if sales_col in df.columns and profit_col in df.columns:
        tmp = df.copy()
        tmp["_sales"] = safe_numeric(tmp[sales_col]).fillna(0)
        tmp["_profit"] = safe_numeric(tmp[profit_col]).fillna(0)
        tmp["_margin"] = tmp.apply(lambda r: profit_margin(r["_profit"], r["_sales"]), axis=1)
        q_sales = tmp["_sales"].quantile(0.75)
        margin_med = tmp["_margin"].median()
        bad = tmp[(tmp["_sales"] >= q_sales) & (tmp["_margin"] < margin_med)]
        if not bad.empty:
            anomalies.append({
                "title": "High-sales low-margin orders detected",
                "severity": "High" if len(bad) > len(df) * 0.1 else "Medium",
                "metric": "profit_margin",
                "value": int(len(bad)),
                "expected_range": "High sales should normally convert to healthy margin",
                "explanation": f"{len(bad)} high-sales records have below-median profit margin.",
                "recommended_check": "Review discount, cost, and product mix for these orders.",
            })
            sample = bad.head(20).drop(columns=["_sales", "_profit", "_margin"], errors="ignore").copy()
            sample["anomaly_metric"] = "high_sales_low_margin"
            unusual_rows.append(sample)

    unusual_df = pd.concat(unusual_rows, ignore_index=True) if unusual_rows else pd.DataFrame()
    return {"anomalies": anomalies, "unusual_rows": unusual_df}
