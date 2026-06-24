from __future__ import annotations

import math
import pandas as pd
import numpy as np
from src.utils import safe_numeric, prepare_analysis_dataset, ensure_datetime


def _roles(roles):
    return roles.get("roles", roles) if isinstance(roles, dict) else {}


def _aggregate_history(df: pd.DataFrame, roles: dict, frequency: str = "M") -> tuple[pd.DataFrame, list[str]]:
    limitations: list[str] = []
    date_col, sales_col, profit_col = roles.get("date_column"), roles.get("sales_column"), roles.get("profit_column")
    if date_col not in df.columns or sales_col not in df.columns:
        return pd.DataFrame(), ["Date and sales columns are required for forecasting."]
    tmp = df.copy()
    tmp[date_col] = ensure_datetime(tmp[date_col])
    tmp = tmp.dropna(subset=[date_col])
    if tmp.empty:
        return pd.DataFrame(), ["No valid dates are available for forecasting."]
    tmp[sales_col] = safe_numeric(tmp[sales_col]).fillna(0)
    if profit_col in tmp.columns:
        tmp[profit_col] = safe_numeric(tmp[profit_col]).fillna(0)
    tmp["period"] = tmp[date_col].dt.to_period(frequency)
    agg = {sales_col: "sum"}
    if profit_col in tmp.columns:
        agg[profit_col] = "sum"
    hist = tmp.groupby("period").agg(agg).sort_index()
    hist = hist.rename(columns={sales_col: "sales", profit_col: "profit"})
    hist.index = hist.index.astype(str)
    hist = hist.reset_index().rename(columns={"period": "period"})
    return hist, limitations


def _predict_series(values: pd.Series, periods: int) -> dict:
    y = safe_numeric(values).dropna().astype(float).reset_index(drop=True)
    if len(y) < 2:
        return {"predictions": [float(y.iloc[-1]) if len(y) else 0.0] * periods, "method": "naive", "mape": None, "mae": None}

    window = min(3, len(y))
    moving_avg = float(y.tail(window).mean())
    exp_level = float(y.ewm(alpha=0.45, adjust=False).mean().iloc[-1])
    x = np.arange(len(y), dtype=float)
    try:
        slope, intercept = np.polyfit(x, y.to_numpy(), 1)
        trend_preds = [max(0.0, float(intercept + slope * (len(y) + i))) for i in range(periods)]
    except Exception:
        trend_preds = [moving_avg] * periods
    seasonal = []
    if len(y) >= 12:
        seasonal = [float(y.iloc[-12 + (i % 12)]) for i in range(periods)]
    else:
        seasonal = [float(y.iloc[-1])] * periods
    preds = []
    for i in range(periods):
        # Conservative ensemble: trend + recent level + seasonality.
        pred = (0.35 * moving_avg) + (0.30 * exp_level) + (0.20 * trend_preds[i]) + (0.15 * seasonal[i])
        preds.append(max(0.0, float(pred)))

    # Rolling one-step backtest for method confidence.
    errors = []
    actuals = []
    if len(y) >= 5:
        for idx in range(3, len(y)):
            train = y.iloc[:idx]
            pred = float(train.tail(min(3, len(train))).mean())
            actual = float(y.iloc[idx])
            errors.append(abs(actual - pred))
            actuals.append(abs(actual))
    mae = float(np.mean(errors)) if errors else None
    mape = float(np.mean([e / a for e, a in zip(errors, actuals) if a > 0]) * 100) if actuals and any(a > 0 for a in actuals) else None
    return {"predictions": preds, "method": "ensemble: moving average + exponential smoothing + trend + seasonal naive", "mape": mape, "mae": mae}


def forecast_sales_profit(df, column_roles, periods: int = 3, frequency: str = "M"):
    """Forecast sales/profit with a transparent conservative ensemble and backtest metrics."""
    df, roles = prepare_analysis_dataset(df, column_roles)
    hist, limitations = _aggregate_history(df, roles, frequency)
    if hist.empty:
        return {"forecast": pd.DataFrame(), "history": hist, "confidence": "Low", "limitations": limitations}
    if len(hist) < 3:
        return {"forecast": pd.DataFrame(), "history": hist, "confidence": "Low", "limitations": limitations + ["Need at least three periods for a useful forecast."]}

    last = pd.Period(hist["period"].iloc[-1], freq=frequency)
    sales_pred = _predict_series(hist["sales"], periods)
    profit_pred = _predict_series(hist["profit"], periods) if "profit" in hist.columns else None
    variability = float(safe_numeric(hist["sales"]).pct_change().replace([np.inf, -np.inf], np.nan).dropna().std() or 0.15)
    variability = min(max(variability, 0.10), 0.35)
    rows = []
    for i in range(periods):
        period = str(last + i + 1)
        ps = sales_pred["predictions"][i]
        pp = profit_pred["predictions"][i] if profit_pred else None
        rows.append({
            "forecast_period": period,
            "predicted_sales": ps,
            "predicted_profit": pp,
            "sales_lower_estimate": ps * (1 - variability),
            "sales_upper_estimate": ps * (1 + variability),
            "profit_lower_estimate": pp * (1 + variability) if pp is not None and pp < 0 else (pp * (1 - variability) if pp is not None else None),
            "profit_upper_estimate": pp * (1 - variability) if pp is not None and pp < 0 else (pp * (1 + variability) if pp is not None else None),
            "method_used": sales_pred["method"],
        })
    mape = sales_pred.get("mape")
    if mape is None:
        confidence = "Medium"
    elif mape <= 15:
        confidence = "High"
    elif mape <= 30:
        confidence = "Medium"
    else:
        confidence = "Low"
    return {
        "forecast": pd.DataFrame(rows),
        "history": hist,
        "backtest": {"sales_mae": sales_pred.get("mae"), "sales_mape_percent": mape, "profit_mae": profit_pred.get("mae") if profit_pred else None, "profit_mape_percent": profit_pred.get("mape") if profit_pred else None},
        "confidence": confidence,
        "limitations": limitations + ["Forecast is an estimate based on historical patterns, not a guaranteed prediction.", "External events, stockouts, campaigns, and price changes are not included unless present in the dataset."],
    }
