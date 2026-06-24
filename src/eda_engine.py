from __future__ import annotations
import pandas as pd
from src.utils import iqr_outlier_mask, safe_correlation_matrix
from src.semantic_layer import build_semantic_model


def generate_eda_summary(df, column_roles):
    semantic = build_semantic_model(df, column_roles)
    numeric_summary = []
    for col in df.select_dtypes(include="number").columns:
        s = df[col]
        numeric_summary.append({"column": col, "min": float(s.min()), "max": float(s.max()), "mean": float(s.mean()), "median": float(s.median()), "std": float(s.std()) if len(s)>1 else 0, "outlier_count": int(iqr_outlier_mask(s).sum())})
    categorical_summary = []
    for col in df.select_dtypes(exclude="number").columns:
        vc = df[col].value_counts(dropna=False).head(5)
        categorical_summary.append({"column": col, "unique_count": int(df[col].nunique(dropna=True)), "top_values": vc.to_dict(), "high_cardinality": df[col].nunique(dropna=True) > max(50, len(df)*0.5)})
    corr = safe_correlation_matrix(df)
    observations = []
    roles = column_roles.get("roles", column_roles) if isinstance(column_roles, dict) else {}
    if roles.get("profit_column") in df.columns:
        observations.append("Profit analysis is available.")
    if roles.get("discount_column") in df.columns and roles.get("profit_column") in df.columns:
        observations.append("Discount impact can be checked against profit.")
    return {
        "dataset_overview": {"row_count": len(df), "column_count": df.shape[1], "detected_dataset_type": semantic["dataset_type"], "granularity": semantic["grain"]},
        "numeric_summary": pd.DataFrame(numeric_summary),
        "categorical_summary": pd.DataFrame(categorical_summary),
        "correlation_matrix": corr,
        "top_positive_correlations": [],
        "top_negative_correlations": [],
        "business_observations": observations,
    }
