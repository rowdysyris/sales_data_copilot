from __future__ import annotations
import pandas as pd
from src.utils import prepare_analysis_dataset
from src.kpi_engine import performance_by_product, performance_by_category, performance_by_subcategory


def analyze_product_portfolio(df, column_roles):
    df, roles = prepare_analysis_dataset(df, column_roles)
    product_df = performance_by_product(df, roles)
    if product_df.empty:
        return {"limitations": ["No product column detected."], "product_portfolio": pd.DataFrame()}
    median_sales = product_df["sales"].median() if "sales" in product_df else 0
    median_profit = product_df["profit"].median() if "profit" in product_df else 0
    def classify(r):
        sales, profit, margin = r.get("sales",0), r.get("profit",0), r.get("profit_margin_percent",0)
        if profit < 0: return "Problem Products"
        if sales >= median_sales and profit >= median_profit: return "Stars"
        if sales >= median_sales and margin < 8: return "Volume Drivers"
        if sales < median_sales and margin >= 20: return "Niche Winners"
        if r.get("average_discount",0) >= 0.3 and margin < 10: return "Discount-Damaged Products"
        return "Review Products"
    product_df = product_df.copy()
    product_df["portfolio_class"] = product_df.apply(classify, axis=1)
    return {
        "product_summary": {"total_products": int(product_df["dimension_value"].nunique())},
        "product_portfolio": product_df,
        "stars": product_df[product_df["portfolio_class"]=="Stars"],
        "volume_drivers": product_df[product_df["portfolio_class"]=="Volume Drivers"],
        "niche_winners": product_df[product_df["portfolio_class"]=="Niche Winners"],
        "problem_products": product_df[product_df["portfolio_class"]=="Problem Products"],
        "discount_damaged_products": product_df[product_df["portfolio_class"]=="Discount-Damaged Products"],
        "category_summary": performance_by_category(df, roles),
        "subcategory_summary": performance_by_subcategory(df, roles),
        "insights": ["Problem products should be reviewed for pricing, cost, or discount action."] if (product_df["portfolio_class"]=="Problem Products").any() else [],
    }
