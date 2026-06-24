from __future__ import annotations


def create_analysis_plan(question, semantic_model, column_roles, resolved_entities=None):
    """Create an analysis plan based on question intent and available semantic model."""
    q = str(question).lower()
    roles = column_roles.get("roles", column_roles) if isinstance(column_roles, dict) else {}
    plan = []
    def add(step, func, reason, req=None, priority=5):
        plan.append({"step_name": step, "analysis_function": func, "reason": reason, "required_columns": req or [], "priority": priority})

    if any(w in q for w in ["profit","loss","margin","profitability","losing money","low profit","negative profit","revenue leakage","bleeding money","bleeding","bad contribution","contribution","margin leakage","leakage","profit drain","money drain","underperforming","unprofitable","loss-making"]):
        add("Profit Intelligence", "analyze_profit_intelligence", "Question asks about profitability.", ["profit_column"], 1)
        add("Metric Relationships", "analyze_metric_relationships", "Profit questions require relationship/driver analysis.", ["profit_column"], 2)
        add("Loss Drivers", "find_loss_drivers", "Find where losses are concentrated.", ["profit_column"], 3)
        add("Recommendations", "generate_recommendations", "Suggest actions.", [], 8)
    elif any(w in q for w in ["customer","loyal","retention","repeat","churn","buyer","client","account"]):
        add("Customer Intelligence", "analyze_customer_loyalty", "Question asks about customers.", [], 1)
    elif any(w in q for w in ["product","item","sku","category","subcategory","portfolio","assortment"]):
        add("Product Intelligence", "analyze_product_portfolio", "Question asks about product/category.", [], 1)
    elif any(w in q for w in ["region","city","state","country","geography","location","zone","segment","market","area"]):
        add("Region Intelligence", "analyze_region_segment_performance", "Question asks about geography/segment.", [], 1)
    elif any(w in q for w in ["discount","offer","rebate","markdown"]):
        add("Metric Relationships", "analyze_metric_relationships", "Question asks about discount impact.", ["discount_column","profit_column"], 1)
    elif any(w in q for w in ["month","year","trend","dropped","increased","decreased","growth","latest","previous"]):
        add("Time Trend", "performance_over_time", "Question asks about time.", ["date_column"], 1)
        add("Anomaly Detection", "detect_business_anomalies", "Find abnormal changes.", [], 2)
    elif any(w in q for w in ["wrong","fix","problem","insights","analyze","management","focus","summary","board","hidden","risk"]):
        add("KPIs", "calculate_kpis", "General diagnostic starts with KPIs.", [], 1)
        if roles.get("profit_column"):
            add("Profit Intelligence", "analyze_profit_intelligence", "Profitability is central to business diagnostics.", ["profit_column"], 2)
        add("Hidden Insights", "generate_hidden_insights", "Surface hidden risks.", [], 3)
        add("Recommendations", "generate_recommendations", "Recommend focus areas.", [], 4)
    else:
        add("KPIs", "calculate_kpis", "Closest general dataset analysis.", [], 1)
        add("Hidden Insights", "generate_hidden_insights", "Best-effort insights.", [], 2)

    return sorted(plan, key=lambda x: x["priority"])
