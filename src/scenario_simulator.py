from __future__ import annotations
from src.utils import safe_numeric, profit_margin, prepare_analysis_dataset


def _base(df, roles):
    df, roles = prepare_analysis_dataset(df, roles)
    s, p = roles.get("sales_column"), roles.get("profit_column")
    sales = float(safe_numeric(df[s]).sum()) if s in df.columns else 0.0
    profit = float(safe_numeric(df[p]).sum()) if p in df.columns else 0.0
    return roles, sales, profit, profit_margin(profit, sales) if sales else 0.0


def _result(bs, bp, ss, sp, assumptions, limitations=None):
    return {"baseline_sales": bs, "baseline_profit": bp, "baseline_margin": profit_margin(bp, bs), "simulated_sales": ss, "simulated_profit": sp, "simulated_margin": profit_margin(sp, ss), "estimated_profit_change": sp-bp, "assumptions": assumptions, "limitations": limitations or ["Scenario estimate, not guaranteed prediction."]}


def simulate_discount_reduction(df, column_roles, reduction_percent):
    roles, bs, bp, bm = _base(df, column_roles)
    disc = roles.get("discount_column")
    lift = bs * (reduction_percent/100) * 0.05 if disc in df.columns else 0
    return _result(bs, bp, bs, bp + lift, [f"Assumes reducing discount by {reduction_percent}% saves 5% of affected sales as profit."])


def simulate_price_increase(df, column_roles, increase_percent):
    roles, bs, bp, bm = _base(df, column_roles)
    added = bs * increase_percent/100
    return _result(bs, bp, bs + added, bp + added, [f"Assumes volume stays constant after {increase_percent}% price increase."])


def simulate_removing_loss_making_products(df, column_roles):
    roles, bs, bp, bm = _base(df, column_roles)
    p = roles.get("profit_column")
    loss = float(safe_numeric(df[p])[safe_numeric(df[p]) < 0].sum()) if p in df.columns else 0
    return _result(bs, bp, bs, bp - loss, ["Assumes loss-making records are removed and no replacement sales occur."])


def simulate_quantity_change(df, column_roles, change_percent):
    roles, bs, bp, bm = _base(df, column_roles)
    factor = 1 + change_percent/100
    return _result(bs, bp, bs*factor, bp*factor, [f"Assumes sales and profit scale linearly with quantity change of {change_percent}%."])


def simulate_discount_cap(df, column_roles, max_discount):
    roles, bs, bp, bm = _base(df, column_roles)
    d = roles.get("discount_column")
    if d not in df.columns:
        return _result(bs, bp, bs, bp, ["No discount column."], ["Discount column missing."])
    disc = safe_numeric(df[d])
    threshold = max_discount if disc.max() > 1 else max_discount/100
    affected_sales_col = roles.get("sales_column")
    affected_sales = float(safe_numeric(df.loc[disc > threshold, affected_sales_col]).sum()) if affected_sales_col in df.columns else 0
    lift = affected_sales * 0.05
    return _result(bs, bp, bs, bp + lift, [f"Assumes capping discounts above {max_discount}% improves profit by 5% of affected sales."])


def simulate_margin_improvement(df, column_roles, target_dimension, target_value, margin_increase_percent):
    roles, bs, bp, bm = _base(df, column_roles)
    s = roles.get("sales_column")
    affected_sales = float(safe_numeric(df.loc[df[target_dimension].astype(str)==str(target_value), s]).sum()) if target_dimension in df.columns and s in df.columns else 0
    lift = affected_sales * margin_increase_percent/100
    return _result(bs, bp, bs, bp + lift, [f"Assumes {target_value} margin improves by {margin_increase_percent} percentage points."])
