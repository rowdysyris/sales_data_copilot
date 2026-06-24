from __future__ import annotations
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_sample_sales_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    categories = {
        "Furniture": ["Chairs", "Tables", "Bookcases", "Furnishings"],
        "Office Supplies": ["Binders", "Paper", "Storage", "Art"],
        "Technology": ["Phones", "Accessories", "Machines", "Copiers"],
    }
    regions = ["North", "South", "East", "West"]
    segments = ["Consumer", "Corporate", "Home Office"]
    states = ["Karnataka", "Maharashtra", "Delhi", "Telangana", "Tamil Nadu"]
    cities = ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai"]
    customers = [f"Customer {i:03d}" for i in range(1, 301)]
    loyal = customers[:30]

    rows = []
    start = datetime(2023, 1, 1)
    for i in range(n):
        cust = rng.choice(loyal if rng.random() < 0.35 else customers)
        cat = rng.choice(list(categories.keys()), p=[0.34,0.33,0.33])
        sub = rng.choice(categories[cat])
        region = rng.choice(regions, p=[0.25,0.30,0.23,0.22])
        seg = rng.choice(segments)
        date = start + timedelta(days=int(rng.integers(0, 720)))
        qty = int(rng.integers(1, 8))
        base = {"Furniture": 2500, "Office Supplies": 700, "Technology": 5000}[cat] * rng.uniform(0.5, 1.8)
        sales = base * qty
        discount = float(rng.choice([0,0.05,0.1,0.15,0.2,0.3,0.4,0.5], p=[.20,.15,.18,.15,.12,.10,.06,.04]))
        margin_rate = {"Furniture": 0.10, "Office Supplies": 0.16, "Technology": 0.24}[cat]
        if sub == "Tables":
            discount = float(rng.choice([0.25,0.3,0.4,0.5], p=[.25,.35,.25,.15]))
            margin_rate = -0.08
        if region == "South":
            margin_rate -= 0.07
        profit = sales * (margin_rate - discount * 0.55) + rng.normal(0, sales * 0.04)
        rows.append({
            "Order ID": f"ORD-{100000+i}",
            "Order Date": date.date(),
            "Ship Date": (date + timedelta(days=int(rng.integers(1, 8)))).date(),
            "Customer ID": f"C-{customers.index(cust)+1:03d}",
            "Customer Name": cust,
            "Segment": seg,
            "Country": "India",
            "City": rng.choice(cities),
            "State": rng.choice(states),
            "Region": region,
            "Category": cat,
            "Sub-Category": sub,
            "Product Name": f"{sub} Model {rng.integers(1, 25)}",
            "Sales": round(sales, 2),
            "Quantity": qty,
            "Discount": discount,
            "Profit": round(profit, 2),
        })
    df = pd.DataFrame(rows)
    # inject missing values, duplicates, outliers
    for col in ["City", "Customer Name"]:
        df.loc[rng.choice(df.index, size=20, replace=False), col] = np.nan
    df = pd.concat([df, df.sample(15, random_state=seed)], ignore_index=True)
    df.loc[rng.choice(df.index, size=5, replace=False), "Sales"] *= 8
    return df


if __name__ == "__main__":
    os.makedirs("sample_data", exist_ok=True)
    out = "sample_data/sample_sales_data.csv"
    generate_sample_sales_data().to_csv(out, index=False)
    print(f"Generated {out}")
