"""Creates OnlineRetail_sample.csv: a SMALL STAND-IN dataset with the same columns as the Kaggle
Online Retail file (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country).
It is simulated, NOT the real data. Put the real OnlineRetail.csv in this folder and the task scripts use it instead.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
rng = np.random.default_rng(seed=42)

products = [(f"{10000 + i}", name, round(float(p), 2)) for i, (name, p) in enumerate(zip(
    ["GIFT BOX SET", "CERAMIC MUG", "CANDLE HOLDER", "WOODEN TOY TRAIN", "PAPER BUNTING", "GLASS JAR",
     "CUSHION COVER", "TEA TOWEL", "STORAGE TIN", "PHOTO FRAME", "BIRD HOUSE", "LUNCH BAG",
     "WALL CLOCK", "DOOR SIGN", "PLANT POT", "COASTER SET", "NOTEBOOK", "WRAPPING PAPER",
     "CHRISTMAS BAUBLE", "ADVENT CALENDAR", "KEY RING", "HOT WATER BOTTLE", "PICNIC BASKET",
     "MIRROR", "VASE", "TRAY", "APRON", "CAKE STAND", "LANTERN", "DOORMAT"],
    rng.lognormal(0.9, 0.7, 30).clip(0.4, 25)))]

countries = ["United Kingdom", "Germany", "France", "EIRE", "Netherlands", "Australia", "Spain", "Belgium", "Switzerland"]
weights = [0.80, 0.055, 0.05, 0.03, 0.02, 0.015, 0.01, 0.01, 0.01]
cust_pool = {c: rng.choice(np.arange(12346, 18288), size=n, replace=False)
             for c, n in zip(countries, [150, 10, 9, 4, 4, 3, 3, 3, 3])}
mult = {"Germany": 1.6, "France": 1.3, "Netherlands": 3.0, "EIRE": 2.0, "Australia": 3.5}   # wholesale-style buyers

n_inv = 720
month_w = np.array([1, 1, 1.1, 1, 1.1, 1.2, 1.1, 1.2, 1.6, 1.9, 2.4, 1.0])   # Dec-10..Nov-11, busier in autumn
days = pd.date_range("2010-12-01", "2011-12-09", freq="D")
day_w = np.array([month_w[(d.month - 12) % 12] for d in days]); day_w = day_w / day_w.sum()
inv_dates = np.sort(rng.choice(days, size=n_inv, p=day_w)) + pd.to_timedelta(rng.integers(8 * 60, 20 * 60, n_inv), unit="m")

rows = []
inv_no = 536365
for k in range(n_inv):
    country = rng.choice(countries, p=weights)
    cust = int(rng.choice(cust_pool[country]))
    if country == "United Kingdom" and rng.random() < 0.10:
        cust = np.nan                                           # some invoices have no CustomerID
    n_lines = int(np.clip(rng.lognormal(1.5, 0.7), 1, 28))
    wholesale = rng.random() < 0.15
    for pid, desc, price in [products[i] for i in rng.choice(30, n_lines, replace=False)]:
        qty = int(rng.choice([1, 2, 3, 4, 6, 8, 12], p=[.2, .2, .15, .15, .1, .1, .1]))
        if wholesale:
            qty *= int(rng.choice([6, 12, 24, 36]))
        qty = int(qty * mult.get(country, 1.0) * (1 if rng.random() < 0.7 else 2))
        rows.append([str(inv_no), pid, desc, qty, inv_dates[k], price, cust, country])
    inv_no += 1

# A few very large bulk orders (order value above 10,000)
for k, country in [(40, "United Kingdom"), (200, "United Kingdom"), (450, "EIRE"), (600, "Germany")]:
    rows.append([str(536365 + k), "10099", "BULK ORDER CHRISTMAS BAUBLE", int(rng.integers(2500, 4000)), inv_dates[k],
                 3.45, rows[[r[0] for r in rows].index(str(536365 + k))][6], country])

# Cancellations (InvoiceNo starts with C, negative quantity)
for _ in range(15):
    src = rows[int(rng.integers(0, len(rows)))]
    rows.append(["C" + str(inv_no), src[1], src[2], -int(rng.integers(1, 10)), src[4] + pd.Timedelta(days=int(rng.integers(1, 20))), src[5], src[6], src[7]])
    inv_no += 1

df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
df = df.sort_values(["InvoiceDate", "InvoiceNo"]).reset_index(drop=True)
df["InvoiceDate"] = df["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S")
df["CustomerID"] = df["CustomerID"].astype("Int64")
df.to_csv(os.path.join(HERE, "OnlineRetail_sample.csv"), index=False)
print(f"Saved OnlineRetail_sample.csv with {len(df):,} rows, {df['InvoiceNo'].nunique()} invoices")
print(df.head(5).to_string(index=False))
print("\nRows per country:\n", df["Country"].value_counts().to_string())
