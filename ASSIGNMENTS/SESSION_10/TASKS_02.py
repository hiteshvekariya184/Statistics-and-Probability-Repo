"""Task 2: Proportional stratified sample of 60 orders from 1000 Flipkart orders, by city."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(seed=11)

# ---- Build 1000 orders labelled by city ----
city_counts = {"Ahmedabad": 340, "Surat": 260, "Vadodara": 210, "Rajkot": 120, "Bhavnagar": 70}
cities = np.repeat(list(city_counts.keys()), list(city_counts.values()))
orders = pd.DataFrame({"order_id": [f"FK{i:04d}" for i in range(1, 1001)],
                       "city": rng.permutation(cities)})
SAMPLE_SIZE = 60

# ---- Step 1: proportion of orders per city ----
proportion = orders["city"].value_counts(normalize=True)

# ---- Step 2: quota per city (largest-remainder method so quotas add up to exactly 60) ----
exact = proportion * SAMPLE_SIZE
quota = exact.astype(int)                                    # whole part
shortfall = SAMPLE_SIZE - quota.sum()                        # seats left to hand out
extra = (exact - quota).sort_values(ascending=False).index[:shortfall]
quota.loc[extra] += 1

summary = pd.DataFrame({"orders": orders["city"].value_counts(),
                        "proportion_%": (proportion * 100).round(1),
                        "exact_quota": exact.round(2),
                        "final_quota": quota})
print(summary, "\n")

# ---- Step 3: draw that many random orders from each city ----
parts = [orders[orders["city"] == c].sample(n=q, random_state=int(rng.integers(1_000_000)))
         for c, q in quota.items()]
sample = pd.concat(parts).sort_values("order_id").reset_index(drop=True)

print(f"Sample size: {len(sample)}")
print("Sample count by city:")
print(sample["city"].value_counts().to_string(), "\n")

check = pd.DataFrame({"population_%": proportion * 100,
                      "sample_%": sample["city"].value_counts(normalize=True) * 100}).round(1)
print("Population vs sample share:")
print(check)

# ---- Chart ----
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(check))
ax.bar(x - 0.2, check["population_%"], 0.4, label="Population (1000 orders)", color="#9bb7d4")
ax.bar(x + 0.2, check["sample_%"], 0.4, label="Sample (60 orders)", color="#2a6f97")
ax.set_xticks(x)
ax.set_xticklabels(check.index)
ax.set_ylabel("Share of orders (%)")
ax.set_title("Stratified sample mirrors the population")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task2.png"), dpi=150)
