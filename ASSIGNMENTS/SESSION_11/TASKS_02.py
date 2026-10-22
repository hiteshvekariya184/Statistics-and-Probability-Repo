"""Task 2: Why the sampling distribution of the mean becomes normal - Zomato order amounts."""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(seed=2)

# Skewed population: Zomato order amounts in rupees (many small orders, few very large party orders)
orders = rng.lognormal(mean=6.0, sigma=0.5, size=100_000)
mu, sigma = orders.mean(), orders.std()
print(f"Population of order amounts: mean = Rs {mu:.0f}, SD = Rs {sigma:.0f}, skewness = {stats.skew(orders):.2f}\n")

# Average of n orders, repeated 5000 times, for growing n
sizes = [1, 2, 5, 30]
rows = []
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True)
for ax, n in zip(axes, sizes):
    means = rng.choice(orders, size=(5000, n)).mean(axis=1)
    rows.append({"n (orders averaged)": n, "Mean of means": means.mean(),
                 "SD of means": means.std(ddof=1), "Theory sigma/sqrt(n)": sigma / np.sqrt(n),
                 "Skewness": stats.skew(means)})
    ax.hist(means, bins=40, color="#2a6f97", edgecolor="white")
    ax.set_title(f"n = {n}")
    ax.set_xlabel("Average order (Rs)")
axes[0].set_ylabel("Frequency")
fig.suptitle("Average of n Zomato orders: skew fades and the shape turns bell-like as n grows")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task2.png"), dpi=150)

print(pd.DataFrame(rows).round(2).to_string(index=False))
