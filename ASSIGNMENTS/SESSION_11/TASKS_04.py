"""Task 4: Compare CLT simulation with n = 5 vs n = 30 (same skewed Instagram likes population)."""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(seed=1)

likes = np.round(rng.lognormal(mean=5.0, sigma=0.8, size=100_000))
pop_mean, pop_sd = likes.mean(), likes.std()


def simulate(n, n_samples=1000):
    return np.array([rng.choice(likes, size=n, replace=False).mean() for _ in range(n_samples)])


results = {n: simulate(n) for n in (5, 30)}

rows = []
for n, m in results.items():
    rows.append({"Sample size n": n, "Mean of means": m.mean(), "SD of means": m.std(ddof=1),
                 "Theory sigma/sqrt(n)": pop_sd / np.sqrt(n), "Skewness": stats.skew(m),
                 "Shapiro-Wilk p": stats.shapiro(m).pvalue})
print(f"Population mean = {pop_mean:.1f}, SD = {pop_sd:.1f}\n")
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\n(Shapiro-Wilk p > 0.05 would mean 'no evidence against normality'; lower = less normal.)")

# Same x-axis for both so the spread can be compared fairly
lo = min(m.min() for m in results.values())
hi = max(m.max() for m in results.values())
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
for ax, (n, m) in zip(axes, results.items()):
    ax.hist(m, bins=35, range=(lo, hi), density=True, color="#2a6f97", edgecolor="white")
    x = np.linspace(lo, hi, 300)
    ax.plot(x, stats.norm.pdf(x, pop_mean, pop_sd / np.sqrt(n)), "r-", lw=2, label="Normal curve (theory)")
    ax.axvline(pop_mean, color="black", ls="--")
    ax.set_title(f"n = {n}  (skewness {stats.skew(m):.2f})")
    ax.set_xlabel("Sample mean of likes")
axes[0].set_ylabel("Density")
axes[0].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task4.png"), dpi=150)
