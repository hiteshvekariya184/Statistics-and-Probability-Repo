"""Task 1: CLT simulation - 1000 samples of size 30 from a skewed 'Instagram likes' dataset."""
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

# ---- Skewed population: likes on 100,000 Instagram posts (most posts low, a few go viral) ----
likes = np.round(rng.lognormal(mean=5.0, sigma=0.8, size=100_000))
pop_mean, pop_sd = likes.mean(), likes.std()

# ---- 1000 random samples of size 30 -> mean of each ----
N_SAMPLES, n = 1000, 30
sample_means = np.array([rng.choice(likes, size=n, replace=False).mean() for _ in range(N_SAMPLES)])

summary = pd.DataFrame({
    "Population (likes)": [pop_mean, pop_sd, stats.skew(likes)],
    f"1000 sample means (n={n})": [sample_means.mean(), sample_means.std(ddof=1), stats.skew(sample_means)],
}, index=["Mean", "Std deviation", "Skewness"]).round(2)
print(summary.to_string(), "\n")
print(f"Theory: SD of sample means = sigma / sqrt(n) = {pop_sd:.2f} / {np.sqrt(n):.3f} = {pop_sd / np.sqrt(n):.2f}")

# ---- Plots ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(likes, bins=60, color="#9bb7d4", edgecolor="white")
axes[0].set_title("Population: likes per post (right-skewed)")
axes[0].set_xlabel("Likes")
axes[0].set_ylabel("Number of posts")

axes[1].hist(sample_means, bins=30, density=True, color="#2a6f97", edgecolor="white", label="Sample means")
x = np.linspace(sample_means.min(), sample_means.max(), 300)
axes[1].plot(x, stats.norm.pdf(x, pop_mean, pop_sd / np.sqrt(n)), "r-", lw=2, label="Normal curve (theory)")
axes[1].axvline(pop_mean, color="black", ls="--", label=f"Population mean {pop_mean:.0f}")
axes[1].set_title(f"Distribution of 1000 sample means (n = {n})")
axes[1].set_xlabel("Sample mean of likes")
axes[1].set_ylabel("Density")
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task1.png"), dpi=150)
