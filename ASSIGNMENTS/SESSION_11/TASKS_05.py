"""Task 5: CLT demo with an exponential (non-normal) distribution.
Code generated with an AI assistant (Claude), then run and checked.
Prompt: "Write Python code that demonstrates the Central Limit Theorem using random
samples from a non-normal distribution such as the exponential distribution."
"""
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(seed=5)

scale = 2.0                       # exponential with mean = SD = 2
mu, sigma = scale, scale
sizes = [1, 5, 30, 100]
n_samples = 5000

fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
print(f"Exponential(scale={scale}): mean = {mu}, SD = {sigma}, skewness = 2.0\n")
print(f"{'n':>4} {'mean of means':>14} {'SD of means':>12} {'sigma/sqrt(n)':>14} {'skewness':>9}")
for ax, n in zip(axes, sizes):
    means = rng.exponential(scale, size=(n_samples, n)).mean(axis=1)
    print(f"{n:>4} {means.mean():>14.3f} {means.std(ddof=1):>12.3f} {sigma / np.sqrt(n):>14.3f} {stats.skew(means):>9.3f}")
    ax.hist(means, bins=40, density=True, color="#2a6f97", edgecolor="white")
    x = np.linspace(means.min(), means.max(), 300)
    ax.plot(x, stats.norm.pdf(x, mu, sigma / np.sqrt(n)), "r-", lw=2)
    ax.set_title(f"n = {n}")
    ax.set_xlabel("Sample mean")
axes[0].set_ylabel("Density")
fig.suptitle("CLT with an exponential distribution (red = normal curve predicted by CLT)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task5.png"), dpi=150)
