"""Task 4: How sample size (50, 200, 500) affects accuracy when estimating mean Swiggy delivery time."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(seed=21)

# ---- 10,000 delivery records (minutes); right-skewed like real delivery times ----
N = 10_000
times = rng.gamma(shape=16, scale=2.0, size=N)       # mean about 32 min
pop_mean, pop_sd = times.mean(), times.std(ddof=1)
print(f"Population: N = {N}, mean = {pop_mean:.2f} min, SD = {pop_sd:.2f} min\n")

rows = []
for n in (50, 200, 500):
    # Theory: standard error with finite population correction (sampling w/o replacement)
    se = pop_sd / np.sqrt(n) * np.sqrt((N - n) / (N - 1))
    moe = 1.96 * se                                   # 95% margin of error

    # Experiment: draw 2000 different samples of size n and see how much the means vary
    means = np.array([rng.choice(times, size=n, replace=False).mean() for _ in range(2000)])
    errors = np.abs(means - pop_mean)
    coverage = np.mean(errors <= moe)

    rows.append({"n": n, "SE (min)": se, "95% margin (+/- min)": moe,
                 "Empirical SD of means": means.std(ddof=1),
                 "Mean abs error (min)": errors.mean(),
                 "Within margin %": coverage * 100})
    if n == 50:
        means_50 = means
    elif n == 200:
        means_200 = means
    else:
        means_500 = means

res = pd.DataFrame(rows).set_index("n").round(3)
print(res.to_string(), "\n")

print(f"Going 50 -> 200 (4x data): margin shrinks by factor {res.loc[50, '95% margin (+/- min)'] / res.loc[200, '95% margin (+/- min)']:.2f}")
print(f"Going 50 -> 500 (10x data): margin shrinks by factor {res.loc[50, '95% margin (+/- min)'] / res.loc[500, '95% margin (+/- min)']:.2f}")

# One concrete sample of each size
print("\nOne example sample of each size:")
for n in (50, 200, 500):
    s = rng.choice(times, size=n, replace=False)
    m = s.mean()
    e = 1.96 * s.std(ddof=1) / np.sqrt(n) * np.sqrt((N - n) / (N - 1))
    print(f"  n = {n:>3}: mean = {m:.2f}, 95% CI = ({m - e:.2f}, {m + e:.2f})")

# ---- Charts ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar([str(n) for n in res.index], res["95% margin (+/- min)"], color="#2a6f97")
axes[0].set_xlabel("Sample size")
axes[0].set_ylabel("95% margin of error (+/- min)")
axes[0].set_title("Bigger sample, smaller error")
axes[1].boxplot([means_50, means_200, means_500], tick_labels=["50", "200", "500"])
axes[1].axhline(pop_mean, color="red", ls="--", label=f"True mean {pop_mean:.1f}")
axes[1].set_xlabel("Sample size")
axes[1].set_ylabel("Sample mean (min)")
axes[1].set_title("Spread of 2000 sample means")
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task4.png"), dpi=150)
