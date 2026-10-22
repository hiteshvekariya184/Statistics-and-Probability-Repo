"""Task 3: Business use of CLT - Flipkart average order value (AOV) A/B test and sample-size planning."""
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(seed=3)

# Order values are skewed, yet CLT lets us treat the SAMPLE MEAN as approximately normal.
control = rng.lognormal(mean=6.50, sigma=0.60, size=400)         # old checkout page
variant = rng.lognormal(mean=6.50, sigma=0.60, size=400) * 1.06   # new page with "frequently bought together"

def mean_ci(x, z=1.96):
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return m, se, (m - z * se, m + z * se)

m_c, se_c, ci_c = mean_ci(control)
m_v, se_v, ci_v = mean_ci(variant)
diff = m_v - m_c
se_diff = np.sqrt(se_c ** 2 + se_v ** 2)
z = diff / se_diff
p_value = 2 * (1 - stats.norm.cdf(abs(z)))

print(pd.DataFrame({"n": [400, 400], "Mean AOV (Rs)": [m_c, m_v], "SE": [se_c, se_v],
                    "95% CI low": [ci_c[0], ci_v[0]], "95% CI high": [ci_c[1], ci_v[1]]},
                   index=["Control", "New checkout"]).round(1).to_string(), "\n")
print(f"Difference in mean AOV = Rs {diff:.1f}  (SE = {se_diff:.1f})")
print(f"95% CI for difference  = ({diff - 1.96 * se_diff:.1f}, {diff + 1.96 * se_diff:.1f})")
print(f"z = {z:.2f}, p-value = {p_value:.4f}  ->  "
      f"{'significant' if p_value < 0.05 else 'not significant'} at 5%\n")

# Sample-size planning: orders needed to estimate AOV within +/- Rs 30 at 95% confidence
sigma_hat = control.std(ddof=1)
for margin in (50, 30, 20):
    need = int(np.ceil((1.96 * sigma_hat / margin) ** 2))
    print(f"Margin +/- Rs {margin}: need about {need} orders")

# Chart: mean AOV with 95% CI
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.bar(["Control", "New checkout"], [m_c, m_v], yerr=[1.96 * se_c, 1.96 * se_v],
       capsize=8, color=["#9bb7d4", "#2a6f97"])
ax.set_ylabel("Average order value (Rs)")
ax.set_title("Flipkart AOV with 95% CI (CLT-based)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task3.png"), dpi=150)
