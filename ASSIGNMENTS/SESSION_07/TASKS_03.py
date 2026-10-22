"""Task 3: bayes_churn_probability() + test + sensitivity analysis (pandas/numpy/matplotlib)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)


def bayes_churn_probability(prior_churn, prob_email_given_churn, prob_email):
    """P(Churn | Email) = P(Email | Churn) * P(Churn) / P(Email)"""
    if not (0 <= prior_churn <= 1 and 0 <= prob_email_given_churn <= 1
            and 0 < prob_email <= 1):
        raise ValueError("Probabilities must be in [0, 1] and prob_email must be > 0.")
    posterior = (prob_email_given_churn * prior_churn) / prob_email
    if posterior > 1:
        raise ValueError("Inputs are inconsistent: resulting probability exceeds 1.")
    return posterior


# ---- Test with the given values ----
result = bayes_churn_probability(prior_churn=0.15,
                                 prob_email_given_churn=0.6,
                                 prob_email=0.25)
print(f"P(Churn | Email) = {result:.4f} ({result:.0%})\n")

# ---- Sensitivity table: how the answer changes with P(Email | Churn) ----
likelihoods = np.array([0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
table = pd.DataFrame({
    "P(Email|Churn)": likelihoods,
    "P(Churn|Email)": [bayes_churn_probability(0.15, l, 0.25) for l in likelihoods],
})
print("Sensitivity analysis (prior = 0.15, P(Email) = 0.25):")
print(table.round(4).to_string(index=False))

# ---- Chart: prior vs posterior ----
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(table["P(Email|Churn)"], table["P(Churn|Email)"], marker="o", color="#2a6f97")
ax.axhline(0.15, ls="--", color="grey", label="Prior P(Churn) = 0.15")
ax.scatter([0.6], [result], color="red", zorder=5, label=f"Given case = {result:.2f}")
ax.set_xlabel("P(Email | Churn)")
ax.set_ylabel("P(Churn | Email)")
ax.set_title("Posterior churn probability vs likelihood")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task3.png"), dpi=150)
