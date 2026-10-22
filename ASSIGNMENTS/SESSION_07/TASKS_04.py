"""Task 4: Fraud alert system - Bayes' Theorem with a pandas confusion matrix."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)

p_fraud = 0.02            # P(F)
tpr = 0.90                # P(A|F)
fpr = 0.05                # P(A|not F)


def p_fraud_given_alert(p_f, tpr, fpr):
    p_alert = tpr * p_f + fpr * (1 - p_f)       # law of total probability
    return (tpr * p_f) / p_alert, p_alert


posterior, p_alert = p_fraud_given_alert(p_fraud, tpr, fpr)
print(f"P(A)     = 0.90*0.02 + 0.05*0.98 = {p_alert:.4f}")
print(f"P(F | A) = {posterior:.4f} ({posterior:.2%})\n")

# ---- Confusion matrix for 100,000 transactions ----
N = 100_000
fraud = int(N * p_fraud)
normal = N - fraud
cm = pd.DataFrame(
    {"Alert": [round(fraud * tpr), round(normal * fpr)],
     "No alert": [round(fraud * (1 - tpr)), round(normal * (1 - fpr))]},
    index=["Fraud", "Normal"],
)
cm["Total"] = cm.sum(axis=1)
cm.loc["Total"] = cm.sum()
print("Confusion matrix (100,000 transactions):")
print(cm)
print(f"\nPrecision from counts = {cm.loc['Fraud', 'Alert'] / cm.loc['Total', 'Alert']:.4f}")

# ---- Base-rate sweep ----
rates = np.linspace(0.001, 0.20, 200)
post = [p_fraud_given_alert(r, tpr, fpr)[0] for r in rates]
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(rates * 100, np.array(post) * 100, color="#2a6f97")
ax.scatter([2], [posterior * 100], color="red", zorder=5,
           label=f"2% fraud -> {posterior:.1%}")
ax.set_xlabel("Fraud rate (%)")
ax.set_ylabel("P(Fraud | Alert) (%)")
ax.set_title("Base-rate effect on alert precision")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task4.png"), dpi=150)
