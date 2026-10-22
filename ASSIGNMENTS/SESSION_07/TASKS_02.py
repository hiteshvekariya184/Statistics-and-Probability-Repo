"""Task 2: P(Smartphone | Headphones) - exact formula + Monte Carlo check (numpy/pandas)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)

# ---- Exact calculation ----
p_headphones = 0.30                         # P(B)
p_phone_given_hp = 0.40                     # given in the question
p_both = p_headphones * p_phone_given_hp    # P(A and B) = 0.12
p_cond = p_both / p_headphones              # P(A|B) = P(A and B) / P(B)
print(f"P(Headphones)                = {p_headphones:.2f}")
print(f"P(Smartphone and Headphones) = {p_both:.2f}")
print(f"P(Smartphone | Headphones)   = {p_cond:.2f} ({p_cond:.0%})\n")

# ---- Simulation with 100,000 users to verify ----
rng = np.random.default_rng(42)
n = 100_000
headphones = rng.random(n) < p_headphones
# Smartphone rate for non-headphone users is NOT given; 0.20 is an illustrative
# assumption and does not affect the answer.
smartphone = np.where(headphones, rng.random(n) < 0.40, rng.random(n) < 0.20)

df = pd.DataFrame({"headphones": headphones, "smartphone": smartphone})
sim_p_b = df["headphones"].mean()
sim_p_ab = (df["headphones"] & df["smartphone"]).mean()
sim_cond = sim_p_ab / sim_p_b

print("Simulation (100,000 users):")
print(f"  P(Headphones)                = {sim_p_b:.4f}")
print(f"  P(Smartphone and Headphones) = {sim_p_ab:.4f}")
print(f"  P(Smartphone | Headphones)   = {sim_cond:.4f}")
print(f"  Groupby check:\n{df.groupby('headphones')['smartphone'].mean().round(4)}")

# Chart
fig, ax = plt.subplots(figsize=(6, 4))
vals = [p_cond, sim_cond]
bars = ax.bar(["Exact", "Simulated"], vals, color=["#2a6f97", "#9bb7d4"])
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
            f"{b.get_height():.3f}", ha="center")
ax.set_ylim(0, 0.55)
ax.set_ylabel("P(Smartphone | Headphones)")
ax.set_title("Exact vs simulated conditional probability")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task2.png"), dpi=150)
