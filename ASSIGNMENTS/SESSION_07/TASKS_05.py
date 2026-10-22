"""Task 5: Food delivery (Swiggy/Zomato) scenario - conditional probability and Bayes with pandas."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)

# Illustrative dataset of 10,000 Friday-night sessions
#   Biryani & searched non-veg = 1400, Biryani & not searched = 600
#   Other   & searched non-veg = 2100, Other   & not searched = 5900
rows = ([("Biryani", 1)] * 1400 + [("Biryani", 0)] * 600 +
        [("Other", 1)] * 2100 + [("Other", 0)] * 5900)
sessions = pd.DataFrame(rows, columns=["ordered", "searched_nonveg"])

print(pd.crosstab(sessions["ordered"], sessions["searched_nonveg"],
                  margins=True, margins_name="Total"), "\n")

is_b = sessions["ordered"] == "Biryani"
is_s = sessions["searched_nonveg"] == 1

p_b = is_b.mean()                          # prior P(Biryani)
p_s = is_s.mean()                          # evidence P(Searched)
p_s_given_b = is_s[is_b].mean()            # likelihood P(Searched | Biryani)

posterior = p_s_given_b * p_b / p_s        # Bayes' Theorem
direct = is_b[is_s].mean()                 # direct conditional probability from data

print(f"Prior      P(Biryani)              = {p_b:.2f}")
print(f"Likelihood P(Searched | Biryani)   = {p_s_given_b:.2f}")
print(f"Evidence   P(Searched)             = {p_s:.2f}")
print(f"Posterior  P(Biryani | Searched)   = {posterior:.2f}")
print(f"Direct check from data             = {direct:.2f}")

# Chart: prior vs posterior
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["Prior\nP(Biryani)", "Posterior\nP(Biryani | Searched non-veg)"],
              [p_b, posterior], color=["#9bb7d4", "#2a6f97"])
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
            f"{b.get_height():.0%}", ha="center")
ax.set_ylim(0, 0.55)
ax.set_ylabel("Probability")
ax.set_title("Bayes update from one user signal")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task5.png"), dpi=150)
