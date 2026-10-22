import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)

# Build a 500-user dataset that matches the given counts
# Both = 50, Accessories only = 120-50 = 70, Phone only = 80-50 = 30, Neither = 350
users = pd.DataFrame(
    [(1, 1)] * 50 + [(1, 0)] * 70 + [(0, 1)] * 30 + [(0, 0)] * 350,
    columns=["bought_accessories", "bought_phone"],
)
assert len(users) == 500

# Contingency table (counts)
table = pd.crosstab(users["bought_accessories"], users["bought_phone"],
                    margins=True, margins_name="Total")
print("Contingency table (rows = accessories, cols = phone):")
print(table, "\n")

n = len(users)
p_acc = users["bought_accessories"].mean()                       # P(B)
p_both = ((users["bought_accessories"] == 1) &
          (users["bought_phone"] == 1)).mean()                   # P(A and B)
p_phone_given_acc = p_both / p_acc                               # P(A|B)

# Same answer straight from pandas by filtering the accessories buyers
check = users.loc[users["bought_accessories"] == 1, "bought_phone"].mean()

print(f"P(Accessories)         = {p_acc:.4f}")
print(f"P(Phone and Acc.)      = {p_both:.4f}")
print(f"P(Phone | Accessories) = {p_phone_given_acc:.4f} ({p_phone_given_acc:.2%})")
print(f"Pandas filter check    = {check:.4f}")

# Chart: unconditional vs conditional probability of buying a phone
p_phone = users["bought_phone"].mean()
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["P(Phone)", "P(Phone | Accessories)"],
              [p_phone, p_phone_given_acc], color=["#9bb7d4", "#2a6f97"])
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
            f"{b.get_height():.2%}", ha="center")
ax.set_ylim(0, 0.6)
ax.set_ylabel("Probability")
ax.set_title("Accessory buyers are more likely to buy a phone")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task1.png"), dpi=150)