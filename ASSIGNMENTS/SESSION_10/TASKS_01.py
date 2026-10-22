"""Task 1: Simple random sample of 20 users from 200 user IDs (Spotify-style feedback survey)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(seed=7)          # seed makes the demo reproducible

# 200 user IDs: U001 ... U200
users = pd.DataFrame({"user_id": [f"U{i:03d}" for i in range(1, 201)]})

# Simple random sampling WITHOUT replacement: every user has the same chance (20/200 = 10%)
picked_idx = rng.choice(users.index, size=20, replace=False)
sample = users.loc[np.sort(picked_idx)].reset_index(drop=True)

print(f"Population size : {len(users)}")
print(f"Sample size     : {len(sample)}")
print(f"Unique IDs      : {sample['user_id'].nunique()} (no duplicates)")
print(f"Selection chance: {len(sample) / len(users):.0%} per user\n")
print("Selected user IDs:")
print(sample["user_id"].to_list())

# Chart: 10 x 20 grid of all users, selected ones highlighted
fig, ax = plt.subplots(figsize=(8, 3.6))
rows, cols = np.divmod(np.arange(200), 20)
chosen = np.isin(np.arange(200), picked_idx)
ax.scatter(cols[~chosen], -rows[~chosen], s=60, color="#c9d6e2", label="Not selected")
ax.scatter(cols[chosen], -rows[chosen], s=90, color="#d1495b", label="Selected for survey (20)")
ax.set_axis_off()
ax.set_title("Random sample of 20 from 200 users")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task1.png"), dpi=150)
