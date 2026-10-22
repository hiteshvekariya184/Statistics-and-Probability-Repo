"""Task 5: Proportional stratified sample of 40 IPL fans grouped by favourite team.
Own logic: numpy index arrays + Hamilton (largest-remainder) quotas, no groupby.sample().
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(seed=2026)

# ---- 1000 IPL fans, each with a favourite team ----
team_sizes = {"CSK": 210, "MI": 190, "RCB": 180, "KKR": 110, "SRH": 80,
              "DC": 70, "RR": 60, "GT": 50, "PBKS": 30, "LSG": 20}
fans = pd.DataFrame({
    "fan_id": [f"F{i:04d}" for i in range(1, 1001)],
    "fav_team": rng.permutation(np.repeat(list(team_sizes), list(team_sizes.values()))),
})
TARGET = 40


def allocate_quotas(sizes, total):
    """Hamilton method: floor of each exact share, then give leftover seats to biggest fractions."""
    teams = list(sizes)
    pop = np.array([sizes[t] for t in teams], dtype=float)
    exact = pop / pop.sum() * total
    base = np.floor(exact).astype(int)
    left = total - base.sum()
    # biggest fractional part first; ties go to the larger team
    order = sorted(range(len(teams)), key=lambda i: (-(exact[i] - base[i]), -pop[i]))
    for i in order[:left]:
        base[i] += 1
    return dict(zip(teams, base)), dict(zip(teams, exact))


actual_sizes = fans["fav_team"].value_counts().to_dict()
quota, exact = allocate_quotas(actual_sizes, TARGET)

# ---- Draw: for each team pick row positions at random without replacement ----
team_col = fans["fav_team"].to_numpy()
chosen_rows = []
for team, q in quota.items():
    positions = np.flatnonzero(team_col == team)         # rows belonging to this team
    chosen_rows.extend(rng.choice(positions, size=q, replace=False))
sample = fans.iloc[sorted(chosen_rows)].reset_index(drop=True)

report = pd.DataFrame({
    "fans": pd.Series(actual_sizes),
    "population_%": pd.Series(actual_sizes) / len(fans) * 100,
    "exact_quota": pd.Series(exact),
    "sample_n": pd.Series(quota),
}).sort_values("fans", ascending=False)
report["sample_%"] = report["sample_n"] / TARGET * 100
print(report.round(2).to_string(), "\n")
print(f"Total sampled: {len(sample)} (target {TARGET})")
print("First 5 sampled fans:")
print(sample.head().to_string(index=False))

# ---- Chart ----
fig, ax = plt.subplots(figsize=(7.5, 4.2))
y = np.arange(len(report))
ax.barh(y + 0.2, report["population_%"], 0.4, color="#9bb7d4", label="Population (1000 fans)")
ax.barh(y - 0.2, report["sample_%"], 0.4, color="#2a6f97", label="Sample (40 fans)")
ax.set_yticks(y)
ax.set_yticklabels(report.index)
ax.invert_yaxis()
ax.set_xlabel("Share (%)")
ax.set_title("IPL fans: stratified sample vs population")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task5.png"), dpi=150)
