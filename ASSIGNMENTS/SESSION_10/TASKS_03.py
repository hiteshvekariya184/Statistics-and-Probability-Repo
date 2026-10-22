"""Task 3: Systematic sampling - every 10th review from 500 Zomato restaurant reviews."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(OUT, exist_ok=True)


def systematic_sample(reviews, step=10, start=None):
    """Return every `step`-th item. Default: the 10th, 20th, 30th ... review.

    start : 0-based index of the first pick. None -> step - 1 (i.e. the 10th review).
            Pass a random number in range(step) for a random-start systematic sample.
    """
    if step < 1:
        raise ValueError("step must be >= 1")
    if start is None:
        start = step - 1
    return reviews.iloc[start::step] if isinstance(reviews, pd.DataFrame) else reviews[start::step]


# ---- 500 Zomato-style reviews ----
rng = np.random.default_rng(seed=3)
reviews = pd.DataFrame({
    "review_id": np.arange(1, 501),
    "rating": rng.choice([1, 2, 3, 4, 5], size=500, p=[0.06, 0.09, 0.20, 0.37, 0.28]),
})

sample = systematic_sample(reviews, step=10)

print(f"Population: {len(reviews)} reviews | Sample: {len(sample)} reviews")
print("Selected review IDs:", sample["review_id"].to_list(), "\n")

# Random-start version for comparison
random_start = int(rng.integers(0, 10))
sample_rs = systematic_sample(reviews, step=10, start=random_start)
print(f"Random-start variant (start index {random_start}): first IDs = "
      f"{sample_rs['review_id'].head(5).to_list()} ... total {len(sample_rs)}\n")

print(f"Average rating - population: {reviews['rating'].mean():.2f}")
print(f"Average rating - sample    : {sample['rating'].mean():.2f}")

# ---- Chart: rating distribution, population vs sample ----
dist = pd.DataFrame({
    "Population": reviews["rating"].value_counts(normalize=True).sort_index() * 100,
    "Sample": sample["rating"].value_counts(normalize=True).sort_index() * 100,
}).fillna(0)
ax = dist.plot(kind="bar", figsize=(7, 4), color=["#9bb7d4", "#2a6f97"], rot=0)
ax.set_xlabel("Rating")
ax.set_ylabel("Share of reviews (%)")
ax.set_title("Every-10th-review sample vs all 500 reviews")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task3.png"), dpi=150)
