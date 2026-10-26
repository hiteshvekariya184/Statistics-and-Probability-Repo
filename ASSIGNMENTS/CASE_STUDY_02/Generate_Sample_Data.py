"""Creates Telco-Customer-Churn_sample.csv: a SIMULATED stand-in with the case-study columns
(CustomerID, Gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, InternetService, Contract,
MonthlyCharges, TotalCharges, Churn). It is NOT the real Kaggle data. Put the real Telco-Customer-Churn.csv
in this folder and the task scripts use it automatically.
"""
import os
import string
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
rng = np.random.default_rng(seed=7)
n = 2000

contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
mean_tenure = pd.Series(contract).map({"Month-to-month": 18, "One year": 38, "Two year": 55}).to_numpy()
tenure = np.clip(rng.gamma(1.6, mean_tenure / 1.6), 0, 72).round().astype(int)

base = pd.Series(internet).map({"DSL": (55, 12), "Fiber optic": (90, 15), "No": (20, 5)})
monthly = np.clip([rng.normal(m, s) for m, s in base], 18.25, 118.75).round(2)
total = (tenure * monthly * rng.normal(1, 0.02, n)).round(2).astype(object)
total[tenure == 0] = ""                                      # blank TotalCharges for brand-new customers

p_base = pd.Series(contract).map({"Month-to-month": 0.43, "One year": 0.11, "Two year": 0.03}).to_numpy()
p = p_base * np.where(internet == "Fiber optic", 1.35, np.where(internet == "No", 0.5, 1.0)) * np.exp(-0.02 * (tenure - mean_tenure))
churn = np.where(rng.random(n) < np.clip(p, 0.01, 0.85), "Yes", "No")

ids = set()
while len(ids) < n:
    ids.add(f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list(string.ascii_uppercase), 5))}")
df = pd.DataFrame({
    "CustomerID": sorted(ids), "Gender": rng.choice(["Male", "Female"], n),
    "SeniorCitizen": (rng.random(n) < 0.16).astype(int),
    "Partner": rng.choice(["Yes", "No"], n, p=[0.48, 0.52]), "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
    "Tenure": tenure, "PhoneService": rng.choice(["Yes", "No"], n, p=[0.9, 0.1]), "InternetService": internet,
    "Contract": contract, "MonthlyCharges": monthly, "TotalCharges": total, "Churn": churn})
df = df.sample(frac=1, random_state=1).reset_index(drop=True)
df.to_csv(os.path.join(HERE, "Telco-Customer-Churn_sample.csv"), index=False)
print(f"Saved Telco-Customer-Churn_sample.csv with {len(df):,} rows")
print(df.head(5).to_string(index=False))
print("\nChurn counts:", df["Churn"].value_counts().to_dict())
