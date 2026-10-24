"""Loading, validation and feature encoding for the CDC BRFSS heart disease survey."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "heart_2020_cleaned.csv"
REPORTS, FIG = ROOT / "reports", ROOT / "reports" / "figures"
for p in (REPORTS, FIG):
    p.mkdir(parents=True, exist_ok=True)

AGE_MID = {"18-24": 21, "25-29": 27, "30-34": 32, "35-39": 37, "40-44": 42, "45-49": 47, "50-54": 52, "55-59": 57,
           "60-64": 62, "65-69": 67, "70-74": 72, "75-79": 77, "80 or older": 85}
GEN_ORD = {"Excellent": 0, "Very good": 1, "Good": 2, "Fair": 3, "Poor": 4}
BINARY = ["Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", "PhysicalActivity", "Asthma", "KidneyDisease", "SkinCancer"]
# Race is deliberately NOT a model feature: it is used only to audit the model for fairness.
FEATURES = ["age", "male", "bmi", "gen_health", "physical_health_days", "mental_health_days", "sleep_hours",
            "diabetes_yes", "diabetes_borderline", "diabetes_pregnancy", *[b.lower() for b in BINARY]]


def load():
    if not RAW.exists():
        print("No data/raw/heart_2020_cleaned.csv -> generating synthetic BRFSS-format data")
        from generate_data import main as gen
        gen()
    df = pd.read_csv(RAW)
    expected = {"HeartDisease", "BMI", "AgeCategory", "GenHealth", "Diabetic", "Sex", "Race", *BINARY}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing}. This project expects the 2020 file (heart_2020_cleaned.csv).")
    log = {"rows": len(df), "exact_duplicates": int(df.duplicated().sum()), "nulls": int(df.isna().sum().sum())}
    return df, log


def encode(df):
    X = pd.DataFrame(index=df.index)
    X["age"] = df.AgeCategory.map(AGE_MID)
    X["male"] = (df.Sex == "Male").astype(int)
    X["bmi"] = df.BMI
    X["gen_health"] = df.GenHealth.map(GEN_ORD)
    X["physical_health_days"] = df.PhysicalHealth
    X["mental_health_days"] = df.MentalHealth
    X["sleep_hours"] = df.SleepTime
    X["diabetes_yes"] = (df.Diabetic == "Yes").astype(int)
    X["diabetes_borderline"] = (df.Diabetic == "No, borderline diabetes").astype(int)
    X["diabetes_pregnancy"] = (df.Diabetic == "Yes (during pregnancy)").astype(int)
    for b in BINARY:
        X[b.lower()] = (df[b] == "Yes").astype(int)
    y = (df.HeartDisease == "Yes").astype(int)
    return X[FEATURES], y


def bmi_class(b):
    return pd.cut(b, [0, 18.5, 25, 30, 35, 40, 200],
                  labels=["Underweight", "Normal", "Overweight", "Obese I", "Obese II", "Obese III"], right=False)
