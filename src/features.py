import pandas as pd
import numpy as np

DATA_PATH = "data/hackathon_preli_dataset.xlsx"
HORIZONS = (2, 3, 4)   # how many weeks ahead we want to predict


def load_data(path=DATA_PATH):
    df = pd.read_excel(path)
    # sort so that "previous week" really means the previous week of the SAME ward
    return df.sort_values(["ward_id", "date"]).reset_index(drop=True)


def build_features(df):
    # groupby ward so a lag never borrows data from a different ward
    g = df.groupby("ward_id")
    out = df[["ward_id", "city", "date",
              "population_density_sqkm", "water_logging_risk_index", "sanitation_score"]].copy()

    # 1. Rainfall k weeks ago (k = 0..8)
    for k in range(9):
        out[f"rain_lag{k}"] = g["rainfall_mm"].shift(k)

    # 2. Total rainfall over the last 2, 4, 8 weeks
    for w in (2, 4, 8):
        out[f"rain_sum_{w}w"] = g["rainfall_mm"].transform(lambda s: s.rolling(w).sum())

    # 3. Temperature and humidity: now, and average of last 4 weeks
    out["temp_now"] = df["avg_temp_c"]
    out["temp_mean_4w"] = g["avg_temp_c"].transform(lambda s: s.rolling(4).mean())
    out["hum_now"] = df["relative_humidity_pct"]
    out["hum_mean_4w"] = g["relative_humidity_pct"].transform(lambda s: s.rolling(4).mean())

        # 5. Season: sine/cosine so that week 52 and week 1 count as neighbours
    week = df["date"].dt.isocalendar().week.astype(int).clip(upper=52)
    out["woy_sin"] = np.sin(2 * np.pi * week / 52)
    out["woy_cos"] = np.cos(2 * np.pi * week / 52)

    # 4. Targets: cases 2, 3, 4 weeks IN THE FUTURE (negative shift = look forward)
    for h in HORIZONS:
        out[f"y_h{h}"] = g["dengue_cases"].shift(-h)

    return out

def feature_columns(frame):
    skip = {"ward_id", "city", "date"}
    return [c for c in frame.columns if c not in skip and not c.startswith("y_h")]