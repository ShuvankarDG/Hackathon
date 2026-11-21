import functools
import sys

import joblib
import pandas as pd

from src import risk
from src.features import load_data, build_features, HORIZONS


@functools.lru_cache(maxsize=1)
def _bundle():
    return joblib.load("models/dengue_models.joblib")        # loaded once, not on every call


@functools.lru_cache(maxsize=1)
def _features():
    return build_features(load_data())                        # computed once, not on every call


def forecast(ward, date):
    b = _bundle()
    f = _features()
    row = f[(f.ward_id == ward) & (f.date == pd.Timestamp(date))]
    if row.empty:
        raise ValueError("Ward/date not found. Dates are Mondays, e.g. 2025-07-07")
    X = row[b["features"]]
    out = []
    for h in HORIZONS:
        cases = float(b["models"][h].predict(X)[0])
        out.append({
            "target_week": (pd.Timestamp(date) + pd.Timedelta(weeks=h)).date(),
            "lead_weeks": h,
            "predicted_cases": round(cases, 1),
            "risk": risk.tier_name(cases, b["cutoffs"]),
        })
    return pd.DataFrame(out)

def drivers(ward, date, lead):
    """How much each group of inputs pushed the forecast up or down (log scale)."""
    b = _bundle()
    f = _features()
    row = f[(f.ward_id == ward) & (f.date == pd.Timestamp(date))]
    contrib = b["models"][lead].predict(row[b["features"]], pred_contrib=True)[0][:-1]
    d = pd.DataFrame({"feature": b["features"], "effect": contrib})

    def group(name):
        if name.startswith("rain"):
            return "Rainfall"
        if name.startswith("temp"):
            return "Temperature"
        if name.startswith("hum"):
            return "Humidity"
        if name.startswith("woy"):
            return "Season"
        return "Ward profile"

    d["group"] = d["feature"].map(group)
    return d.groupby("group")["effect"].sum().sort_values()

if __name__ == "__main__":
    ward, date = sys.argv[1], sys.argv[2]
    print(forecast(ward, date).to_string(index=False))