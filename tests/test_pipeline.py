import numpy as np
import pandas as pd

from src.features import load_data, build_features
from src.predict import forecast


def test_lags_match_raw_weather():
    raw = load_data()
    f = build_features(raw)
    w = raw[raw.ward_id == "CCC_Ward_01"].reset_index(drop=True)
    fw = f[f.ward_id == "CCC_Ward_01"].reset_index(drop=True)
    for k in (1, 3, 8):
        assert np.allclose(fw[f"rain_lag{k}"].iloc[k:], w.rainfall_mm.shift(k).iloc[k:])
    assert np.allclose(fw["rain_sum_4w"].iloc[3:], w.rainfall_mm.rolling(4).sum().iloc[3:])


def test_targets_are_future_cases():
    raw = load_data()
    f = build_features(raw)
    w = raw[raw.ward_id == "CCC_Ward_01"].reset_index(drop=True)
    fw = f[f.ward_id == "CCC_Ward_01"].reset_index(drop=True)
    assert np.array_equal(fw["y_h3"].iloc[:-3].values, w.dengue_cases.iloc[3:].values)


def test_future_weather_does_not_change_past_features():
    raw = load_data()
    base = build_features(raw)
    changed = raw.copy()
    changed.loc[changed.date > pd.Timestamp("2024-06-30"), "rainfall_mm"] *= 3
    new = build_features(changed)
    past = base.date <= pd.Timestamp("2024-06-30")
    cols = [c for c in base.columns if c.startswith("rain_")]
    assert np.allclose(base.loc[past, cols].fillna(-1), new.loc[past, cols].fillna(-1))


def test_forecast_output_is_valid():
    fc = forecast("CCC_Ward_01", "2025-07-07")
    assert len(fc) == 3
    assert set(fc.risk) <= {"Low", "Medium", "High", "Critical"}
    assert (fc.predicted_cases >= 0).all()