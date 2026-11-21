"""Data loading + multi-week weather lag pipeline (vectorised, per-ward)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data" / "hackathon_preli_dataset.xlsx"
CACHE = ROOT / "data" / "processed.parquet"
HORIZONS = (2, 3, 4)  # forecast lead in weeks (brief: 2-4 week early warning)
STATIC = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"]
RAIN_LAGS = range(0, 9)  # rainfall at t-0 ... t-8 weeks (breeding + incubation + onset)
CITY_CODE = {"Chattogram": 0, "Dhaka North": 1, "Dhaka South": 2, "Gazipur": 3, "Narayanganj": 4}


def load_raw(path: Path = RAW_XLSX, use_cache: bool = True) -> pd.DataFrame:
    """Read the weekly ward-level table. The slow xlsx parse happens once; later runs read parquet."""
    path = Path(path)
    if use_cache and path == RAW_XLSX and CACHE.exists() and CACHE.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_parquet(CACHE)
    df = pd.read_excel(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ward_id", "date"]).reset_index(drop=True)
    if use_cache and path == RAW_XLSX:
        df.to_parquet(CACHE, index=False)
    return df


def build_features(df: pd.DataFrame, with_cases: bool) -> pd.DataFrame:
    """Features known at forecast origin t (end of week t). No future information is used
    in the inputs; only the y_h* target columns look forward."""
    df = df.sort_values(["ward_id", "date"]).reset_index(drop=True)
    g = df.groupby("ward_id", sort=False)
    out = df[["ward_id", "city", "date"] + STATIC].copy()
    out["city_code"] = df["city"].map(CITY_CODE).fillna(-1).astype(int)

    # --- rainfall lags & rolling accumulations (the "vector breeding clock") ---
    for k in RAIN_LAGS:
        out[f"rain_lag{k}"] = g["rainfall_mm"].shift(k) if k else df["rainfall_mm"]
    for w in (2, 4, 8):
        out[f"rain_sum_{w}w"] = g["rainfall_mm"].transform(lambda s, w=w: s.rolling(w, min_periods=w).sum())
    out["rain_sum_wk3_5"] = out[["rain_lag3", "rain_lag4", "rain_lag5"]].sum(axis=1, min_count=3)
    out["rain_x_waterlog_4w"] = out["rain_sum_4w"] * df["water_logging_risk_index"]

    # --- temperature & humidity (current, lagged, smoothed) ---
    for col, name in (("avg_temp_c", "temp"), ("relative_humidity_pct", "hum")):
        out[f"{name}_lag0"] = df[col]
        out[f"{name}_lag2"] = g[col].shift(2)
        out[f"{name}_lag4"] = g[col].shift(4)
        out[f"{name}_mean_4w"] = g[col].transform(lambda s: s.rolling(4, min_periods=4).mean())
    out["tmin_mean_4w"] = g["min_temp_c"].transform(lambda s: s.rolling(4, min_periods=4).mean())

    # --- seasonality ---
    woy = df["date"].dt.isocalendar().week.astype(int).clip(upper=52)
    out["woy_sin"] = np.sin(2 * np.pi * woy / 52)
    out["woy_cos"] = np.cos(2 * np.pi * woy / 52)

    # --- optional surveillance feedback (recently reported cases) ---
    if with_cases:
        for k in range(0, 4):
            out[f"cases_lag{k}"] = g["dengue_cases"].shift(k) if k else df["dengue_cases"]
        out["cases_mean_4w"] = g["dengue_cases"].transform(lambda s: s.rolling(4, min_periods=4).mean())

    # --- targets: cases h weeks after the origin ---
    for h in HORIZONS:
        out[f"y_h{h}"] = g["dengue_cases"].shift(-h)
    return out


def feature_columns(frame: pd.DataFrame) -> list[str]:
    drop = {"ward_id", "city", "date"}
    return [c for c in frame.columns if c not in drop and not c.startswith("y_h")]


GROUPS = {
    "Rainfall (lags & accumulation)": lambda c: c.startswith("rain"),
    "Temperature": lambda c: c.startswith("temp") or c.startswith("tmin"),
    "Humidity": lambda c: c.startswith("hum"),
    "Ward vulnerability": lambda c: c in STATIC or c == "city_code",
    "Season": lambda c: c.startswith("woy"),
    "Recent reported cases": lambda c: c.startswith("cases"),
}


def group_of(col: str) -> str:
    for name, fn in GROUPS.items():
        if fn(col):
            return name
    return "Other"
