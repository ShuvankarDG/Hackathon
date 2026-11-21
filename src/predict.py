"""Fast local inference: serialized LightGBM models -> risk tier, score, band and weather drivers.

    python -m src.predict --ward CCC_Ward_01 --date 2025-07-07
"""
from __future__ import annotations

import argparse
import functools

import joblib
import numpy as np
import pandas as pd

from . import risk
from .features import HORIZONS, ROOT, build_features, group_of, load_raw

MODEL_PATH = ROOT / "models" / "dengue_models.joblib"


@functools.lru_cache(maxsize=1)
def load_bundle() -> dict:
    """Models are deserialised once per process (joblib), never retrained at request time."""
    return joblib.load(MODEL_PATH)


@functools.lru_cache(maxsize=2)
def feature_table(variant: str) -> pd.DataFrame:
    """Lag features for every ward-week, computed once per process and reused by all queries."""
    return build_features(load_raw(), with_cases=variant == "weather_plus_cases")


def forecast_frame(feats: pd.DataFrame, variant: str = "weather_only") -> pd.DataFrame:
    """Vectorised forecast for many origin rows -> one row per (origin row, horizon)."""
    vb = load_bundle()["variants"][variant]
    X = feats[vb["features"]]
    rows = []
    for h in HORIZONS:
        m = vb["models"][h]
        p = m["mean"].predict(X)
        lo = np.minimum(m["q10"].predict(X), p)
        hi = np.maximum(m["q90"].predict(X), p)
        rows.append(pd.DataFrame({
            "ward_id": feats["ward_id"].values, "city": feats["city"].values, "origin_date": feats["date"].values,
            "horizon": h, "target_date": feats["date"].values + pd.to_timedelta(7 * h, unit="D"),
            "pred_cases": p, "lo": lo, "hi": hi,
            "tier": [risk.TIERS[i] for i in risk.tier_index(p, vb["cutoffs"])],
            "risk_score": risk.score(p, vb["ecdf"]),
        }))
    return pd.concat(rows, ignore_index=True)


def drivers(feat_row: pd.DataFrame, horizon: int, variant: str = "weather_only") -> pd.DataFrame:
    """Per-feature contributions (LightGBM native TreeSHAP, log-expected-cases scale), grouped into drivers."""
    vb = load_bundle()["variants"][variant]
    X = feat_row[vb["features"]]
    c = vb["models"][horizon]["mean"].predict(X, pred_contrib=True)[0][:-1]
    df = pd.DataFrame({"feature": vb["features"], "contribution": c, "value": X.iloc[0].values})
    df["driver_group"] = df["feature"].map(group_of)
    return df


def whatif_row(raw: pd.DataFrame, ward: str, origin: pd.Timestamp, rain_mult: float = 1.0,
               weeks: int = 4, variant: str = "weather_only") -> pd.DataFrame:
    """Re-run the lag pipeline for ONE ward with the last `weeks` of rainfall scaled (scenario analysis)."""
    w = raw[(raw.ward_id == ward) & (raw.date <= origin)].tail(24).copy()
    w.iloc[-weeks:, w.columns.get_loc("rainfall_mm")] *= rain_mult
    f = build_features(w, with_cases=variant == "weather_plus_cases")
    return f.tail(1)


def window_summary(fc: pd.DataFrame, variant: str = "weather_only") -> dict:
    """Aggregate the 2-4 week window for one ward/origin: mean weekly cases -> tier."""
    vb = load_bundle()["variants"][variant]
    mean_cases = float(fc["pred_cases"].mean())
    return {"mean_weekly_cases": mean_cases, "tier": risk.tier_name(mean_cases, vb["cutoffs"]),
            "risk_score": float(risk.score([mean_cases], vb["ecdf"])[0]),
            "peak_tier": risk.TIERS[int(max(risk.tier_index(fc["pred_cases"], vb["cutoffs"])))]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ward", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--variant", default="weather_only")
    a = ap.parse_args()
    ft = feature_table(a.variant)
    row = ft[(ft.ward_id == a.ward) & (ft.date == pd.Timestamp(a.date))]
    if row.empty:
        raise SystemExit("ward/date not found (dates are Mondays, e.g. 2025-07-07)")
    fc = forecast_frame(row, a.variant)
    print(fc[["target_date", "horizon", "pred_cases", "lo", "hi", "tier", "risk_score"]].round(2).to_string(index=False))
    print("window:", window_summary(fc, a.variant))
