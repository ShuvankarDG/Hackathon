"""Risk tiers and 0-100 scores derived from the training distribution (not hand-set)."""
from __future__ import annotations

import numpy as np

TIERS = ["Low", "Medium", "High", "Critical"]
TIER_QUANTILES = (0.60, 0.85, 0.95)  # of weekly ward cases in the training period
TIER_COLORS = {"Low": "#2e8b57", "Medium": "#e6b800", "High": "#e67300", "Critical": "#c0141c"}


def fit_cutoffs(y_train: np.ndarray) -> list[float]:
    cuts = [float(np.quantile(y_train, q)) for q in TIER_QUANTILES]
    for i in range(1, len(cuts)):  # strictly increasing so every tier is reachable
        cuts[i] = max(cuts[i], cuts[i - 1] + 1.0)
    return cuts


def tier_index(cases, cuts) -> np.ndarray:
    return np.searchsorted(np.asarray(cuts), np.asarray(cases, dtype=float), side="right")


def tier_name(cases: float, cuts) -> str:
    return TIERS[int(tier_index([cases], cuts)[0])]


def score(cases, ecdf_sorted: np.ndarray) -> np.ndarray:
    """0-100 percentile of predicted weekly cases within the historical distribution."""
    return 100.0 * np.searchsorted(ecdf_sorted, np.asarray(cases, dtype=float), side="right") / len(ecdf_sorted)
