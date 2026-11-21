import numpy as np

TIERS = ["Low", "Medium", "High", "Critical"]


def fit_cutoffs(train_cases):
    """Tier boundaries = 60th, 85th, 95th percentile of weekly cases in the TRAINING years."""
    return [float(np.quantile(train_cases, q)) for q in (0.60, 0.85, 0.95)]


def tier_index(cases, cutoffs):
    """0=Low, 1=Medium, 2=High, 3=Critical"""
    return np.searchsorted(np.asarray(cutoffs), np.asarray(cases, dtype=float), side="right")


def tier_name(cases, cutoffs):
    return TIERS[int(tier_index([cases], cutoffs)[0])]