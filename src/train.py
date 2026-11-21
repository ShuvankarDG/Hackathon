"""Train horizon-specific LightGBM models with a strict chronological split.

    python -m src.train            # train on <=2022, early-stop on 2023, report on untouched 2024-2025
    python -m src.train --refit-all  # production refit on all data (no honest test score afterwards)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from . import risk
from .features import HORIZONS, ROOT, build_features, feature_columns, load_raw

TRAIN_END = pd.Timestamp("2022-12-31")
VAL_END = pd.Timestamp("2023-12-31")
VARIANTS = {"weather_only": False, "weather_plus_cases": True}
SEED = 42
warnings.filterwarnings("ignore")
BASE = dict(learning_rate=0.04, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=2.0, random_state=SEED, verbose=-1, n_jobs=2)


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE": float(np.mean(np.abs(y - p))), "RMSE": float(np.sqrt(np.mean((y - p) ** 2)))}


def tier_metrics(y, p, cuts) -> dict:
    ty, tp = risk.tier_index(y, cuts), risk.tier_index(p, cuts)
    crit = ty == 3
    return {
        "tier_accuracy": float(np.mean(ty == tp)),
        "within_one_tier": float(np.mean(np.abs(ty - tp) <= 1)),
        "critical_recall": float(np.mean(tp[crit] == 3)) if crit.any() else None,
        "critical_or_high_recall_on_critical": float(np.mean(tp[crit] >= 2)) if crit.any() else None,
        "critical_precision": float(np.mean(ty[tp == 3] == 3)) if (tp == 3).any() else None,
        # operational alert = predicted High or Critical
        "alert_recall_high_or_above": float(np.mean(tp[ty >= 2] >= 2)) if (ty >= 2).any() else None,
        "alert_precision_high_or_above": float(np.mean(ty[tp >= 2] >= 2)) if (tp >= 2).any() else None,
    }


def main(refit_all: bool = False) -> None:
    t0 = time.time()
    raw = load_raw()
    bundle = {"horizons": list(HORIZONS), "variants": {}, "split": {
        "train_end": str(TRAIN_END.date()), "val_end": str(VAL_END.date()), "refit_all": refit_all}}
    report: dict = {"split": bundle["split"], "variants": {}}
    test_rows = []

    for vname, with_cases in VARIANTS.items():
        feats = build_features(raw, with_cases)
        cols = feature_columns(feats)
        feats = feats.dropna(subset=["rain_lag8", "rain_sum_8w", "temp_mean_4w"] + (["cases_mean_4w"] if with_cases else []))
        vb = {"features": cols, "models": {}}
        report["variants"][vname] = {}

        # tier cutoffs + score ECDF come from the TRAIN period only (no leakage)
        tr_all = feats[feats.date <= TRAIN_END]
        y_pool = np.concatenate([tr_all[f"y_h{h}"].dropna().values for h in HORIZONS])
        vb["cutoffs"] = risk.fit_cutoffs(y_pool)
        vb["ecdf"] = np.sort(y_pool)

        for h in HORIZONS:
            tgt = f"y_h{h}"
            d = feats.dropna(subset=[tgt])
            tr, va, te = d[d.date <= TRAIN_END], d[(d.date > TRAIN_END) & (d.date <= VAL_END)], d[d.date > VAL_END]

            m = lgb.LGBMRegressor(objective="poisson", n_estimators=1500, **BASE)
            m.fit(tr[cols], tr[tgt], eval_set=[(va[cols], va[tgt])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            n_best = int(m.best_iteration_ or 300)

            fit_df = d if refit_all else tr
            mean_m = lgb.LGBMRegressor(objective="poisson", n_estimators=n_best, **BASE).fit(fit_df[cols], fit_df[tgt])
            q_models = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=n_best, **BASE)
                        .fit(fit_df[cols], fit_df[tgt]) for q in (0.1, 0.9)}
            vb["models"][h] = {"mean": mean_m, "q10": q_models[0.1], "q90": q_models[0.9], "n_estimators": n_best}

            # ---- honest test-set evaluation (2024-2025, never used for fitting or early stopping) ----
            p = mean_m.predict(te[cols])
            lo, hi = np.minimum(q_models[0.1].predict(te[cols]), p), np.maximum(q_models[0.9].predict(te[cols]), p)
            y = te[tgt].values
            clim = tr.groupby(["ward_id", tr.date.dt.isocalendar().week.clip(upper=52)])[tgt].mean()
            key = list(zip(te.ward_id, te.date.dt.isocalendar().week.clip(upper=52)))
            p_clim = np.array([clim.get(k, np.nan) for k in key])
            p_clim = np.where(np.isnan(p_clim), tr[tgt].mean(), p_clim)
            cur = te["cases_lag0"].values if with_cases else None
            r = {"model": metrics(y, p), "seasonal_climatology_baseline": metrics(y, p_clim),
                 "interval_80_coverage": float(np.mean((y >= np.floor(lo)) & (y <= np.ceil(hi)))),
                 "tiers": tier_metrics(y, p, vb["cutoffs"]), "n_test_rows": int(len(te)), "n_estimators": n_best}
            if cur is not None:
                r["persistence_baseline"] = metrics(y, cur)
            report["variants"][vname][f"h{h}"] = r
            test_rows.append(pd.DataFrame({"variant": vname, "horizon": h, "ward_id": te.ward_id.values, "city": te.city.values,
                                           "origin_date": te.date.values, "actual": y, "pred": p, "lo": lo, "hi": hi}))
            print(f"[{vname} h={h}] MAE {r['model']['MAE']:.3f} (clim {r['seasonal_climatology_baseline']['MAE']:.3f}) "
                  f"RMSE {r['model']['RMSE']:.3f} tier-acc {r['tiers']['tier_accuracy']:.2f} cov80 {r['interval_80_coverage']:.2f}")
        bundle["variants"][vname] = vb

    # global feature importance (gain) for the documentation / dashboard
    imp = {}
    for vname, vb in bundle["variants"].items():
        m = vb["models"][3]["mean"]
        imp[vname] = dict(sorted(zip(vb["features"], map(float, m.booster_.feature_importance("gain"))), key=lambda kv: -kv[1])[:15])
    report["top_features_gain_h3"] = imp

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    joblib.dump(bundle, ROOT / "models" / "dengue_models.joblib", compress=3)
    pd.concat(test_rows).to_parquet(ROOT / "reports" / "test_predictions.parquet", index=False)
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit-all", action="store_true")
    main(ap.parse_args().refit_all)
