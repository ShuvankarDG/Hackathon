import os
import warnings

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from src import risk
from src.features import load_data, build_features, feature_columns, HORIZONS

warnings.filterwarnings("ignore")
TRAIN_END = pd.Timestamp("2022-12-31")
VAL_END = pd.Timestamp("2023-12-31")


def main():
    f = build_features(load_data())
    f = f.dropna(subset=["rain_lag8", "rain_sum_8w"])
    cols = feature_columns(f)

    # tier boundaries from training years only, pooled over the three lead times
    train_all = f[f.date <= TRAIN_END]
    pooled = np.concatenate([train_all[f"y_h{h}"].dropna().values for h in HORIZONS])
    cutoffs = risk.fit_cutoffs(pooled)
    print("Tier cut-offs (cases/week):", cutoffs)

    models = {}
    for h in HORIZONS:
        target = f"y_h{h}"
        d = f.dropna(subset=[target])
        train = d[d.date <= TRAIN_END]
        val = d[(d.date > TRAIN_END) & (d.date <= VAL_END)]
        test = d[d.date > VAL_END]

        model = lgb.LGBMRegressor(
            objective="poisson", n_estimators=1500, learning_rate=0.04, num_leaves=15,
            min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
        model.fit(train[cols], train[target],
                  eval_set=[(val[cols], val[target])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        models[h] = model

        pred = model.predict(test[cols])
        mae = np.abs(test[target] - pred).mean()
        true_tier = risk.tier_index(test[target], cutoffs)
        pred_tier = risk.tier_index(pred, cutoffs)
        print(f"{h}-week | MAE {mae:.3f} | tier accuracy {np.mean(true_tier == pred_tier):.2f} "
              f"| within one tier {np.mean(np.abs(true_tier - pred_tier) <= 1):.2f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump({"models": models, "features": cols, "cutoffs": cutoffs},
                "models/dengue_models.joblib")
    print("Saved models/dengue_models.joblib")


if __name__ == "__main__":
    main()