import numpy as np
import pandas as pd
import lightgbm as lgb

from src.features import load_data, build_features, feature_columns, HORIZONS

TRAIN_END = pd.Timestamp("2022-12-31")
VAL_END = pd.Timestamp("2023-12-31")

f = build_features(load_data())
f = f.dropna(subset=["rain_lag8", "rain_sum_8w"])
cols = feature_columns(f)
f["week_of_year"] = f["date"].dt.isocalendar().week.astype(int).clip(upper=52)

for h in HORIZONS:
    target = f"y_h{h}"
    d = f.dropna(subset=[target])
    train = d[d.date <= TRAIN_END]
    val = d[(d.date > TRAIN_END) & (d.date <= VAL_END)]
    test = d[d.date > VAL_END]

    model = lgb.LGBMRegressor(
        objective="poisson",     # for count data (cases per week)
        n_estimators=1500,       # upper limit; early stopping decides the real number
        learning_rate=0.04,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    model.fit(
        train[cols], train[target],
        eval_set=[(val[cols], val[target])],          # 2023 is used only to decide when to stop
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    pred = model.predict(test[cols])
    mae = np.abs(test[target] - pred).mean()
    rmse = np.sqrt(((test[target] - pred) ** 2).mean())

    # same baseline as before, for comparison
    clim = train.groupby(["ward_id", "week_of_year"])[target].mean().rename("base")
    base = test.join(clim, on=["ward_id", "week_of_year"])["base"].fillna(train[target].mean())
    base_mae = np.abs(test[target] - base).mean()

    print(f"{h}-week ahead | trees used {model.best_iteration_} | "
          f"model MAE {mae:.3f} RMSE {rmse:.3f} | baseline MAE {base_mae:.3f}")