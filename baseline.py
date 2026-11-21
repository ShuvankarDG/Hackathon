import numpy as np
import pandas as pd
from src.features import load_data, build_features, HORIZONS

TRAIN_END = pd.Timestamp("2022-12-31")
VAL_END = pd.Timestamp("2023-12-31")

f = build_features(load_data())
f = f.dropna(subset=["rain_lag8", "rain_sum_8w"])   # drop first weeks that have no history
f["week_of_year"] = f["date"].dt.isocalendar().week.astype(int).clip(upper=52)

for h in HORIZONS:
    target = f"y_h{h}"
    d = f.dropna(subset=[target])

    # time-based split: never shuffle
    train = d[d.date <= TRAIN_END]
    val = d[(d.date > TRAIN_END) & (d.date <= VAL_END)]
    test = d[d.date > VAL_END]

    # baseline: average cases for this ward in this week of the year (training years only)
    clim = train.groupby(["ward_id", "week_of_year"])[target].mean().rename("pred")
    t = test.join(clim, on=["ward_id", "week_of_year"])
    t["pred"] = t["pred"].fillna(train[target].mean())

    mae = (t[target] - t["pred"]).abs().mean()
    rmse = np.sqrt(((t[target] - t["pred"]) ** 2).mean())
    print(f"{h}-week ahead | train {len(train)}  val {len(val)}  test {len(test)} | "
          f"baseline MAE {mae:.3f}  RMSE {rmse:.3f}")