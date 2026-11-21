from src.features import load_data, build_features

df = load_data()
f = build_features(df)

print(f.shape)
ward = "CCC_Ward_01"
w_raw = df[df.ward_id == ward].reset_index(drop=True)
w_feat = f[f.ward_id == ward].reset_index(drop=True)

# rain_lag3 in row 10 should equal rainfall from row 7
print(w_feat.loc[10, "rain_lag3"], w_raw.loc[7, "rainfall_mm"])

# y_h3 in row 10 should equal cases from row 13
print(w_feat.loc[10, "y_h3"], w_raw.loc[13, "dengue_cases"])