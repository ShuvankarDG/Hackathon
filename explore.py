import pandas as pd

df = pd.read_excel("data/hackathon_preli_dataset.xlsx")
df = df.sort_values(["ward_id", "date"])

print("Shape:", df.shape)
print(df.dtypes)
print("Missing values:", df.isna().sum().sum())
print("Share of weeks with zero cases:", (df.dengue_cases == 0).mean())

print("\nAverage cases and rain by month:")
print(df.groupby(df.date.dt.month)[["dengue_cases", "rainfall_mm"]].mean())

print("\nCorrelation of rainfall k weeks ago with cases now:")
for k in range(9):
    lagged_rain = df.groupby("ward_id").rainfall_mm.shift(k)
    print(k, round(lagged_rain.corr(df.dengue_cases), 3))