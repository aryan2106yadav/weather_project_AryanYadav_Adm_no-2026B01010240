import numpy as np
import pandas as pd

# 1. Load the data
df = pd.read_csv("weather_data.csv")

# 2. Basic information
print("Shape (rows, columns):", df.shape)
print("\nColumn names and data types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())

# 3. Missing values
print("\nMissing values per column:")
print(df.isna().sum())

# 4. Duplicate rows
print("\nDuplicate date+city rows:", df.duplicated(subset=["date", "city"]).sum())

# 5. Impossible values
print("\nHumidity outside 0-100:", (~df["humidity_pct"].dropna().between(0, 100)).sum())
print("Negative rainfall:", (df["rainfall_mm"] < 0).sum())
print("Max temp below min temp:", (df["temp_max_c"] < df["temp_min_c"]).sum())

# 6. Summary statistics
print("\nSummary statistics:")
print(df.describe().round(2))