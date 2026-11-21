"""
Weather data backend - uses ONLY numpy and pandas.

A WeatherBackend object loads a weather CSV, cleans it, and answers queries
(filtering, summaries, monthly averages, extremes, correlations, export).

Run the demo:
    python weather_backend.py
"""

import numpy as np
import pandas as pd

NUM_COLS = ["temp_max_c", "temp_min_c", "temp_avg_c",
            "humidity_pct", "rainfall_mm", "wind_speed_kmh", "pressure_hpa"]
REQUIRED_COLS = ["date", "city"] + NUM_COLS


class WeatherBackend:
    def __init__(self, csv_path=None):
        self.df = None
        self.source = None
        if csv_path:
            self.load(csv_path)

    # ------------------------------------------------------------------
    # Loading and cleaning
    # ------------------------------------------------------------------
    def load(self, csv_path):
        """Read a CSV file, clean it and keep it in memory."""
        raw = pd.read_csv(csv_path)
        self.df = self.clean(raw)
        self.source = csv_path
        return self

    @staticmethod
    def clean(df):
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df[REQUIRED_COLS].copy()

        # Fix data types
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["city"] = df["city"].astype(str).str.strip().str.title()
        df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors="coerce")

        # Remove duplicates and bad dates, then sort
        df = df.drop_duplicates(subset=["date", "city"]).dropna(subset=["date"])
        df = df.sort_values(["city", "date"]).reset_index(drop=True)

        # Mark impossible values as NaN
        df["humidity_pct"] = np.where(df["humidity_pct"].between(0, 100),
                                      df["humidity_pct"], np.nan)
        df["rainfall_mm"] = np.where(df["rainfall_mm"] < 0, np.nan, df["rainfall_mm"])
        swapped = df["temp_max_c"] < df["temp_min_c"]
        df.loc[swapped, ["temp_max_c", "temp_min_c"]] = np.nan

        # Outliers (3 x IQR); rainfall is skipped on purpose
        for col in ["temp_max_c", "temp_min_c", "wind_speed_kmh", "pressure_hpa"]:
            if df[col].notna().sum() < 4:
                continue
            q1, q3 = np.nanpercentile(df[col], [25, 75])
            iqr = q3 - q1
            low, high = q1 - 3 * iqr, q3 + 3 * iqr
            df[col] = np.where((df[col] < low) | (df[col] > high), np.nan, df[col])

        # Fill missing values city by city
        fill_cols = ["temp_max_c", "temp_min_c", "humidity_pct",
                     "wind_speed_kmh", "pressure_hpa"]
        df[fill_cols] = df.groupby("city")[fill_cols].transform(
            lambda s: s.interpolate(limit_direction="both")
        )
        df["rainfall_mm"] = df["rainfall_mm"].fillna(0)

        # Keep the average consistent with max and min
        df["temp_avg_c"] = np.round((df["temp_max_c"] + df["temp_min_c"]) / 2, 1)

        # Helper columns
        df["month"] = df["date"].dt.month
        df["season"] = np.select(
            [df["month"].isin([12, 1, 2]),
             df["month"].isin([3, 4, 5]),
             df["month"].isin([6, 7, 8, 9])],
            ["Winter", "Summer", "Monsoon"],
            default="Post-monsoon",
        )
        df["rainy_day"] = np.where(df["rainfall_mm"] > 0, 1, 0)
        return df

    def _data(self):
        if self.df is None:
            raise RuntimeError("No data loaded. Call load('file.csv') first.")
        return self.df

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def cities(self):
        return sorted(self._data()["city"].unique().tolist())

    def get_weather(self, city=None, start=None, end=None, season=None,
                    limit=None, offset=0):
        """Filter daily rows. Dates are 'YYYY-MM-DD' strings."""
        df = self._data()
        if city:
            df = df[df["city"].str.lower() == city.strip().lower()]
        if start:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end:
            df = df[df["date"] <= pd.to_datetime(end)]
        if season:
            df = df[df["season"].str.lower() == season.strip().lower()]
        df = df.iloc[offset:]
        if limit:
            df = df.iloc[:limit]
        return df.reset_index(drop=True)

    def summary(self, **filters):
        """Per-city statistics. Accepts the same filters as get_weather()."""
        df = self.get_weather(**filters)
        if df.empty:
            return pd.DataFrame()
        return df.groupby("city").agg(
            days=("date", "count"),
            avg_temp_c=("temp_avg_c", "mean"),
            max_temp_c=("temp_max_c", "max"),
            min_temp_c=("temp_min_c", "min"),
            avg_humidity_pct=("humidity_pct", "mean"),
            total_rainfall_mm=("rainfall_mm", "sum"),
            rainy_days=("rainy_day", "sum"),
            avg_wind_kmh=("wind_speed_kmh", "mean"),
            avg_pressure_hpa=("pressure_hpa", "mean"),
        ).round(2).reset_index()

    def monthly(self, city=None):
        """Monthly averages and rainfall totals per city."""
        df = self.get_weather(city=city)
        if df.empty:
            return pd.DataFrame()
        return df.groupby(["city", "month"]).agg(
            avg_temp_c=("temp_avg_c", "mean"),
            avg_humidity_pct=("humidity_pct", "mean"),
            total_rainfall_mm=("rainfall_mm", "sum"),
            rainy_days=("rainy_day", "sum"),
        ).round(2).reset_index()

    def extremes(self, n=5, column="temp_max_c", hottest=True, city=None):
        """Top n days by a column (hottest by default; hottest=False for lowest)."""
        df = self.get_weather(city=city)
        if hottest:
            return df.nlargest(n, column).reset_index(drop=True)
        return df.nsmallest(n, column).reset_index(drop=True)

    def correlations(self, city=None):
        """Correlation matrix between the weather measurements."""
        cols = ["temp_avg_c", "humidity_pct", "rainfall_mm",
                "wind_speed_kmh", "pressure_hpa"]
        return self.get_weather(city=city)[cols].corr().round(2)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def save_clean(self, path="weather_clean.csv"):
        out = self._data().copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out.to_csv(path, index=False)
        return path

    @staticmethod
    def to_json(df):
        """DataFrame -> JSON text (dates as text, NaN as null)."""
        out = df.copy()
        for col in out.select_dtypes(include="datetime").columns:
            out[col] = out[col].dt.strftime("%Y-%m-%d")
        return out.to_json(orient="records")


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    api = WeatherBackend("weather_sample_2025.csv")
    print("Cities:", api.cities())

    print("\nSummary per city:")
    print(api.summary())

    print("\nMonsoon summary:")
    print(api.summary(season="monsoon"))

    print("\nMumbai monthly (first 6 rows):")
    print(api.monthly(city="Mumbai").head(6))

    print("\n5 hottest days overall:")
    print(api.extremes(n=5)[["date", "city", "temp_max_c"]])

    print("\nDelhi, 1-3 July as JSON:")
    print(WeatherBackend.to_json(
        api.get_weather(city="Delhi", start="2025-07-01", end="2025-07-03")))

    print("\nCorrelations (all cities):")
    print(api.correlations())

    print("\nSaved:", api.save_clean())