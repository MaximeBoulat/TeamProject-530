from pathlib import Path
from Globals import DATASETS_ROOT
import pandas as pd


class DataPreprocessing:

    def __init__(self):
        pass

    def preprocess_data(self):
        
        daily = pd.read_csv(Path(DATASETS_ROOT) / "daily_dataset.csv")
        households = pd.read_csv(Path(DATASETS_ROOT) / "informations_households.csv")
        weather_daily = pd.read_csv(Path(DATASETS_ROOT) / "weather_daily_darksky.csv")
        holidays = pd.read_csv(Path(DATASETS_ROOT) / "uk_bank_holidays.csv")

        # date cleaning

        daily["day"] = pd.to_datetime(daily["day"])
        
        # Extract month for season calculation before converting to date
        daily["month"] = daily["day"].dt.month
        
        # Map month to season
        def get_season(month):
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            else:  # 9, 10, 11
                return "Fall"
        
        daily["season"] = daily["month"].apply(get_season)
        
        weather_daily["temperatureMaxTime"] = pd.to_datetime(weather_daily["temperatureMaxTime"])
        weather_daily["temperatureMinTime"] = pd.to_datetime(weather_daily["temperatureMinTime"])
        weather_daily["day"] = weather_daily["temperatureMaxTime"].dt.date

        # Add day_of_week before converting to date
        daily["day_of_week"] = daily["day"].dt.dayofweek  # 0=Monday, 6=Sunday
        
        daily["day"] = daily["day"].dt.date

        # Create normalized daily dataset - aggregate across all households per day
        daily_norm = (
            daily
                .groupby("day")
                .agg(
                    total_kwh_per_day=("energy_sum", "sum"),
                    num_active_households=("LCLid", "nunique"),
                    month=("month", "first"),
                    day_of_week=("day_of_week", "first"),
                    season=("season", "first")
                )
                .reset_index()
            )

        # Compute normalized average kWh per household per day
        daily_norm["avg_kwh_per_household_per_day"] = (
            daily_norm["total_kwh_per_day"] / daily_norm["num_active_households"]
        )

        # Merge weather features to the normalized daily data
        weather_features = weather_daily[[
            "day",
            "temperatureHigh",
            "temperatureLow",
            "temperatureMax",
            "temperatureMin",
            "apparentTemperatureHigh",
            "apparentTemperatureLow",
            "humidity",
            "windSpeed",
            "cloudCover",
            "pressure",
            "visibility"
        ]].copy()

        daily_with_weather = daily_norm.merge(
            weather_features,
            on="day",
            how="left"
        )

        # drop NAs
        daily_with_weather = daily_with_weather.dropna()

        # Merge holidays
        holidays = holidays.rename(columns={
            "Bank holidays": "day",
            "Type": "holiday_name"
        })

        # Parse date
        holidays["day"] = pd.to_datetime(holidays["day"]).dt.date

        daily_complete = daily_with_weather.merge(
            holidays, on="day", how="left"
        )

        # Flag holidays
        daily_complete["is_holiday"] = daily_complete["holiday_name"].notna()

        # remove last row
        daily_complete = daily_complete.iloc[:-1]

        return daily_complete
