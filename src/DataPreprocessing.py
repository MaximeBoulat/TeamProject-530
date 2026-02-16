from pathlib import Path
from enum import Enum
from Globals import DATASETS_ROOT
import pandas as pd


class ModelType(Enum):
    LSTM = "lstm"
    NEURAL_NETWORK = "neural_network"


class DataPreprocessing:

    def __init__(self):
        pass

    def preprocess_data(self, model_type=ModelType.LSTM):
        """
        Load and preprocess data for the specified model type.

        Args:
            model_type: ModelType.LSTM for global daily aggregation,
                        ModelType.NEURAL_NETWORK for ACORN-segmented aggregation

        Returns:
            Preprocessed DataFrame ready for model-specific preparation
        """
        daily = pd.read_csv(Path(DATASETS_ROOT) / "daily_dataset.csv")
        households = pd.read_csv(Path(DATASETS_ROOT) / "informations_households.csv")
        weather_daily = pd.read_csv(Path(DATASETS_ROOT) / "weather_daily_darksky.csv")
        holidays = pd.read_csv(Path(DATASETS_ROOT) / "uk_bank_holidays.csv")

        # Common date cleaning
        daily["day"] = pd.to_datetime(daily["day"])

        weather_daily["temperatureMaxTime"] = pd.to_datetime(weather_daily["temperatureMaxTime"])
        weather_daily["temperatureMinTime"] = pd.to_datetime(weather_daily["temperatureMinTime"])
        weather_daily["day"] = weather_daily["temperatureMaxTime"].dt.date

        holidays = holidays.rename(columns={
            "Bank holidays": "day",
            "Type": "holiday_name"
        })
        holidays["day"] = pd.to_datetime(holidays["day"]).dt.date

        if model_type == ModelType.LSTM:
            return self._preprocess_lstm(daily, weather_daily, holidays)
        elif model_type == ModelType.NEURAL_NETWORK:
            return self._preprocess_nn(daily, households, weather_daily, holidays)

    def _preprocess_lstm(self, daily, weather_daily, holidays):
        """Global daily aggregation for LSTM."""

        daily["month"] = daily["day"].dt.month
        daily["day_of_week"] = daily["day"].dt.dayofweek

        def get_season(month):
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            else:
                return "Fall"

        daily["season"] = daily["month"].apply(get_season)
        daily["day"] = daily["day"].dt.date

        # Aggregate across all households per day
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

        daily_norm["avg_kwh_per_household_per_day"] = (
            daily_norm["total_kwh_per_day"] / daily_norm["num_active_households"]
        )

        # Merge weather
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
            weather_features, on="day", how="left"
        )

        daily_with_weather = daily_with_weather.dropna()

        # Merge holidays
        daily_complete = daily_with_weather.merge(
            holidays, on="day", how="left"
        )

        daily_complete["is_holiday"] = daily_complete["holiday_name"].notna()

        # Remove last row
        daily_complete = daily_complete.iloc[:-1]

        return daily_complete

    def _preprocess_nn(self, daily, households, weather_daily, holidays):
        """ACORN-segmented aggregation with seasonally adjusted percentile target."""

        # Merge with household info
        daily_ml = daily.merge(
            households[["LCLid", "Acorn_grouped"]],
            on="LCLid",
            how="left"
        )

        # Filter to valid ACORN groups
        valid_acorn_groups = ["Adversity", "Comfortable", "Affluent"]
        daily_ml = daily_ml[daily_ml["Acorn_grouped"].isin(valid_acorn_groups)]

        # Aggregate by day and Acorn_grouped
        ml_dataset = (
            daily_ml
            .groupby(["day", "Acorn_grouped"])
            .agg(
                total_kwh_per_day=("energy_sum", "sum"),
                num_active_households=("LCLid", "nunique")
            )
            .reset_index()
        )

        ml_dataset["avg_kwh_per_household_per_day"] = (
            ml_dataset["total_kwh_per_day"] / ml_dataset["num_active_households"]
        )

        ml_dataset["day"] = pd.to_datetime(ml_dataset["day"])

        # Temporal features
        ml_dataset["month"] = ml_dataset["day"].dt.month
        ml_dataset["day_of_week"] = ml_dataset["day"].dt.dayofweek

        def get_season(month):
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            else:
                return "Fall"

        ml_dataset["season"] = ml_dataset["month"].apply(get_season)

        # Remove last row
        ml_dataset = ml_dataset.iloc[:-1]

        # Merge weather
        weather_ml = weather_daily[[
            "day", "temperatureHigh", "temperatureLow",
            "humidity", "windSpeed", "cloudCover", "pressure"
        ]].copy()
        weather_ml["day"] = pd.to_datetime(weather_ml["day"])

        ml_dataset = ml_dataset.merge(weather_ml, on="day", how="left")

        # Merge holidays
        holidays_dt = holidays.copy()
        ml_dataset["day_only"] = ml_dataset["day"].dt.date

        ml_dataset = ml_dataset.merge(
            pd.DataFrame({"day": holidays_dt["day"], "is_holiday": True}),
            left_on="day_only",
            right_on="day",
            how="left",
            suffixes=("", "_holiday")
        )

        ml_dataset["is_holiday"] = ml_dataset["is_holiday"].fillna(False)
        ml_dataset = ml_dataset.drop(
            columns=["day_only", "day_holiday"], errors="ignore"
        )

        ml_dataset = ml_dataset.dropna()

        # Create seasonally adjusted consumption percentile
        ml_dataset["seasonal_consumption_percentile"] = (
            ml_dataset
            .groupby(["Acorn_grouped", "season"])["avg_kwh_per_household_per_day"]
            .rank(pct=True)
        )

        # Bin into 3 consumption level classes
        ml_dataset["consumption_level"] = pd.cut(
            ml_dataset["seasonal_consumption_percentile"],
            bins=[0, 1/3, 2/3, 1.0],
            labels=[0, 1, 2],
            include_lowest=True
        ).astype(int)

        return ml_dataset
