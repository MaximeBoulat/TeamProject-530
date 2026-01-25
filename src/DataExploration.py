import pandas as pd
import matplotlib.pyplot as plt
import math

from pathlib import Path

from Globals import DATASETS_ROOT

class DataExploration:

    def __init__(self):
        self.df = self._load_data()

        self.df.info()

        pass

    def explore_data(self):

        variables = [
            "energy_min",
            "energy_max",
            "energy_mean",
            "energy_median",
            "energy_sum",
            "energy_std"
        ]

        print("Exploring data...")
        # plt = self._explore_seasonal_variation(variables)
        # plt.show()
        # plt = self._explore_acorn_groups(variables)
        # plt.show()
        # plt = self._explore_tariff_types(variables)
        # plt.show()

        # create df with sum for each day according to Acorn group

        acorn_grouped = self.df.groupby(['day', 'Acorn_grouped']).agg(
            total_energy=('energy_sum', 'sum'),
            num_households=('LCLid', 'nunique')
        )
        acorn_grouped['avg_energy_per_household'] = acorn_grouped['total_energy'] / acorn_grouped['num_households']
        acorn_daily_avg = acorn_grouped['avg_energy_per_household'].unstack().rolling(30).mean()

        acorn_daily_avg.plot()
        plt.show()

        tariff_grouped = self.df.groupby(['day', 'stdorToU']).agg(
            total_energy=('energy_sum', 'sum'),
            num_households=('LCLid', 'nunique')
        )
        tariff_grouped['avg_energy_per_household'] = tariff_grouped['total_energy'] / tariff_grouped['num_households']
        tariff_daily_avg = tariff_grouped['avg_energy_per_household'].unstack().rolling(30).mean()

        tariff_daily_avg.plot()
        plt.show()



    
    def _explore_seasonal_variation(self, variables):

        key = 'season'
        values = ['Winter', 'Spring', 'Summer', 'Fall']

        plt = self._drawBoxplotsByGroup(self.df, variables, key, values, 1)
        return plt

    def _explore_acorn_groups(self, variables):

        key = 'Acorn_grouped'
        values = ['Adversity', 'Comfortable', 'Affluent']

        plt = self._drawBoxplotsByGroup(self.df, variables, key, values, 1)
        return plt
    
    def _explore_tariff_types(self, variables):

        key = 'stdorToU'
        values = ['ToU', 'Std']

        plt = self._drawBoxplotsByGroup(self.df, variables, key, values, 1)
        return plt


    def _load_data(self):
        path = Path(DATASETS_ROOT) / "daily_dataset.csv"
        df = pd.read_csv(path)

        # cleanup
        df["day"] = pd.to_datetime(df["day"])
        df = df.dropna()

        # Add season column

        df['season'] = df['day'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })

        # Add the Acorn and Tariff columns

        path = Path(DATASETS_ROOT) / "informations_households.csv"

        household_info_df = pd.read_csv(path)

        df = df.merge(household_info_df, on='LCLid', how='left')

        # drop ambiguous Acorn groups
        df = df[~df['Acorn_grouped'].isin(['ACORN-U', 'ACORN-'])]

        return df


    
    def _drawBoxplotsByGroup(self, df, variables, group_key, group_values, num_col):
        n_plots = len(variables)
        num_rows = int(math.ceil(n_plots / num_col))
        
        plt.figure(figsize=(12, 12))
        for i in range(n_plots):
            plt.subplot(num_rows, num_col, i+1)

            variable = variables[i]

            data_by_group = self._create_Groups(df, variable, group_key, group_values)

            plt.boxplot(data_by_group, labels=group_values, orientation='horizontal')
            plt.xlabel(variable)
            plt.xticks(rotation=45)

        plt.tight_layout()

        return plt



    def _create_Groups(self, dataset, variable, group_key, group_values):
        
        data_by_group = [dataset[dataset[group_key] == value][variable].dropna() for value in group_values]
        return data_by_group



