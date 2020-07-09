"""
quant_test.functions.py
~~~~~~~~~~~~~~~~~~~~~~

This module contains an example class and function that can be imported.
"""

import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype


class Adjustments:

    def __init__(self, df: pd.DataFrame):
        self.data = df

    def encode_categorical_vars(self, exclude: list = None) -> pd.DataFrame:
        if not exclude:
            exclude = []
        categorical_columns = []
        for column in self.data.columns:
            if column in exclude:
                continue
            # Ignore dates & numbers
            if is_string_dtype(self.data.dtypes[column]):
                # Check to see if the column looks like categories
                this = self.data[column].value_counts()
                # Disqualify column if there's too many values or they're too disparate
                if (
                        #len(this) > 100 or
                        max(this) < 5 or
                        max(this) / len(self.data) < 0.05
                ):
                    continue

                categorical_columns.append(column)

        if len(categorical_columns):
            dummy_df = pd.get_dummies(self.data, columns=categorical_columns)
            out_df = pd.merge(self.data, dummy_df)
            self.data = out_df
        return self.data

    def normalize_numerical_vars(self, exclude: list = None):
        if not exclude:
            exclude = []
        numeric_columns = []
        for column in self.data.columns:
            if column in exclude:
                continue
            # Ignore dates & strings
            if is_numeric_dtype(self.data.dtypes[column]):
                numeric_columns.append(column)

        if len(numeric_columns):
            self.data[numeric_columns] = self.data[numeric_columns] .transform(self._zscore)

        return self.data

    @staticmethod
    def _zscore(series: pd.Series) -> float:
        return (series - series.mean()) / series.std(ddof=0)
