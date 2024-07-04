import numpy as np
import pandas as pd


def find_closest_numeric_column(df, value):
    numeric_columns = {}
    for col in df.columns:
        try:
            numeric_col = float(col)
            numeric_columns[col] = numeric_col
        except ValueError:
            continue

    if not numeric_columns:
        return None

    differences = {col: abs(numeric_col - value)
                   for col, numeric_col in numeric_columns.items()}
    closest_column = min(differences, key=differences.get)
    return closest_column


def can_convert_to_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


@pd.api.extensions.register_dataframe_accessor("s")
class SpectraAccessor:
    def __init__(self, pandas_obj):
        # self._validate(pandas_obj)
        self.df = pandas_obj

    @staticmethod
    def _validate(obj):
        pass

    def get_wns(self):
        return np.array([col for col in self.df.columns
                         if can_convert_to_float(col)]).astype("float")

    def get_spectra(self):
        float_columns = [col for col in self.df.columns
                         if can_convert_to_float(col)]
        return self.df[float_columns].values

    def __getitem__(self, wn):
        return self.df[find_closest_numeric_column(self.df, wn)].values
