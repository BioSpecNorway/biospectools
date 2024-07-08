import numpy as np
import pandas as pd


def find_closest_numeric_column(df, wn_value):
    num_cols = {}
    for col in df.columns:
        try:
            num_cols[col] = float(col)
        except ValueError:
            continue

    if not num_cols:
        return None

    return list(num_cols.keys())[
           np.argmin(np.abs(np.array(list(num_cols.values())) - wn_value))]


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

    def _get_float_columns(self):
        return [col for col in self.df.columns if can_convert_to_float(col)]

    def get_wns(self):
        return np.array(self._get_float_columns()).astype("float")

    def get_spectra(self):
        return self.df[self._get_float_columns()].values

    def __getitem__(self, wn_value):

        if type(wn_value) is not tuple:
            return self.df[find_closest_numeric_column(self.df, wn_value)].values

        wn_list = []
        for val in wn_value:
            wn_list.append(find_closest_numeric_column(self.df, val))
        return self.df[wn_list].values




