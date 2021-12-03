"""
base.py
====================================
The core module of my example project.
"""
from typing import Hashable, List
import numpy as np
import pandas as pd

from .methods import (CategoryDuper, ConstantDuper, DatetimeDuper, FloatDuper,
                      IntDuper, RegExDuper)


class Duper(object):
    """The main class of data-duper. Use this to fit a data set and dupe it."""

    def __init__(self):
        self._columns = []
        self.dtypes = {}
        self.methods = {}

    def __str__(self) -> str:
        if len(self.methods) > 0:
            descr = pd.DataFrame(index=self.columns)
            descr["dtypes"] = descr.index.map(
                {k: v.__str__() for k, v in self.dtypes.items()}
            )
            descr["methods"] = descr.index.map(
                {k: v.__repr__() for k, v in self.methods.items()}
            )
            return (
                f"{self.__class__.__name__}, {len(self.methods)} columns:\n"
                f"{descr}"
            )
        else:
            return f"{self.__class__.__name__}, unfitted"

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @property
    def columns(self) -> List[Hashable]:
        return self._columns

    def fit(self, df: pd.DataFrame, category_threshold: float = 0.05):
        """
        Fit the data generator on a provided dataset.

        Parameters
        ---------
        df: pd.DataFrame
            training dataset with realistic data.
        category_threshold: float, default=0.05
            Fraction of unique values until which category duper is perferred,
            should be in [0,1].
        """
        self._columns = list(df.columns)
        self.dtypes = df.dtypes.to_dict()

        self.methods = {}
        for col in df.columns:

            value_counts = df[col].value_counts(dropna=False)
            unique_values = value_counts.index[~value_counts.index.isna()]

            if len(unique_values) == 0:
                self.methods[col] = ConstantDuper(
                    value=df[col].unique()[0], na_rate=1.0
                )

            elif len(unique_values) == 1:
                na_rate = value_counts.get(np.nan, 0) / len(df[col])
                self.methods[col] = ConstantDuper(
                    value=unique_values[0], na_rate=na_rate
                )

            elif (
                self.dtypes[col].name == "bool"
                or len(unique_values) / len(df) < category_threshold
            ):
                self.methods[col] = CategoryDuper(data=df[col].values)

            elif self.dtypes[col].name.startswith("float"):
                self.methods[col] = FloatDuper(data=df[col])

            elif self.dtypes[col].name.startswith("int"):
                self.methods[col] = IntDuper(data=df[col])

            elif self.dtypes[col].name.startswith("datetime"):
                self.methods[col] = DatetimeDuper(data=df[col])

            elif (
                self.dtypes[col].name in ["string", "object"]
                and len(set(map(len, unique_values))) == 1
            ):
                self.methods[col] = RegExDuper(data=df[col])

            else:
                self.methods[col] = CategoryDuper(data=df[col].values)

    def make(self, size=0, with_na=False) -> pd.DataFrame:
        """Create a new random pandas DataFrame after fitting the generator.

        Parameters
        ---------

        Returns
        ---------
        d.DataFrame
            Data set of a similar structure as the fitted one.
        """
        df = pd.DataFrame(data=None, index=None, columns=self.columns).astype(
            self.dtypes
        )

        for col in df.columns:
            try:
                df[col] = self.methods[col].make(n=size, with_na=with_na)
            except Exception as e:
                raise Exception(*e.args, f"column='{col}'")

        return df
