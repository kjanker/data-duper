"""
base.py
====================================
The core module of my example project.
"""
from typing import Dict, Hashable, List

import pandas as pd
from numpy.typing import DTypeLike

from .analysis import choose_method
from .methods import BaseDuper


class Duper:
    """The main class of data-duper. Use this to fit a data set and dupe it."""

    def __init__(self) -> None:
        self._columns = []
        self._dtypes = {}
        self._methods = {}

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    @property
    def columns(self) -> List[Hashable]:
        return self._columns

    @property
    def dtypes(self) -> Dict[Hashable, DTypeLike]:
        return self._dtypes

    @property
    def methods(self) -> Dict[Hashable, BaseDuper]:
        return self._methods

    def fit(self, df: pd.DataFrame, category_threshold: float = 0.05) -> None:
        """Fit the data generator on a provided dataset.

        Parameters
        ---------
        df: pd.DataFrame
            training dataset with realistic data.
        category_threshold: float, default=0.05
            Fraction of unique values until which category duper is perferred,
            should be in [0,1].
        """
        self._columns = list(df.columns)
        self._dtypes = df.dtypes.to_dict()

        self._methods = {}
        for col in df.columns:
            self._methods[col] = choose_method(
                data=df[col], category_threshold=category_threshold
            )

    def make(self, size: int, with_na: bool = False) -> pd.DataFrame:
        """Create a new random pandas DataFrame after fitting the generator.

        Parameters
        ---------
        size: int
            the size of the new data frame
        with_na: bool, default=False
            defines, if NA values will be replicated to the new data frame

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
