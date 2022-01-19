"""
The core module of data-duper.
"""
from __future__ import annotations

from typing import Dict, Hashable, List

import pandas as pd
from numpy.typing import DTypeLike

from . import analysis
from .generator.base import Generator


class Duper:
    """The main class of data-duper. Use this to fit a data set and dupe it.

    Start by fitting the duper to your real pandas DataFrame, :meth:`fit()`.
    Afterwards, you can generate new similar DataFrames of any size via
    :meth:`make()`.

    It is also possible to review the fitted duper and change the generator
    of single columns.

    Examples:
        >>> df_real = ... # your pandas DataFrame
        >>> duper = Duper()
        >>> duper.fit(df=df_real)
        >>> df_dupe = duper.make(size=10000)
        >>> print(df_dupe)
    """

    def __init__(self):
        """Creates a new empty duper instance."""
        self._generators: Dict[Hashable, Generator] = {}

    def __getitem__(self, item: Hashable):
        return self.generators[item]

    def __setitem__(self, item: Hashable, value: Generator):
        if isinstance(value, Generator):
            self.generators[item] = value
        else:
            raise TypeError(f"value should be an instance of {Generator}")

    def __str__(self) -> str:
        if len(self.generators) > 0:
            descr = pd.DataFrame(index=self.columns)
            descr["dtypes"] = descr.index.map(
                {k: v.__str__() for k, v in self.dtypes.items()}
            )
            descr["generators"] = descr.index.map(
                {k: v.__repr__() for k, v in self.generators.items()}
            )
            return (
                f"{self.__class__.__name__}, {len(self.generators)} columns:\n"
                f"{descr}"
            )
        else:
            return f"{self.__class__.__name__}, unfitted"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    @property
    def generators(self) -> Dict[Hashable, Generator]:
        """Dict: generators of the duper's columns."""
        return self._generators

    @property
    def columns(self) -> List[Hashable]:
        """List: names of the duper's columns."""
        return list(self.generators.keys())

    @property
    def dtypes(self) -> Dict[Hashable, DTypeLike]:
        """Dict: dtypes of the duper's columns."""
        return {k: v.dtype for k, v in self.generators.items()}

    def fit(self, df: pd.DataFrame, category_threshold: float = 0.05) -> None:
        """Fit the duper on the provided pandas DataFrame. First, the
        best generator is derived from data and type of each column. Second, the
        generator is fitted to the data.

        Args:
            df (pd.DataFrame): pandas DataFrame with your data.
            category_threshold (float, optional): fraction of unique values
                until which category duper is perferred, should be in [0,1].
                Defaults to 0.05.
        """
        self._generators = {}
        for col in df.columns:
            data = df[col]
            self._generators[col] = analysis.find_best_generator(
                data=data, category_threshold=category_threshold
            ).from_data(data)

    def make(self, size: int, with_na: bool = False) -> pd.DataFrame:
        """Create a new random pandas DataFrame after fitting the generator.

        Args:
            size (int): the number of rows in the new data frame
            with_na (bool, optional): NA values can be replicated to the new
            data frame. Defaults to False.

        Returns:
            pd.DataFrame: generated new data set
        """
        df = pd.DataFrame(data=None, index=None, columns=self.columns).astype(
            self.dtypes
        )

        for col in self.columns:
            try:
                df[col] = self.generators[col].make(size=size, with_na=with_na)
            except Exception as e:
                raise Exception(*e.args, f"column='{col}'")

        return df
