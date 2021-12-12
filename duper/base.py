"""
The core module of data-duper.
"""
from typing import Dict, Hashable, List

import pandas as pd
from numpy.typing import DTypeLike

from . import analysis
from .generator.base import Generator


class Duper:
    """The main class of data-duper. Use this to fit a data set and dupe it."""

    def __init__(self) -> None:
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
        return self._generators

    @property
    def columns(self) -> List[Hashable]:
        return list(self.generators.keys())

    @property
    def dtypes(self) -> Dict[Hashable, DTypeLike]:
        return {k: v.dtype for k, v in self.generators.items()}

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
        self._generators = {}
        for col in df.columns:
            self._generators[col] = analysis.fit_generator(
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

        for col in self.columns:
            try:
                df[col] = self.generators[col].make(size=size, with_na=with_na)
            except Exception as e:
                raise Exception(*e.args, f"column='{col}'")

        return df
