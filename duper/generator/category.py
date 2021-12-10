"""
A generator for data with few different values, e.g. category, status.
"""
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .base import Generator


class Category(Generator):
    """Recommended for string data of few different values.
    Draws values based on their occurence in the data.

    """

    def __init__(self, data: NDArray) -> None:
        super().__init__(data=data)
        self._choices = pd.value_counts(data, normalize=True, dropna=False)

    @classmethod
    def from_data(cls, data: NDArray):
        return cls(data=data)

    def __str__(self) -> str:
        catagories_str = self.categories(with_na=True).to_string(
            header=False,
            index=True,
            length=False,
            dtype=False,
            name=False,
            na_rep="NA",
        )
        return (
            f"{self.__class__.__name__} with "
            f"{len(self.categories(with_na=True))} values:\n"
            f"{catagories_str}"
        )

    def categories(self, with_na=True) -> pd.Series:
        if with_na:
            return self._choices
        else:
            choices = self._choices[~self._choices.index.isna()]
            return choices / choices.sum()

    def make(self, size: int, with_na=False) -> NDArray:
        return np.random.choice(
            a=self.categories(with_na=with_na).index,
            size=size,
            p=self.categories(with_na=with_na),
        )
