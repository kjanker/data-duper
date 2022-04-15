"""
A generator for data with few different values, e.g. category, status.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike, NDArray

from .base import Generator


class Category(Generator):
    """Recommended for string data of few different values.
    Can be used to replicate, e.g. category or status. Draws values based on
    their occurence in the data.
    """

    def __init__(
        self,
        vals: NDArray,
        p: NDArray,
        dtype: DTypeLike = None,
        na_rate: float = 0.0,
    ):
        """Formal initializer. Consider using :meth:`from_data()` instead.

        Args:
            vals (NDArray): 1-dimensional array of value options
            p (NDArray): propability array of floats, same shape as **vals**
            dtype (DTypeLike, optional): by default, dtype is derived from
                **vals**. Optionally, a valid numpy dtype can be provided.
            na_rate (float, optional): Rate at which NA occour in the data.
                Must be in [0,1]. Defaults to 0.0.
        """
        vals = np.asarray(vals, dtype=dtype) if dtype else np.asarray(vals)
        p = np.asarray(p, dtype=np.float_)

        if len(vals.shape) != 1:
            raise ValueError("vals must be 1-dimensional")
        if p.shape != vals.shape:
            raise ValueError("vals and bins do not have the same shape")

        self._choices = pd.Series(p, index=vals)
        self.dtype = dtype or np.array(vals).dtype
        if na_rate < 0 or na_rate > 1:
            raise ValueError("na_rate must be in [0,1]")
        self.na_rate = na_rate

    @classmethod
    def from_data(cls, data: NDArray):
        """Initializes a new category generator from the provided data. This is
        a convenience interface and the preferred method to create a new
        generator instance.

        Args:
            data (ArrayLike): Set of valid data to create and fit the generator.
        """
        data = cls.validate(data=data)
        _choices = pd.value_counts(data, normalize=True, dropna=True)
        na_rate = pd.isna(data).sum() / data.size
        return cls(vals=_choices.index, p=_choices, na_rate=na_rate)

    def __str__(self) -> str:
        catagories_str = self.categories.to_string(
            header=False,
            index=True,
            length=False,
            dtype=False,
            name=False,
            na_rep="NA",
        )
        return (
            f"{self.__class__.__name__} with "
            f"{len(self.categories)} values:\n"
            f"{catagories_str}"
        )

    @property
    def categories(self) -> pd.Series:
        """pd.Series: lists values and propability"""
        return self._choices

    def _make(self, size: int) -> NDArray:
        """Hidden maker method. Creates an array of the given **size** by
        choosing values with their given propability.

        Args:
            size (int): number of elements in returned array.

        Returns:
            NDArray: 1-d array composed from the generator's **vals**.
        """
        return np.random.choice(
            a=self.categories.index,
            size=size,
            p=self.categories,
        )
