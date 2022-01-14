"""
A generator for data containing a single value.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import DTypeLike, NDArray

from .base import Generator


class Constant(Generator):
    """A simple constant generator. Replicates data with a single constant
    value (except for NA).
    """

    def __init__(
        self, value: Any, dtype: DTypeLike = None, na_rate: float = 0.0
    ):
        """Initializes a new generator based on the provided constant **value**.

        Args:
            value (Any): constant value of the generator
            dtype (DTypeLike, optional): By default, the dtype is derived from
                **value**. Optionally, a valid numpy dtype can be provided.
            na_rate (float, optional): Rate at which NA occour in the data.
                Must be in [0,1]. Defaults to 0.0.
        """
        if dtype:
            np.array(value, dtype=dtype)
        self.dtype = dtype or np.array(value).dtype
        self.value = value
        self.na_rate = na_rate

    @classmethod
    def from_data(cls, data: NDArray):
        """Initializes a new generator from the provided **data**. This is
        a convenience interface and the preferred method to create a new
        generator instance.

        Args:
            data (ArrayLike): set of valid data to create and fit the generator.

        Raise:
            ValueError: if data holds more than one unique value that is not NA.
        """
        data = cls.validate(data=data)
        # using pandas isna here since dtype might be object/string
        na_rate = sum(pd.isna(data)) / len(data)
        unique_values = pd.unique(data[~pd.isna(data)])

        if len(unique_values) > 1:
            raise ValueError("Cannot be inferred from non constant data.")

        if len(unique_values) < 1:
            unique_values = pd.unique(data)

        return cls(value=unique_values[0], dtype=data.dtype, na_rate=na_rate)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} with value '{self.value}'"

    def _make(self, size: int) -> NDArray:
        """Hidden maker method. Simply creates an array of the given **size**
        repeating the generator's constant value.

        Args:
            size (int): number of elements in returned array.

        Returns:
            NDArray: array with one unique value.
        """
        return np.full(shape=size, fill_value=self.value)
