"""
A generator for data containing a single value.
"""
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .base import Generator


class Constant(Generator):
    """Simplest generator method. Replicates data with a constant value."""

    def __init__(self, value, dtype=None, na_rate: float = 0.0) -> None:
        self.value = value
        self.dtype = dtype if dtype else np.array(value).dtype
        self.na_rate = na_rate

    @classmethod
    def from_data(cls, data: NDArray):
        """Init class from data set."""

        Generator.validate(data=data)
        dtype = data.dtype
        # using pandas isna here since dtype might be object/string
        na_rate = sum(pd.isna(data)) / len(data)
        unique_values = pd.unique(data[~pd.isna(data)])

        if len(unique_values) > 1:
            raise ValueError("Cannot be inferred from non constant data.")

        if len(unique_values) < 1:
            unique_values = pd.unique(data)

        return cls(value=unique_values[0], dtype=dtype, na_rate=na_rate)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} with value '{self.value}'"

    def _make(self, size: int) -> NDArray:
        return np.full(shape=size, fill_value=self.value)
