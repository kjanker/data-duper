"""
Module containing the abstract base class for all generators.
"""
import numpy as np
from numpy.typing import NDArray


class Generator:
    """Abstract base class of the value generators."""

    def __init__(self, data: NDArray) -> None:
        self.dtype: np.dtype = data.dtype
        self.na_rate: float = 0.0
        self.nan = (
            np.datetime64("NaT")
            if np.issubdtype(self.dtype, np.datetime64)
            else np.nan
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def _make(self, size: int) -> NDArray:
        pass

    def make(self, size: int, with_na: bool = False) -> NDArray:
        data = self._make(size=size).astype(self.dtype)
        if with_na:
            isna = np.random.uniform(size=size) < self.na_rate
            data[isna] = self.nan
        return data
