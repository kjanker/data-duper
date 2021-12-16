"""
Module containing the abstract base class for all generators.
"""
import numpy as np
from numpy.typing import ArrayLike, NDArray


class Generator:
    """Abstract base class of the value generators."""

    def __init__(self, data: NDArray) -> None:
        self.dtype: np.dtype = data.dtype
        self.na_rate: float = 0.0

    @classmethod
    def from_data(cls, data: NDArray):
        return cls(data=data)

    @property
    def nan(self):
        if np.issubdtype(self.dtype, np.datetime64):
            return np.datetime64("NaT")
        else:
            return np.nan

    def _make(self, size: int) -> NDArray:
        raise NotImplementedError

    def make(self, size: int, with_na: bool = False) -> NDArray:
        data = self._make(size=size).astype(self.dtype)
        if with_na:
            isna = np.random.uniform(size=size) < self.na_rate
            if any(isna):
                if np.issubdtype(self.dtype, np.int_):
                    data = data.astype(np.float_)
                data[isna] = self.nan
        return data

    @staticmethod
    def validate(data: ArrayLike, dtype=None) -> NDArray:

        adata = np.asarray(data)

        if adata.size == 0:
            raise ValueError("data must not be empty")

        if dtype and not np.issubdtype(adata.dtype, dtype):
            raise TypeError(f"data elements must be subdtypes of {dtype}")

        return adata
