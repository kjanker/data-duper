"""
Module containing the abstract base class for all generators.
"""
from __future__ import annotations

from typing import List

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray


class Generator:
    """Abstract base class of the value generators."""

    DATA_DTYPES: List[DTypeLike] = []
    """Accepted data types of the generator."""

    def __init__(self, data: NDArray) -> None:
        self.dtype: DTypeLike = data.dtype
        self.na_rate: float = 0.0

    @classmethod
    def from_data(cls, data: NDArray):
        return cls(data=data)

    @property
    def nan(self):
        """The NA value of the generator, either NaN or NaT."""
        if np.issubdtype(self.dtype, np.datetime64):
            return np.datetime64("NaT")
        else:
            return np.nan

    def _make(self, size: int) -> NDArray:
        """Hidden maker method. This method does the actual work before
        :meth:`make()` continues with the postprocessing.

        Args:
            size (int): number of elements in returned array

        Raises:
            NotImplementedError: method must be overwritten.

        Returns:
            NDArray: valid array of given size
        """
        raise NotImplementedError

    def make(self, size: int, with_na: bool = False) -> NDArray:
        """Creates a new data array of a given size. The values are generatored
        randomly for each execution. NA values can be inserted optionally.

        Args:
            size (int): number of elements in returned array
            with_na (bool, optional): Allows to replicate NA occurrence in data.
                If True, NA values are randomly inserted in the data. The rate
                is fitted from the data. Defaults to False.

        Returns:
            NDArray: array with newly generatored values
        """
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
