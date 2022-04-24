"""
Module containing the abstract base class for all generators.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, DTypeLike, NDArray


class Generator:
    """Abstract base class of the value generators."""

    DATA_DTYPES: List[DTypeLike] = []
    """Accepted data types of the generator."""

    dtype: DTypeLike = None
    """Data type of the generated array."""

    na_rate: float = 0.0
    """Ratio of NA values in generated array."""

    @classmethod
    def from_data(cls, data: NDArray):
        raise NotImplementedError

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

    def make(self, size: int, with_na: bool = True) -> pd.Series:
        """Creates a new data array of a given size. The values are generatored
        randomly for each execution. NA values can be inserted optionally.

        Args:
            size (int): number of elements in returned array
            with_na (bool, optional): Allows to replicate NA occurrence in data.
                If True, NA values are randomly inserted in the data. The rate
                is fitted from the data. Defaults to True.

        Returns:
            pd.Series: array with newly generatored values
        """
        if with_na:
            is_value = np.random.uniform(size=size) > self.na_rate
            s = pd.Series(data=np.empty(size), dtype=self.dtype)
            s.loc[is_value] = pd.Series(
                data=self._make(size=sum(is_value)), dtype=self.dtype
            )
            s.loc[~is_value] = pd.NA
            return s
        else:
            return pd.Series(data=self._make(size=size), dtype=self.dtype)

    @classmethod
    def validate(cls, data: ArrayLike) -> NDArray:
        """Validates the provided data to be processed in :meth:`from_data()`.
        The data is checked to be not empty and of the correct type.

        Args:
            data (ArrayLike): data set of a type matching the generator

        Raises:
            ValueError: if data set is empty
            TypeError: if data type does not match the generator

        Returns:
            NDArray: input data as valid NDArray
        """
        data = np.asarray(data)

        if data.size == 0:
            raise ValueError("data must not be empty")

        if len(cls.DATA_DTYPES) > 0 and not any(
            [np.issubdtype(data.dtype, dt) for dt in cls.DATA_DTYPES]
        ):
            dtypes_repr = ", ".join(map(str, cls.DATA_DTYPES))
            raise TypeError(f"data must be subdtypes of {dtypes_repr}")

        return data
