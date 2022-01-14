"""
Generators for numeric data that can be inferred from empiric distribution.
"""
from __future__ import annotations

from typing import List, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .. import helper
from .base import Generator


class QuantileGenerator(Generator):
    """Abstract generator class for numerical data. Do not use directly.

    Replicates the data by drawing from the linear interpolated quantile.

    It initiates a reduced step function to draw values. This is more efficient
    compared to np.quantile if the data contains doublicate values.

    """

    DATA_DTYPES: List[DTypeLike] = [np.int_, np.float_, np.datetime64]

    def __init__(
        self,
        vals: ArrayLike,
        bins: ArrayLike = None,
        dtype: DTypeLike = None,
        na_rate: float = 0.0,
    ):
        """Formal initializer. Consider using :meth:`from_data()` instead.

        Args:
            vals (ArrayLike): Array of data points
            bins (ArrayLike, optional): By default, bins is calculated from a
                linear space. Optionally, an array of bins with same length as
                **vals** can be provided.
            dtype (DTypeLike, optional): By default, the dtype is derived from
                **vals**. Optionally, a valid numpy dtype can be provided.
            na_rate (float, optional): Rate at which NaN occour in the data.
                Must be in [0,1]. Defaults to 0.0.
        """
        _vals = np.asarray(vals, dtype=dtype)
        if len(_vals.shape) != 1:
            raise ValueError("vals must be 1-dimensional")
        _n = _vals.size
        if _n < 2:
            raise ValueError("vals must have at least two valid elements")
        _bins = np.linspace(0, 1, _n) if bins is None else np.asarray(bins)
        if _bins.shape != _vals.shape:
            raise ValueError("vals and bins do not have the same shape")

        _vals = np.sort(_vals)
        _mask = np.r_[True, _vals[2:] - _vals[:-2] != np.full(_n - 2, 0), True]

        self.vals = _vals[_mask]
        self.bins = _bins[_mask]
        self.dtype = dtype or _vals.dtype

        if na_rate < 0 or na_rate > 1:
            raise ValueError("na_rate must be in [0,1]")
        self.na_rate = na_rate

    @classmethod
    def from_data(cls, data: ArrayLike):
        """Initializes a new quantile generator from the provided data. This is
        a convenience interface and the preferred method to create a new
        generator instance.

        Args:
            data (ArrayLike): Set of valid data to create and fit the generator.
        """
        data = cls.validate(data=data)
        vals = np.asarray(data[~np.isnan(data)])
        dtype = data.dtype
        na_rate = 1 - vals.size / data.size
        return cls(vals=vals, dtype=dtype, na_rate=na_rate)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} from empiric quantiles"

    def _make(self, size: int) -> NDArray:
        p = np.random.uniform(0, 1, size)
        return helper.interp(p, self.bins, self.vals).astype(self.dtype)


class Numeric(QuantileGenerator):
    """Generator class recommended to replicate int and float data sets.

    A new generator instance is best initialized via :meth:`from_data()` from a
    given set of numeric data.

    The generator draws new values from an interpolated empirical distribution
    fitted on the provided data and
    """

    DATA_DTYPES = [np.int_, np.float_]

    @property
    def precision(self) -> Union[int, float]:
        """Union[int, float]: Precision of the generator, interpreted as the
        greatest common divisor."""
        return helper.gcd(self.vals)

    def _make(self, size: int) -> NDArray:
        """Hidden maker method. This wraps the quantile _make with a rounding
        function to ensure the values are rounded their initial precision (gcd).

        Args:
            size (int): number of elements in returned array

        Returns:
            NDArray: numeric array rounded to the generators precision
        """
        return helper.roundx(super()._make(size=size), x=self.precision)


class Datetime(QuantileGenerator):
    """Generator class recommended to replicate datetime data.

    A new generator instance is best initialized via :meth:`from_data()` from a
    given set of datetime data.

    The generator draws new values from an interpolated empirical distribution
    fitted on the provided data and
    """

    DATA_DTYPES = [np.datetime64]

    @property
    def precision(self) -> str:
        """str: Datetime precision of the generators, represented as *ns*, *ms*,
        *s*, *m*, *h*, *D*, *M*, and *Y*."""
        return helper.datetime_precision(self.vals)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} from empiric quantiles, "
            f"precision={self.precision}"
        )

    def _make(self, size: int) -> NDArray:
        """Hidden maker method. This wraps the quantile _make with a rounding
        function to ensure the datetimes are rounded their initial precision.

        Args:
            size (int): number of elements in returned array

        Returns:
            NDArray: datetime array rounded to the generators precision
        """
        return super()._make(size=size).astype(f"datetime64[{self.precision}]")
