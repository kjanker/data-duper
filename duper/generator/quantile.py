"""
Generators for numeric data that can be inferred from empiric distribution.
"""
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .. import helper
from .base import Generator


class QuantileGenerator(Generator):
    """Abstract generator class for numerical data. Do not use directly.

    Replicates the data by drawing from the linear interpolated quantile.

    It initiates a reduced step function to draw values. This is more efficient
    compared to np.quantile if the data contains doublicate values.

    """

    def __init__(
        self,
        vals: ArrayLike,
        bins: ArrayLike = None,
        dtype=None,
        na_rate: float = 0.0,
    ) -> None:

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
        _data = np.asarray(data)
        if _data.size == 0:
            raise ValueError("data must not be empty")
        vals = np.asarray(_data[~np.isnan(_data)])
        dtype = _data.dtype
        na_rate = 1 - vals.size / _data.size
        return cls(vals=vals, dtype=dtype, na_rate=na_rate)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} from empiric quantiles"

    def _make(self, size: int) -> NDArray:
        p = np.random.uniform(0, 1, size)
        return helper.interp(p, self.bins, self.vals).astype(self.dtype)


class Numeric(QuantileGenerator):
    """Generator class recommended to replicate continous float data.

    This is directly based on the meta QuantileGenerator class.

    """

    def __init__(
        self,
        vals: ArrayLike,
        bins: ArrayLike = None,
        dtype=None,
        na_rate: float = 0.0,
    ) -> None:
        super().__init__(vals, bins, dtype, na_rate)

        self.gcd = self.dtype.type(helper.gcd_float(self.vals))

    @classmethod
    def from_data(cls, data: ArrayLike):
        _data = np.asarray(data)
        if not (
            np.issubdtype(_data.dtype, np.float_)
            or np.issubdtype(_data.dtype, np.int_)
        ):
            raise TypeError("data elements must be of numerical dtype")
        return super().from_data(data=_data)

    def _make(self, size: int) -> NDArray:
        return helper.roundx(super()._make(size=size), x=self.gcd)


class Float(QuantileGenerator):
    """DEPRECATED - Use Numeric instead

    Generator class recommended to replicate continous float data.

    This is directly based on the meta QuantileGenerator class.

    """

    def __init__(
        self,
        vals: ArrayLike,
        bins: ArrayLike = None,
        dtype=None,
        na_rate: float = 0.0,
    ) -> None:
        super().__init__(vals, bins, dtype, na_rate)

        self.gcd = helper.gcd_float(self.vals)

    @classmethod
    def from_data(cls, data: ArrayLike):
        _data = np.asarray(data)
        if not np.issubdtype(_data.dtype, np.float_):
            raise TypeError("data elements must be of floating dtype")
        return super().from_data(data=_data)

    def _make(self, size: int) -> NDArray:
        return helper.roundx(super()._make(size=size), x=self.gcd)


class Integer(QuantileGenerator):
    """DEPRECATED - Use Numeric instead

    Generator class recommended to replicate integer data.

    This is based on the meta QuantileGenerator class.

    """

    def __init__(
        self,
        vals: ArrayLike,
        bins: ArrayLike = None,
        dtype=None,
        na_rate: float = 0.0,
    ) -> None:
        super().__init__(vals, bins, dtype, na_rate)

        self.gcd = np.gcd.reduce(self.vals)

    @classmethod
    def from_data(cls, data: ArrayLike):
        _data = np.asarray(data)
        if not np.issubdtype(_data.dtype, np.int_):
            raise TypeError("data elements must be of integer dtype")
        return super().from_data(data=_data)

    def _make(self, size: int) -> NDArray:
        return helper.roundx(super()._make(size=size), x=self.gcd)


class Datetime(QuantileGenerator):
    """Generator class recommended to replicate datetime data.

    This is based on the meta QuantileGenerator class.

    """

    def __init__(
        self,
        vals: ArrayLike,
        bins: ArrayLike = None,
        dtype=None,
        na_rate: float = 0.0,
    ) -> None:
        super().__init__(vals, bins, dtype, na_rate)

        self.freq = helper.datetime_precision(self.vals)

    @classmethod
    def from_data(cls, data: ArrayLike):
        _data = np.asarray(data)
        if not np.issubdtype(_data.dtype, np.datetime64):
            raise TypeError("data elements must be of datetime dtype")
        return super().from_data(data=_data)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} from empiric quantiles, "
            f"freq={self.freq}"
        )

    def _make(self, size: int) -> NDArray:
        return super()._make(size=size).astype(f"datetime64[{self.freq}]")
