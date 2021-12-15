"""
Generators for numeric data that can be inferred from empiric distribution.
"""
import numpy as np
from numpy.typing import NDArray

from .. import helper
from .base import Generator


class QuantileGenerator(Generator):
    """Abstract generator class for numerical data. Do not use directly.

    Replicates the data by drawing from the linear interpolated quantile.

    It initiates a reduced step function to draw values. This is more efficient
    compared to np.quantile if the data contains doublicate values.

    """

    def __init__(
        self, bins: NDArray, vals: NDArray, dtype=None, na_rate: float = 0.0
    ) -> None:
        self.bins = bins
        self.vals = vals
        self.dtype = dtype if dtype else vals.dtype
        self.na_rate = na_rate

    @classmethod
    def from_data(cls, data: NDArray):

        Generator.validate(data=data)
        dtype = data.dtype
        na_rate = sum(np.isnan(data)) / len(data)

        vals = np.sort(data[~np.isnan(data)])
        n = len(vals)
        bins = np.linspace(0, 1, n)
        mask = np.r_[False, vals[2:] - vals[:-2] == np.full(n - 2, 0), False]
        return cls(
            bins=bins[~mask], vals=vals[~mask], dtype=dtype, na_rate=na_rate
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__} from empiric quantiles"

    def _make(self, size: int) -> NDArray:
        p = np.random.uniform(0, 1, size)
        i = np.searchsorted(self.bins, p)
        return (p - self.bins[i - 1]) / (self.bins[i] - self.bins[i - 1]) * (
            self.vals[i] - self.vals[i - 1]
        ) + self.vals[i - 1]


class Float(QuantileGenerator):
    """Generator class recommended to replicate continous float data.

    This is directly based on the meta QuantileGenerator class.

    """

    pass


class Integer(QuantileGenerator):
    """Generator class recommended to replicate integer data.

    This is based on the meta QuantileGenerator class.

    """

    def __init__(
        self, bins: NDArray, vals: NDArray, dtype=None, na_rate: float = 0.0
    ) -> None:
        super().__init__(bins, vals, dtype, na_rate)

        self.gcd = np.gcd.reduce(self.vals)

    def _make(self, size: int) -> NDArray:
        return helper.roundx(super()._make(size=size), x=self.gcd)


class Datetime(QuantileGenerator):
    """Generator class recommended to replicate datetime data.

    This is based on the meta QuantileGenerator class.

    """

    def __init__(
        self, bins: NDArray, vals: NDArray, dtype=None, na_rate: float = 0.0
    ) -> None:
        super().__init__(bins, vals, dtype, na_rate)

        self.freq = "ns"
        for freq in ["ms", "s", "m", "h", "D", "M", "Y"]:
            if any(vals != vals.astype(f"datetime64[{freq}]")):
                break
            else:
                self.freq = freq

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} from empiric quantiles, "
            f"freq={self.freq}"
        )

    def _make(self, size: int) -> NDArray:
        return super()._make(size=size).astype(f"datetime64[{self.freq}]")
