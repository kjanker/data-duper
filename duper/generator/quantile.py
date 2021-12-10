"""
Generators for numeric data that can be inferred from empiric distribution.
"""
import numpy as np
from numpy.typing import NDArray

from .base import Generator


class QuantileGenerator(Generator):
    """Abstract generator class for numerical data. Do not use directly.

    Replicates the data by drawing from the linear interpolated quantile.

    """

    def __init__(self, data: NDArray) -> None:
        super().__init__(data=data)
        self.data = data[~np.isnan(data)]
        self.na_rate = 1 - len(self.data) / len(data)

    @classmethod
    def from_data(cls, data: NDArray):
        return cls(data=data)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} from empiric quantiles"

    def _make(self, size: int) -> NDArray:
        p = np.random.uniform(0, 1, size)
        return np.quantile(self.data, p, interpolation="linear")


class Float(QuantileGenerator):
    """Generator class recommended to replicate continous float data.

    This is directly based on the meta QuantileGenerator class.

    """

    pass


class Integer(QuantileGenerator):
    """Generator class recommended to replicate integer data.

    This is based on the meta QuantileGenerator class.

    """

    def _make(self, size: int) -> NDArray:
        return super()._make(size=size).round()


class Datetime(QuantileGenerator):
    """Generator class recommended to replicate datetime data.

    This is based on the meta QuantileGenerator class.

    """

    def __init__(self, data: NDArray[np.datetime64], freq: str = None) -> None:
        super().__init__(data=data)
        if freq is None:
            self._set_auto_freq()
        else:
            self.data = self.data.astype(f"datetime64[{freq}]")
            self.freq = freq

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__} from empiric quantiles, "
            f"freq={self.freq}"
        )

    def _set_auto_freq(self) -> None:
        """Derive datetime frequency dtype from data"""
        self.freq = "ns"
        for freq in ["ms", "s", "m", "h", "D", "M", "Y"]:
            if any(self.data != self.data.astype(f"datetime64[{freq}]")):
                break
            else:
                self.data = self.data.astype(f"datetime64[{freq}]")
                self.freq = freq

    def _make(self, size: int) -> NDArray:
        return super()._make(size=size).astype(f"datetime64[{self.freq}]")
