"""
A generator for data containing a single value.
"""
import numpy as np
from numpy.typing import NDArray

from .base import Generator


class Constant(Generator):
    """Simplest generator method. Replicates data with a constant value."""

    def __init__(self, value, na_rate: float = 0.0) -> None:
        super().__init__(data=np.array(value))
        self.value = value
        self.na_rate = na_rate

    def __str__(self) -> str:
        return f"{self.__class__.__name__} with value '{self.value}'"

    def _make(self, size: int) -> NDArray:
        return np.full(shape=size, fill_value=self.value)
