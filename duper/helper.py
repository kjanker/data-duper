import numpy as np
from numpy.typing import NDArray


def roundx(a: NDArray, x: int = 10) -> NDArray:
    """
    Rounds values to the closest multiple of x.
    """
    return np.array(x * np.around(a / x)).astype(a.dtype)
