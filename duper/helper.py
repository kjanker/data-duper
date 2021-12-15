import numpy as np
from numpy.typing import NDArray


def roundx(a: NDArray, x: int = 10) -> NDArray:
    """
    Rounds values to the closest multiple of x.
    """
    return np.array(x * np.around(a / x)).astype(a.dtype)


def datetime_precision(a: NDArray) -> str:
    """
    Returns the highest datetime precision present in the data.
    """
    freq = "ns"
    for f in ["ms", "s", "m", "h", "D", "M", "Y"]:
        if any(a != a.astype(f"datetime64[{f}]")):
            break
        else:
            freq = f
    return freq
