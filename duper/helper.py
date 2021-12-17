from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


def roundx(a: ArrayLike, x: Union[int, float] = 1) -> NDArray:
    """
    Rounds values to the closest multiple of x.
    """
    a = np.asarray(a)
    return np.around(x * np.around(a / x), number_precision((x,)))


def number_precision(a: ArrayLike, max: int = 16) -> int:
    """
    Returns the decimal index to which rounding does not change the values.

    This can be used to count decimals of floats.
    """
    a = np.asarray(a)[~np.isnan(np.asarray(a))]
    d = -int(np.ceil(np.log10(np.abs(np.amax(a)))))
    while any(a != np.around(a, d)) and d < max:
        d += 1
    return d


def gcd_float(a: ArrayLike) -> float:
    """
    A float generalisation of the greatest common divisor.
    """
    a = np.asarray(a)
    n = number_precision(a)
    int_repr = np.around(a * (10 ** n)).astype(np.int_)
    return np.around(np.gcd.reduce(int_repr) / (10 ** n), decimals=n)


def datetime_precision(a: ArrayLike) -> str:
    """
    Returns the highest datetime precision present in the data.
    """
    a = np.asarray(a)
    freq = "ns"
    for f in ["ms", "s", "m", "h", "D", "M", "Y"]:
        if any(a != a.astype(f"datetime64[{f}]")):
            break
        else:
            freq = f
    return freq


def interp(x: ArrayLike, xp: ArrayLike, fp: ArrayLike) -> NDArray:
    """
    One-dimensional linear interpolation for monotonically increasing points.

    Note: numpy.interp does not work with datetime values.
    """
    x = np.asarray(x)
    xp = np.asarray(xp)
    fp = np.asarray(fp)
    # TODO: add checks and exceptions
    i = np.searchsorted(xp, x)
    return np.asarray(
        (x - xp[i - 1]) / (xp[i] - xp[i - 1]) * (fp[i] - fp[i - 1]) + fp[i - 1]
    )
